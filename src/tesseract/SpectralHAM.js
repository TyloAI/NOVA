// N.O.V.A. Tesseract — GPU-backed holographic associative memory
// This wraps the spectral WGSL kernels: binding, superposition, recall, and mask pruning.

import { NovaTensor } from "../core/NovaTensor.js";
import { ToposManager } from "../layers/ToposManager.js";
import { PhaseInjector } from "./PhaseInjector.js";
import { Positioner } from "./Positioner.js";

export class SpectralHAM {
  constructor(ctx, opts = {}) {
    this.ctx = ctx;
    this.device = ctx.device;
    this.queue = ctx.queue;
    this.dModel = opts.dModel ?? 1024;
    this.injector = new PhaseInjector(this.device, this.dModel);
    this.positioner = opts.positioner ?? new Positioner(this.device, this.dModel);
    this.topos = new ToposManager(this.dModel, {
      maxLayers: opts.maxLayers ?? 8,
      threshold: opts.threshold ?? 0.05
    });

    this.state = null;
    this.recallBuf = null;
    this.maskBuf = null;

    this.meta = null;
    this.maskMeta = null;

    this.encodePipeline = null;
    this.recallPipeline = null;
    this.maskPipeline = null;
    this.decayPipeline = null;
    this.decayFactor = opts.decay ?? 0.9; // stronger default decay to suppress stale interference
    this.decayMeta = null;
  }

  async init() {
    const holoCode = await fetch(new URL("../ops/holographic.wgsl", import.meta.url)).then((r) => r.text());
    const maskCode = await fetch(new URL("../ops/consistency.wgsl", import.meta.url)).then((r) => r.text());
    const decayCode = await fetch(new URL("../ops/state_decay.wgsl", import.meta.url)).then((r) => r.text());

    this.encodePipeline = this.device.createComputePipeline({
      label: "tesseract-encode-bind",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: holoCode }), entryPoint: "encode_bind" }
    });
    this.recallPipeline = this.device.createComputePipeline({
      label: "tesseract-recall",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: holoCode }), entryPoint: "recall_decode" }
    });
    this.maskPipeline = this.device.createComputePipeline({
      label: "tesseract-consistency",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: maskCode }), entryPoint: "main" }
    });
    this.decayPipeline = this.device.createComputePipeline({
      label: "tesseract-decay",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: decayCode }), entryPoint: "main" }
    });

    // Complex buffers are stored as [dModel, 2] (real, imag).
    this.state = new NovaTensor(this.device, [this.dModel, 2]);
    this.state.zero();
    this.recallBuf = new NovaTensor(this.device, [this.dModel, 2]);
    this.maskBuf = new NovaTensor(this.device, [this.dModel]);
    this.maskBuf.write(this.topos.currentMask());

    this.meta = this.#makeUniform(new Uint32Array([this.dModel, 0, 0, 0]));
    this.maskMeta = this.#makeUniform(new Uint32Array([this.dModel, 0, 0, 0]));
    this.decayMeta = this.#makeUniform(new Float32Array([this.decayFactor, this.dModel, 0, 0]));
  }

  resetState() {
    this.state.zero();
  }

  setDecay(factor) {
    this.decayFactor = factor;
    this.queue.writeBuffer(this.decayMeta, 0, new Float32Array([this.decayFactor, this.dModel, 0, 0]).buffer);
  }

  #makeUniform(array) {
    const padded = Math.ceil(array.byteLength / 16) * 16;
    const buffer = this.device.createBuffer({
      size: padded,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.queue.writeBuffer(buffer, 0, array.buffer);
    return buffer;
  }

  updateMask() {
    const mask = this.topos.currentMask();
    this.maskBuf.write(mask);
  }

  enterScope(label) {
    this.topos.enterScope(label);
    this.updateMask();
  }

  exitScope() {
    this.topos.exitScope();
    this.updateMask();
  }

  async encode(tokenTensor, positionTensor) {
    const encoder = this.device.createCommandEncoder();

    // Decay stale interference before writing new binding
    if (this.decayPipeline) {
      const decayGroup = this.device.createBindGroup({
        layout: this.decayPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.state.buffer } },
          { binding: 1, resource: { buffer: this.decayMeta } }
        ]
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.decayPipeline);
      pass.setBindGroup(0, decayGroup);
      const groups = Math.ceil(this.dModel / 256);
      pass.dispatchWorkgroups(groups);
      pass.end();
    }
    const bindGroup = this.device.createBindGroup({
      layout: this.encodePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tokenTensor.buffer } },
        { binding: 1, resource: { buffer: positionTensor.buffer } },
        { binding: 2, resource: { buffer: this.state.buffer } },
        { binding: 3, resource: { buffer: this.recallBuf.buffer } }, // unused in encode path
        { binding: 4, resource: { buffer: this.meta } }
      ]
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.encodePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1); // full FFT runs inside a single invocation
    pass.end();
    this.queue.submit([encoder.finish()]);
  }

  async recall(positionTensor) {
    const encoder = this.device.createCommandEncoder();
    const bindGroup = this.device.createBindGroup({
      layout: this.recallPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: positionTensor.buffer } }, // reused slot
        { binding: 1, resource: { buffer: positionTensor.buffer } },
        { binding: 2, resource: { buffer: this.state.buffer } },
        { binding: 3, resource: { buffer: this.recallBuf.buffer } },
        { binding: 4, resource: { buffer: this.meta } }
      ]
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.recallPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    this.queue.submit([encoder.finish()]);
    return this.recallBuf;
  }

  async applyMask() {
    const encoder = this.device.createCommandEncoder();
    const bindGroup = this.device.createBindGroup({
      layout: this.maskPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.state.buffer } },
        { binding: 1, resource: { buffer: this.maskBuf.buffer } },
        { binding: 2, resource: { buffer: this.maskMeta } }
      ]
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.maskPipeline);
    pass.setBindGroup(0, bindGroup);
    const groups = Math.ceil(this.dModel / 256);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.queue.submit([encoder.finish()]);
  }

  /**
   * Convenience bridge: real-domain vector -> complex tensor with phase -> encode.
   */
  async encodeRealVector(realVec, tokenId, positionTensor) {
    const complexTensor = this.injector.tensorFrom(realVec, tokenId);
    await this.encode(complexTensor, positionTensor);
    complexTensor.destroy?.();
  }

  /**
   * Real vector + position index convenience (uses cached positional holograms).
   */
  async encodeRealAtPosition(realVec, tokenId, posIndex) {
    const posTensor = this.positioner.getPositionTensor(posIndex);
    await this.encodeRealVector(realVec, tokenId, posTensor);
  }

  async recallAtPosition(posIndex) {
    const posTensor = this.positioner.getPositionTensor(posIndex);
    return this.recall(posTensor);
  }
}
