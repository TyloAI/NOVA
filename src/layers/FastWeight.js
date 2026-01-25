import { NovaTensor } from "../core/NovaTensor.js";

// WGSL shader definitions

const rmsnormShader = `
struct Vector { data: array<f32>; };
@group(0) @binding(0) var<storage, read> x : Vector;
@group(0) @binding(1) var<storage, read> gamma : Vector;
@group(0) @binding(2) var<storage, read_write> out : Vector;
@group(0) @binding(3) var<uniform> meta : vec4<u32>; // length, padding

var<workgroup> sumSq : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {
  if (wid.x > 0u) { return; }
  let len = meta.x;
  var acc: f32 = 0.0;
  var idx: u32 = lid.x;
  let stride: u32 = 256u;
  while (idx < len) {
    let v = x.data[idx];
    acc = acc + v * v;
    idx = idx + stride;
  }
  sumSq[lid.x] = acc;
  workgroupBarrier();

  var offset: u32 = 128u;
  loop {
    if (offset == 0u) { break; }
    if (lid.x < offset) {
      sumSq[lid.x] = sumSq[lid.x] + sumSq[lid.x + offset];
    }
    offset = offset / 2u;
    workgroupBarrier();
  }
  let meanSq = sumSq[0] / f32(len);
  let scale = inverseSqrt(meanSq + 1e-5);

  var writeIdx: u32 = lid.x;
  while (writeIdx < len) {
    out.data[writeIdx] = x.data[writeIdx] * scale * gamma.data[writeIdx];
    writeIdx = writeIdx + stride;
  }
}
`;

const sigmoidShader = `
struct Vector { data: array<f32>; };
@group(0) @binding(0) var<storage, read_write> vec : Vector;
@group(0) @binding(1) var<uniform> meta : vec4<u32>; // length

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= meta.x) { return; }
  let x = vec.data[idx];
  vec.data[idx] = 1.0 / (1.0 + exp(-x));
}
`;

const addShader = `
struct Vector { data: array<f32>; };
@group(0) @binding(0) var<storage, read> a : Vector;
@group(0) @binding(1) var<storage, read> b : Vector;
@group(0) @binding(2) var<storage, read_write> out : Vector;
@group(0) @binding(3) var<uniform> meta : vec4<u32>; // length

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= meta.x) { return; }
  out.data[idx] = a.data[idx] + b.data[idx];
}
`;

const mulShader = `
struct Vector { data: array<f32>; };
@group(0) @binding(0) var<storage, read> a : Vector;
@group(0) @binding(1) var<storage, read> b : Vector;
@group(0) @binding(2) var<storage, read_write> out : Vector;
@group(0) @binding(3) var<uniform> meta : vec4<u32>; // length

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= meta.x) { return; }
  out.data[idx] = a.data[idx] * b.data[idx];
}
`;

const swishShader = `
struct Vector { data: array<f32>; };
@group(0) @binding(0) var<storage, read_write> x : Vector;
@group(0) @binding(1) var<uniform> meta : vec4<u32>; // length

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= meta.x) { return; }
  let val = x.data[idx];
  x.data[idx] = val / (1.0 + exp(-val));
}
`;

const copyShader = `
struct Vector { data: array<f32>; };
@group(0) @binding(0) var<storage, read> src : Vector;
@group(0) @binding(1) var<storage, read_write> dst : Vector;
@group(0) @binding(2) var<uniform> meta : vec4<u32>; // length

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= meta.x) { return; }
  dst.data[idx] = src.data[idx];
}
`;

// Holographic mixer: complex rotation write + rotated read + gating
const holoMixerShader = `
struct Vector { data: array<f32>; };
@group(0) @binding(0) var<storage, read_write> state : Vector; // complex: [real, imag] interleaved
@group(0) @binding(1) var<storage, read> r : Vector;           // rotation
@group(0) @binding(2) var<storage, read> v : Vector;           // value
@group(0) @binding(3) var<storage, read> g : Vector;           // gate
@group(0) @binding(4) var<storage, read_write> out : Vector;   // output
@group(0) @binding(5) var<uniform> meta : vec4<f32>;           // length, decay

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let len = u32(meta.x);
  if (idx >= len) { return; }
  let angle = r.data[idx];
  let c = cos(angle);
  let s = sin(angle);
  let stateIdx = idx * 2u;
  let real = state.data[stateIdx];
  let imag = state.data[stateIdx + 1u];
  let vreal = v.data[idx];
  let decay = meta.y;
  let writeReal = decay * real + vreal * c;
  let writeImag = decay * imag + vreal * s;
  state.data[stateIdx] = writeReal;
  state.data[stateIdx + 1u] = writeImag;
  let readReal = writeReal * c + writeImag * s;
  out.data[idx] = readReal * g.data[idx];
}
`;

// Symplectic flow: conserve spinor magnitude; rotate old memory before blending new stimulus
const symplecticFlowShader = `
struct ComplexField { data: array<vec2<f32>>; };
struct Vector { data: array<f32>; };
@group(0) @binding(0) var<storage, read_write> state : ComplexField;
@group(0) @binding(1) var<storage, read> rotation : Vector;
@group(0) @binding(2) var<storage, read> stimulus : Vector;
@group(0) @binding(3) var<storage, read> gate : Vector;
@group(0) @binding(4) var<storage, read_write> out : Vector;
@group(0) @binding(5) var<uniform> meta : vec4<f32>; // length, gain, imagScale, readMix

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let len = u32(meta.x);
  if (idx >= len) { return; }
  let theta = rotation.data[idx];
  let g = gate.data[idx];
  let c = cos(theta);
  let s = sin(theta);
  let prev = state.data[idx];
  let rotated = vec2<f32>(
    prev.x * c - prev.y * s,
    prev.x * s + prev.y * c
  );
  let lambda = max(0.0, g) * meta.y;
  let gated = rotated * vec2<f32>(lambda, lambda);
  let stim = stimulus.data[idx];
  let injected = vec2<f32>(stim, stim * meta.z);
  let updated = gated + injected;
  state.data[idx] = updated;
  let readMix = clamp(meta.w, 0.0, 1.0);
  out.data[idx] = updated.x * readMix + updated.y * (1.0 - readMix);
}
`;

// Active inference: update manifold state based on surprise
const activeInferenceShader = `
struct Vector { data: array<f32>; };
@group(0) @binding(0) var<storage, read_write> state : Vector; // complex state
@group(0) @binding(1) var<storage, read> r : Vector;           // rotation
@group(0) @binding(2) var<storage, read> prediction : Vector;  // predicted input
@group(0) @binding(3) var<storage, read> input : Vector;       // real input
@group(0) @binding(4) var<uniform> meta : vec4<f32>;           // length, lr

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let len = u32(meta.x);
  if (idx >= len) { return; }
  let surprise = input.data[idx] - prediction.data[idx];
  let angle = r.data[idx];
  let c = cos(angle);
  let s = sin(angle);
  let stateIdx = idx * 2u;
  let real = state.data[stateIdx];
  let imag = state.data[stateIdx + 1u];
  let lr = meta.y;
  state.data[stateIdx] = real + lr * surprise * c;
  state.data[stateIdx + 1u] = imag + lr * surprise * s;
}
`;

const gatherShader = `
struct Matrix { data: array<f32>; };
@group(0) @binding(0) var<storage, read> embedding : Matrix;
@group(0) @binding(1) var<storage, read_write> out : Matrix;
@group(0) @binding(2) var<uniform> meta : vec4<u32>; // tokenId, dim, padding

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let dim = meta.y;
  if (idx >= dim) { return; }
  let base = meta.x * dim + idx;
  out.data[idx] = embedding.data[base];
}
`;

// FastWeightLayer definition

export class FastWeightLayer {
  constructor(ctx, modelConfig) {
    this.context = ctx;
    this.device = ctx.device;
    this.queue = ctx.queue;

    this.vocabSize = modelConfig.vocabSize;
    this.inputSize = modelConfig.inputSize || modelConfig.vocabSize;
    this.outputSize = modelConfig.outputSize || modelConfig.vocabSize;
    this.dModel = modelConfig.dModel;
    this.ffnHidden = modelConfig.ffnHidden;
    this.thinkingSteps = modelConfig.thinkingSteps || modelConfig.layers || 12;

    this.manifoldDecay = 0.9;
    this.activeInferenceRate = 0.001;
    this.spinorImagScale = 0.25;
    this.spinorReadMix = 1.0;

    this.matmulPipeline = null;
    this.rmsnormPipeline = null;
    this.sigmoidPipeline = null;
    this.swishPipeline = null;
    this.holoMixerPipeline = null;
    this.symplecticFlowPipeline = null;
    this.activeInferencePipeline = null;
    this.addPipeline = null;
    this.mulPipeline = null;
    this.copyPipeline = null;
    this.gatherPipeline = null;

    this.embedding = null;
    this.normGamma = null;

    this.wr = null;
    this.wv = null;
    this.wg = null;
    this.ffnUp1 = null;
    this.ffnUp2 = null;
    this.ffnDown = null;
    this.wPredict = null;

    this.baseManifold = null;
    this.manifoldState = null;

    this.buffers = {};
    this.uniforms = {};
  }

  createUniform(array, label) {
    const padded = Math.ceil(array.byteLength / 16) * 16;
    const buffer = this.device.createBuffer({
      label,
      size: padded,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false
    });
    this.queue.writeBuffer(buffer, 0, array.buffer);
    return buffer;
  }

  dispatch1d(pass, pipeline, bindGroup, length, workgroup = 256) {
    const groups = Math.ceil(length / workgroup);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groups);
  }

  dispatch2d(pass, pipeline, bindGroup, x, y, wgX = 8, wgY = wgX) {
    const gx = Math.ceil(x / wgX);
    const gy = Math.ceil(y / wgY);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(gx, gy, 1);
  }

  dispatchRmsnorm(pass, pipeline, bindGroup) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
  }

  makeBindGroup(pipeline, entries) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries
    });
  }

  setHebbianScalars(decay, learningRate) {
    this.manifoldDecay = decay;
    this.activeInferenceRate = learningRate;
    if (this.uniforms.holoMeta) {
      this.queue.writeBuffer(
        this.uniforms.holoMeta,
        0,
        new Float32Array([this.dModel, this.manifoldDecay, 0, 0]).buffer
      );
    }
    if (this.uniforms.symplecticMeta) {
      this.queue.writeBuffer(
        this.uniforms.symplecticMeta,
        0,
        new Float32Array([this.dModel, this.manifoldDecay, this.spinorImagScale, this.spinorReadMix]).buffer
      );
    }
    if (this.uniforms.activeMeta) {
      this.queue.writeBuffer(
        this.uniforms.activeMeta,
        0,
        new Float32Array([this.dModel, this.activeInferenceRate, 0, 0]).buffer
      );
    }
  }

  resetFastWeights() {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.baseManifold.buffer,
      0,
      this.manifoldState.buffer,
      0,
      this.baseManifold.byteLength
    );
    this.queue.submit([encoder.finish()]);
  }

  commitFastWeightsToBase() {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.manifoldState.buffer,
      0,
      this.baseManifold.buffer,
      0,
      this.manifoldState.byteLength
    );
    this.queue.submit([encoder.finish()]);
  }

  async init() {
    const matmulCode = await fetch(new URL("../ops/matmul.wgsl", import.meta.url))
      .then((r) => r.text())
      .catch(() => {
        return `
struct Matrix { data: array<f32>, };
@group(0) @binding(0) var<storage, read> a : Matrix;
@group(0) @binding(1) var<storage, read> b : Matrix;
@group(0) @binding(2) var<storage, read_write> out : Matrix;
@group(0) @binding(3) var<uniform> dims : vec3<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= dims.x || col >= dims.z) { return; }
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < dims.y; k = k + 1u) {
    acc = acc + a.data[row * dims.y + k] * b.data[k * dims.z + col];
  }
  out.data[row * dims.z + col] = acc;
}
`;
      });

    this.matmulPipeline = this.device.createComputePipeline({
      label: "matmul",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: matmulCode }), entryPoint: "main" }
    });

    this.rmsnormPipeline = this.device.createComputePipeline({
      label: "rmsnorm",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: rmsnormShader }), entryPoint: "main" }
    });

    this.sigmoidPipeline = this.device.createComputePipeline({
      label: "sigmoid",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: sigmoidShader }), entryPoint: "main" }
    });

    this.swishPipeline = this.device.createComputePipeline({
      label: "swish",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: swishShader }), entryPoint: "main" }
    });

    this.holoMixerPipeline = this.device.createComputePipeline({
      label: "holo-mixer",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: holoMixerShader }), entryPoint: "main" }
    });

    this.symplecticFlowPipeline = this.device.createComputePipeline({
      label: "symplectic-flow",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: symplecticFlowShader }), entryPoint: "main" }
    });

    this.activeInferencePipeline = this.device.createComputePipeline({
      label: "active-inference",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: activeInferenceShader }), entryPoint: "main" }
    });

    this.addPipeline = this.device.createComputePipeline({
      label: "vec-add",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: addShader }), entryPoint: "main" }
    });

    this.mulPipeline = this.device.createComputePipeline({
      label: "vec-mul",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: mulShader }), entryPoint: "main" }
    });

    this.copyPipeline = this.device.createComputePipeline({
      label: "vec-copy",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: copyShader }), entryPoint: "main" }
    });

    this.gatherPipeline = this.device.createComputePipeline({
      label: "embedding-gather",
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: gatherShader }), entryPoint: "main" }
    });

    this.embedding = new NovaTensor(this.device, [this.vocabSize, this.dModel]);
    this.embedding.initXavier({ fanIn: this.dModel, fanOut: this.dModel });

    this.normGamma = new NovaTensor(this.device, [this.dModel]).initOnes();

    this.wr = new NovaTensor(this.device, [this.dModel, this.dModel]).initXavier({
      fanIn: this.dModel,
      fanOut: this.dModel
    });
    this.wv = new NovaTensor(this.device, [this.dModel, this.dModel]).initXavier({
      fanIn: this.dModel,
      fanOut: this.dModel
    });
    this.wg = new NovaTensor(this.device, [this.dModel, this.dModel]).initXavier({
      fanIn: this.dModel,
      fanOut: this.dModel
    });

    this.ffnUp1 = new NovaTensor(this.device, [this.ffnHidden, this.dModel]).initXavier({
      fanIn: this.dModel,
      fanOut: this.ffnHidden
    });
    this.ffnUp2 = new NovaTensor(this.device, [this.ffnHidden, this.dModel]).initXavier({
      fanIn: this.dModel,
      fanOut: this.ffnHidden
    });
    this.ffnDown = new NovaTensor(this.device, [this.dModel, this.ffnHidden]).initResidual({
      fanIn: this.ffnHidden,
      fanOut: this.dModel,
      numLayers: this.thinkingSteps
    });

    this.wPredict = new NovaTensor(this.device, [this.dModel, this.dModel]).initXavier({
      fanIn: this.dModel,
      fanOut: this.dModel
    });

    this.baseManifold = new NovaTensor(this.device, [this.dModel, 2]);
    this.baseManifold.initUnitSpinor();
    this.manifoldState = new NovaTensor(this.device, [this.dModel, 2]);
    this.manifoldState.zero();

    this.buffers.input = new NovaTensor(this.device, [this.dModel]);
    this.buffers.target = new NovaTensor(this.device, [this.dModel]);
    this.buffers.stream = new NovaTensor(this.device, [this.dModel]);
    this.buffers.norm = new NovaTensor(this.device, [this.dModel]);
    this.buffers.r = new NovaTensor(this.device, [this.dModel]);
    this.buffers.v = new NovaTensor(this.device, [this.dModel]);
    this.buffers.g = new NovaTensor(this.device, [this.dModel]);
    this.buffers.holoOut = new NovaTensor(this.device, [this.dModel]);
    this.buffers.ffnGate = new NovaTensor(this.device, [this.ffnHidden]);
    this.buffers.ffnValue = new NovaTensor(this.device, [this.ffnHidden]);
    this.buffers.ffnProd = new NovaTensor(this.device, [this.ffnHidden]);
    this.buffers.ffnOut = new NovaTensor(this.device, [this.dModel]);
    this.buffers.cellOut = new NovaTensor(this.device, [this.dModel]);
    this.buffers.resid = new NovaTensor(this.device, [this.dModel]);
    this.buffers.prediction = new NovaTensor(this.device, [this.dModel]);
    this.buffers.logits = new NovaTensor(
      this.device,
      [this.vocabSize],
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );

    this.uniforms.projDims = this.createUniform(
      new Uint32Array([this.dModel, this.dModel, 1, 0]),
      "proj-dims"
    );
    this.uniforms.ffnUpDims = this.createUniform(
      new Uint32Array([this.ffnHidden, this.dModel, 1, 0]),
      "ffn-up-dims"
    );
    this.uniforms.ffnDownDims = this.createUniform(
      new Uint32Array([this.dModel, this.ffnHidden, 1, 0]),
      "ffn-down-dims"
    );
    this.uniforms.predDims = this.createUniform(
      new Uint32Array([this.dModel, this.dModel, 1, 0]),
      "pred-dims"
    );
    this.uniforms.logitDims = this.createUniform(
      new Uint32Array([this.vocabSize, this.dModel, 1, 0]),
      "logit-dims"
    );

    this.uniforms.rmsMeta = this.createUniform(
      new Uint32Array([this.dModel, 0, 0, 0]),
      "rms-meta"
    );
    this.uniforms.sigmoidMeta = this.createUniform(
      new Uint32Array([this.dModel, 0, 0, 0]),
      "sigmoid-meta"
    );
    this.uniforms.addMeta = this.createUniform(
      new Uint32Array([this.dModel, 0, 0, 0]),
      "add-meta"
    );
    this.uniforms.mulMeta = this.createUniform(
      new Uint32Array([this.ffnHidden, 0, 0, 0]),
      "mul-meta"
    );
    this.uniforms.copyMeta = this.createUniform(
      new Uint32Array([this.dModel, 0, 0, 0]),
      "copy-meta"
    );
    this.uniforms.holoMeta = this.createUniform(
      new Float32Array([this.dModel, this.manifoldDecay, 0, 0]),
      "holo-meta"
    );
    this.uniforms.symplecticMeta = this.createUniform(
      new Float32Array([this.dModel, this.manifoldDecay, 0.25, 1.0]),
      "symplectic-meta"
    );
    this.uniforms.activeMeta = this.createUniform(
      new Float32Array([this.dModel, this.activeInferenceRate, 0, 0]),
      "active-meta"
    );
    this.uniforms.gatherInputMeta = this.createUniform(
      new Uint32Array([0, this.dModel, 0, 0]),
      "gather-input"
    );
    this.uniforms.gatherTargetMeta = this.createUniform(
      new Uint32Array([0, this.dModel, 0, 0]),
      "gather-target"
    );

    this.resetFastWeights();
  }

  async forward(inputTensor, tokenId = 0, applyHebbian = false) {
    return this.forwardToken(tokenId, { applyHebbian });
  }

  async forwardToken(tokenId, opts = {}) {
    const { targetTokenId = null, applyHebbian = true, thinkingSteps = this.thinkingSteps } = opts;

    this.queue.writeBuffer(
      this.uniforms.gatherInputMeta,
      0,
      new Uint32Array([tokenId, this.dModel, 0, 0]).buffer
    );
    if (targetTokenId !== null) {
      this.queue.writeBuffer(
        this.uniforms.gatherTargetMeta,
        0,
        new Uint32Array([targetTokenId, this.dModel, 0, 0]).buffer
      );
    }

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    const gatherInput = this.makeBindGroup(this.gatherPipeline, [
      { binding: 0, resource: { buffer: this.embedding.buffer } },
      { binding: 1, resource: { buffer: this.buffers.input.buffer } },
      { binding: 2, resource: { buffer: this.uniforms.gatherInputMeta } }
    ]);
    this.dispatch1d(pass, this.gatherPipeline, gatherInput, this.dModel);

    if (targetTokenId !== null) {
      const gatherTarget = this.makeBindGroup(this.gatherPipeline, [
        { binding: 0, resource: { buffer: this.embedding.buffer } },
        { binding: 1, resource: { buffer: this.buffers.target.buffer } },
        { binding: 2, resource: { buffer: this.uniforms.gatherTargetMeta } }
      ]);
      this.dispatch1d(pass, this.gatherPipeline, gatherTarget, this.dModel);
    }

    const streamInit = this.makeBindGroup(this.copyPipeline, [
      { binding: 0, resource: { buffer: this.buffers.input.buffer } },
      { binding: 1, resource: { buffer: this.buffers.stream.buffer } },
      { binding: 2, resource: { buffer: this.uniforms.copyMeta } }
    ]);
    this.dispatch1d(pass, this.copyPipeline, streamInit, this.dModel);

    for (let t = 0; t < thinkingSteps; t += 1) {
      const preNorm = this.makeBindGroup(this.rmsnormPipeline, [
        { binding: 0, resource: { buffer: this.buffers.stream.buffer } },
        { binding: 1, resource: { buffer: this.normGamma.buffer } },
        { binding: 2, resource: { buffer: this.buffers.norm.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.rmsMeta } }
      ]);
      this.dispatchRmsnorm(pass, this.rmsnormPipeline, preNorm);

      const rProj = this.makeBindGroup(this.matmulPipeline, [
        { binding: 0, resource: { buffer: this.wr.buffer } },
        { binding: 1, resource: { buffer: this.buffers.norm.buffer } },
        { binding: 2, resource: { buffer: this.buffers.r.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.projDims } }
      ]);
      this.dispatch2d(pass, this.matmulPipeline, rProj, this.dModel, 1, 8, 1);

      const vProj = this.makeBindGroup(this.matmulPipeline, [
        { binding: 0, resource: { buffer: this.wv.buffer } },
        { binding: 1, resource: { buffer: this.buffers.norm.buffer } },
        { binding: 2, resource: { buffer: this.buffers.v.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.projDims } }
      ]);
      this.dispatch2d(pass, this.matmulPipeline, vProj, this.dModel, 1, 8, 1);

      const gProj = this.makeBindGroup(this.matmulPipeline, [
        { binding: 0, resource: { buffer: this.wg.buffer } },
        { binding: 1, resource: { buffer: this.buffers.norm.buffer } },
        { binding: 2, resource: { buffer: this.buffers.g.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.projDims } }
      ]);
      this.dispatch2d(pass, this.matmulPipeline, gProj, this.dModel, 1, 8, 1);

      const gateSigmoid = this.makeBindGroup(this.sigmoidPipeline, [
        { binding: 0, resource: { buffer: this.buffers.g.buffer } },
        { binding: 1, resource: { buffer: this.uniforms.sigmoidMeta } }
      ]);
      this.dispatch1d(pass, this.sigmoidPipeline, gateSigmoid, this.dModel);

      const symplecticFlow = this.makeBindGroup(this.symplecticFlowPipeline, [
        { binding: 0, resource: { buffer: this.manifoldState.buffer } },
        { binding: 1, resource: { buffer: this.buffers.r.buffer } },
        { binding: 2, resource: { buffer: this.buffers.v.buffer } },
        { binding: 3, resource: { buffer: this.buffers.g.buffer } },
        { binding: 4, resource: { buffer: this.buffers.holoOut.buffer } },
        { binding: 5, resource: { buffer: this.uniforms.symplecticMeta } }
      ]);
      this.dispatch1d(pass, this.symplecticFlowPipeline, symplecticFlow, this.dModel);

      const ffnGate = this.makeBindGroup(this.matmulPipeline, [
        { binding: 0, resource: { buffer: this.ffnUp1.buffer } },
        { binding: 1, resource: { buffer: this.buffers.holoOut.buffer } },
        { binding: 2, resource: { buffer: this.buffers.ffnGate.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.ffnUpDims } }
      ]);
      this.dispatch2d(pass, this.matmulPipeline, ffnGate, this.ffnHidden, 1, 8, 1);

      const ffnValue = this.makeBindGroup(this.matmulPipeline, [
        { binding: 0, resource: { buffer: this.ffnUp2.buffer } },
        { binding: 1, resource: { buffer: this.buffers.holoOut.buffer } },
        { binding: 2, resource: { buffer: this.buffers.ffnValue.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.ffnUpDims } }
      ]);
      this.dispatch2d(pass, this.matmulPipeline, ffnValue, this.ffnHidden, 1, 8, 1);

      const swishGate = this.makeBindGroup(this.swishPipeline, [
        { binding: 0, resource: { buffer: this.buffers.ffnGate.buffer } },
        { binding: 1, resource: { buffer: this.uniforms.mulMeta } }
      ]);
      this.dispatch1d(pass, this.swishPipeline, swishGate, this.ffnHidden);

      const ffnGateValueMul = this.makeBindGroup(this.mulPipeline, [
        { binding: 0, resource: { buffer: this.buffers.ffnGate.buffer } },
        { binding: 1, resource: { buffer: this.buffers.ffnValue.buffer } },
        { binding: 2, resource: { buffer: this.buffers.ffnProd.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.mulMeta } }
      ]);
      this.dispatch1d(pass, this.mulPipeline, ffnGateValueMul, this.ffnHidden);

      const ffnDown = this.makeBindGroup(this.matmulPipeline, [
        { binding: 0, resource: { buffer: this.ffnDown.buffer } },
        { binding: 1, resource: { buffer: this.buffers.ffnProd.buffer } },
        { binding: 2, resource: { buffer: this.buffers.ffnOut.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.ffnDownDims } }
      ]);
      this.dispatch2d(pass, this.matmulPipeline, ffnDown, this.dModel, 1, 8, 1);

      const cellAdd = this.makeBindGroup(this.addPipeline, [
        { binding: 0, resource: { buffer: this.buffers.holoOut.buffer } },
        { binding: 1, resource: { buffer: this.buffers.ffnOut.buffer } },
        { binding: 2, resource: { buffer: this.buffers.cellOut.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.addMeta } }
      ]);
      this.dispatch1d(pass, this.addPipeline, cellAdd, this.dModel);

      const residAdd = this.makeBindGroup(this.addPipeline, [
        { binding: 0, resource: { buffer: this.buffers.stream.buffer } },
        { binding: 1, resource: { buffer: this.buffers.cellOut.buffer } },
        { binding: 2, resource: { buffer: this.buffers.resid.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.addMeta } }
      ]);
      this.dispatch1d(pass, this.addPipeline, residAdd, this.dModel);

      const postNorm = this.makeBindGroup(this.rmsnormPipeline, [
        { binding: 0, resource: { buffer: this.buffers.resid.buffer } },
        { binding: 1, resource: { buffer: this.normGamma.buffer } },
        { binding: 2, resource: { buffer: this.buffers.stream.buffer } },
        { binding: 3, resource: { buffer: this.uniforms.rmsMeta } }
      ]);
      this.dispatchRmsnorm(pass, this.rmsnormPipeline, postNorm);

      if (applyHebbian) {
        const predProj = this.makeBindGroup(this.matmulPipeline, [
          { binding: 0, resource: { buffer: this.wPredict.buffer } },
          { binding: 1, resource: { buffer: this.buffers.stream.buffer } },
          { binding: 2, resource: { buffer: this.buffers.prediction.buffer } },
          { binding: 3, resource: { buffer: this.uniforms.predDims } }
        ]);
        this.dispatch2d(pass, this.matmulPipeline, predProj, this.dModel, 1, 8, 1);

        const activeInput = targetTokenId !== null ? this.buffers.target : this.buffers.input;
        const activeInference = this.makeBindGroup(this.activeInferencePipeline, [
          { binding: 0, resource: { buffer: this.manifoldState.buffer } },
          { binding: 1, resource: { buffer: this.buffers.r.buffer } },
          { binding: 2, resource: { buffer: this.buffers.prediction.buffer } },
          { binding: 3, resource: { buffer: activeInput.buffer } },
          { binding: 4, resource: { buffer: this.uniforms.activeMeta } }
        ]);
        this.dispatch1d(pass, this.activeInferencePipeline, activeInference, this.dModel);
      }
    }

    const logitsProj = this.makeBindGroup(this.matmulPipeline, [
      { binding: 0, resource: { buffer: this.embedding.buffer } },
      { binding: 1, resource: { buffer: this.buffers.stream.buffer } },
      { binding: 2, resource: { buffer: this.buffers.logits.buffer } },
      { binding: 3, resource: { buffer: this.uniforms.logitDims } }
    ]);
    this.dispatch2d(pass, this.matmulPipeline, logitsProj, this.vocabSize, 1, 8, 1);

    pass.end();
    this.queue.submit([encoder.finish()]);
    return this.buffers.logits;
  }
}
