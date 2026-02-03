// N.O.V.A. Tesseract — Phase Injector
// Bridges real-domain embeddings into complex spectral vectors by applying a deterministic phase.

import { NovaTensor } from "../core/NovaTensor.js";

export class PhaseInjector {
  constructor(device, dModel, opts = {}) {
    this.device = device;
    this.dModel = dModel;
    this.phaseTable = new Map();
    this.seed = opts.seed ?? 0x9e3779b1; // golden ratio hash base
  }

  /**
   * Deterministic phase per token id (stable within a session).
   */
  phaseFor(tokenId) {
    if (this.phaseTable.has(tokenId)) return this.phaseTable.get(tokenId);
    // Simple hash → [0, 2π)
    let x = (tokenId ^ this.seed) >>> 0;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    const frac = (x >>> 0) / 0xffffffff;
    const theta = 2 * Math.PI * frac;
    this.phaseTable.set(tokenId, theta);
    return theta;
  }

  /**
   * Injects phase into a real vector, returning a complex Float32Array [real, imag] interleaved.
   * realVec length must equal dModel.
   */
  inject(realVec, tokenId) {
    if (realVec.length !== this.dModel) {
      throw new Error(`PhaseInjector: expected real vector length ${this.dModel}, got ${realVec.length}`);
    }
    const theta = this.phaseFor(tokenId);
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const out = new Float32Array(this.dModel * 2);
    for (let i = 0; i < this.dModel; i += 1) {
      const v = realVec[i];
      out[i * 2] = v * c;
      out[i * 2 + 1] = v * s;
    }
    return out;
  }

  /**
   * Builds a GPU tensor for the phase-injected complex vector.
   */
  tensorFrom(realVec, tokenId) {
    const complex = this.inject(realVec, tokenId);
    const tensor = new NovaTensor(this.device, [this.dModel, 2]);
    tensor.write(complex);
    return tensor;
  }
}
