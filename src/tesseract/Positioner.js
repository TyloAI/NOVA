// N.O.V.A. Tesseract — positional hologram generator
// Produces approximately orthogonal complex codes for binding.

import { NovaTensor } from "../core/NovaTensor.js";

export class Positioner {
  constructor(device, dModel, opts = {}) {
    this.device = device;
    this.dModel = dModel;
    this.seed = opts.seed ?? 0x51b3a1c3;
    this.cache = new Map(); // posIndex -> NovaTensor
  }

  #phaseFor(posIndex, dimIndex) {
    // Hash (posIndex, dimIndex) deterministically into [0, 2π)
    let x = (posIndex * 0x9e3779b1) ^ (dimIndex * 0x7f4a7c15) ^ this.seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    const frac = (x >>> 0) / 0xffffffff;
    return 2 * Math.PI * frac;
  }

  /**
   * Deterministic positional hologram for a given index.
   * Ensures binding/unbinding reuse the same code each time.
   */
  buildPositionVec(posIndex) {
    const out = new Float32Array(this.dModel * 2);
    for (let i = 0; i < this.dModel; i += 1) {
      const phase = this.#phaseFor(posIndex, i);
      out[i * 2] = Math.cos(phase);
      out[i * 2 + 1] = Math.sin(phase);
    }
    return out;
  }

  /**
   * Returns a cached GPU tensor for the positional hologram at posIndex.
   */
  getPositionTensor(posIndex) {
    if (this.cache.has(posIndex)) return this.cache.get(posIndex);
    const vec = this.buildPositionVec(posIndex);
    const tensor = new NovaTensor(this.device, [this.dModel, 2]);
    tensor.write(vec);
    this.cache.set(posIndex, tensor);
    return tensor;
  }
}
