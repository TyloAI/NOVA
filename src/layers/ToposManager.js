// N.O.V.A. Tesseract — Spectral scope manager (band-pass masks per logical depth)

export class ToposManager {
  constructor(dModel, opts = {}) {
    this.dModel = dModel;
    this.maxLayers = Math.max(1, opts.maxLayers ?? 8);
    this.threshold = opts.threshold ?? 0.05;
    this.stack = [];
    this.fullMask = this.#buildMask(0, dModel);
  }

  #bandWidth() {
    return Math.max(1, Math.floor(this.dModel / this.maxLayers));
  }

  #buildMask(start, stop) {
    const safeStart = Math.max(0, Math.min(start, this.dModel));
    const safeStop = Math.max(safeStart, Math.min(stop, this.dModel));
    const mask = new Float32Array(this.dModel);
    for (let i = safeStart; i < safeStop; i += 1) {
      mask[i] = 1;
    }
    return mask;
  }

  enterScope(label = "scope") {
    const depth = this.stack.length;
    const width = this.#bandWidth();
    const start = depth * width;
    const stop = Math.min(this.dModel, start + width);
    const mask = this.#buildMask(start, stop);
    this.stack.push({ label, mask, start, stop });
    return mask;
  }

  exitScope() {
    this.stack.pop();
    return this.currentMask();
  }

  currentMask() {
    if (!this.stack.length) return this.fullMask;
    return this.stack[this.stack.length - 1].mask;
  }

  /**
   * Measures out-of-band energy to flag scope bleed.
   * @param {Float32Array} stateVec - magnitude or power spectrum
   */
  spillEnergy(stateVec) {
    const mask = this.currentMask();
    let spill = 0;
    const len = Math.min(stateVec.length, mask.length);
    for (let i = 0; i < len; i += 1) {
      const permitted = mask[i];
      if (permitted === 1) continue;
      const v = stateVec[i];
      spill += v < 0 ? -v : v;
    }
    return spill;
  }

  /**
   * Prunes out-of-band components in-place.
   */
  applyMask(stateVec) {
    const mask = this.currentMask();
    const len = Math.min(stateVec.length, mask.length);
    for (let i = 0; i < len; i += 1) {
      stateVec[i] *= mask[i];
    }
    return stateVec;
  }
}
