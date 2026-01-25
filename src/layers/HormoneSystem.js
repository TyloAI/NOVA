export class HormoneSystem {
  constructor(labels, baseline = []) {
    this.labels = labels;
    this.levels = new Map();
    labels.forEach((label, idx) => {
      this.levels.set(label, baseline[idx] ?? 0);
    });
  }

  setLevel(label, value) {
    if (this.levels.has(label)) {
      this.levels.set(label, value);
    }
  }

  adjust(label, delta) {
    if (this.levels.has(label)) {
      this.levels.set(label, this.levels.get(label) + delta);
    }
  }

  /**
   * Projects the hormone levels onto a bias vector with the desired length.
   * Bias is tiled across the vector to keep GPU-friendly layout.
   */
  biasVector(length) {
    const values = Array.from(this.levels.values());
    const bias = new Float32Array(length);
    for (let i = 0; i < length; i += 1) {
      bias[i] = values[i % values.length];
    }
    return bias;
  }
}
