// N.O.V.A. Tesseract — Sequencer
// Ensures consistent position indexing (ring buffer) so bind/unbind never mismatch.

export class Sequencer {
  constructor(ham, opts = {}) {
    this.ham = ham;
    this.maxPositions = Math.max(1, opts.maxPositions ?? ham.dModel); // capacity
    this.strictNoWrap = opts.strictNoWrap ?? true; // if true, throw on wrap to avoid destructive reuse
    this.step = 0;
  }

  currentIndex(offset = 0) {
    const idx = this.step + offset;
    const wrapped = ((idx % this.maxPositions) + this.maxPositions) % this.maxPositions;
    if (this.strictNoWrap && idx >= this.maxPositions) {
      throw new Error(`Sequencer: position capacity exceeded (${this.maxPositions}). Reset or slide window before reuse.`);
    }
    return wrapped;
  }

  /**
   * Encode a token at the current position index, then advance the step.
   */
  async encode(realVec, tokenId) {
    const posIndex = this.currentIndex();
    await this.ham.encodeRealAtPosition(realVec, tokenId, posIndex);
    this.step += 1;
    return posIndex;
  }

  /**
   * Recall using a relative offset from the latest step (offset -1 = last token).
   */
  async recall(offsetFromLast = -1) {
    const posIndex = this.currentIndex(offsetFromLast);
    return this.ham.recallAtPosition(posIndex);
  }

  /**
   * Clears holographic state and resets position counter.
   */
  reset() {
    this.ham.resetState();
    this.step = 0;
  }
}
