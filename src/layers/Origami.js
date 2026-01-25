export class OrigamiMemory {
  constructor(stateSize) {
    this.stateSize = stateSize;
    this.stack = [];
  }

  cloneState(state) {
    const arr = state instanceof Float32Array ? state : new Float32Array(state);
    return new Float32Array(arr);
  }

  enterScope(state) {
    this.stack.push(this.cloneState(state));
  }

  exitScope(fallback) {
    if (this.stack.length === 0) {
      return fallback;
    }
    return this.stack.pop();
  }

  /**
   * Detects braces and folds/unfolds memory.
   * Returns the possibly updated state.
   */
  route(char, state) {
    if (char === "{") {
      this.enterScope(state);
      return state;
    }
    if (char === "}") {
      return this.exitScope(state);
    }
    return state;
  }
}
