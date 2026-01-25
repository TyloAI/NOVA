export class ModelAdapter {
  constructor(name, options = {}) {
    this.name = name;
    this.options = options;
  }

  async predict(_text) {
    throw new Error('predict() must be implemented by subclasses');
  }

  async train(_text, _target) {
    return { updated: false };
  }
}
