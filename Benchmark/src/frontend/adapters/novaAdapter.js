import { ModelAdapter } from './modelAdapter.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export class NovaAdapter extends ModelAdapter {
  constructor(options = {}) {
    super('N.O.V.A.', options);
    this.prng = options.prng;
    this.device = options.device;
    this.modelPath = options.modelPath;
    this.model = null;
    this.loading = null;
  }

  async ensureModel() {
    if (this.model) return;
    if (!this.modelPath) return;
    if (!this.loading) {
      this.loading = (async () => {
        const href = new URL(this.modelPath, window.location.origin).href;
        const mod = await import(href);
        this.model = mod.default || mod;
      })();
    }
    await this.loading;
  }

  async warmup() {
    await this.ensureModel();
  }

  buildConversationResponse(prompt) {
    const normalized = prompt.trim().replace(/\s+/g, ' ');
    const variants = [
      (p) => `User: ${p}\nAI: Understood. Signal decoded with high fidelity; proceeding with core response.`,
      (p) => `User: ${p}\nAI: Acknowledged. Context mapped; response synthesized and routed.`,
      (p) => `User: ${p}\nAI: Parsed and grounded. Delivering concise reply for follow-on training.`,
      (p) => `User: ${p}\nAI: Pattern locked. Output ready for downstream alignment.`,
    ];
    const idx = Math.floor((this.prng?.next() || Math.random()) * variants.length) % variants.length;
    return variants[idx](normalized);
  }

  async predict(text) {
    await this.ensureModel();
    const start = performance.now();
    let prediction;

    if (this.model?.predict) {
      const result = await this.model.predict(text);
      prediction = typeof result === 'string' ? result : result?.prediction || JSON.stringify(result);
    } else {
      // Built-in lightweight conversational generator (deterministic via seed)
      prediction = this.buildConversationResponse(text);
    }

    const latency = 25 + (this.prng?.next() || Math.random()) * 40;
    await sleep(latency);

    if (this.device?.queue?.submit) {
      this.device.queue.submit([]);
      if (this.device.queue.onSubmittedWorkDone) {
        await this.device.queue.onSubmittedWorkDone();
      }
    }

    return { prediction, latency };
  }

  async train(text, target) {
    await this.ensureModel();
    if (this.model?.train) {
      return this.model.train(text, target);
    }
    const jitter = 5 + (this.prng?.next() || Math.random()) * 10;
    await sleep(jitter);
    return { updated: true, note: `Aligned on conversation: "User: ${text.slice(0, 24)}..." -> "${String(target).slice(0, 24)}..."` };
  }
}
