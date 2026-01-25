import { ModelAdapter } from './modelAdapter.js';

export class TransformerAdapter extends ModelAdapter {
  constructor(options = {}) {
    super('Transformer', options);
    this.pipeline = null;
    this.loading = null;
    this.modelPath = options.modelPath || null;
  }

  async detectModelSource() {
    // Prefer user-provided model path or local /models mount
    const localBase = this.modelPath || '/models/Xenova/gpt2';
    try {
      const res = await fetch(`${localBase}/config.json`, { method: 'HEAD' });
      if (res.ok) return { id: localBase, source: 'local' };
    } catch (err) {
      console.warn('[STATUS] Local GPT-2 not found at', localBase, err?.message || err);
    }
    return { id: 'Xenova/gpt2', source: 'cdn' };
  }

  async warmup() {
    await this.ensurePipeline();
    // Minimal generation to pull weights/compile graph before the arena loop.
    await this.pipeline('Warmup', {
      max_new_tokens: 4,
      temperature: 1.0,
      top_k: 10,
      top_p: 0.9,
    });
  }

  async ensurePipeline() {
    if (this.pipeline) return;
    if (!this.loading) {
      this.loading = (async () => {
        const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0/dist/transformers.min.js');
        const { pipeline } = mod;
        const { id, source } = await this.detectModelSource();
        try {
          this.pipeline = await pipeline('text-generation', id, {
            quantized: true,
          });
          console.log(`[STATUS] GPT-2 baseline loaded from ${source === 'local' ? 'local /models' : 'CDN'}.`);
        } catch (err) {
          throw new Error(`Transformer baseline failed to load (${id}): ${err?.message || err}`);
        }
      })();
    }
    await this.loading;
    if (!this.pipeline) {
      throw new Error('Transformer baseline unavailable (pipeline failed to load)');
    }
  }

  async predict(text) {
    const start = performance.now();
    await this.ensurePipeline();
    const outputs = await this.pipeline(text, {
      max_new_tokens: 40,
      temperature: 0.9,
      top_k: 50,
      top_p: 0.95,
    });
    const latency = performance.now() - start;
    const generated = Array.isArray(outputs) && outputs.length > 0
      ? outputs[0]?.generated_text || ''
      : '';
    if (!generated) {
      throw new Error('Transformer baseline returned empty generation');
    }
    return { prediction: generated, latency };
  }

  async train(_text, _target) {
    return { updated: false };
  }
}
