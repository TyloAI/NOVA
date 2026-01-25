export class DeviceContext {
  constructor() {
    this.adapter = null;
    this.device = null;
    this.queue = null;
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported in this browser.");
    }
    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) {
      throw new Error("No GPU adapter available.");
    }
    this.device = await this.adapter.requestDevice();
    this.queue = this.device.queue;
    return this;
  }

  createShader(code) {
    return this.device.createShaderModule({ code });
  }

  createBindGroupLayout(entries) {
    return this.device.createBindGroupLayout({ entries });
  }

  createPipeline(label, layout, module, entryPoint = "main") {
    return this.device.createComputePipeline({
      label,
      layout,
      compute: {
        module,
        entryPoint
      }
    });
  }
}
