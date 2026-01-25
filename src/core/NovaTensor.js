export class NovaTensor {
  constructor(device, shape, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    this.device = device;
    this.shape = shape;
    const length = shape.reduce((a, b) => a * b, 1);
    this.size = length;
    this.byteLength = length * Float32Array.BYTES_PER_ELEMENT;
    this.buffer = device.createBuffer({
      size: this.byteLength,
      usage,
      mappedAtCreation: false
    });
  }

  write(data) {
    const array = data instanceof Float32Array ? data : new Float32Array(data);
    if (!this.buffer) {
      throw new Error("Tensor has been disposed.");
    }
    if (array.byteLength !== this.byteLength) {
      throw new Error(`Tensor write size mismatch: expected ${this.byteLength} bytes, got ${array.byteLength} bytes.`);
    }
    // Use the view's byte range to avoid overrunning GPU buffer on shared ArrayBuffers.
    this.device.queue.writeBuffer(this.buffer, 0, array.buffer, array.byteOffset, array.byteLength);
  }

  zero() {
    this.device.queue.writeBuffer(this.buffer, 0, new ArrayBuffer(this.byteLength));
  }

  static randn(length, mean = 0, std = 1) {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i += 2) {
      const u1 = Math.random() || Number.EPSILON;
      const u2 = Math.random() || Number.EPSILON;
      const mag = Math.sqrt(-2.0 * Math.log(u1));
      const z0 = mag * Math.cos(2.0 * Math.PI * u2);
      const z1 = mag * Math.sin(2.0 * Math.PI * u2);
      out[i] = z0 * std + mean;
      if (i + 1 < length) {
        out[i + 1] = z1 * std + mean;
      }
    }
    return out;
  }

  fillRandomNormal(mean = 0, std = 1) {
    const data = NovaTensor.randn(this.size, mean, std);
    this.write(data);
    return data;
  }

  initXavier({ fanIn, fanOut, gain = 1, scale = 1 } = {}) {
    const inferredFanIn = fanIn ?? (this.shape.length > 1 ? this.shape[this.shape.length - 1] : this.shape[0]);
    const inferredFanOut = fanOut ?? (this.shape.length > 1 ? this.shape[this.shape.length - 2] : this.shape[0]);
    const std = Math.sqrt(2 / (inferredFanIn + inferredFanOut)) * gain * scale;
    this.fillRandomNormal(0, std);
    return this;
  }

  initResidual({ fanIn, fanOut, gain = 1, numLayers = 12 } = {}) {
    const inferredFanIn = fanIn ?? (this.shape.length > 1 ? this.shape[this.shape.length - 1] : this.shape[0]);
    const inferredFanOut = fanOut ?? (this.shape.length > 1 ? this.shape[this.shape.length - 2] : this.shape[0]);
    const residualScale = 1 / Math.sqrt(2 * numLayers);
    const std = Math.sqrt(2 / (inferredFanIn + inferredFanOut)) * gain * residualScale;
    this.fillRandomNormal(0, std);
    return this;
  }

  initOnes() {
    this.write(new Float32Array(this.size).fill(1));
    return this;
  }

  static randomUnitSpinor(length, radius = 1) {
    const out = new Float32Array(length * 2);
    for (let i = 0; i < length; i += 1) {
      const theta = Math.random() * 2 * Math.PI;
      out[i * 2] = Math.cos(theta) * radius;
      out[i * 2 + 1] = Math.sin(theta) * radius;
    }
    return out;
  }

  initUnitSpinor(radius = 1) {
    if (this.size % 2 !== 0) {
      throw new Error("UnitSpinor init requires an even-sized tensor for vec2 packing.");
    }
    const pairs = this.size / 2;
    const data = NovaTensor.randomUnitSpinor(pairs, radius);
    this.write(data);
    return data;
  }

  /**
   * Safe readback that maps a temporary buffer and destroys it to avoid GPU memory leaks.
   */
  async read() {
    if (!this.buffer) {
      throw new Error("Tensor has been disposed.");
    }
    // Copy GPU data into a temporary readback buffer.
    const readBuffer = this.device.createBuffer({
      size: this.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    try {
      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(this.buffer, 0, readBuffer, 0, this.byteLength);
      this.device.queue.submit([commandEncoder.finish()]);

      await readBuffer.mapAsync(GPUMapMode.READ);
      
      const copyArray = readBuffer.getMappedRange();
      // Clone before unmapping; the mapped view is invalid once unmapped.
      const result = new Float32Array(copyArray.slice(0));

      readBuffer.unmap();

      return result;
    } finally {
      // Release the staging buffer to prevent VRAM leaks.
      readBuffer.destroy();
    }
  }

  dispose() {
    if (this.buffer) {
      this.buffer.destroy();
      this.buffer = null;
    }
  }
}
