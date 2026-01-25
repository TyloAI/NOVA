const shader = /* wgsl */ `
struct Matrix {
  values: array<f32, 16>;
};

@group(0) @binding(0) var<storage, read> A : Matrix;
@group(0) @binding(1) var<storage, read> B : Matrix;
@group(0) @binding(2) var<storage, read_write> C : Matrix;

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= 4u || col >= 4u) {
    return;
  }
  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    sum = sum + A.values[row * 4u + k] * B.values[k * 4u + col];
  }
  C.values[row * 4u + col] = sum;
}
`;

function trimAverage(samples) {
  if (!samples.length) return 0;
  if (samples.length <= 2) {
    return samples.reduce((a, b) => a + b, 0) / samples.length;
  }
  const sorted = [...samples].sort((a, b) => a - b);
  const trimmed = sorted.slice(1, -1);
  return trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
}

export async function calibrateHardware({ runs = 10 } = {}) {
  try {
    if (!navigator.gpu) {
      console.log('[STATUS] WebGPU unavailable, falling back to CPU estimates.');
      return {
        hardwareScore: 0,
        samples: [],
        device: null,
        adapterInfo: { renderer: 'No WebGPU adapter' },
      };
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      console.log('[STATUS] No GPU adapter available.');
      return {
        hardwareScore: 0,
        samples: [],
        device: null,
        adapterInfo: { renderer: 'Unavailable' },
      };
    }

    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: shader });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });

    const matrixA = new Float32Array(16).map((_, i) => (i % 5) + 1);
    const matrixB = new Float32Array(16).map((_, i) => ((i * 3) % 7) + 1);

    const bufferSize = matrixA.byteLength;
    const bufferA = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufferB = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufferC = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    device.queue.writeBuffer(bufferA, 0, matrixA.buffer);
    device.queue.writeBuffer(bufferB, 0, matrixB.buffer);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferC } },
      ],
    });

    const samples = [];
    for (let i = 0; i < runs; i += 1) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1, 1, 1);
      pass.end();
      encoder.copyBufferToBuffer(bufferC, 0, readback, 0, bufferSize);
      const start = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const end = performance.now();
      samples.push(end - start);
      console.log(`[PROGRESS] ${Math.min(5 + (i / runs) * 10, 15).toFixed(1)} Calibration ${i + 1}/${runs}`);
    }

    const avg = trimAverage(samples);
    const hardwareScore = avg > 0 ? Math.max(10, 1000 / avg) : 0;

    const adapterInfo = {
      renderer: adapter.info?.description || adapter.name || 'Unknown GPU',
      vendor: adapter.info?.vendor,
      architecture: adapter.info?.architecture,
    };

    await readback.mapAsync(GPUMapMode.READ);
    readback.unmap();

    return { hardwareScore, samples, device, adapterInfo };
  } catch (err) {
    console.error('[ERROR] Calibration failed', err);
    return {
      hardwareScore: 0,
      samples: [],
      device: null,
      adapterInfo: { renderer: `Calibration failed: ${err?.message || err}` },
    };
  }
}
