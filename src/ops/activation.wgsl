struct Buffer {
  data: array<f32>,
};

@group(0) @binding(0) var<storage, read_write> values : Buffer;
@group(0) @binding(1) var<uniform> dims : vec2<u32>; // length, mode (0 = SiLU, 1 = GeLU)

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

fn gelu(x: f32) -> f32 {
  let k0 = 0.79788456; // sqrt(2 / pi)
  return 0.5 * x * (1.0 + tanh(k0 * (x + 0.044715 * pow(x, 3.0))));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= dims.x) {
    return;
  }
  let mode = dims.y;
  let x = values.data[idx];
  values.data[idx] = select(silu(x), gelu(x), mode == 1u);
}
