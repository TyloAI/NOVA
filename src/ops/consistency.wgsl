// N.O.V.A. Tesseract — consistency pruning pass
// Applies spectral mask elementwise to suppress out-of-scope energy.

struct ComplexBuf { data: array<vec2<f32>>, };
struct MaskBuf { data: array<f32>, };

@group(0) @binding(0) var<storage, read_write> stateVec : ComplexBuf;
@group(0) @binding(1) var<storage, read> maskVec : MaskBuf;
@group(0) @binding(2) var<uniform> meta : vec4<u32>; // len

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let n = meta.x;
  if (idx >= n) { return; }
  let m = maskVec.data[idx];
  let val = stateVec.data[idx];
  stateVec.data[idx] = val * vec2<f32>(m, m);
}
