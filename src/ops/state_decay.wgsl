// N.O.V.A. Tesseract — state decay (global scaling) to reduce stale interference

struct ComplexBuf { data: array<vec2<f32>>, };

@group(0) @binding(0) var<storage, read_write> stateVec : ComplexBuf;
@group(0) @binding(1) var<uniform> meta : vec4<f32>; // decay, len, unused, unused

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let decay = meta.x;
  let len = u32(meta.y);
  if (idx >= len) { return; }
  stateVec.data[idx] = stateVec.data[idx] * vec2<f32>(decay, decay);
}
