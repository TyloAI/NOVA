struct Matrix {
  data: array<f32>,
};

@group(0) @binding(0) var<storage, read> a : Matrix;
@group(0) @binding(1) var<storage, read> b : Matrix;
@group(0) @binding(2) var<storage, read_write> out : Matrix;
@group(0) @binding(3) var<uniform> dims : vec3<u32>; // aRows, aCols (=bRows), bCols

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= dims.x || col >= dims.z) {
    return;
  }

  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < dims.y; k = k + 1u) {
    let aIdx = row * dims.y + k;
    let bIdx = k * dims.z + col;
    acc = acc + a.data[aIdx] * b.data[bIdx];
  }
  let outIdx = row * dims.z + col;
  out.data[outIdx] = acc;
}
