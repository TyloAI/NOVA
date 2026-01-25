struct Matrix {
  data: array<f32>,
};

@group(0) @binding(0) var<storage, read> outputVec : Matrix;
@group(0) @binding(1) var<storage, read> inputVec : Matrix;
@group(0) @binding(2) var<storage, read_write> fastWeights : Matrix;
@group(0) @binding(3) var<uniform> dims : vec4<u32>; // outSize, inSize, strideOut, strideIn
@group(0) @binding(4) var<uniform> scalars : vec2<f32>; // decay, learningRate

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.x;
  let i = gid.y;
  if (o >= dims.x || i >= dims.y) {
    return;
  }
  let idx = o * dims.y + i;
  let prev = fastWeights.data[idx];
  let update = scalars.y * outputVec.data[o] * inputVec.data[i];
  fastWeights.data[idx] = scalars.x * prev + update;
}
