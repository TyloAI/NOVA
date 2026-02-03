// N.O.V.A. Tesseract — Holographic associative memory kernels
// Entry points:
// - encode_bind: bind token with position via circular convolution and superpose into state
// - recall_decode: unbind a position from state to recover a token estimate

const PI : f32 = 3.141592653589793;
const MAX_FFT : u32 = 2048u; // must cover d_model (power-of-two)

struct ComplexBuf {
  data: array<vec2<f32>>,
};

struct MaskBuf {
  data: array<f32>,
};

@group(0) @binding(0) var<storage, read> tokenVec : ComplexBuf;
@group(0) @binding(1) var<storage, read> posVec : ComplexBuf;
@group(0) @binding(2) var<storage, read_write> stateVec : ComplexBuf;
@group(0) @binding(3) var<storage, read_write> outVec : ComplexBuf;
@group(0) @binding(4) var<uniform> meta : vec4<u32>; // len, scratch, unused, unused

fn c_mul(a : vec2<f32>, b : vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn c_inv(a : vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x, -a.y);
}

fn bit_reverse(idx : u32, bits : u32) -> u32 {
  var x = idx;
  var rev : u32 = 0u;
  var i : u32 = 0u;
  loop {
    if (i >= bits) { break; }
    rev = (rev << 1u) | (x & 1u);
    x = x >> 1u;
    i = i + 1u;
  }
  return rev;
}

fn twiddle(k : u32, n : u32, direction : i32) -> vec2<f32> {
  // direction: +1 forward FFT, -1 inverse FFT
  let sign = select(1.0, -1.0, direction < 0);
  let angle = 2.0 * PI * f32(k) / f32(n) * sign;
  return vec2<f32>(cos(angle), sin(angle));
}

// In-place radix-2 Cooley–Tukey FFT on a fixed local buffer.
fn fft(data : ptr<function, array<vec2<f32>, MAX_FFT>>, n : u32, direction : i32) {
  if (n == 0u || n > MAX_FFT || (n & (n - 1u)) != 0u) {
    return;
  }
  let bits = u32(log2(f32(n)));
  var i : u32 = 0u;
  loop {
    if (i >= n) { break; }
    let j = bit_reverse(i, bits);
    if (j > i) {
      let tmp = (*data)[i];
      (*data)[i] = (*data)[j];
      (*data)[j] = tmp;
    }
    i = i + 1u;
  }

  var len : u32 = 2u;
  loop {
    if (len > n) { break; }
    let half = len >> 1u;
    var base : u32 = 0u;
    loop {
      if (base >= n) { break; }
      var j : u32 = 0u;
      loop {
        if (j >= half) { break; }
        let idx1 = base + j;
        let idx2 = idx1 + half;
        let w = twiddle(j, len, direction);
        let t = c_mul((*data)[idx2], w);
        let u = (*data)[idx1];
        (*data)[idx1] = u + t;
        (*data)[idx2] = u - t;
        j = j + 1u;
      }
      base = base + len;
    }
    len = len << 1u;
  }

  if (direction < 0) {
    let invN = 1.0 / f32(n);
    var k : u32 = 0u;
    loop {
      if (k >= n) { break; }
      (*data)[k] = (*data)[k] * invN;
      k = k + 1u;
    }
  }
}

@compute @workgroup_size(1)
fn encode_bind() {
  let n = meta.x;
  if (n == 0u || n > MAX_FFT) { return; }
  var bufToken : array<vec2<f32>, MAX_FFT>;
  var bufPos : array<vec2<f32>, MAX_FFT>;
  var bufBind : array<vec2<f32>, MAX_FFT>;

  var i : u32 = 0u;
  loop {
    if (i >= n) { break; }
    bufToken[i] = tokenVec.data[i];
    bufPos[i] = posVec.data[i];
    i = i + 1u;
  }

  fft(&bufToken, n, 1);
  fft(&bufPos, n, 1);

  i = 0u;
  loop {
    if (i >= n) { break; }
    bufBind[i] = c_mul(bufToken[i], bufPos[i]);
    i = i + 1u;
  }

  fft(&bufBind, n, -1);

  i = 0u;
  loop {
    if (i >= n) { break; }
    stateVec.data[i] = stateVec.data[i] + bufBind[i];
    i = i + 1u;
  }
}

@compute @workgroup_size(1)
fn recall_decode() {
  let n = meta.x;
  if (n == 0u || n > MAX_FFT) { return; }
  var bufState : array<vec2<f32>, MAX_FFT>;
  var bufPos : array<vec2<f32>, MAX_FFT>;
  var bufFreq : array<vec2<f32>, MAX_FFT>;

  var i : u32 = 0u;
  loop {
    if (i >= n) { break; }
    bufState[i] = stateVec.data[i];
    // Use conjugate for inverse binding
    bufPos[i] = c_inv(posVec.data[i]);
    i = i + 1u;
  }

  fft(&bufState, n, 1);
  fft(&bufPos, n, 1);

  i = 0u;
  loop {
    if (i >= n) { break; }
    bufFreq[i] = c_mul(bufState[i], bufPos[i]);
    i = i + 1u;
  }

  fft(&bufFreq, n, -1);

  i = 0u;
  loop {
    if (i >= n) { break; }
    outVec.data[i] = bufFreq[i];
    i = i + 1u;
  }
}
