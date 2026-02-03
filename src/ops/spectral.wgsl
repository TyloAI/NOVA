// N.O.V.A. Tesseract – Spectral primitives (FFT + complex ops)
// These kernels are scaffolding for the spectral fast-weight path.

const PI : f32 = 3.141592653589793;
const MAX_FFT : u32 = 2048u; // must cover chosen d_model (power of two)

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

// In-place radix-2 Cooley–Tukey FFT. Expects n to be power-of-two and <= MAX_FFT.
fn fft(data : ptr<function, array<vec2<f32>, MAX_FFT>>, n : u32, direction : i32) {
  if (n == 0u || n > MAX_FFT || (n & (n - 1u)) != 0u) {
    // Invalid length: skip.
    return;
  }

  // Bit-reversal permutation
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

  // Iterative Danielson–Lanczos stages
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

  // Normalize inverse FFT
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
