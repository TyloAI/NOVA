// Tesseract Echo Test — validates holographic recall without language model noise.
// Usage: run in browser console with WebGPU enabled or adapt to your bundler.

import { DeviceContext } from "../src/core/Device.js";
import { SpectralHAM } from "../src/tesseract/SpectralHAM.js";
import { Positioner } from "../src/tesseract/Positioner.js";
import { PhaseInjector } from "../src/tesseract/PhaseInjector.js";

async function holographicEchoTest({
  dModel = 256,
  seqLen = 5,
  recallIndex = 2
} = {}) {
  const ctx = await new DeviceContext().init();
  const ham = new SpectralHAM(ctx, { dModel });
  await ham.init();
  const positioner = new Positioner(ctx.device, dModel);
  const injector = new PhaseInjector(ctx.device, dModel);

  // Prepare random real vectors and positional tensors
  const realVectors = [];
  const positions = [];
  for (let i = 0; i < seqLen; i += 1) {
    const v = new Float32Array(dModel);
    for (let j = 0; j < dModel; j += 1) v[j] = Math.random() * 2 - 1;
    realVectors.push(v);
    positions.push(positioner.getPositionTensor(i));
  }

  // Encode all tokens
  for (let i = 0; i < seqLen; i += 1) {
    const tokenId = i; // synthetic id for stable phase
    const complexTensor = injector.tensorFrom(realVectors[i], tokenId);
    await ham.encode(complexTensor, positions[i]);
  }

  // Recall target
  const targetPos = positions[recallIndex];
  const recalledTensor = await ham.recall(targetPos);
  const recalled = await recalledTensor.read();

  // Compute cosine similarity vs. original target (magnitude only on real part)
  const target = realVectors[recallIndex];
  let dot = 0;
  let nr = 0;
  let nt = 0;
  for (let i = 0; i < dModel; i += 1) {
    const r = recalled[i * 2]; // real component
    const t = target[i];
    dot += r * t;
    nr += r * r;
    nt += t * t;
  }
  const cos = dot / (Math.sqrt(nr) * Math.sqrt(nt) + 1e-9);
  console.log(`Tesseract echo cos=${cos.toFixed(4)} (expect >0.9 when clean)`);
  return cos;
}

window.holographicEchoTest = holographicEchoTest;
