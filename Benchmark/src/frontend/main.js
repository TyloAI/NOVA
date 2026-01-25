import { PRNG } from './prng.js';
import { calibrateHardware } from './calibration.js';
import { NovaAdapter } from './adapters/novaAdapter.js';
import { TransformerAdapter } from './adapters/transformerAdapter.js';
import { runBenchmark } from './arena.js';

const statusEl = document.getElementById('status');
const metaEl = document.getElementById('meta');

const params = new URLSearchParams(window.location.search);
const seed = Number(params.get('seed')) || Date.now();
const modelA = params.get('modelA') || 'N.O.V.A. (builtin)';
const modelB = params.get('modelB') || 'Transformer (baseline)';
const modelAPath = params.get('modelAPath') || '';
const modelBPath = params.get('modelBPath') || '';

const meta = {
  seed,
  modelA,
  modelB,
  modelAPath,
  modelBPath,
  startedAt: new Date().toISOString(),
  browser: navigator.userAgent,
};

const updateStatus = (text) => {
  if (statusEl) statusEl.textContent = text;
};

window.__BENCHMARK_RESULT__ = { done: false };
metaEl.textContent = `Seed=${seed} | A=${modelA} | B=${modelB}`;
updateStatus('Calibrating WebGPU…');
console.log('[STATUS] Entering Ghost Runner');
console.log('[PROGRESS] 2 Warmup');

async function run() {
  const prng = new PRNG(seed);
  const calibration = await calibrateHardware({ runs: 10 });
  meta.gpu = calibration.adapterInfo?.renderer || 'Unknown GPU';

  const adapterA = new NovaAdapter({ prng, device: calibration.device, modelPath: params.get('modelAPath') });
  let adapterB;
  if (modelB.startsWith('Transformer')) {
    adapterB = new TransformerAdapter({ prng, modelPath: modelBPath });
  } else if (modelB.includes('N.O.V.A.')) {
    adapterB = new NovaAdapter({ prng, device: calibration.device, modelPath: params.get('modelBPath') });
  } else {
    adapterB = new TransformerAdapter({ prng, modelPath: modelBPath });
  }

  updateStatus('Running arena rounds…');
  if (adapterA.warmup) await adapterA.warmup();
  if (adapterB.warmup) await adapterB.warmup();
  const runs = [];
  const totalRuns = 3;
  for (let i = 0; i < totalRuns; i += 1) {
    const arenaResult = await runBenchmark({
      adapterA,
      adapterB,
      prng,
      calibration,
      meta,
      runIndex: i + 1,
      totalRuns,
    });
    runs.push(arenaResult);
  }

  const summary = {
    novaScore: runs.reduce((a, r) => a + r.finalScoreNova, 0) / runs.length,
    gpt2Score: runs.reduce((a, r) => a + r.finalScoreGpt2, 0) / runs.length,
    avgLatencyA: runs.reduce((a, r) => a + r.avgLatencyA, 0) / runs.length,
    avgLatencyB: runs.reduce((a, r) => a + r.avgLatencyB, 0) / runs.length,
    avgLoss: runs.reduce((a, r) => a + r.avgLoss, 0) / runs.length,
  };

  updateStatus('Compiling report data…');
  console.log('[PROGRESS] 98 Finalizing');

  window.__BENCHMARK_RESULT__ = {
    done: true,
    metadata: meta,
    hardware: {
      hardwareScore: calibration.hardwareScore,
      samples: calibration.samples,
      gpu: meta.gpu,
    },
    runs,
    summary,
  };
  console.log('[PROGRESS] 100 Ready');
  updateStatus('Benchmark completed. Closing soon.');
}

run().catch((err) => {
  console.error('[ERROR]', err);
  window.__BENCHMARK_RESULT__ = {
    done: true,
    error: err?.message || String(err),
    metadata: meta,
    runs: [],
    curve: [],
    logs: [`Failed during execution: ${err?.message || err}`],
    hardware: { hardwareScore: 0, samples: [], gpu: 'unknown' },
    finalScore: 0,
  };
});
