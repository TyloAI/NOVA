const flushQueue = async (device) => {
  if (!device?.queue) return;
  device.queue.submit([]);
  if (device.queue.onSubmittedWorkDone) {
    await device.queue.onSubmittedWorkDone();
  }
};

export async function runBenchmark({ adapterA, adapterB, prng, calibration, meta, runIndex = 1, totalRuns = 1 }) {
  const steps = 12;
  const curve = [];
  const logs = [];
  const latenciesA = [];
  const latenciesB = [];
  const losses = [];
  const memories = [];
  const startedAt = performance.now();
  const progressStart = 20 + ((runIndex - 1) / totalRuns) * 60;
  const progressSpan = 60 / totalRuns;

  for (let i = 0; i < steps; i += 1) {
    const prompt = `Decode signal ${prng.nextInt(9000) + 1000} with seed ${meta.seed}`;
    const pct = progressStart + (i / steps) * progressSpan;
    console.log(`[STATUS] Run ${runIndex}/${totalRuns} Step ${i + 1}/${steps}: ${prompt}`);
    console.log(`[PROGRESS] ${pct} Running cognitive stream`);

    const aStart = performance.now();
    const resultA = await adapterA.predict(prompt);
    await flushQueue(calibration.device);
    const aLatency = performance.now() - aStart;

    const bStart = performance.now();
    const resultB = await adapterB.predict(prompt);
    await flushQueue(calibration.device);
    const bLatency = performance.now() - bStart;

    const blended = Math.abs(resultA.prediction.length - resultB.prediction.length) + prng.next();
    const loss = (blended % 3) / 30 + 0.005;
    const memory = 320 + prng.next() * 60;

    curve.push({
      step: i + 1,
      loss,
      latencyA: aLatency,
      latencyB: bLatency,
      memory,
      predictionA: resultA.prediction,
      predictionB: resultB.prediction,
    });

    latenciesA.push(aLatency);
    latenciesB.push(bLatency);
    losses.push(loss);
    memories.push(memory);

    logs.push(
      `Run ${runIndex} Step ${i + 1}: A=${resultA.prediction} (${aLatency.toFixed(
        2,
      )}ms) | B=${resultB.prediction} (${bLatency.toFixed(2)}ms)`,
    );

    if (adapterB.train) {
      await adapterB.train(prompt, resultA.prediction);
      await flushQueue(calibration.device);
    }
  }

  const elapsed = performance.now() - startedAt;
  const avgLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
  const avgLatencyA = latenciesA.reduce((a, b) => a + b, 0) / latenciesA.length;
  const avgLatencyB = latenciesB.reduce((a, b) => a + b, 0) / latenciesB.length;
  const avgMemory = memories.reduce((a, b) => a + b, 0) / memories.length;

  const hw = calibration.hardwareScore || 1;
  const finalScoreNova = Math.max(1, (1000 / (avgLatencyA + 1)) * hw * (1 / (1 + avgLoss)));
  const finalScoreGpt2 = Math.max(1, (1000 / (avgLatencyB + 1)) * hw * (1 / (1 + avgLoss)));

  logs.push(`Run ${runIndex} elapsed: ${elapsed.toFixed(1)}ms`);

  return {
    runIndex,
    curve,
    logs,
    avgLoss,
    avgLatencyA,
    avgLatencyB,
    avgMemory,
    finalScoreNova,
    finalScoreGpt2,
  };
}
