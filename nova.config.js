export const novaConfig = {
  model: {
    vocabSize: 50257,
    inputSize: 50257,
    outputSize: 50257,
    dModel: 768,
    heads: 12,
    headDim: 64,
    ffnHidden: 3072,
    layers: 12,
    hiddenSize: 768
  },
  training: {
    epochs: 2,
    contextWindow: 10,
    contextDecay: 0.7, // Higher decay so it does not only remember the latest token
    codeBoost: 1,
    promptWeight: 0.7,
    maxLines: 800,
    shuffle: true,
    tokenStride: 1,
    logEvery: 1200,
    yieldEvery: 40
  },
  runtime: {
    thinkingSteps: 12,
    maxReplyTokens: 220,
    minReplyTokens: 3,
    temperature: 0.45, // Cooler sampling to suppress off-topic picks
    topK: 12, // Looser topK to avoid getting stuck
    repeatPenalty: 1.2,
    repeatWindow: 32,
    contextWindow: 10,
    contextDecay: 0.7, // Higher decay so it does not only remember the latest token
    frequencyBias: 0.05,
    userContextWeight: 0.8,
    userContextLimit: 18,
    firstTokenUserScale: 1.0,
    promptTokenBias: 0.8,
    anchorBonus: 0.6,
    commonPenalty: 1.5,
    enableHeuristics: false,
    startQuotePenalty: 1.4,
    punctuationPenalty: 1.2,
    doublePunctuationPenalty: 2.2,
    startPunctuationPenalty: 1.6,
    minWordsBeforePunct: 1,
    punctuationHardBlock: true,
    minWordTokens: 8,
    maxSentences: 3,
    bigramWeight: 1.3, // Stronger n-gram constraints to curb tangents
    bigramFilter: true,
    strictBigram: false,
    trigramWeight: 1.5, // Stronger n-gram constraints to curb tangents
    trigramFilter: true,
    strictTrigram: false,
    exemplarTopK: 0,
    exemplarWeight: 0.0, // Disable exemplar bias to avoid copying training samples
    exemplarFilter: false,
    logitNorm: true,
    logitNormEps: 0.000001,
    logitScale: 2.0,
    codeOnly: false,
    learningRate: 0.001,   // Boost online updates to keep learning syntax
    decay: 0.9             // Preserve state longer to avoid rapid forgetting
  },
  hormone: {
    mood: ["energy", "respect", "sarcasm"],
    baseline: [0.1, 0.1, 0.1]
  },
  snapshot: {
    enabled: true,           // Enable snapshot support
    autoLoad: true,          // Try to load snapshots automatically
    autoExport: false,       // Auto-export (set to true for open-source builds)
    url: "./model.snapshot"  // Snapshot file path
  }
};
