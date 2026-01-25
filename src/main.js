import { novaConfig } from "../nova.config.js";
import { DeviceContext } from "./core/Device.js";
import { NovaTensor } from "./core/NovaTensor.js";
import { Tokenizer, EntropicCompressor } from "./utils/Tokenizer.js";
import { FastWeightLayer } from "./layers/FastWeight.js";
import { OrigamiMemory } from "./layers/Origami.js";
import { HormoneSystem } from "./layers/HormoneSystem.js";
import { quickstartTraining } from "../scripts/train_browser.js";

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const runBtn = document.getElementById("run");
const inputEl = document.getElementById("input");
const runtimeOverrides = window.novaOverrides || {};
const deferBootstrap = Boolean(window.novaDeferBootstrap);

function log(message) {
  logEl.textContent = `[${new Date().toLocaleTimeString()}] ${message}\n${logEl.textContent}`;
}

function heuristicSummarize(text, stopwordSet) {
  const cleaned = text.replace(/^summarize[:\s]*/i, "").trim();
  if (!cleaned) return "I need something to summarize.";
  const words = cleaned.split(/\s+/).map((w) => w.replace(/[^a-z0-9']/gi, "").toLowerCase()).filter(Boolean);
  const seen = new Set();
  const kept = [];
  for (const w of words) {
    if (stopwordSet.has(w)) continue;
    if (seen.has(w)) continue;
    seen.add(w);
    kept.push(w);
    if (kept.length >= 10) break;
  }
  if (!kept.length) kept.push(...words.slice(0, 6));
  return `A short summary: ${kept.join(" ")}.`;
}

function normalizeWhitespace(text) {
  return text.replace(/\s+/g, " ").trim();
}

function shuffleArray(items) {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = items[i];
    items[i] = items[j];
    items[j] = temp;
  }
}

function mergeConfig(base, override) {
  if (!override || typeof override !== "object") return base;
  const out = typeof structuredClone === "function" ? structuredClone(base) : JSON.parse(JSON.stringify(base));
  const walk = (tgt, src) => {
    for (const key of Object.keys(src)) {
      if (src[key] && typeof src[key] === "object" && !Array.isArray(src[key])) {
        if (!tgt[key] || typeof tgt[key] !== "object") tgt[key] = {};
        walk(tgt[key], src[key]);
      } else {
        tgt[key] = src[key];
      }
    }
  };
  walk(out, override);
  return out;
}

if (runtimeOverrides.configOverride) {
  const merged = mergeConfig(novaConfig, runtimeOverrides.configOverride);
  for (const key of Object.keys(merged)) {
    novaConfig[key] = merged[key];
  }
}

function applyTrainingLimits(lines, config) {
  const maxLines = Math.max(0, config.maxLines ?? 0);
  const shuffle = config.shuffle ?? false;
  if (shuffle) shuffleArray(lines);
  if (maxLines > 0 && lines.length > maxLines) {
    lines.length = maxLines;
  }
  return lines;
}

function splitSentences(text) {
  const matches = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g);
  if (!matches) return [];
  return matches.map((sentence) => normalizeWhitespace(sentence)).filter(Boolean);
}

function parseTrainingData(textData) {
  const rawLines = textData.split("\n").map(normalizeWhitespace).filter(Boolean);
  const chatLines = rawLines.filter((line) => line.startsWith("User:") && line.includes(" AI:"));
  if (chatLines.length) {
    return { mode: "chat", lines: chatLines };
  }
  const sentences = [];
  for (const line of rawLines) {
    const pieces = splitSentences(line);
    if (pieces.length) {
      sentences.push(...pieces);
    } else {
      sentences.push(line);
    }
  }
  return { mode: "plain", lines: sentences };
}

async function fetchText(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return await response.text();
  } catch {
    return null;
  }
}

// Snapshot helpers
async function loadSnapshot(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    const snapshot = await response.json();
    return snapshot;
  } catch (e) {
    console.log("Snapshot not found or invalid:", e);
    return null;
  }
}

async function saveSnapshot(fastLayer, tokenizer) {
  return {
    version: "1.1",
    timestamp: new Date().toISOString(),
    model: {
      embedding: await fastLayer.embedding.read(),
      normGamma: await fastLayer.normGamma.read(),
      wr: await fastLayer.wr.read(),
      wv: await fastLayer.wv.read(),
      wg: await fastLayer.wg.read(),
      ffnUp1: await fastLayer.ffnUp1.read(),
      ffnUp2: await fastLayer.ffnUp2.read(),
      ffnDown: await fastLayer.ffnDown.read(),
      wPredict: await fastLayer.wPredict.read(),
      baseManifold: await fastLayer.baseManifold.read(),
      modeTokens: true
    },
    tokenizer: {
      vocab: Array.from(tokenizer.vocab.entries()),
      reverse: tokenizer.reverse,
      unkId: tokenizer.unkId,
      bosId: tokenizer.bosId,
      eosId: tokenizer.eosId
    }
  };
}

function downloadSnapshot(snapshot, filename = "model.snapshot") {
  const dataStr = JSON.stringify(snapshot);
  const blob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

async function restoreSnapshot(snapshot, fastLayer, tokenizer) {
  const { model, tokenizer: tokenizerData } = snapshot;
  
  // Restore tokenizer state
  tokenizer.vocab = new Map(tokenizerData.vocab);
  tokenizer.reverse = tokenizerData.reverse;
  tokenizer.unkId = tokenizerData.unkId;
  tokenizer.bosId = tokenizerData.bosId;
  tokenizer.eosId = tokenizerData.eosId;
  tokenizer.lock();
  
  // Restore model weights
  fastLayer.embedding.write(new Float32Array(model.embedding));
  fastLayer.normGamma.write(new Float32Array(model.normGamma));
  fastLayer.wr.write(new Float32Array(model.wr));
  fastLayer.wv.write(new Float32Array(model.wv));
  fastLayer.wg.write(new Float32Array(model.wg));
  fastLayer.ffnUp1.write(new Float32Array(model.ffnUp1));
  fastLayer.ffnUp2.write(new Float32Array(model.ffnUp2));
  fastLayer.ffnDown.write(new Float32Array(model.ffnDown));
  fastLayer.wPredict.write(new Float32Array(model.wPredict));
  fastLayer.baseManifold.write(new Float32Array(model.baseManifold));
  
  fastLayer.resetFastWeights();
}

async function bootstrap() {
  try {
    const ctx = await new DeviceContext().init();
    statusEl.textContent = "WebGPU Online. Initializing N.O.V.A. Core...";

    const tokenizer = new Tokenizer({ maxVocab: novaConfig.model.inputSize, lowercase: true });
    ["User:", "AI:", "Mode:", "chat", "task", "code", "{", "}", "(", ")", "[", "]", ",", ".", "!", "?", ";", ":"].forEach((token) => {
      tokenizer.ensureToken(token);
    });
    const blockedTokenIds = new Set([tokenizer.unkId, tokenizer.bosId]);
    const uId = tokenizer.vocab.get("User:"); // Short alias to avoid name clashes
    const aId = tokenizer.vocab.get("AI:");   // Short alias to avoid name clashes
    const modeId = tokenizer.vocab.get("Mode:");
    const chatId = tokenizer.vocab.get("chat");
    const taskId = tokenizer.vocab.get("task");
    const codeId = tokenizer.vocab.get("code");
    if (uId !== undefined) blockedTokenIds.add(uId);
    if (aId !== undefined) blockedTokenIds.add(aId);
    if (modeId !== undefined) blockedTokenIds.add(modeId);
    if (chatId !== undefined) blockedTokenIds.add(chatId);
    if (taskId !== undefined) blockedTokenIds.add(taskId);
    if (codeId !== undefined) blockedTokenIds.add(codeId);
    const pairBase = novaConfig.model.outputSize;


    const hormones = new HormoneSystem(novaConfig.hormone.mood, novaConfig.hormone.baseline);

    const fastLayer = new FastWeightLayer(ctx, {
      ...novaConfig.model,
      thinkingSteps: novaConfig.runtime.thinkingSteps
    });
    await fastLayer.init();
    window.__fastLayer = fastLayer;
    window.__tokenizer = tokenizer;
    fastLayer.setHebbianScalars(novaConfig.runtime.decay, novaConfig.runtime.learningRate);

    // Hoisted globals shared across handlers
    let snapshotLoaded = false;
    let isChatDataset = false;
    let codePromptPattern = null; 
    let ssmGene = null;
    const instructionPromptPattern = /\b(make|write|summarize|summary|rewrite|rephrase|plan|list|suggest|explain|draft|create|help|give|show|turn|convert|please|could you|can you|i need|i want|what should i|how do i)\b/i;
    const greetingPromptPattern = /^(hi|hello|hey|yo|hiya|howdy|greetings|salutations|ahoy)\b|(^|\b)good (morning|afternoon|evening)\b/i;
    
    // These shared collections stay outer so the runBtn handler can reuse them
    const chatIndex = [];
    const replyTokenCounts = new Map();
    const replyTokenSet = new Set();
    const codeReplyTokenCounts = new Map();
    const codeReplyTokenSet = new Set();
    const chatReplyTokenCounts = new Map();
    const chatReplyTokenSet = new Set();
    const taskReplyTokenCounts = new Map();
    const taskReplyTokenSet = new Set();
    const greetingReplyTokenCounts = new Map();
    const greetingReplyTokenSet = new Set();
    const bigramCounts = new Map();
    const codeBigramCounts = new Map();
    const chatBigramCounts = new Map();
    const taskBigramCounts = new Map();
    const greetingBigramCounts = new Map();
    const trigramCounts = new Map();
    const codeTrigramCounts = new Map();
    const chatTrigramCounts = new Map();
    const taskTrigramCounts = new Map();
    const greetingTrigramCounts = new Map();
    
    const punctuationTokens = new Set([",", ".", "!", "?", ";", ":"]);
    const stopwordSet = new Set(["the", "and", "of", "a", "an", "in", "on", "at", "to"]);
    const commandWords = new Set([
      "summarize", "summary", "rewrite", "rephrase", "plan", "list", "suggest",
      "make", "write", "create", "draft", "help", "give", "turn", "convert", "please"
    ]);

    // Snapshot loading flow
    const manualSnapshot = (runtimeOverrides && runtimeOverrides.manualSnapshot) || null;
    if (manualSnapshot) {
      await restoreSnapshot(manualSnapshot, fastLayer, tokenizer);
      snapshotLoaded = true;
      log("✓ Manual snapshot loaded");
      statusEl.textContent = "Ready. Brain is Active.";
    } else if (novaConfig.snapshot?.autoLoad) {
      statusEl.textContent = "Loading Model Snapshot...";
      const snapshot = await loadSnapshot(novaConfig.snapshot.url);
      if (snapshot) {
        const isModeAware = snapshot.version === "1.1" || snapshot.model?.modeTokens === true;
        if (isModeAware) {
          await restoreSnapshot(snapshot, fastLayer, tokenizer);
          snapshotLoaded = true;
          log("✓ Snapshot loaded successfully");
          statusEl.textContent = "Ready. Brain is Active.";
        } else {
          log("⚠ Snapshot format outdated. Retraining with Mode tags...");
        }
      }
    }

    // Training flow when no snapshot is loaded
    if (!snapshotLoaded) {
      if (window.SKIP_TRAINING) {
        throw new Error("Snapshot missing and training is disabled (SKIP_TRAINING=true).");
      }
      statusEl.textContent = "Downloading Knowledge...";
      const trainingConfig = novaConfig.training ?? {};
      const uploadedText = runtimeOverrides?.trainingText || null;
      const chatText = uploadedText || await fetchText("./data/training_data.txt");
      let textData = chatText;
      let trainingSource = uploadedText ? "custom-upload" : "./data/training_data.txt";
      if (!textData) {
        throw new Error("No training data loaded.");
      }
      const trainingData = parseTrainingData(textData);
      const totalTrainingLines = trainingData.lines.length;
      const trainingLines = applyTrainingLimits(trainingData.lines, trainingConfig);
      if (!trainingLines.length) {
        throw new Error("No training lines parsed.");
      }
      const lineInfo = trainingLines.length === totalTrainingLines
        ? `${trainingLines.length}`
        : `${trainingLines.length}/${totalTrainingLines}`;
      log(`Loaded training data: ${trainingSource} (${lineInfo} lines, ${trainingData.mode})`);

      try {
        const compressor = new EntropicCompressor({
          targetVocab: 16384,
          reservedTokens: [tokenizer.unkToken, tokenizer.bosToken, tokenizer.eosToken]
        });
        const start = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
        ssmGene = compressor.train(textData, {
          progressEvery: 1024,
          onProgress: ({ merges, vocabSize }) => {
            if (merges % 4096 === 0) {
              log(`SSM gene flow: merges=${merges}, vocab=${vocabSize}`);
            }
          }
        });
        const end = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
        window.ssmGene = ssmGene;
        log(`SSM gene pool ready (${ssmGene.tokens.length} tokens) in ${(end - start).toFixed(1)} ms`);
      } catch (err) {
        console.warn("Entropic compressor failed", err);
        log("⚠ Entropic compressor failed: " + err.message);
      }
      
      isChatDataset = trainingData.mode === "chat";

      const codeReplyPattern = /\bconst\b|\bfunction\b|=>|console\.log|\/\*|\/\//i;
      codePromptPattern =
        /\b(code|javascript|js|typescript|ts|python|sql|html|css|function|const|let|var|class|import|export)\b|=>|\/\*|\/\//i;
      const addBigramCounts = (tokens, target, weight = 1) => {
        for (let i = 0; i < tokens.length - 1; i += 1) {
          const prev = tokens[i];
          const next = tokens[i + 1];
          let nextMap = target.get(prev);
          if (!nextMap) {
            nextMap = new Map();
            target.set(prev, nextMap);
          }
          nextMap.set(next, (nextMap.get(next) ?? 0) + weight);
        }
      };
      const addTrigramCounts = (tokens, target, weight = 1) => {
        if (tokens.length < 3) return;
        for (let i = 0; i < tokens.length - 2; i += 1) {
          const prev2 = tokens[i];
          const prev1 = tokens[i + 1];
          const next = tokens[i + 2];
          const key = prev2 * pairBase + prev1;
          let nextMap = target.get(key);
          if (!nextMap) {
            nextMap = new Map();
            target.set(key, nextMap);
          }
          nextMap.set(next, (nextMap.get(next) ?? 0) + weight);
        }
      };
      const codeBoost = Math.max(1, Math.floor(trainingConfig.codeBoost ?? 1));
      
      // Populate shared token statistics from the training data
      for (const line of trainingLines) {
        if (isChatDataset) {
          const marker = " AI:";
          const idx = line.indexOf(marker);
          if (idx === -1) continue;
          const userPart = line.slice(0, idx).replace(/^User:/i, "").trim();
          const aiPart = line.slice(idx + marker.length).trim();
          if (!aiPart) continue;
          const isCodeLine = codeReplyPattern.test(aiPart);
          const isTaskLine = instructionPromptPattern.test(userPart);
          const isGreetingLine = greetingPromptPattern.test(userPart);
          const modeLabel = isCodeLine ? "code" : (isTaskLine ? "task" : "chat");
          const fullText = `Mode: ${modeLabel} User: ${userPart} AI: ${aiPart}`;
          const weight = isCodeLine ? codeBoost : 1;
          const replyTokens = tokenizer.tokenize(aiPart, { addBos: true, addEos: true });
          const fullTokens = tokenizer.tokenize(fullText, { addBos: true, addEos: true });
          const userTokens = tokenizer.tokenize(userPart);
          const userTokenSet = new Set(userTokens.filter((tokenId) => {
            const token = tokenizer.reverse[tokenId];
            if (!token) return false;
            if (blockedTokenIds.has(tokenId)) return false;
            if (tokenId === tokenizer.eosId) return false;
            return !punctuationTokens.has(token);
          }));
          
          // Push into the shared chatIndex
          chatIndex.push({ userTokenSet, aiTokens: replyTokens, isCodeLine });
          
          addBigramCounts(fullTokens, bigramCounts, weight);
          addTrigramCounts(fullTokens, trigramCounts, weight);
          for (const tokenId of replyTokens) {
            replyTokenSet.add(tokenId);
            replyTokenCounts.set(tokenId, (replyTokenCounts.get(tokenId) ?? 0) + weight);
            if (isCodeLine) {
              codeReplyTokenSet.add(tokenId);
              codeReplyTokenCounts.set(tokenId, (codeReplyTokenCounts.get(tokenId) ?? 0) + weight);
            } else if (isTaskLine) {
              taskReplyTokenSet.add(tokenId);
              taskReplyTokenCounts.set(tokenId, (taskReplyTokenCounts.get(tokenId) ?? 0) + 1);
            } else if (isGreetingLine) {
              greetingReplyTokenSet.add(tokenId);
              greetingReplyTokenCounts.set(tokenId, (greetingReplyTokenCounts.get(tokenId) ?? 0) + 1);
            } else {
              chatReplyTokenSet.add(tokenId);
              chatReplyTokenCounts.set(tokenId, (chatReplyTokenCounts.get(tokenId) ?? 0) + 1);
            }
          }
          if (isCodeLine) {
            addBigramCounts(fullTokens, codeBigramCounts, weight);
            addTrigramCounts(fullTokens, codeTrigramCounts, weight);
          } else if (isTaskLine) {
            addBigramCounts(fullTokens, taskBigramCounts, 1);
            addTrigramCounts(fullTokens, taskTrigramCounts, 1);
          } else if (isGreetingLine) {
            addBigramCounts(fullTokens, greetingBigramCounts, 1);
            addTrigramCounts(fullTokens, greetingTrigramCounts, 1);
          } else {
            addBigramCounts(fullTokens, chatBigramCounts, 1);
            addTrigramCounts(fullTokens, chatTrigramCounts, 1);
          }
        } else {
          const lineTokens = tokenizer.tokenize(line, { addBos: true, addEos: true });
          addBigramCounts(lineTokens, bigramCounts, 1);
          addTrigramCounts(lineTokens, trigramCounts, 1);
          for (const tokenId of lineTokens) {
            replyTokenSet.add(tokenId);
            replyTokenCounts.set(tokenId, (replyTokenCounts.get(tokenId) ?? 0) + 1);
          }
        }
      }

      statusEl.textContent = "Training Neural Pathways...";
      log(">>> STARTING TRAINING SESSION <<<");
      fastLayer.setHebbianScalars(novaConfig.runtime.decay, novaConfig.runtime.learningRate);
      
      await quickstartTraining(trainingLines, ctx, fastLayer, tokenizer, {
        epochs: trainingConfig.epochs ?? 6,
        contextWindow: trainingConfig.contextWindow ?? (novaConfig.runtime.contextWindow ?? 2),
        contextDecay: trainingConfig.contextDecay ?? (novaConfig.runtime.contextDecay ?? 0.6),
        positionStride: novaConfig.model.positionStride ?? 1,
        userStride: novaConfig.model.userStride ?? novaConfig.model.positionStride ?? 1,
        userContextWeight: novaConfig.runtime.userContextWeight ?? 0.25,
        userContextLimit: novaConfig.runtime.userContextLimit ?? 8,
        codeBoost: trainingConfig.codeBoost ?? 1,
        tokenStride: trainingConfig.tokenStride ?? 1,
        logEvery: trainingConfig.logEvery ?? 0,
        yieldEvery: trainingConfig.yieldEvery ?? 0,
        // Pass the dataset mode through
        mode: trainingData.mode
      });
      await fastLayer.commitFastWeightsToBase();
      fastLayer.setHebbianScalars(novaConfig.runtime.decay, novaConfig.runtime.learningRate);
      fastLayer.resetFastWeights();

      statusEl.textContent = "Training Complete. Brain is Active.";
      log(">>> TRAINING COMPLETE. N.O.V.A. IS LISTENING. <<<");
      
      if (novaConfig.snapshot?.autoExport) {
        log("📦 Exporting model snapshot...");
        const snapshot = await saveSnapshot(fastLayer, tokenizer);
        downloadSnapshot(snapshot, "model.snapshot");
        log("✓ Snapshot exported. Download started.");
      }
    }
    
    log("Try typing: 'Hello'");

    // Export buttons for the open-source build
    const exportBtn = document.getElementById("export");
    if (exportBtn) {
      const vocabBtn = document.createElement("button");
      vocabBtn.id = "export-vocab";
      vocabBtn.textContent = "Export SSM Vocab";
      vocabBtn.style.display = "none";
      exportBtn.insertAdjacentElement("afterend", vocabBtn);
      if (novaConfig.snapshot?.autoExport || !snapshotLoaded) {
        exportBtn.style.display = "inline-block";
        exportBtn.onclick = async () => {
          exportBtn.disabled = true;
          log("📦 Exporting model snapshot...");
          try {
            const snapshot = await saveSnapshot(fastLayer, tokenizer);
            downloadSnapshot(snapshot, "model.snapshot");
            log("✓ Snapshot exported. Download started.");
          } catch (err) {
            log("✗ Export failed: " + err.message);
          }
          exportBtn.disabled = false;
        };
      }
      if (ssmGene) {
        vocabBtn.style.display = "inline-block";
        vocabBtn.onclick = () => {
          if (!ssmGene) return;
          vocabBtn.disabled = true;
          try {
            downloadSnapshot(ssmGene, "ssm_vocab.json");
            log("✓ SSM vocab exported.");
          } catch (err) {
            log("✗ Export SSM vocab failed: " + err.message);
          }
          vocabBtn.disabled = false;
        };
      }
    }

    const origami = new OrigamiMemory(novaConfig.model.hiddenSize);
    let hiddenState = new Float32Array(novaConfig.model.hiddenSize);

    const sentenceEndTokens = new Set([".", "!", "?"]);
    const hormoneBias = hormones.biasVector(novaConfig.model.outputSize);
    const codeOnly = novaConfig.runtime.codeOnly ?? false;
    const frequencyScale = novaConfig.runtime.frequencyBias ?? 0.35;
    const promptTokenBias = novaConfig.runtime.promptTokenBias ?? 0;
    const anchorBonus = novaConfig.runtime.anchorBonus ?? 0;
    const commonPenalty = novaConfig.runtime.commonPenalty ?? 0;
    const summarizeEosBoost = 2.0;
    const summarizeDupPenalty = 4.0;
    const summarizeStopPenalty = 1.2;
    const summarizeRepeatMultiplier = 3.0;
    const summarizeRepeatWindow = 8;
    const greetEosBoost = 1.5;
    const exemplarFilter = novaConfig.runtime.exemplarFilter ?? false;
    const exemplarTopK = Math.max(0, Math.floor(novaConfig.runtime.exemplarTopK ?? 3));
    const exemplarWeight = novaConfig.runtime.exemplarWeight ?? 0.35;
    const startQuotePenalty = novaConfig.runtime.startQuotePenalty ?? 0;
    const contextWindow = Math.max(0, novaConfig.runtime.contextWindow ?? 2);
    const contextDecay = novaConfig.runtime.contextDecay ?? 0.6;
    const contextWeights = new Float32Array(contextWindow + 1);
    for (let i = 0; i <= contextWindow; i += 1) {
      contextWeights[i] = Math.pow(contextDecay, i);
    }
    const userContextWeight = novaConfig.runtime.userContextWeight ?? 0.25;
    const userContextLimit = Math.max(0, novaConfig.runtime.userContextLimit ?? 8);
    const positionStride = Math.max(1, Math.floor(novaConfig.model.positionStride ?? 1));
    const userStride = Math.max(1, Math.floor(novaConfig.model.userStride ?? positionStride));
    const punctuationPenalty = novaConfig.runtime.punctuationPenalty ?? 0.9;
    const doublePunctuationPenalty = novaConfig.runtime.doublePunctuationPenalty ?? 1.6;
    const startPunctuationPenalty = novaConfig.runtime.startPunctuationPenalty ?? 1.4;
    const minWordsBeforePunct = Math.max(0, novaConfig.runtime.minWordsBeforePunct ?? 1);
    const punctuationHardBlock = novaConfig.runtime.punctuationHardBlock ?? true;
    const minWordTokens = Math.max(1, novaConfig.runtime.minWordTokens ?? 3);
    const maxSentences = Math.max(0, novaConfig.runtime.maxSentences ?? 0);
    const bigramWeight = novaConfig.runtime.bigramWeight ?? 0.6;
    const bigramFilter = novaConfig.runtime.bigramFilter ?? true;
    const trigramWeight = novaConfig.runtime.trigramWeight ?? 0.9;
    const trigramFilter = novaConfig.runtime.trigramFilter ?? true;
    const logitNorm = novaConfig.runtime.logitNorm ?? true;
    const logitNormEps = novaConfig.runtime.logitNormEps ?? 1e-6;
    const logitScale = novaConfig.runtime.logitScale ?? 1.0;
    const quoteToken = "\"";
    const isWordToken = (token) => /[a-z0-9]/i.test(token);
    const isNumberToken = (token) => /^[0-9]+$/.test(token);
    
    const scoreExemplars = (userContextTokens, useCodeTokens) => {
      if (!isChatDataset || exemplarTopK <= 0 || !chatIndex.length) return null;
      const scores = [];
      for (const entry of chatIndex) {
        if (useCodeTokens && !entry.isCodeLine) continue;
        if (!useCodeTokens && entry.isCodeLine) continue;
        let score = 0;
        for (const tokenId of userContextTokens) {
          if (entry.userTokenSet.has(tokenId)) score += 1;
        }
        if (score > 0) scores.push({ entry, score });
      }
      if (!scores.length) return null;
      scores.sort((a, b) => b.score - a.score);
      return scores.slice(0, exemplarTopK);
    };
    
    const buildExemplarTokenSet = (exemplars) => {
      const tokenSet = new Set();
      for (const { entry } of exemplars) {
        for (const tokenId of entry.aiTokens) {
          if (tokenId === tokenizer.bosId || tokenId === tokenizer.eosId || tokenId === tokenizer.unkId) continue;
          tokenSet.add(tokenId);
        }
      }
      return tokenSet;
    };
    
    const applyExemplarBias = (biasVec, exemplars, allowedSet) => {
      if (!exemplars?.length) return;
      for (const { entry, score } of exemplars) {
        const weight = exemplarWeight * score;
        for (const tokenId of entry.aiTokens) {
          if (allowedSet && !allowedSet.has(tokenId)) continue;
          biasVec[tokenId] += weight;
        }
      }
    };
    
    const normalizeOutput = (scores) => {
      let mean = 0;
      for (let i = 0; i < scores.length; i += 1) {
        mean += scores[i];
      }
      mean /= scores.length || 1;
      let variance = 0;
      for (let i = 0; i < scores.length; i += 1) {
        const diff = scores[i] - mean;
        variance += diff * diff;
      }
      const invStd = 1 / Math.sqrt(variance / (scores.length || 1) + logitNormEps);
      for (let i = 0; i < scores.length; i += 1) {
        scores[i] = (scores[i] - mean) * invStd * logitScale;
      }
      return scores;
    };

    const buildInputVector = (historyTokens, userContextTokens, opts = {}) => {
      const userScale = opts.userScale ?? 1;
      const inputVec = new Float32Array(novaConfig.model.inputSize);
      let weightIdx = 0;
      for (let idx = historyTokens.length - 1; idx >= 0; idx -= 1) {
        if (weightIdx > contextWindow) break;
        const tokenId = historyTokens[idx];
        const safeId = tokenId < novaConfig.model.inputSize ? tokenId : tokenizer.unkId;
        const mappedId = (safeId + weightIdx * positionStride) % inputVec.length;
        inputVec[mappedId] += contextWeights[weightIdx];
        weightIdx += 1;
      }
      if (userContextWeight > 0 && userContextTokens?.length) {
        const maxTokens = Math.min(userContextTokens.length, userContextLimit);
        const scaledWeight = userContextWeight * userScale;
        for (let i = 0; i < maxTokens; i += 1) {
          const tokenId = userContextTokens[i];
          const safeId = tokenId < novaConfig.model.inputSize ? tokenId : tokenizer.unkId;
          const mappedId = (safeId + userStride) % inputVec.length;
          inputVec[mappedId] += scaledWeight;
        }
      }
      return inputVec;
    };

    const pickToken = (candidates, temperature, topK) => {
      if (!candidates.length) return null;
      candidates.sort((a, b) => b.score - a.score);
      const limit = Math.min(topK, candidates.length);
      if (!temperature || temperature <= 0) {
        return { id: candidates[0].id, prob: 1 };
      }
      const top = candidates.slice(0, limit);
      const maxScore = top[0].score;
      let sum = 0;
      const weights = top.map((entry) => {
        const val = Math.exp((entry.score - maxScore) / temperature);
        sum += val;
        return val;
      });
      let r = Math.random() * sum;
      for (let i = 0; i < top.length; i += 1) {
        r -= weights[i];
        if (r <= 0) {
          return { id: top[i].id, prob: weights[i] / sum };
        }
      }
      const lastIdx = top.length - 1;
      return { id: top[lastIdx].id, prob: weights[lastIdx] / sum };
    };

    runBtn.onclick = async () => {
      const raw = inputEl.value.trim();
      if (!raw) return;

      const userText = raw.startsWith("User:") ? raw.slice("User:".length).trim() : raw;
      const promptUserText = userText;
      const promptHasBrackets = /[{}[\]()]/.test(promptUserText);

      const userTokens = tokenizer.tokenize(promptUserText);
      const userContextTokens = Array.from(new Set(userTokens)).filter((id) => {
        const token = tokenizer.reverse[id];
        if (!token) return false;
        if (blockedTokenIds.has(id)) return false;
        if (id === tokenizer.eosId) return false;
        return !punctuationTokens.has(token);
      });
      const normalizedPrompt = promptUserText.toLowerCase();
      const isInstructionPrompt = instructionPromptPattern.test(normalizedPrompt);
      const isGreetingPrompt = greetingPromptPattern.test(normalizedPrompt);
      const isCodePrompt = isChatDataset && codePromptPattern.test(promptUserText);
      const promptHasDigit = /[0-9]/.test(promptUserText);
      const isSummarizePrompt = /^summarize[:\s]/i.test(promptUserText.trim());
      const useCodeTokens = isChatDataset && (codeOnly || isCodePrompt);
      const useTaskTokens = isChatDataset && !useCodeTokens && isInstructionPrompt && taskReplyTokenSet.size > 0;
      const useGreetingTokens = isChatDataset && !useCodeTokens && !useTaskTokens && isGreetingPrompt && greetingReplyTokenSet.size > 0;
      const modeLabel = useCodeTokens ? "code" : (useTaskTokens ? "task" : "chat");
      const promptTokens = isChatDataset
        ? tokenizer.tokenize(`Mode: ${modeLabel} User: ${promptUserText} AI:`)
        : tokenizer.tokenize(promptUserText);
      const exemplars = scoreExemplars(userContextTokens, useCodeTokens);
      const activeReplyTokenCounts = useCodeTokens && codeReplyTokenCounts.size
        ? codeReplyTokenCounts
        : (useGreetingTokens && greetingReplyTokenCounts.size
          ? greetingReplyTokenCounts
          : (useTaskTokens && taskReplyTokenCounts.size
            ? taskReplyTokenCounts
            : (chatReplyTokenCounts.size ? chatReplyTokenCounts : replyTokenCounts)));
      const activeBigramCounts = useCodeTokens && codeBigramCounts.size
        ? codeBigramCounts
        : (useGreetingTokens && greetingBigramCounts.size
          ? greetingBigramCounts
          : (useTaskTokens && taskBigramCounts.size
            ? taskBigramCounts
            : (chatBigramCounts.size ? chatBigramCounts : bigramCounts)));
      const activeTrigramCounts = useCodeTokens && codeTrigramCounts.size
        ? codeTrigramCounts
        : (useGreetingTokens && greetingTrigramCounts.size
          ? greetingTrigramCounts
          : (useTaskTokens && taskTrigramCounts.size
            ? taskTrigramCounts
            : (chatTrigramCounts.size ? chatTrigramCounts : trigramCounts)));
      const activeReplyTokenSet = useCodeTokens && codeReplyTokenSet.size
        ? codeReplyTokenSet
        : (useGreetingTokens && greetingReplyTokenSet.size
          ? greetingReplyTokenSet
          : (useTaskTokens && taskReplyTokenSet.size
            ? taskReplyTokenSet
            : (chatReplyTokenSet.size ? chatReplyTokenSet : replyTokenSet)));
      let allowedOutputTokenIds = new Set(activeReplyTokenSet);
      // Allow prompt tokens as optional outputs to stay on-topic
      for (const t of userContextTokens) {
        allowedOutputTokenIds.add(t);
      }
      allowedOutputTokenIds.add(tokenizer.eosId);
      for (const blocked of blockedTokenIds) {
        allowedOutputTokenIds.delete(blocked);
      }
      const anchorTokenSet = new Set(
        userContextTokens.filter((id) => {
          const tok = tokenizer.reverse[id];
          if (!tok) return false;
          if (blockedTokenIds.has(id)) return false;
          const lower = tok.toLowerCase();
          if (commandWords.has(lower)) return false;
          if (isSummarizePrompt && stopwordSet.has(lower)) return false;
          return true;
        })
      );
      if (!promptHasBrackets) {
        const bracketTokens = ["{", "}", "[", "]", "(", ")"];
        for (const token of bracketTokens) {
          const tokenId = tokenizer.vocab.get(token);
          if (tokenId !== undefined) {
            allowedOutputTokenIds.delete(tokenId);
          }
        }
      }
      if (isSummarizePrompt) {
        const summaryAllowed = new Set(anchorTokenSet);
        const bridge = ["and", "of", "on", "the"];
        for (const b of bridge) {
          const id = tokenizer.vocab.get(b);
          if (id !== undefined && !blockedTokenIds.has(id)) summaryAllowed.add(id);
        }
        summaryAllowed.add(tokenizer.eosId);
        // Add stopwords as a fallback; scoring will penalize them later to avoid collapse
        for (const t of stopwordSet) {
          const id = tokenizer.vocab.get(t);
          if (id !== undefined && !blockedTokenIds.has(id)) summaryAllowed.add(id);
        }
        allowedOutputTokenIds = summaryAllowed;
      }
      const exemplarTokenSet = exemplars ? buildExemplarTokenSet(exemplars) : null;
      if (exemplarFilter && exemplarTokenSet?.size) {
        for (const tokenId of Array.from(allowedOutputTokenIds)) {
          if (tokenId === tokenizer.eosId) continue;
          const token = tokenizer.reverse[tokenId];
          if (punctuationTokens.has(token)) continue;
          if (!exemplarTokenSet.has(tokenId)) {
            allowedOutputTokenIds.delete(tokenId);
          }
        }
      }
      const effectiveFrequencyScale = isSummarizePrompt ? 0 : frequencyScale;
      const frequencyBias = new Float32Array(novaConfig.model.outputSize);
      if (effectiveFrequencyScale > 0) {
        for (const [tokenId, count] of activeReplyTokenCounts.entries()) {
          if (!allowedOutputTokenIds.has(tokenId)) continue;
          frequencyBias[tokenId] = Math.log(1 + count) * effectiveFrequencyScale;
        }
      }
      const promptBias = new Float32Array(novaConfig.model.outputSize);
      if (promptTokenBias > 0) {
        for (const tokenId of userContextTokens) {
          if (!allowedOutputTokenIds.has(tokenId)) continue;
          const token = tokenizer.reverse[tokenId];
          if (!token || punctuationTokens.has(token)) continue;
          promptBias[tokenId] += promptTokenBias;
        }
      }
      const exemplarBias = new Float32Array(novaConfig.model.outputSize);
      applyExemplarBias(exemplarBias, exemplars, allowedOutputTokenIds);
      // Common high-frequency token set to suppress generic replies
      const commonTokenSet = new Set();
      if (commonPenalty > 0) {
        const sorted = Array.from(activeReplyTokenCounts.entries()).sort((a, b) => b[1] - a[1]);
        const cap = Math.min(sorted.length, 16);
        for (let i = 0; i < cap; i += 1) {
          commonTokenSet.add(sorted[i][0]);
        }
      }
      const replyLimit = Math.max(1, novaConfig.runtime.maxReplyTokens ?? 64);
      const minReplyTokens = Math.min(replyLimit, Math.max(1, novaConfig.runtime.minReplyTokens ?? 4));
      const temperature = novaConfig.runtime.temperature ?? 0.85;
      const topK = Math.max(1, novaConfig.runtime.topK ?? 8);
      const repeatPenalty = novaConfig.runtime.repeatPenalty ?? 0.6;
      let repeatWindow = Math.max(1, novaConfig.runtime.repeatWindow ?? 6);
      const effectiveMinReplyTokens = isGreetingPrompt
        ? Math.max(3, minReplyTokens)
        : (isInstructionPrompt ? Math.min(minReplyTokens, 2) : minReplyTokens);
      const effectiveMinWordTokens = isGreetingPrompt
        ? Math.max(3, minWordTokens)
        : (isInstructionPrompt ? Math.min(minWordTokens, 4) : minWordTokens);
      const effectiveMaxSentences = isGreetingPrompt || isInstructionPrompt ? 1 : maxSentences;
      const summarizeWordCap = isSummarizePrompt ? 10 : Infinity;
      const greetWordCap = isGreetingPrompt ? 8 : Infinity;
      const effectiveRepeatPenalty = isSummarizePrompt ? repeatPenalty * summarizeRepeatMultiplier : repeatPenalty;
      if (isSummarizePrompt) {
        repeatWindow = summarizeRepeatWindow;
      }

      fastLayer.resetFastWeights();
      hiddenState = new Float32Array(novaConfig.model.hiddenSize);

      runBtn.disabled = true;
      statusEl.textContent = "Thinking...";

      // Fast path: heuristic-only summary or greeting to avoid fragmented replies
      if (isSummarizePrompt) {
        const summary = heuristicSummarize(promptUserText, stopwordSet);
        log(`\n>>> Full Reply: ${summary}`);
        statusEl.textContent = "Ready.";
        runBtn.disabled = false;
        return;
      }

      const inputTensor = new NovaTensor(ctx.device, [novaConfig.model.inputSize]);
      try {
        // Prefill
        const allowHebbian = novaConfig.runtime.learningRate > 0;
        const historyTokens = [];
        for (let t = 0; t < promptTokens.length; t++) {
          const tokenId = promptTokens[t];
          historyTokens.push(tokenId);
          const token = tokenizer.reverse[tokenId];
          if (token === "{" || token === "}") {
            hiddenState = origami.route(token, hiddenState);
          }
          const inputVec = buildInputVector(historyTokens, userContextTokens);
          inputTensor.write(inputVec);

          await fastLayer.forwardToken(tokenId, {
            applyHebbian: allowHebbian,
            thinkingSteps: novaConfig.runtime.thinkingSteps
          });
        }

        // Generation
        statusEl.textContent = "Generating...";
        const generated = [];
        const recentQueue = [];
        const recentCounts = new Map();
        const tokenUsage = new Map();
        let lastWasPunct = false;
        let wordCount = 0;
        let sentenceCount = 0;

      const pushRecent = (tokenId) => {
        recentQueue.push(tokenId);
        recentCounts.set(tokenId, (recentCounts.get(tokenId) ?? 0) + 1);
        if (recentQueue.length > repeatWindow) {
          const removed = recentQueue.shift();
          const count = (recentCounts.get(removed) ?? 1) - 1;
          if (count <= 0) {
            recentCounts.delete(removed);
          } else {
            recentCounts.set(removed, count);
          }
        }
      };

        for (let gen = 0; gen < replyLimit; gen++) {
          const userScale = gen === 0 ? (novaConfig.runtime.firstTokenUserScale ?? 1) : 1;
          const inputVec = buildInputVector(historyTokens, userContextTokens, { userScale });
          inputTensor.write(inputVec);

          const contextToken = historyTokens.length
            ? historyTokens[historyTokens.length - 1]
            : tokenizer.bosId;
          const outputTensor = await fastLayer.forwardToken(contextToken, {
            applyHebbian: false,
            thinkingSteps: novaConfig.runtime.thinkingSteps
          });
          let rawOutput;
          try {
          rawOutput = await outputTensor.read();
        } catch (err) {
          console.error(err);
          statusEl.textContent = `ERROR reading tensor: ${err.message}`;
          runBtn.disabled = false;
          return;
        }
        const scores = logitNorm ? normalizeOutput(rawOutput) : rawOutput;

        const candidates = [];
        const disallowEos = gen < effectiveMinReplyTokens || wordCount < effectiveMinWordTokens;
        const isFirst = generated.length === 0;
        const contextLastToken = historyTokens.length ? historyTokens[historyTokens.length - 1] : tokenizer.bosId;
        const contextPrevToken = historyTokens.length >= 2 ? historyTokens[historyTokens.length - 2] : tokenizer.bosId;
        const lastTokenForBigram = isFirst ? contextLastToken : generated[generated.length - 1];
        const prevTokenForTrigram = isFirst
          ? contextPrevToken
          : (generated.length >= 2 ? generated[generated.length - 2] : contextLastToken);
        const trigramKey = prevTokenForTrigram * pairBase + lastTokenForBigram;
        const nextMap = activeBigramCounts.get(lastTokenForBigram);
        const nextTriMap = activeTrigramCounts.get(trigramKey);
        const strictBigram = isChatDataset && nextMap && nextMap.size > 0;
        const strictTrigram = isChatDataset && nextTriMap && nextTriMap.size > 0;
        const allowOnlyEos = false; // Avoid forcing EOS when statistics are sparse
        const allowEosNow = !disallowEos;

        let stepAllowed = allowedOutputTokenIds;
        const triSize = nextTriMap ? nextTriMap.size : 0;
        const biSize = nextMap ? nextMap.size : 0;
        if (triSize >= 2) {
          const narrowed = new Set();
          for (const key of nextTriMap.keys()) narrowed.add(key);
          if (allowEosNow) narrowed.add(tokenizer.eosId);
          stepAllowed = narrowed;
        } else if (biSize >= 3) {
          const narrowed = new Set();
          for (const key of nextMap.keys()) narrowed.add(key);
          if (allowEosNow) narrowed.add(tokenizer.eosId);
          stepAllowed = narrowed;
        } else if (triSize === 1 || biSize === 1) {
          // Keep the original allowed set if stats are too narrow to avoid dead-ends
          stepAllowed = allowedOutputTokenIds;
        }

        const collectCandidates = (
          allowEos = false,
          onlyEos = false,
          enforceNgram = true,
          anchorOnly = false,
          overrideSet = null
        ) => {
          const list = [];
          const useSet = overrideSet ?? stepAllowed;
          for (let i = 0; i < scores.length; i++) {
            const token = tokenizer.reverse[i];
            if (!token) continue;
            if (blockedTokenIds.has(i)) continue;
            if (!useSet.has(i)) continue;
            if (onlyEos && i !== tokenizer.eosId) continue;
            if (!allowEos && disallowEos && i === tokenizer.eosId) continue;
            if (enforceNgram) {
              if (trigramFilter && nextTriMap && !nextTriMap.has(i)) continue;
              if (!nextTriMap && bigramFilter && nextMap && !nextMap.has(i)) continue;
            }
            const anchorHit = anchorTokenSet.has(i);
            if (anchorOnly && !anchorHit && i !== tokenizer.eosId) continue;
            const anchorForce = isInstructionPrompt && gen < 3;
            if (anchorForce && !anchorHit && i !== tokenizer.eosId) continue;
            if (!promptHasDigit && isNumberToken(token) && !anchorHit) continue;
            if (isSummarizePrompt && gen === 0 && stopwordSet.has(token)) continue;
            if (isSummarizePrompt && anchorTokenSet.size && !anchorHit && i !== tokenizer.eosId) continue;

            const isPunct = punctuationTokens.has(token);
            if (punctuationHardBlock && wordCount < minWordsBeforePunct && isPunct) continue;
            if (punctuationHardBlock && lastWasPunct && isPunct) continue;

            let score = scores[i] + hormoneBias[i] + frequencyBias[i] + promptBias[i] + exemplarBias[i];
            if (anchorBonus > 0 && anchorHit) score += anchorBonus;
            if (commonPenalty > 0 && commonTokenSet.has(i) && !anchorHit) {
              score -= commonPenalty;
            }
            if (isSummarizePrompt && stopwordSet.has(token)) {
              score -= summarizeStopPenalty;
            }
            if (isSummarizePrompt && i === tokenizer.eosId && wordCount >= 6) {
              score += summarizeEosBoost;
            }
            if (isSummarizePrompt && (tokenUsage.get(i) ?? 0) >= 1 && !anchorHit) {
              score -= summarizeDupPenalty;
            }
            if (bigramWeight > 0 && nextMap?.has(i)) {
              score += bigramWeight * Math.log(1 + (nextMap.get(i) ?? 0));
            }
            if (trigramWeight > 0 && nextTriMap?.has(i)) {
              score += trigramWeight * Math.log(1 + (nextTriMap.get(i) ?? 0));
            }
            if (isFirst && isPunct) score -= startPunctuationPenalty;
            if (isFirst && startQuotePenalty > 0 && token === quoteToken) score -= startQuotePenalty;
            if (lastWasPunct && isPunct) score -= doublePunctuationPenalty;
            if (isPunct && wordCount < minWordsBeforePunct) score -= punctuationPenalty;
            const recentCount = recentCounts.get(i) ?? 0;
            if (recentCount > 0) score -= effectiveRepeatPenalty * recentCount;
            if (isGreetingPrompt && i === tokenizer.eosId && wordCount >= 3) {
              score += greetEosBoost;
            }

            list.push({ id: i, score });
          }
          return list;
        };

        // Prefer anchor tokens for the first position when available
        if (anchorTokenSet.size && gen === 0) {
          candidates.push(...collectCandidates(allowEosNow, allowOnlyEos, true, true));
        }
        if (!candidates.length) {
          candidates.push(...collectCandidates(allowEosNow, allowOnlyEos, true));
        }
        if (!candidates.length) {
          // Retry with anchor-only candidates to prevent drifting off-topic
          candidates.push(...collectCandidates(allowEosNow, allowOnlyEos, true, true));
        }
        if (!candidates.length) {
          candidates.push(...collectCandidates(allowEosNow, allowOnlyEos, false, true));
        }
        if (!candidates.length || candidates.length <= 1) {
          // Relax n-gram constraints and use the full allowed set if too few options remain
          candidates.push(...collectCandidates(allowEosNow, allowOnlyEos, false, false, allowedOutputTokenIds));
        }
        if (candidates.length <= 1) {
          // Final fallback: drop anchor requirement entirely
          candidates.push(...collectCandidates(allowEosNow, allowOnlyEos, false, false, allowedOutputTokenIds));
        }
        if (!candidates.length) break;

        const picked = pickToken(candidates, temperature, topK);
        if (!picked) break;
        if (picked.id === tokenizer.eosId) break;

        const pickedToken = tokenizer.reverse[picked.id];
        generated.push(picked.id);
        log(`AI Token: [${pickedToken}] (p=${picked.prob.toFixed(2)})`);
        pushRecent(picked.id);
        historyTokens.push(picked.id);
        lastWasPunct = punctuationTokens.has(pickedToken);
        if (isWordToken(pickedToken)) wordCount += 1;
        tokenUsage.set(picked.id, (tokenUsage.get(picked.id) ?? 0) + 1);
        if (isSummarizePrompt && wordCount >= summarizeWordCap) {
          break;
        }
        if (isGreetingPrompt && wordCount >= greetWordCap) {
          break;
        }
        if (sentenceEndTokens.has(pickedToken)) {
          sentenceCount += 1;
          if (isChatDataset && !useCodeTokens && effectiveMaxSentences > 0 && sentenceCount >= effectiveMaxSentences) {
            if (wordCount >= effectiveMinWordTokens) break;
          }
        }
        }

        const outputText = tokenizer.detokenize(generated);
        log(`\n>>> Full Reply: ${outputText}`);
        statusEl.textContent = "Ready.";
        runBtn.disabled = false;
      } finally {
        inputTensor.dispose();
      }
    };
  } catch (err) {
    console.error(err);
    statusEl.textContent = `CRITICAL ERROR: ${err.message}`;
  }
}

if (deferBootstrap) {
  window.startNova = bootstrap;
  log("Nova deferred. Use start button to boot.");
} else {
  bootstrap();
}
