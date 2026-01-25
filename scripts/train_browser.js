import { NovaTensor } from "../src/core/NovaTensor.js";

export async function quickstartTraining(lines, ctx, fastLayer, tokenizer, opts = {}) {
  const epochs = opts.epochs ?? 3;
  const contextWindow = Math.max(0, opts.contextWindow ?? 0);
  const contextDecay = opts.contextDecay ?? 0.6;
  const positionStride = Math.max(1, Math.floor(opts.positionStride ?? 1));
  const userStride = Math.max(1, Math.floor(opts.userStride ?? 1));
  const mode = opts.mode ?? "plain";
  const userContextWeight = opts.userContextWeight ?? 0.25;
  const userContextLimit = Math.max(0, Math.floor(opts.userContextLimit ?? 8));
  const codeBoost = Math.max(1, Math.floor(opts.codeBoost ?? 1));
  const tokenStride = Math.max(1, Math.floor(opts.tokenStride ?? 1));
  const punctuationTokens = new Set([",", ".", "!", "?", ";", ":"]);
  const instructionPromptPattern = /\b(make|write|summarize|summary|rewrite|rephrase|plan|list|suggest|explain|draft|create|help|give|show|turn|convert|please|could you|can you|i need|i want|what should i|how do i)\b/i;
  const contextWeights = [];
  const totalLines = lines.length;
  const logEvery = Math.max(0, Math.floor(opts.logEvery ?? 0));
  const yieldEvery = Math.max(0, Math.floor(opts.yieldEvery ?? 0));
  const yieldToUI = () => new Promise((resolve) => setTimeout(resolve, 0));
  
  for (let i = 0; i <= contextWindow; i += 1) {
    contextWeights.push(Math.pow(contextDecay, i));
  }

  console.log(">>> SURGICAL TRAINING STARTED <<<");
  
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    console.log(`Epoch ${epoch + 1}/${epochs}`);
    for (let lineIdx = 0; lineIdx < lines.length; lineIdx += 1) {
      const line = lines[lineIdx];
      
      if (logEvery > 0 && lineIdx % logEvery === 0) {
        console.log(`Learning sequence ${lineIdx + 1}/${totalLines}: "${line}"`);
      }
      
      const repeats = /\b(code|javascript|function)\b/i.test(line) ? codeBoost : 1;
      for (let r = 0; r < repeats; r += 1) {
        if (mode === "chat") {
          const marker = " AI:";
          const idx = line.indexOf(marker);
          if (idx === -1) continue;
          const userPart = line.slice(0, idx).replace(/^User:/i, "").trim();
          const aiPart = line.slice(idx + marker.length).trim();
          if (!aiPart) continue;

          const isTaskLine = instructionPromptPattern.test(userPart);
          const modeLabel = isTaskLine ? "task" : "chat";
          const fullText = `Mode: ${modeLabel} User: ${userPart} AI: ${aiPart}`;
          const fullTokens = tokenizer.tokenize(fullText, { addBos: true, addEos: true });
          const aiTokenId = tokenizer.vocab.get("AI:");
          let aiStartIndex = -1;
          if (aiTokenId !== undefined) {
            for (let i = 0; i < fullTokens.length; i += 1) {
              if (fullTokens[i] === aiTokenId) {
                aiStartIndex = i + 1;
                break;
              }
            }
          }

          for (let i = 0; i < fullTokens.length - 1; i += tokenStride) {
            const prevToken = fullTokens[i];
            const nextToken = fullTokens[i + 1];
            const nextIndex = i + 1;
            const shouldUpdate = aiStartIndex >= 0 && nextIndex >= aiStartIndex;

            await fastLayer.forwardToken(prevToken, {
              targetTokenId: nextToken,
              applyHebbian: shouldUpdate
            });
          }
        } else {
          const tokens = tokenizer.tokenize(line, { addBos: true, addEos: true });

          for (let i = 0; i < tokens.length - 1; i += tokenStride) {
            const prevToken = tokens[i];
            const nextToken = tokens[i + 1];

            // Apply the Hebbian update step
            await fastLayer.forwardToken(prevToken, {
              targetTokenId: nextToken,
              applyHebbian: true
            });
          }
        }
      }
      
      if (yieldEvery > 0 && (lineIdx + 1) % yieldEvery === 0) {
        await yieldToUI();
      }
    }
  }
  
  console.log(">>> TRAINING COMPLETE <<<");
}
