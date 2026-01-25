export class Tokenizer {
  constructor(opts = {}) {
    this.maxVocab = opts.maxVocab ?? 256;
    this.lowercase = opts.lowercase ?? true;
    this.vocab = new Map();
    this.reverse = [];
    this.locked = false;
    this.unkToken = "<unk>";
    this.bosToken = "<bos>";
    this.eosToken = "<eos>";

    this.unkId = this.reserveToken(this.unkToken);
    this.bosId = this.reserveToken(this.bosToken);
    this.eosId = this.reserveToken(this.eosToken);
  }

  reserveToken(token) {
    const id = this.reverse.length;
    this.vocab.set(token, id);
    this.reverse[id] = token;
    return id;
  }

  lock() {
    this.locked = true;
  }

  normalizeText(text) {
    if (!text) return "";
    let out = text.replace(/User:/g, "User: ").replace(/AI:/g, "AI: ");
    out = out.replace(/\s+/g, " ").trim();
    return out;
  }

  normalizeToken(token) {
    if (token === "User:" || token === "AI:") return token;
    let out = token;
    if (this.lowercase) out = out.toLowerCase();
    if (!/^[\x20-\x7E]+$/.test(out)) return this.unkToken;
    return out;
  }

  split(text) {
    const cleaned = this.normalizeText(text);
    if (!cleaned) return [];
    const pattern = /User:|AI:|[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?|\/\/|\/\*|\*\/|===|!==|==|!=|<=|>=|=>|\+\+|--|\+=|-=|\*=|\/=|&&|\|\||[{}()[\],.!?;:"'<>+=\-*/%&|^~]/g;
    const raw = cleaned.match(pattern);
    if (!raw) return [];
    return raw.map((token) => this.normalizeToken(token));
  }

  ensureToken(token) {
    const normalized = this.normalizeToken(token);
    const existing = this.vocab.get(normalized);
    if (existing !== undefined) return existing;
    if (this.locked) return this.unkId;
    if (this.reverse.length >= this.maxVocab) return this.unkId;
    const id = this.reverse.length;
    this.vocab.set(normalized, id);
    this.reverse[id] = normalized;
    return id;
  }

  tokenize(text, opts = {}) {
    const { addBos = false, addEos = false } = opts;
    const tokens = this.split(text);
    const ids = [];
    if (addBos) ids.push(this.bosId);
    for (const token of tokens) {
      ids.push(this.ensureToken(token));
    }
    if (addEos) ids.push(this.eosId);
    return ids;
  }

  detokenize(ids) {
    const tokens = [];
    for (const id of ids) {
      const token = this.reverse[id];
      if (!token) continue;
      if (token === this.unkToken || token === this.bosToken || token === this.eosToken) continue;
      tokens.push(token);
    }
    return this.join(tokens);
  }

  join(tokens) {
    if (!tokens.length) return "";
    const noSpaceBefore = new Set([",", ".", "!", "?", ";", ":", ")", "]", "}"]);
    const noSpaceAfter = new Set(["(", "[", "{"]);
    let text = "";
    let prev = "";
    let inQuote = false;
    for (const token of tokens) {
      if (!text) {
        text = token;
        prev = token;
        if (token === "\"") inQuote = true;
        continue;
      }
      if (token === "\"") {
        if (inQuote) {
          text += token;
          inQuote = false;
        } else {
          if (noSpaceAfter.has(prev) || prev === "User:" || prev === "AI:") {
            text += token;
          } else {
            text += ` ${token}`;
          }
          inQuote = true;
        }
        prev = token;
        continue;
      }
      if (inQuote) {
        if (prev === "\"") {
          text += token;
        } else if (noSpaceBefore.has(token)) {
          text += token;
        } else {
          text += ` ${token}`;
        }
        prev = token;
        continue;
      }
      if (noSpaceBefore.has(token)) {
        text += token;
      } else if (noSpaceAfter.has(prev) || prev === "User:" || prev === "AI:") {
        text += token;
      } else {
        text += ` ${token}`;
      }
      prev = token;
    }
    return text;
  }
}

/**
 * EntropicCompressor
 * 贪婪字节级 BPE 训练器，用于为特定语料自适应地产生 16k 级别的高密度基因表。
 */
export class EntropicCompressor {
  constructor(opts = {}) {
    this.targetVocab = opts.targetVocab ?? 16384;
    this.reservedTokens = opts.reservedTokens ?? ["<unk>", "<bos>", "<eos>"];
    this.minFrequency = opts.minFrequency ?? 1;
  }

  renderTokenLabel(bytes, fallbackId) {
    if (!bytes?.length) return `<r${fallbackId}>`;
    const printable = bytes.length <= 12 && bytes.every((b) => b >= 32 && b < 127);
    if (printable) {
      const text = String.fromCharCode(...bytes).replace(/"/g, '\\"');
      return `text:"${text}"`;
    }
    const hex = bytes.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `hex:${hex}`.slice(0, 64);
  }

  mergeTokens(tokens, pair, newId) {
    const merged = [];
    for (let i = 0; i < tokens.length; i += 1) {
      if (i < tokens.length - 1 && tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
        merged.push(newId);
        i += 1;
      } else {
        merged.push(tokens[i]);
      }
    }
    return merged;
  }

  train(text, opts = {}) {
    const encoder = new TextEncoder();
    const corpusBytes = encoder.encode(text ?? "");
    if (corpusBytes.length === 0) {
      throw new Error("EntropicCompressor: corpus is empty.");
    }

    const targetVocab = Math.max(this.reservedTokens.length + 256, opts.targetVocab ?? this.targetVocab);
    const minFrequency = Math.max(1, opts.minFrequency ?? this.minFrequency);
    const progressEvery = Math.max(0, opts.progressEvery ?? 0);
    const onProgress = typeof opts.onProgress === "function" ? opts.onProgress : null;

    const vocab = [...this.reservedTokens];
    const idToBytes = [];
    for (let i = 0; i < this.reservedTokens.length; i += 1) idToBytes.push([]);
    for (let b = 0; b < 256; b += 1) {
      vocab.push(`b:${b.toString(16).padStart(2, "0")}`);
      idToBytes.push([b]);
    }
    let tokens = Array.from(corpusBytes, (b) => this.reservedTokens.length + b);
    const merges = [];

    while (vocab.length < targetVocab && tokens.length > 1) {
      let bestPair = null;
      let bestCount = 0;
      const pairCounts = new Map();

      for (let i = 0; i < tokens.length - 1; i += 1) {
        const left = tokens[i];
        const right = tokens[i + 1];
        const key = (left << 16) ^ right;
        const count = (pairCounts.get(key) ?? 0) + 1;
        pairCounts.set(key, count);
        if (count > bestCount) {
          bestCount = count;
          bestPair = [left, right];
        }
      }

      if (!bestPair || bestCount < minFrequency) break;
      if (vocab.length >= targetVocab) break;

      const newId = vocab.length;
      const mergedBytes = [
        ...(idToBytes[bestPair[0]] ?? []),
        ...(idToBytes[bestPair[1]] ?? [])
      ];
      vocab.push(this.renderTokenLabel(mergedBytes, newId));
      idToBytes.push(mergedBytes);
      merges.push({ id: newId, pair: bestPair, freq: bestCount });

      tokens = this.mergeTokens(tokens, bestPair, newId);

      if (onProgress && progressEvery > 0 && merges.length % progressEvery === 0) {
        onProgress({
          merges: merges.length,
          vocabSize: vocab.length,
          bestCount,
          sequenceLength: tokens.length
        });
      }
    }

    const tokenMap = {};
    vocab.forEach((tok, id) => {
      tokenMap[tok] = id;
    });

    return {
      version: "ssm-entropy-1",
      meta: {
        targetVocab,
        actualVocab: vocab.length,
        merges: merges.length,
        corpusBytes: corpusBytes.length,
        reserved: this.reservedTokens.length,
        baseAlphabet: 256
      },
      vocab,
      tokens: vocab.map((token, id) => ({ id, token, bytes: idToBytes[id] ?? [] })),
      merges,
      map: tokenMap
    };
  }

  encode(text, geneMap) {
    if (!geneMap) throw new Error("EntropicCompressor.encode: missing geneMap.");
    const encoder = new TextEncoder();
    const baseOffset = this.reservedTokens.length;
    let tokens = Array.from(encoder.encode(text ?? ""), (b) => baseOffset + b);
    for (const merge of geneMap.merges ?? []) {
      tokens = this.mergeTokens(tokens, merge.pair, merge.id);
    }
    return tokens;
  }
}
