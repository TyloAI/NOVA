<div align="center">
<h1>
  <img src="https://tyloai.com/logo.png" alt="Logo" width="50" style="vertical-align: middle; margin-right: 15px;">
  PROTOETH/K
</h1>
<p><em>Building AI that thinks deeply and acts responsibly.</em></p>


![WebGPU Required](https://img.shields.io/badge/WebGPU-required-e65100)
![Status](https://img.shields.io/badge/status-experimental-f06292)
![Memory](https://img.shields.io/badge/memory-fast--weight-3949ab)
![Stack](https://img.shields.io/badge/stack-WebGPU%20%7C%20ESM-455a64)
![License](https://img.shields.io/badge/license-AGPL--3.0-2e7d32)
![Version](https://img.shields.io/badge/Version-0.1.0-blue)

# N.O.V.A.
### Neural Organic Virtual Architecture
**Technical Report: On-Device Agile Reasoning via WebGPU-Accelerated Hebbian Fast Weights**

</div>

# N.O.V.A. (Native Object Vector Architecture)

Other languages: [Simplified Chinese](zh-Hans.md) | [Traditional Chinese](zh-Hant.md)

## Abstract

N.O.V.A. is a browser-resident fast-weight dialogue stack that runs entirely on WebGPU, performs token-level Hebbian adaptation, and exposes every component for inspection. A custom tokenizer plus an entropic byte-pair compressor train a compact vocabulary on the fly, while a symplectic fast-weight manifold, hormone-guided biasing, and Origami-style scope memory sustain short-context reasoning without a server. The goal is to provide a technical, reproducible, and modifiable reference for client-side learning, not a turnkey black-box assistant.

## Introduction

### System overview (technical report, not a tutorial)
- **Compute substrate.** `src/core/Device.js` wraps WebGPU adapter/device/queue; `NovaTensor` manages GPU buffers with Xavier/residual init, unit-spinor states, and safe `read`/`dispose`. All math runs in-browser; no server path exists.
- **Fast-weight cell.** `src/layers/FastWeight.js` chains RMSNorm → rotational projections `(wr, wv, wg)` → symplectic flow for complex-valued manifold updates → gated FFN `(ffnUp1/ffnUp2/ffnDown)` with swish gating → active inference adjustment using `wPredict` → logits via shared embedding matrix. Thinking steps iterate this cell per token for short “inner-loop” reasoning.
- **Memory + biasing.** Fast weights capture token-to-token associations with decay/learning-rate control; `HormoneSystem` tiles affective scalars across the bias vector; `OrigamiMemory` pushes/pops hidden state on `{}` scopes to mimic structured retention.
- **Tokenizer and vocabulary.** `Tokenizer` enforces ASCII cleanliness, reserves `<unk>/<bos>/<eos>` and special mode tokens, and locks vocabulary after training. `EntropicCompressor` (byte-level BPE) learns a 16k “gene” table from the provided corpus before training to compress frequent patterns.
- **Runtime heuristics.** `src/main.js` applies n-gram anchoring (bigram/trigram counts), repetition penalties, punctuation gating, anchor bonuses, exemplar bias (optional), and mode tags (`Mode: chat/task/code`) to stabilize sampling. Temperature/top-k plus logit normalization govern stochasticity.
- **Training loop.** `scripts/train_browser.js` performs quickstart training in-browser: token-by-token Hebbian updates conditioned on whether tokens are inside the AI span, with optional code-boost replays, context decay, and yield points to keep UI responsive.
- **Snapshots and reproducibility.** Snapshot load/export includes model weights and tokenizer vocab; `nova.config.js` controls auto-load/export and file location. No IndexedDB persistence is assumed—explicit download/upload is the reproducibility path.

**Limitations and scope.** Model size is small (word-level, short context) and susceptible to dataset bias; WebGPU is mandatory; safety/guardrails are minimal; persistence depends on snapshots; outputs remain stochastic despite n-gram constraints. Not suitable for safety-critical or privacy-sensitive use without additional review.

**Operational expectations.** Use ASCII chat-formatted data (`User:... AI:...`) for stability; long-form or multilingual data will collapse to `<unk>`; GPU memory dictates feasible `dModel`/`layers`; browser flags may be required to enable WebGPU on some platforms.

## Mathematical Formulation

Let the tokenizer map a text stream into tokens \(t_1,\dots,t_T\), embeddings \(x_t = E(t_t)\), and a normalized stream \(\hat{h}_t = \mathrm{RMSNorm}(x_t)\). The fast-weight cell computes
\[
r_t = W_r \hat{h}_t,\quad v_t = W_v \hat{h}_t,\quad g_t = \sigma(W_g \hat{h}_t),
\]
followed by a symplectic memory flow
\[
s_t = \lambda\,R(r_t)\,s_{t-1} + \beta\,[v_t,\,\kappa v_t],\qquad
m_t = g_t \odot \mathrm{read}(s_t,\phi),
\]
where \(R(\cdot)\) is a complex rotation, \(\lambda\) the manifold decay, and \(\phi\) the readout mix. A gated feed-forward block applies
\[
u_t = \mathrm{swish}(W_{u1} m_t) \odot (W_{u2} m_t),\quad
c_t = W_d u_t,\quad
h_t = \mathrm{RMSNorm}(m_t + c_t).
\]
Active inference adjusts the manifold toward predicted embeddings \(\hat{x}_{t+1} = W_p h_t\) by gradient-free correction. Hebbian consolidation updates the memory manifold and projections with
\[
M \leftarrow \rho M + \eta\, (h_t \otimes e_{t+1}),
\]
using decay \(\rho\) and learning rate \(\eta\). Generation draws logits
\[
\ell_t = E^\top h_t + b_{\text{ngram}} + b_{\text{bias}},
\]
where \(b_{\text{ngram}}\) encodes bigram/trigram priors and \(b_{\text{bias}}\) aggregates repetition, punctuation, anchor, and hormone-derived penalties/bonuses. Sampling uses temperature/top-\(k\) with repeat constraints and optional exemplar filtering. Mode tokens (`Mode: chat/task/code`) condition both training and inference when present.

**Data path.** Raw text → tokenizer (with optional entropic compression pre-pass) → token IDs with `<bos>/<eos>` → fast-weight forward + Hebbian updates (training) → snapshot (weights + vocab). Inference reuses the same path without updates unless explicitly enabled.

## How to Run

1. **Prerequisites.** A WebGPU-capable browser (Chrome/Edge recent), served via `https` or `localhost`. No backend is required.
2. **Start a static server.** For example:
   - Node: `npx http-server .`
   - Python: `python -m http.server 8000`
   Then visit `http://localhost:8080` or `http://localhost:8000`.
3. **Prepare data.** Default corpus: `data/training_data.txt`, each line `User:... AI:...` in ASCII. Short, natural replies improve stability. Non-chat text is auto-split into sentences.
4. **Train and chat.** Click “Pulse” in `index.html`. If `nova.config.js` enables `snapshot.autoLoad`, the UI first attempts to restore `model.snapshot`; otherwise it trains in-browser via `scripts/train_browser.js`. Snapshots include model weights and tokenizer vocab and can be exported for reuse.
5. **Tune behavior.** Adjust `nova.config.js` to set model width/depth (`dModel`, `layers`, `ffnHidden`), Hebbian rates (`runtime.learningRate`, `runtime.decay`), context windows, n-gram weighting, exemplar bias, temperature/top-\(k\), punctuation guards, and hormone baselines. `training.codeBoost` can upsample code-like lines; `runtime.thinkingSteps` controls inner-loop iterations per token.
6. **Observe/log.** UI log shows data loading, entropic compressor progress, snapshot load status, and training line counts; console logs expose per-epoch progress and n-gram statistics.
6. **Troubleshooting.** If WebGPU is unavailable, switch browsers or enable the WebGPU flag; blank outputs often indicate missing training data or blocked tokens; non-ASCII lines will be mapped to `<unk>` and reduce quality.

## License

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). Copyright © 2026 Protoethik Co., Ltd. Use, modification, and networked deployment must comply with AGPL-3.0, including source disclosure requirements.

## Citation

If you reference this work, please cite as:

```
@techreport{protoethik2026nova,
  title   = {N.O.V.A.: Native Object Vector Architecture for Browser-Resident Fast-Weight Dialogue},
  author  = {Protoethik Co., Ltd.},
  year    = {2026},
  note    = {Version 1.1, WebGPU fast-weight prototype.}
}
```

## Contact

For technical questions, open an issue in the project repository or reach out to Protoethik Co., Ltd. through your standard support channel. This software is experimental; deploy only after your own verification and compliance review.
