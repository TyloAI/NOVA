<div align="center">

<img src="https://tyloai.com/logo.png" alt="PROTOETH/K Logo" height="60" style="vertical-align:middle; margin-bottom:10px;">

<h1 style="font-size: 3em; margin-top: 0;">P R O T O E T H / K</h1>

<p style="font-size: 1.2em; font-style: italic; color: #666;">
Building AI that thinks deeply and acts responsibly.
</p>

[![Full Paper](https://img.shields.io/badge/📄-Paper-blue)](https://doi.org/10.5281/zenodo.18366793)

---

![WebGPU Required](https://img.shields.io/badge/Hardware-WebGPU%20Required-e65100?style=for-the-badge&logo=webgpu)
![Status](https://img.shields.io/badge/Status-Experimental%20Research-f06292?style=for-the-badge)
![Memory](https://img.shields.io/badge/Architecture-Hebbian%20Fast%20Weights-3949ab?style=for-the-badge)
![License](https://img.shields.io/badge/License-AGPL%20v3.0-2e7d32?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-0.1.0-blue?style=for-the-badge)

# N.O.V.A.
### Native Object Vector Architecture
**Technical Report: On-Device Agile Reasoning via WebGPU-Accelerated Hebbian Fast Weights**
</div>

---

## 1. Abstract

**N.O.V.A.** (Native Object Vector Architecture) is a browser-resident, fast-weight dialogue stack executing entirely on the WebGPU compute substrate. Unlike static Transformer models, N.O.V.A. performs **token-level Hebbian adaptation** in real-time, utilizing a symplectic manifold for context retention and "Origami-style" symbolic scoping for structural logic. This architecture eliminates the need for server-side inference, offering a fully transparent, inspection-ready reference implementation for client-side agile reasoning. This report details the system's mathematical formulation, memory dynamics, and deployment methodology.

## 2. Architectural Specification

N.O.V.A. is engineered to provide a reproducible, modifiable reference for edge-based learning, moving beyond the paradigm of "black-box" turnkey assistants.

### 2.1 Compute Substrate (`src/core/`)
* **`Device.js`**: A low-level wrapper around the WebGPU Adapter, Device, and Queue.
* **`NovaTensor`**: Manages raw GPU buffers with specialized initialization kernels (Xavier, Residual, Unit-Spinor). It implements safe memory lifecycles (`read`/`dispose`), ensuring all mathematical operations remain strictly in-browser with zero server dependency.

### 2.2 The Fast-Weight Cell (`src/layers/FastWeight.js`)
The core reasoning unit iterates per token, forming a "Thinking Step" loop:
1.  **Projection:** Input embeddings pass through RMSNorm and project into rotational components $(W_r, W_v, W_g)$.
2.  **Symplectic Flow:** Updates a complex-valued manifold using energy-preserving rotations, preventing gradient decay.
3.  **Gated FFN:** A Swish-gated Feed-Forward Network ($W_{up1}, W_{up2}, W_{down}$) processes the manifold state.
4.  **Active Inference:** The system predicts its own next state via $W_{predict}$ and adjusts the manifold based on prediction error.

### 2.3 Memory & Biasing Dynamics
* **Hebbian Consolidation:** Fast weights capture transient token-to-token associations dynamically, governed by variable decay and learning rates.
* **HormoneSystem:** Tiles affective scalar values across the bias vector, allowing for "emotional" modulation of generation probability.
* **OrigamiMemory:** A symbolic stack that pushes/pops hidden states upon detecting structural delimiters (e.g., `{`, `}`), enabling robust handling of nested logic in code generation.

### 2.4 Tokenization & Runtime Heuristics
* **Entropic Compression:** A byte-level BPE compressor learns a compact ~16k vocabulary "gene" table from the corpus, optimizing for high-density concepts.
* **Sampling Strategy:** Inference is stabilized via `src/main.js` using n-gram anchoring (bigram/trigram), repetition penalties, anchor bonuses, and mode-specific tags (`Mode: chat/task/code`).

---

## 3. Mathematical Formulation

Let the tokenizer map a text stream into tokens $t_1,\dots,t_T$, producing embeddings $x_t = E(t_t)$. The normalized stream is defined as $\hat{h}_t = \mathrm{RMSNorm}(x_t)$.

### 3.1 Projection & Gating
The fast-weight cell computes the rotational ($r$), value ($v$), and gate ($g$) vectors:
$$
r_t = W_r \hat{h}_t,\quad v_t = W_v \hat{h}_t,\quad g_t = \sigma(W_g \hat{h}_t)
$$

### 3.2 Symplectic Memory Flow
State retention is governed by a complex rotation $R(\cdot)$ on the manifold $s$, ensuring long-term stability:
$$
s_t = \lambda\,R(r_t)\,s_{t-1} + \beta\,[v_t,\,\kappa v_t]
$$
$$
m_t = g_t \odot \mathrm{read}(s_t,\phi)
$$
Where $\lambda$ represents manifold decay and $\phi$ is the readout mixture coefficient.

### 3.3 Gated FFN & Active Inference
The manifold output is processed via a Swish-gated block:
$$
u_t = \mathrm{swish}(W_{u1} m_t) \odot (W_{u2} m_t),\quad c_t = W_d u_t
$$
$$
h_t = \mathrm{RMSNorm}(m_t + c_t)
$$
The system performs **Active Inference**, adjusting the manifold towards predicted embeddings $\hat{x}_{t+1} = W_p h_t$ via gradient-free correction.

### 3.4 Hebbian Update Rule
The memory manifold and projections are updated in real-time during the forward pass:
$$
M \leftarrow \rho M + \eta\, (h_t \otimes e_{t+1})
$$
Where $\rho$ is the decay factor and $\eta$ is the learning rate.

### 3.5 Logit Generation
Final logits are computed by projecting back to the vocabulary space, modulated by n-gram priors and bias vectors:
$$
\ell_t = E^\top h_t + b_{\text{ngram}} + b_{\text{bias}}
$$

---

## 4. Deployment & Reproduction

### Prerequisites
* **Hardware:** WebGPU-capable GPU (Integrated or Discrete).
* **Software:** Recent Chromium-based browser (Chrome/Edge).

### Operational Steps
1.  **Serve Static Files:**
    No backend logic is required. Serve the root directory via any HTTP server.
    ```bash
    # Node.js
    npx http-server .
    # Python
    python -m http.server 8000
    ```
2.  **Initialize Runtime:**
    Navigate to `http://localhost:8000` and click **"Pulse"** to initialize the compute shaders.
3.  **Training & Inference:**
    * **Auto-Load:** If `nova.config.js` enables `snapshot.autoLoad`, the system restores `model.snapshot`.
    * **In-Browser Training:** Otherwise, `scripts/train_browser.js` executes token-by-token Hebbian updates on `data/training_data.txt`.
4.  **Configuration:**
    Adjust hyperparameters (dModel, layers, learningRate) in `nova.config.js`.

---

## 5. Limitations & Scope

* **Experimental Nature:** N.O.V.A. is a research prototype focusing on architectural novelty (Fast Weights/WebGPU) rather than scale.
* **Data Sensitivity:** The model is susceptible to dataset bias and assumes ASCII-formatted chat data.
* **Persistence:** Model state is transient unless explicitly exported via Snapshots.
* **Safety:** Minimal guardrails are implemented. Not suitable for safety-critical applications without further review.



---

## 6. License
Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). Copyright © 2026 Protoethik Co., Ltd.
Use, modification, and networked deployment must comply with AGPL-3.0, including source disclosure requirements.

---

## 7. Citation

If you use this work in your research, please cite:

```bibtex
Wu, Y. (2026). Context Is Geometry (1.0). Zenodo. https://doi.org/10.5281/zenodo.18366793
```

**Direct link**: [https://doi.org/10.5281/zenodo.18366793](https://doi.org/10.5281/zenodo.18366793)
