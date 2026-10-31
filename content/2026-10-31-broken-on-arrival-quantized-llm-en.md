Title: Broken on Arrival: Non-Functional Quantized LLM Artifacts in Public Model Registries
Date: 2026-10-31
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: 2.3% of official Ollama registry tags for code-capable small LLMs are completely non-functional due to uncalibrated low-bit quantization — a problem invisible to tensor-level reproducibility checks.

## Why This Paper Matters

Every day, thousands of developers pull quantized LLMs from public registries like Ollama or Hugging Face to run locally on their hardware. The GGUF format (llama.cpp's single-file model standard) has become the de facto packaging for these models, supporting quantization levels from 2 to 8 bits per weight. This ecosystem has grown enormously — internet measurements find over 155,000 Ollama instances exposed publicly.

But here's the uncomfortable truth: **none of the registries examined documented functional testing of their quantized files**. Downloads aren't verified. Likes don't imply correctness. A file could be reproducible bit-for-bit from its source recipe and still solve zero coding tasks.

This paper provides the first systematic registry-scale execution audit of quantized LLM artifacts, and discovers that the failure rate is small but real: 7 of 305 official Ollama tags (2.3%) solve at most 6 of 164 HumanEval+ tasks. All are low-bit quantizations (q2_K, q3_K_M) of models with at most 4B parameters. The root cause is not corruption — it's a missing calibration step. This paper's contribution is not just diagnosing the problem; it's releasing a 30-second screening tool (`quantcheck`) so registries can catch it.

## Core Technical Contribution

### GGUF k-quants and the Importance Matrix

GGUF "k-quants" quantize model weights in **blocks**, assigning each block a per-block scale. A quantization level like `q3_K_M` specifies a fixed *mix* of different block types across tensor types — it is not a single uniform bit width. The "K" in the name indicates use of "k-quants" (block quantization), and the suffix (S/M/L) specifies the mix intensity.

When llama.cpp quantizes without calibration, it chooses block scales by minimizing a simple error metric on the weights alone. The **importance matrix** (imatrix) adds a critical ingredient: activation statistics from a calibration corpus. This allows the quantizer to weight the rounding error by how much each block actually *matters* during inference — blocks whose outputs influence downstream computations heavily get more precise scales.

The imatrix's effect is largest at the lowest bit widths. For Qwen2.5-Coder-3B at q3_K_M and below:
- **Without imatrix**: 0 solved tasks on HumanEval+ (complete failure)
- **With imatrix**: 76–113 solved tasks (46–69% success rate)
- 4-bit (`q4_K_M`) without imatrix: ~80% success — functional

This is the core finding: **the same model at 4 bits works fine without calibration; at 3 bits it collapses completely**.

### The Study Pipeline

The paper's methodology is rigorous and systematic:

1. **Census**: All 305 eligible GGUF tags across 35 model variants in 15 official Ollama model lines, plus 22 community HF files. Eligibility: instruct/code-capable, explicit quantization suffix, file ≤8 GB.

2. **15-task screen**: Tuned on 24 Qwen2.5-Coder configurations of known health. The screen selects 15 HumanEval+ tasks solved by all 23 known-healthy configs. Taking ~30 seconds of generation per artifact, the screen catches suspects with:
   - **Stage 1**: fewer than 9/15 tasks solved → flagged
   - **Stage 2**: at most 2/15 AND group baseline ≥8 → "suspect"
   - "Non-functional" if also solves <10% of full 164-task HumanEval+ benchmark

3. **Recipe-matched rebuild**: For each suspect, the paper downloads the original checkpoint and quantizes it *with the same recipe* (no imatrix, matching the registry metadata). This produces a "referee" build. If the registry file matches the referee tensor-for-tensor, it's a **recipe collapse**, not a file corruption.

4. **Calibration rescue**: A calibrated rebuild (with imatrix) of the same checkpoint at the same level tests whether the collapse is reversible.

### Key Classification Rules

- **File defect**: suspect ≤2/15, plain rebuild ≥5 and ≥4 more than suspect
- **Collapse**: plain rebuild also fails comparably → **calibration-rescued** if imatrix build succeeds
- **Low baseline**: group lacks functional high-precision builds within 8 GB frame

## Comparison to Prior Work

| Study | Full Registry Census | Controls Recipe | Attributes Cause |
|---|---|---|---|
| Singh et al. 2025 | partial | no | no |
| Liu & Tsai 2025 | no | yes (reproducibility) | no |
| **This paper** | **yes** | **yes** | **yes** |

The key gap prior work left open was: are failing quantized models corrupted files, or correct-but-broken recipes? This paper proves definitively that the five Qwen and Phi3.5 suspects are **exact uncalibrated rebuilds** — not corrupted, just under-calibrated. Previous reports (like SynConfRoute's 0% on Qwen2.5-Coder-3B q2_K) had observed the symptom but not diagnosed the cause.

## Reading the Results

**The 2.3% headline**: 7 of 305 official Ollama tags non-functional. This sounds small — but these 7 tags include the q2_K and q3_K_M builds of one of the most popular small coding models (Qwen2.5-Coder-3B), meaning many developers have silently downloaded broken tools.

**The calibration rescue**: Adding an imatrix raised three Qwen2.5-Coder-3B builds from **0% to 46–69%** of HumanEval+. This is dramatic — the difference between a model that produces syntactically broken code on every task and one that correctly solves two-thirds.

**Why 1.5B survives**: The Qwen2.5-Coder-1.5B model, smaller than the 3B, remains functional without imatrix at low bits. The paper notes that **"small models trained on many tokens degrade most under low-bit quantization"** — 3B models with dense pretraining data are more sensitive to precision loss than 1.5B models. This is counterintuitive: bigger isn't automatically more robust to quantization.

**The screen's precision**: 0 of 66 sampled healthy tags were false positives. The 15-task screen flagged 39 of 305 as stage-1 suspects; the full evaluation confirmed 7 as non-functional. This gives the screen practical value as a registry gate.

## Key Notes of This Paper

### Why "Reproducible" ≠ "Functional"

The paper introduces a subtle but important distinction. For each of the five rebuilt suspects, the paper computes a tensor-for-tensor comparison between the registry file and its recipe-matched rebuild:

```
For each tensor t:
  max_absolute_error(registry_file[t], plain_rebuild[t]) == 0
```

This is exact bitwise reproduction. Yet the models solve 0% of HumanEval+. This proves that **reproducibility is a necessary but not sufficient condition for functional correctness** in quantized models. The quality gap lives in the *recipe choice*, not the execution of the recipe.

### The Imatrix Mechanism

The imatrix calibration works by recording, for each weight block, how much its output variance affects the model's loss on a calibration corpus. Let:
- $W$ be a weight matrix block
- $X$ be typical input activations to that block
- $\mathbf{h} = WX$ be the output

The calibration computes the expected squared output for each column $j$ of $W$:
$$\text{imatrix}[j] = \mathbb{E}_{X}\left[\sum_i (W_{ij} x_i)^2\right]$$

During quantization, the rounding error for each weight is scaled inversely by its importance score — blocks that produce large outputs get higher-fidelity encoding. Without this, the quantizer treats all blocks equally, which catastrophically degrades the most information-dense blocks at 2–3 bits.

### The Failure Pattern

All 7 non-functional tags share these properties:
1. **Low-bit**: q2_K or q3_K_M (≤ ~3.35 bits/weight average)
2. **Small model**: ≤4B parameters
3. **Dense pretraining**: models like Qwen2.5-Coder trained on very large token counts
4. **No imatrix in metadata**: the file's quantize metadata lacks `quantize.imatrix` keys

The correlation between density of pretraining and low-bit fragility is the paper's most theoretically interesting finding.

## Limitations

The paper explicitly acknowledges:
- **Two license-gated models unattributed**: llama3.2-1B's two non-functional tags could not be rebuilt from source (gated checkpoint), so their failure cause is unconfirmed.
- **Census date-stamped**: inventory generated on August 10, 2026; results may not reflect the current registry state as files are updated.
- **Code tasks only**: the screen and HumanEval+ evaluation measure code generation capability. A model could pass this screen but fail on other tasks.
- **Ollama library focus**: the community HF survey (22 files) is not a systematic census.
- **Screen tuned on Qwen**: the 15-task subset was selected based on Qwen2.5-Coder configurations; it may not generalize to all architectures.

## Future Work

**Authors' directions**: The paper recommends that registries adopt the `quantcheck` screen as a publication gate for all low-bit tags, and calibrate or withhold failures. They suggest standardizing imatrix generation as part of the GGUF conversion pipeline.

**Additional promising directions**:
- **Per-architecture importance matrices**: Different model families may benefit from architecture-specific calibration corpora (e.g., code models calibrated on code, math models on math). The current practice of generic text calibration may leave performance on the table.
- **Automated registry CI**: Integrating `quantcheck` into Hugging Face's Model Hub as an automated CI step (similar to how PyPI runs installation tests) would catch these failures before publication rather than after discovery.
- **Cross-backend validation**: The paper tests on CUDA and Metal. Results for ARM backends (Snapdragon, Apple Silicon via llama.cpp's Metal, and embedded ARM Cortex-M) may differ, as the quantization arithmetic is implemented differently in each backend.
- **Adversarial quantization detection**: The paper notes that quantization can trigger malicious behavior. A calibration-aware security screen could check both functionality and safety-relevant behavioral changes under quantization.

## Implications for Edge / On-Device Deployment

This paper has immediate practical consequences for edge deployment:

1. **Don't trust the download count**: Popular models with many downloads can still be non-functional at the bit level you need. Verify before shipping to production.

2. **Always imatrix at ≤3-bit**: If you're targeting 3-bit or below (common for embedding LLMs on 4–8 GB devices), uncalibrated quantization is a silent killer. Build with imatrix from a domain-relevant calibration corpus.

3. **The 4B barrier**: Models with ≤4B parameters appear most vulnerable to uncalibrated low-bit collapse. This is precisely the range targeted for edge deployment — the 1B–4B range that fits on smartphones and embedded devices.

4. **30-second screen in CI**: The `quantcheck` tool's cost (30s per artifact) is negligible in a deployment pipeline. Add it as a gate before distributing quantized models internally.

5. **Reproducibility audits are insufficient**: If your validation process is "compare tensors to the source recipe," you are not catching this failure class. You need functional execution tests.

## Links

[Original Paper](https://huggingface.co/papers/2609.05881)
[ArXiv](https://arxiv.org/abs/2609.05881)
[quantcheck Tool & Audit Dataset](https://huggingface.co/datasets/aditiPatodiya/quantcheck-audit)
[GitHub](https://github.com/aditi-p31/quantcheck)
