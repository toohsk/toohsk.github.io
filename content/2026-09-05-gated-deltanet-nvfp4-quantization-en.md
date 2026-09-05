Title: Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for Hybrid LLMs
Date: 2026-09-05
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: The Minima recipe shows that fully quantizing a hybrid 27B LLM — including its recurrent Gated DeltaNet layers — to NVFP4 W4A4 (4-bit weights and activations) matches BF16 accuracy while shrinking the model to 17.5 GiB and cutting prefill latency by 14–19%, because the delta-rule recurrence actively erases quantization error rather than accumulating it.

## Why This Paper Matters

Serving large language models cheaply enough for real-world deployment has forced the community toward aggressive quantization. NVFP4 (4-bit weights and activations on native hardware tensor cores) is the current frontier — but until now, no one had dared apply it to the recurrent half of a hybrid LLM.

Hybrid models like Qwen3.8-27B pair standard softmax attention layers with **Gated DeltaNet (GDN)** linear-attention layers that summarize the entire context in a fixed-size recurrent state matrix. In this model, 48 of 64 layers are GDN — the recurrent component dominates the architecture, not attention. Every public 4-bit build exempted those 48 layers from aggressive quantization, keeping GDN gates in BF16 or FP8, based on a seemingly reasonable intuition: a recurrence carries its state across tens of thousands of tokens, so per-step quantization errors should accumulate and compound.

This paper shows the intuition is **exactly backwards**, and explains why in mechanistic detail. Understanding this is consequential: if recurrent-state mixers are naturally quantization-robust, then the entire class of hybrid architectures — Mamba, GLA, RetNet, DeltaNet variants — is ripe for deeper compression than the community assumed.

## Core Technical Contribution

The paper introduces **Minima**: an NVFP4 W4A4 quantization recipe for Qwen3.8-27B that quantizes **all 496 linear layers**, GDN gates included, using post-training calibration (no QAT, no distillation).

### What NVFP4 Actually Does

NVFP4 (E2M1 format) stores weights and activations at 4 bits, with one E4M3 (8-bit float) scale per 16-element block. The key property: **a block's largest element sets the scale**, so an outlier degrades only its 15 neighbors rather than the entire tensor. This block-scoped localization is what makes extreme outliers (kurtosis ~1,560 in GDN activation streams) survivable.

### The Four-Part Mechanistic Chain

The paper runs a controlled mechanism study — capturing real activations on 32K-token documents, then replaying the recurrence in FP32 with injected fake quantization noise — and finds four interlocking reasons GDN is safe:

**1. Block scaling localizes residual-stream outliers.**
GDN inputs share the residual stream with attention, and that stream is viciously spiky (median max/RMS = 63.5, 10–32% of 16-element blocks dominated by a single value). But NVFP4's block granularity confines each outlier to its own block, equalizing the per-layer activation quantization error to a flat 7.5–9.2% — identical across every layer role. The inputs are not the easy part.

**2. Gate nonlinearities compress what reaches the control signals.**
The two gate projections `a` (controlling forget rate α) and `b` (controlling write strength β) are parameterized in log space:
$$g_t = -\exp(A_{\log}) \cdot \text{softplus}(a_t + \text{dt\_bias}), \quad \alpha_t = e^{g_t} \in (0,1), \quad \beta_t = \sigma(b_t) \in (0,1)$$

A ~11% GEMM error on `a` becomes only a ~2% error on β after the softplus + exponential chain. The sigmoid on `b` provides similar squashing. These gates were being protected precisely because they are *control* signals — but the nonlinear parameterization (chosen originally for training stability) provides natural quantization shielding at inference.

**3. The delta-rule recurrence bounds and erases noise.**
The recurrent state evolves as:
$$S_t = \alpha_t S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top$$

This is not a simple exponential moving average. The **write term** is a *correction*: it replaces what the state currently predicts for key $k_t$ with $v_t$. Crucially, this means every new write **overwrites the state along the current key direction**, actively erasing errors stored in that direction. In the lockstep FP32 experiment: a 1% state impulse injected at token 1,024 falls to 1/e within 80–1,382 steps — far faster than the decay-gate horizons of 44K–62K tokens.

The full-quantization state error plateaus immediately at 12.6% and stays flat over 32K tokens. There is **no accumulation** — the plateau is an equilibrium between error injection (per-step quantization) and error erasure (delta-rule overwrites).

**4. End-to-end, context washes the error out.**
The per-token NLL gap (Minima vs. BF16) is +0.081 nats in the first half of the 32K window and +0.011 in the second. In the final 2K tokens, Minima actually scores *better* than BF16 (−0.053 nats). This is the opposite of the accumulation story: a filled GDN state absorbs the 4-bit cost.

### A Key Serving-Stack Bug (and Fix)

The paper identifies a silent corruption that affects any NVFP4 recipe quantizing GDN: `llm-compressor` calibrates one FP32 global scale *per linear module*, but `vLLM` **fuses adjacent modules** into a single GEMM at serving time and takes the maximum scale without rescaling the others. For GDN's fused `in_proj_b+a` group, the paired scales differ by 2.75× in every one of the 48 layers. The corrupted model produces plausible results — reasoning degrades moderately, but long-context perplexity *improves* (because a broken forget gate makes the state hold everything). The fix is a checkpoint-side rewrite: rescale each fused group to the shared maximum and fold the ratio into the per-block E4M3 scales.

## Comparison to Prior Work

| Recipe | GDN precision | Attention precision | Size (GiB) | TTFT @32K (s) | 5-task avg vs. BF16 |
|---|---|---|---|---|---|
| BF16 (reference) | BF16 | BF16 | 50.1 | 6.90 | — |
| Unsloth Dynamic v3 | FP8 W8A8 | FP8 W8A8 | ~18.8 | ~4.90 | −0.64 |
| RadixArk (ModelOpt) | FP8 W8A8 | FP8 W8A8 | ~18.8 | ~4.87 | −0.67 |
| **Minima (ours)** | **NVFP4 W4A4** | **NVFP4 W4A4** | **17.5** | **4.03** | **−0.52** |

All quantized recipes sit within seed noise of BF16 on every task. Minima is the only recipe that quantizes GDN gates; it is the smallest and fastest at prefill, while matching BF16 AIME'25 exactly (26/30 on all four seeds).

## Reading the Results

**What matters most:**
- The 5-task average gap of −0.52 across all quantized models is less than one AIME problem (3.3 points). Quantization is not hurting reasoning.
- Minima's TTFT drops from 6.90 s to 4.03 s at 32K (+14–19% prefill throughput over community FP8 builds). This is the efficiency win that only happens when GDN is at 4-bit.
- KV-cache size increases to 1.81M cacheable tokens on a single card, vs. BF16's much lower limit.
- Adding calibrated FP8 KV-cache scales (free at serving time, 32 tensors) recovers 83% of the perplexity penalty at 32K, dropping from 10.84 to 10.50.

**Perplexity as the honest residual:** Minima pays +0.72 PPL at 4K context and +0.49 at 32K. Both gaps shrink with context length. No task score separates any pair of models at statistical confidence.

## Key Notes of This Paper

### The delta-rule recurrence as an error-erasure mechanism

The core insight that makes GDN quantization-safe is the self-correcting nature of the write operation. In a standard RNN (like an LSTM or simple GRU), state errors from step $t$ are merely *decayed* by subsequent steps — they persist at a rate determined by the forget gate. In Gated DeltaNet, writes are **directional corrections**:
$$S_t = \alpha_t S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top$$

The term $(v_t - S_{t-1}^\top k_t)$ is the prediction error — how far the current state's guess for key $k_t$ is from the true value $v_t$. The write *replaces* the state along the key direction $k_t$ rather than blending $v_t$ in. So if an old error happened to be stored along a direction that a later write targets, that write erases it completely. The paper calls this "active erasure" — quantization noise is not just forgotten, it is actively overwritten key by key as new tokens arrive.

Contrast this with the sensitivity to direct noise on $\alpha$: a 0.1% multiplicative noise on the gate value itself (not on the pre-activation) produces 22% state error, because with $\alpha \approx 1$, a tiny $\delta\alpha$ is an enormous relative change in the effective context horizon $1/(1-\alpha)$. The log-space parameterization is crucial: the softplus + exponential chain means the same 11% GEMM error on the pre-activation of $a$ produces only 3.6% state error — a 3× buffer built into the architecture.

### Block scaling as a containment mechanism

NVFP4's E4M3 scale per 16-element block sets the scale to `blockmax / 6`. A single outlier 100× the median channel amplitude fixes the scale of its block and forces the 15 neighboring values to be represented with a scale that is 100× too large for them — but it does not touch any other block. Within a block, the worst-case ratio of max to RMS is bounded by $\sqrt{16} = 4$, which bounds how "one-hot" a block can be. This geometric argument means block scaling is not just a practical trick; it provides a formal containment guarantee for outlier damage.

## Limitations

- **Single model and format:** All results are for Qwen3.8-27B and NVFP4. Whether the mechanism generalizes to other recurrent architectures (Mamba, GLA, RetNet) or other quantization formats (GPTQ, AWQ, INT4) is an open question.
- **Context length:** Evaluated to 32K perplexity and 64K retrieval. The bounded-error mechanism predicts safety at 128K+, but this is extrapolation. Longer-context stress tests were not run.
- **Concurrent QAT checkpoint not benchmarked:** A QAT recipe (QUASAR) that also quantizes all 496 layers via distillation appeared after the measurement campaign. The paper argues PTQ suffices, but a head-to-head comparison with QAT under a controlled harness was not completed.
- **Decode throughput:** All three NVFP4 builds are within 4% in decode throughput (weight-bandwidth bound). Minima does not win on decode.

## Future Work

**Authors' directions:** Extending to 3-bit MLP codebooks with a distillation-healed low-rank residual (a concurrent engineering project mentioned in §8) brings the weight footprint to 12.3 GiB while sustaining 1,200+ tokens/s on a single GPU. Longer-context evaluation at 128K+ would test the bounded-error prediction.

**Additional promising directions:**
- **Generalization to other recurrent mixers:** The mechanism (log-space gate parameterization + correction-based writes) may apply to Mamba's selective SSM, which also uses softplus-parameterized gates. A similar study for Mamba2 and Hawk/Griffin hybrids could unlock a class of fully W4A4 long-context models.
- **Mixed-precision QAT that skips protecting GDN gates:** Given that GDN gates are the *safest* projections, QAT recipes could concentrate their budget on the attention and MLP projections that carry the largest quantization errors (out/qkv/z).
- **The global-scale mismatch as a broader audit item:** Any hybrid model served by vLLM that fuses adjacent GDN modules should be audited for this bug. A systematic fix in the calibration tooling (not just post-hoc checkpoint repair) would benefit the community.

## Implications for Edge / On-Device Deployment

The practical significance is direct: Minima fits in **17.5 GiB** — versus 50.1 GiB for BF16 — making a 27B model deployable on a single 24 GB consumer GPU (RTX 4090 / PRO 6000 class) with room for a generous KV cache. For edge servers and workstations, this changes the economics of running frontier-class reasoning models locally.

The key takeaways for practitioners:
1. **Quantize everything in hybrid models.** The GDN block (23% of decode weight bytes) is the easy half, not the hard half.
2. **Ship calibrated KV scales.** 32 tensors, free at serving time, recover 83% of the long-context perplexity penalty.
3. **Audit fused-GEMM scale consistency.** Any NVFP4 checkpoint of a hybrid model should verify that fused serving groups share a common global scale.
4. **The compound error intuition is wrong for correction-based recurrences.** Architectures that write corrections (not accumulations) are naturally quantization-robust at long contexts.

For on-device deployment where power and memory are the hard constraints, hybrid architectures with recurrent mixers may ultimately outperform pure-attention models at equal parameter counts — they carry less KV memory (only 16 of 64 layers cache KV here), and as this paper shows, they are fully compressible to 4-bit without accuracy loss.

## Links

[Original Paper](https://arxiv.org/abs/2609.04098)
