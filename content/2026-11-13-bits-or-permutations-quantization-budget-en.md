Title: Bits or Permutations? How to Spend a Fixed Inference Budget on Small Local LLMs
Date: 2026-11-13
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: A principled budget-allocation study on Qwen3-0.6B and 1.7B finds that 6-bit quantization loses no accuracy versus fp16, while 4-bit causes slot skew that four-pass cyclic option averaging can largely recover — recommending 6-bit weights plus up to four orderings as the best combined strategy.

## Why This Paper Matters

Multiple-choice question answering (MCQ) on local language models exposes two independent sources of error that are rarely studied together:

1. **Quantization error:** Lower-bit weights introduce rounding noise that can change the predicted answer.
2. **Option-order bias:** LLMs are known to prefer answer options in certain slots (e.g., always selecting "A"). Averaging predictions over several reorderings reduces this bias.

Both interventions cost inference time and memory. If your budget is fixed — say, you can run N total weight-read operations — should you spend it on higher-bit weights or on more orderings? The two approaches compete for the same computational budget, and no prior work had studied the interaction.

This paper provides the first systematic comparison. The practical guidance is actionable: **use 6-bit weights and spend the remaining budget on up to four cyclic orderings.** This beats both alternatives (pure quantization improvement or pure permutation averaging) in accuracy per byte.

## Core Technical Contribution

The experimental setup systematically varies two dimensions:

**Weight precision:** fp16 (reference), 8-bit, 6-bit, 4-bit, and 3-bit, using round-to-nearest simulated quantization.

**Option orderings:** 1 pass (no averaging), 2, 3, or 4 cyclic permutations of the answer options (A/B/C/D → B/C/D/A → etc.), with the final prediction computed by averaging logits across orderings.

**Models:** Qwen3-0.6B and Qwen3-1.7B.

**Benchmarks:** ARC-Easy, ARC-Challenge, MMLU, and HellaSwag — 300 items each from their multiple-choice subsets. Answers are read from the logits of the letters A–D (not generated text), making evaluation clean and deterministic.

**Cost model:** Weight bytes read × number of passes. This is architecture-independent (no hardware assumptions) and naturally combines the two dimensions: running a 4-bit model for four passes costs the same as running a 6-bit model for two passes (approximately).

## Comparison to Prior Work

The paper frames itself as a first study of this combination. The authors searched the literature and found no prior work examining quantization together with option-order averaging. The two literatures have been developed separately:

- Quantization work (GPTQ, AWQ, etc.) typically targets generation tasks, not MCQ classification.
- Option-order averaging work (self-consistency, permutation ensembles) typically operates at full precision.

The baselines are therefore the two pure strategies:
- **Pure quantization improvement:** moving from 4-bit to 6-bit or 8-bit.
- **Pure permutation averaging:** running more orderings at the same precision.

## Reading the Results

**Quantization accuracy vs. fp16 (baseline differences, 95% CI):**

| Bits | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| 8-bit | -0.7 (-1.1 to -0.3) | -0.4 (-0.7 to -0.2) |
| 6-bit | +1.8 (+1.0 to +2.5) | +0.9 (+0.3 to +1.5) |
| 4-bit | (substantial degradation + slot skew) | (moderate degradation + slot skew) |

**The 6-bit finding is the most surprising:** 6-bit quantization slightly *improves* accuracy vs. fp16 on these benchmarks. The authors explicitly decline to explain this — they note it without providing a mechanistic account. It likely reflects regularization effects from quantization noise that happen to help MCQ accuracy, but that's speculation.

**Slot skew at 4-bit:**

Slot skew (bias toward specific answer positions) rises sharply at 4-bit:
- Qwen3-0.6B: 0.15 → 0.53 (3.5× increase)
- Qwen3-1.7B: 0.07 → 0.12 (1.7× increase)

A Gaussian noise control matched to the same weight SNR as 4-bit quantization gives skew of 0.15–0.19 (Qwen3-0.6B). The fact that actual 4-bit quantization (0.53) far exceeds the Gaussian control (0.15–0.19) suggests that round-to-nearest quantization noise has **structured correlations** that specifically amplify slot bias — not just random noise. This is an interesting finding in its own right.

**Recovery via permutation averaging at 4-bit:**

| Orderings | Qwen3-0.6B at 4-bit | Qwen3-0.6B at fp16 |
|---|---|---|
| 1 pass | (baseline) | (baseline) |
| 4 cyclic passes | +12.6 points | +5.3 points |

The recovery is dramatic: 12.6 accuracy points from four cyclic orderings at 4-bit, versus 5.3 points at fp16. Permutation averaging is *more valuable* at lower precision because there's more slot skew to correct.

**Budget allocation decision table:**

The paper reduces the tradeoff to a simple rule: **extra orderings beat extra bits only when weights have 6 or more bits.** At 4-bit, spending your budget on more orderings is better than upgrading bits. At 6-bit, the remaining budget is best spent on orderings (up to 4). The practical recommendation: store weights at 6-bit, run up to 4 orderings.

## Key Notes of This Paper

**The cost model is the methodological anchor.** By defining cost as "weight bytes read × passes," the authors create a hardware-agnostic comparison that can be evaluated analytically. A 4-bit model at 4 passes costs (4 × 0.5) = 2 units; a 6-bit model at 2 passes costs (2 × 0.75) = 1.5 units. This lets them construct the decision table from a small number of data points.

**The slot-skew metric is under-appreciated in quantization literature.** Most quantization papers report aggregate accuracy; this paper reports per-slot bias separately. The finding that round-to-nearest creates *more structured* slot bias than equivalent Gaussian noise (0.53 vs. 0.15 for 0.6B) is a novel observation that likely has implications for quantization method design.

**The logit-reading evaluation protocol is strict.** Many MCQ evaluations use generation (the model writes "A" or "The answer is A"). This paper reads the raw logit of the letter tokens — a protocol that is more sensitive to quantization effects because there's no post-processing to absorb noise. Results may differ from generation-based evaluation.

**The 6-bit benefit is unexplained.** The +1.8 point gain from 6-bit vs. fp16 is statistically significant (CI: +1.0 to +2.5) but mechanistically unexplained. This warrants investigation — if 6-bit quantization reliably regularizes MCQ performance, it could be exploited intentionally rather than accepted as an artifact.

## Limitations

- **Simulated quantization only.** Round-to-nearest is the simplest quantizer; real deployments use more sophisticated methods (GPTQ, AWQ, block-fp quantization) that have different error structures.
- **MCQ tasks only.** Option-order averaging is specific to multiple-choice; the approach doesn't directly apply to generation tasks (summarization, translation, code).
- **Two models from one family.** Qwen3-0.6B and 1.7B are from the same training pipeline. Results may not generalize to other architectures (Llama, Mistral, Phi, etc.).
- **No wall-clock timing.** The cost model is theoretical (bytes × passes); actual latency depends on hardware, batch size, and implementation.
- **3-bit results not fully reported.** The abstract mentions 3-bit in the setup but the main text focuses on 4–8 bit.

## Future Work

**Authors' implied directions:** Validating on larger models and generation tasks; explaining the 6-bit accuracy gain; testing with hardware-deployed quantizers rather than simulated round-to-nearest.

**Promising follow-on directions:**
- **Non-uniform bit allocation with permutation averaging.** If 6-bit is the sweet spot for quantization and permutations handle residual bias, can we apply 6-bit only to attention layers (most sensitive) and 4-bit to FFN layers (larger, less sensitive) while adding permutation passes? This could push below 6-bit average precision while maintaining quality.
- **Understanding structured slot skew from quantization.** The finding that 4-bit round-to-nearest creates far more slot bias than equivalent Gaussian noise suggests something fundamental about how uniform quantization interacts with softmax normalization. A theoretical analysis could lead to quantization-aware calibration that suppresses this effect.
- **Extending to generation with token-level confidence.** The cyclic-ordering trick is MCQ-specific, but a generation analog exists: running the same prefix with slight temperature or quantization variation and aggregating. Combining this with dynamic precision (as in the companion paper 2611.14418) could generalize both ideas.
- **Optimal number of orderings beyond 4.** The paper tests up to 4; is there diminishing returns before that, or would 8 orderings help at 3-bit? The cost-matched framework makes this easy to analyze.

## Implications for Edge / On-Device Deployment

**6-bit is the practical takeaway for on-device MCQ tasks.** Most on-device quantization deployments use either 4-bit (for maximum compression) or 8-bit (for safety). This paper provides strong evidence that 6-bit is the better operating point: no accuracy penalty versus fp16, and roughly 25% memory reduction versus 8-bit.

For applications that use LLMs to classify or select among options (intent detection, document classification, medical triage, router logic in agentic systems), option-order averaging at 6-bit costs 4× the latency of a single pass but can deliver substantially higher accuracy than 4-bit without averaging. On real-time mobile applications, this may still be acceptable if queries are batched.

The structured slot skew finding at 4-bit is important for safety-critical applications: a quantized on-device model may systematically favor certain output slots, introducing bias that standard aggregate accuracy metrics would miss. Per-slot calibration analysis should be part of on-device model validation.

## Links

[Original Paper](https://arxiv.org/abs/2611.14413)
