Title: Think Twice: Metis-1.6 and the Looped MoE Architecture for Efficient Language Models
Date: 2026-11-03
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Metis-1.6 introduces two small MoE language models (Praxis: 2.94B total/0.46B active; Logos: 10.23B total/1.18B active) that run their hybrid Mamba-2/attention stack twice per token, achieving Pareto-optimal efficiency—2nd in math out of 30 models—on only 1.02T tokens, built by a single researcher.

## Why This Paper Matters

The dominant paradigm for improving language models is simple: add more parameters, train on more data. Frontier labs announce parameter counts like trophies. But this approach has a ceiling—or at least a cost ceiling—that only the largest organizations can afford to push against.

Metis-1.6 is a direct challenge to that paradigm. One researcher, Giulianno Vollmer at Lernex, built two models from scratch—training infrastructure, tokenizer, data pipeline, evaluation harness, and all—on AMD Instinct MI300A accelerators, and got results that sit on the *Pareto frontier* of the efficiency-vs-capability space.

The hard problem here is multi-dimensional:

1. **Compute efficiency**: You want more effective computation per parameter per token without just adding layers.
2. **Data efficiency**: Training a capable model without the 10T+ token budgets of large labs.
3. **Architecture novelty**: Can a genuinely new design beat established dense/MoE baselines rather than recombining existing parts?

The paper also doubles as an unusually honest postmortem—detailing an optimizer bug that ran for the entire pretraining campaign, a 1.77B-parameter memory component the model learned to disable, five SFT failure modes, and a subtle sign-inversion bug in length rewards that penalized correct answers.

## Core Technical Contribution

### The Looped Architecture

Both Praxis and Logos use the same architectural pattern at different scales: a hybrid Mamba-2/attention MoE stack that runs **twice per token**, with shared weights between the two passes.

- **Logos**: 20 layers (3 attention at layers 5, 10, 15; 17 Mamba-2), d_model=2,560, 10.23B total / 1.18B active (2.36B across both passes)
- **Praxis**: 12 layers (2 attention at layers 4, 8; 10 Mamba-2), d_model=2,048, 2.94B total / 0.46B active

Key idea: instead of making the model *wider or deeper*, the same stack processes each token twice. This gives each token a "second look"—the second pass sees information from the first via a **cross-pass depth memory** (4 heads, 256 dimensions, gate initialized at -6 to start near zero). The depth memory is small and learned; it carries first-pass notes into the second pass without requiring separate parameters per pass.

The efficiency gain: Logos stores 10.23B parameters but only activates 2.36B per token across both passes—fewer active parameters than any model that beats it on the benchmark suite.

### Latent Mixture-of-Experts

Every layer ends in a MoE block operating in a compressed 1,024-dimensional latent space:

```
FFN(x) = W_up(S(z) + sum_{e in TopK(r(x), 4)} g_hat_e(x) * E_e(z))
```

where `z = W_down * x` (projection into latent space), `S` is a shared SwiGLU expert (always active), and `E_e` are routed SwiGLU experts—192 of width 768 in Logos, 128 of width 512 in Praxis—with top-4 routing. Load is balanced via auxiliary-loss-free bias updates (rate 1e-3) plus a router z-loss (coefficient 1e-3), following the DeepSeek approach. This is where most of the parameter efficiency comes from.

### Hybrid Mamba-2 + Attention

Mamba-2 layers handle the majority of sequence mixing (17/20 in Logos, 10/12 in Praxis). Key Mamba-2 parameters: expansion 2, state size 128, head dim 64, width-4 causal convolution, chunk size 256. Only 3 layers in Logos (2 in Praxis) keep a KV cache—so memory grows slowly with context. Attention uses 40 query / 10 KV heads in Logos (32/8 in Praxis), head dim 64, **no positional encoding at all**.

The Mamba-2 layers provide implicit position information through their recurrent state, which is why removing positional encodings entirely works—and why extending from 4K to 131K context only required a 18B-token continued pretraining phase rather than a full re-architecture.

To allow the two passes to behave differently through the same attention weights, each attention layer adds a **pass-indexed rank-8 low-rank update** to its Q/K/V projection. The total parameter cost of this adaptation mechanism is tiny.

### Residual Streams and Hyper-Connections

Each sub-layer reads from and writes to **four residual streams** of full model width (not just one as in a standard transformer). A learned 4×4 mixing matrix—projected toward doubly stochastic matrices via 8 Sinkhorn-Knopp iterations—combines information across streams. A 64-dimensional pass embedding conditions the controller so it behaves differently on pass 1 vs. pass 2.

### Training Setup

**Data curriculum** (804.76B tokens total):
- Phase A (68.9%): broad filtered web text, books, scientific articles, code
- Phase B (26.1%): up-weighted math, science, software engineering, structured analysis
- Phase C (5.0%): curated technical prose, formal proofs, competition math, reference docs

All shards decontaminated via 13-gram shingle matching against evaluation benchmarks.

**Context extension**: 18B additional tokens on documents up to 163,840 tokens. Logos needle-in-haystack accuracy rose from 19.53% to 68.75% at 131K tokens; RULER reached 60.14% at 128K.

**Post-training**: 120B tokens cold-start SFT on long reasoning traces (math/science/code), then 80B tokens broad SFT across all domains including tool-calling and instruction following.

## Comparison to Prior Work

Logos was compared against 17 post-trained models and 30 base models using a single evaluation harness with identical prompts, scorers, and verifiers across all models.

**Post-trained comparison (40-benchmark mean)**:

| Model | Score | Training Tokens | Active Params/token |
|---|---|---|---|
| Gemma 4 12B | 64.06 | ~? | higher |
| Qwen3.5 9B | 61.90 | ~? | higher |
| Qwen3.5 4B | 58.51 | ~? | higher |
| Qwen3 8B | 55.09 | ~? | higher |
| Gemma 4 E4B | 54.59 | ~? | higher |
| Qwen3 4B | 53.68 | ~? | higher |
| **Logos (10.23B total)** | **52.93** | **1.02T** | **2.36B** |
| Granite 4.1 8B | 46.72 | higher | higher |

The key comparison: Logos is 7th of 17, within 0.75 points of Qwen3-4B, but trained on ~1/12 the median training tokens of the comparators that publish their data count (median: ~12T tokens; Logos: 1.02T). Every model that beats Logos and publishes data was trained on at least 28T tokens.

**Mathematics specifically** (2nd of 30 models):
- AIME 2025: 67.50% (avg@8)
- HMMT February 2026: 63.64% (avg@8, 1st in comparison set)
- HMMT February 2025: 62.92% (1st in comparison set)
- IMO-AnswerBench: 57.00% (1st in comparison set)
- GPQA Diamond: 55.05% (5th in comparison set)

## Reading the Results

The Pareto frontier framing is the right lens here. On a scatter plot of (training tokens, mean benchmark score), Logos sits in the shaded corner: better than the median model, trained on far less data. Same pattern for (active parameters, score).

**Where the looped design wins**: Math and formal reasoning. The two-pass structure gives the model a natural chain-of-thought over the architecture itself—the first pass builds representations, the second refines them with the depth memory notes. This may explain why olympiad math, which requires multi-step symbolic manipulation, benefits disproportionately.

**Where it falls short**: Code generation (17th of 30), instruction following (18th), safety (19th), long context (24th in think mode). These correlate strongly with training data volume—models trained on 2.5T–8T tokens already beat Logos in these domains. The weakness is data-limited, not architecture-limited.

**Praxis (2.94B total)** averages 36.62 on the same benchmarks. AIME 2025: 27.08%, HMMT February 2026: 24.24%, GSM8K: 76.80%, HumanEval: 62.80%. At 0.46B active parameters per pass, it sits on the compute Pareto frontier in four of eight domains.

## Key Notes of This Paper

### The Sign Inversion Bug in Length Rewards (Proposition 1)

This is the most technically precise finding and worth understanding carefully.

The length reward setup: within a group of G samples for the same prompt, correct short samples get reward 1, correct long samples get reward `1-λ`, incorrect samples get 0. Advantages are GRPO-style group-mean centered: `A_i = (r_i - r_bar) / σ`.

The proposition (proved in the paper): with a median split (q=1/2, so half of correct samples are "short") and λ=1/2, **every long correct sample gets a negative advantage** when the group pass rate `p > 2/3`.

Why? The group mean `r_bar = p * [1 - λ(1-q)]`. A long correct sample has reward `1-λ = 0.5`. When enough samples are correct that `p > 2/3`, this reward falls *below* the group mean, so the advantage becomes negative. The gradient then lowers the probability of producing those long-but-correct derivations.

In other words: **once a problem becomes easy enough (2/3+ of samples pass), showing your work is penalized**. The model learns to answer briefly rather than derive carefully.

The one-line fix:

```python
A_i = max(0, A_i)  # for every correct sample i
```

This removes the negative signal on long correct samples while keeping the positive signal on short ones and the negative signal on incorrect ones. Simple, but not obvious from the reward design alone.

### The Optimizer That Never Ran

The entire 804.76B-token pretraining ran with the wrong optimizer. The config said `optimizer.name: adamuon` but the trainer always built a plain Muon/AdamW hybrid. Four discrepancies:

1. Attention matrices (which the name-matching regex missed) trained with AdamW throughout
2. Fused Mamba projections were orthogonalized as a single block, coupling gate/state/step-size singular vectors
3. No second-moment state (AdaMuon) was ever accumulated
4. Weight norms drifted significantly across parameter types (experts: 0.88-0.91×, Mamba out: 1.24×, convolutions: 0.57×)

The loss curve never showed any anomaly. The lesson: assert every parameter's optimizer group membership at startup.

### N-Gram Memory: A Negative Result

The models included hashed n-gram conditional memory (1.77B parameters in Logos): for each position, canonicalized 2-grams and 3-grams are hashed into lookup tables; retrieved values are injected into residual streams via learned gates.

The models learned to close the gates entirely. Mean gate activation across all injection sites: 0.0001–0.0031 after pretraining, at most 0.0141 after post-training. Removing the memory changed held-out loss by less than 10^-3 nats on Logos. The released models ship without the memory (Logos: 20.80 GB vs. 24.35 GB with memory).

Likely causes: (1) Mamba-2 convolutions in layers 0-1 already compute 2-3-gram compositions before the first injection point; (2) hash collisions at 1.73M rows across 805B training tokens create noisy lookups that cost more than they add.

### The SFT Failure with Off-Policy Cross-Entropy

One subtle finding: standard token-level cross-entropy on teacher text causes unbounded gradient magnitude on low-probability stylistic tokens, collapsing generation entropy before RL begins. The fix is a reference-anchored sampling objective:

```
L(x, y*) = -1/|y*| * sum_t clip((π_θ(y*_t|...) / π_ref(y*_t|...))^0.25, 0.2, 2.5) * log π_θ(y*_t|...)
```

This clips the probability ratio (with 0.25 power for smoothing, bounds [0.2, 2.5]) before multiplying by the log-probability. Reference log-probabilities are precomputed offline in FP16, so the reference model costs no accelerator memory during training.

## Limitations

The paper is admirably candid:

1. **Code and instruction following are weak**: Ranks 17th/18th of 30 in these domains. Models with 2.5T–8T tokens already beat Logos here. The architecture isn't the bottleneck—data is.
2. **Long context in think mode is poor**: RULER at 128K drops to 34.17% in think mode (vs. 60.14% in direct mode). The KV cache advantage of having only 3 attention layers partially reverses for very long reasoning chains.
3. **Factual recall is low**: SimpleQA Verified: 3.78—consistent with training on 1/12 the median data.
4. **All conclusions are conditioned on the wrong optimizer**: The Muon/AdaMuon distinction was never evaluated cleanly. The Metis-1.7 ablations (Section 10) measure what the corrected optimizer is worth, but Metis-1.6 itself ran with AdamW on attention throughout.
5. **Single-researcher constraint**: Metis-1.6 can out-iterate large labs but cannot out-compute them. The Pareto efficiency claim is real, but the absolute capability ceiling is far below frontier.

## Future Work

**Authors' stated directions** (Metis-1.7 and MoRE):
- **MoRE (Mixture of Recursive Experts)**: The looped architecture evolves into variable-depth recursion, routing tokens to different numbers of passes. The paper includes 10B-token ablations on this design.
- **Larger n-gram memory off-accelerator**: 50B-parameter table in ternary format (1.85 bits/value, BitNet b1.58 quantization), stored off-accelerator to hold long-tail facts without competing for HBM.
- **Corrected optimizer**: Running AdaMuon correctly on attention matrices, plus properly separated Mamba fused projections.
- **Better SFT data**: Native direct-mode answers (not truncated reasoning), filtered traces (≤2 post-solution verifications), newer teacher models only.

**Promising follow-on directions** not explicit in the paper:
- **Adaptive pass depth with compute budget**: If two passes is good, adaptive 1-3 passes based on token difficulty could optimize a per-token compute budget—essentially turning the looped architecture into a dynamic-depth model at inference time.
- **Depth memory scaling**: The cross-pass depth memory (4 heads, 256 dims) is tiny. Scaling it, or making it attend to a longer buffer of first-pass states, could strengthen the reasoning benefit of the two-pass design.
- **Joint Mamba-2/attention training dynamics**: The optimizer audit revealed that attention and Mamba matrices behave very differently under the same learning rate schedule. An optimizer that dynamically adjusts per-module based on geometry (e.g., Muon's orthogonalization applied correctly to each sub-component) may be the clearest efficiency win.
- **Ternary or quantized experts**: The n-gram memory experiment with BitNet b1.58 ternary weights is a proof of concept. Applying similar quantization to the 192 routed experts in Logos could significantly reduce memory footprint while retaining accuracy, since MoE experts are inherently sparse.

## Implications for Edge / On-Device Deployment

Metis-1.6 is not primarily an on-device model—it was trained at scale on MI300A accelerators. But several design decisions have direct implications for efficient deployment:

**Favorable for edge**:
- **Selective KV cache**: Only 3 of 20 layers in Logos keep a KV cache. At 128K context, this dramatically reduces the memory bandwidth bottleneck that kills transformer inference on memory-constrained devices.
- **Low active parameter count**: Praxis activates 0.46B parameters per pass, and 0.92B total across two passes—comparable to a dense 1B model in compute cost, but with 2.94B total parameters available for routing. This is a favorable tradeoff for devices that can hold the full weights in flash storage but have limited SRAM/HBM.
- **Mamba-2 dominance**: Mamba-2 layers have O(1) state size per token at inference (vs. O(n) KV cache for attention), which is a fundamental advantage for streaming or long-session on-device inference.
- **No positional encoding**: Simpler to implement, one fewer hyperparameter to tune during adaptation.

**Challenging for edge**:
- The two-pass design doubles the sequential depth per token, which increases latency even if active FLOPs are the same. For real-time applications with tight latency budgets, this matters.
- The full Logos checkpoint at 20.80 GB (BF16) does not fit in typical mobile device RAM. Praxis at 6.16 GB is borderline for high-end devices.
- MoE routing requires sparse dispatch, which is harder to implement efficiently on fixed-function NPUs designed for dense matrix operations.

The Praxis size class (2.94B total, 0.46B active) is the more immediately deployable variant. With INT4/INT8 quantization, it could plausibly run on high-end mobile hardware. The architectural choices—Mamba-heavy hybrid, latent MoE, no positional encoding—seem designed with this possibility in mind, even if that is not the stated goal.

## Links

[Original Paper](https://arxiv.org/abs/2610.37903)
[HuggingFace Paper Page](https://huggingface.co/papers/2610.37903)
[Models (Lernex)](https://huggingface.co/Lernex)
