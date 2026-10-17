Title: PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference
Date: 2026-10-17
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: PN-Mix introduces a query-aware, RoPE channel-pair-level saliency criterion for mixed-precision KV cache quantization that identifies critical key channels once during prefill and reuses them throughout decoding — achieving accuracy comparable to 2.32–2.39-bit methods at only 2.26-bit average precision, and graceful degradation down to 1.5 bits.

## Why This Paper Matters

Long-output inference — generating thousands of tokens in a single pass — is now a core capability of modern LLMs. Reasoning models, code generators, and long-form writing assistants all depend on it. But long-output decoding has a severe bottleneck: the KV cache grows linearly with sequence length, and at batch size 32 with 4K+ generation length, a Qwen3-8B model in FP16 already exhausts a 40GB A100.

KV cache quantization compresses this cache by storing keys and values at reduced precision. The fundamental challenge is that **not all cache elements are equally sensitive to quantization errors**. Uniform low-bit quantization (e.g., 2-bit across the board) degrades accuracy severely, because errors in the wrong channels propagate through attention into dramatically wrong output distributions.

Mixed-precision quantization is the natural answer: allocate more bits to the sensitive channels, fewer bits to the rest. But here's what makes this genuinely hard: **which channels are sensitive depends on the query**, and queries change at every generation step. Prior work identifies salient key channels using only key statistics (average magnitude, channel range), ignoring the query-key interaction that actually determines attention logits. This is a fundamental mismatch: the channels that matter are the ones where quantization errors cause large attention-logit perturbations — and that depends on both keys AND queries.

PN-Mix fixes this by deriving a query-aware channel-pair saliency criterion directly from attention-logit perturbation analysis.

## Core Technical Contribution

### The RoPE Structure Insight

Rotary Position Embeddings (RoPE) are nearly universal in modern LLMs. RoPE works by applying a position-dependent block-diagonal rotation to adjacent pairs of query/key dimensions. This means channels are coupled in pairs: dimensions 0-1, 2-3, 4-5, etc. form pairs after the RoPE rotation.

Prior mixed-precision methods treated channels independently — they scored channel `c` in isolation. PN-Mix operates at the **channel-pair level**, consistent with RoPE's actual mathematical structure. This is not just an implementation choice; it's a principled design decision that reflects how quantization errors propagate through the RoPE-transformed attention computation.

### The Perturbation Analysis

Given a query head `h`, the attention logit between query position `t` and key position `s` decomposes as:

```
Q̃_{h,t} · K̃_{h,s}^T = Σᵢ ||q^t_i||₂ · ||k^s_i||₂ · cos(φ^{t,s}_i)
```

where `i` indexes RoPE channel pairs, `||q^t_i||₂` and `||k^s_i||₂` are the L2 norms of the i-th query and key pair vectors, and `φ^{t,s}_i` is the angle between them.

When the key cache is quantized, channel pair `i` at position `s` becomes `k'^s_i`. Writing the quantization perturbations as `||k'^s_i||₂ = ||k^s_i||₂ + ε^s_i` (norm change) and `cos(φ'^{t,s}_i) = cos(φ^{t,s}_i) + δ^{t,s}_i` (angle change), the resulting attention-logit error is:

```
Q̃_{h,t} · K̃_{h,s}^T - Q̃_{h,t} · K̃'_{h,s}^T 
= -Σᵢ ||q^t_i||₂ · ||k^s_i||₂ · (δ^{t,s}_i + cos(φ^{t,s}_i) · ρ^s_i + ρ^s_i · δ^{t,s}_i)
```

The key insight: the query-key pair-norm product `||q^t_i||₂ · ||k^s_i||₂` **scales** the contribution of each channel pair's quantization error to the attention-logit perturbation. Channel pairs with large query-key pair-norms are disproportionately sensitive — quantization errors there hurt most.

### The PN-Mix Importance Score

The exact perturbation terms (ρ and δ) depend on the reconstructed quantized key and aren't available when you're deciding which channels get high precision. So PN-Mix uses `||q^t_i||₂ · ||k^s_i||₂` as an attention-aware proxy.

Computing this over all causal (t, s) query-key pairs is expensive. PN-Mix uses a separable approximation:

```
σ_{h,i} = E_t[||q^t_i||₂] · E_s[||k^s_i||₂]
```

This approximation independently averages query pair-norms and key pair-norms over observed tokens, avoiding explicit causal-pair enumeration. The paper validates that this separable form closely tracks the true causal-joint score (Spearman rank correlation ≥ 0.996, top-10% overlap ≥ 96.2% across tested models).

### Prefill-Based Saliency with Decoding Reuse

Crucially, `σ_{h,i}` is computed **once during prefill** (the prompt processing phase) and reused throughout the entire decoding phase. This is possible because saliency patterns are relatively stable across the decoding sequence — the channels that were salient during prefill remain salient during generation.

This design has two major benefits:
1. **No decoding-time overhead:** No saliency recomputation during the generation loop.
2. **No offline calibration:** Unlike some prior methods (ChanMix), PN-Mix doesn't require a separate calibration dataset. It adapts to each input prompt.

### Quantization Scheme

PN-Mix uses asymmetric group quantization. For a group `X ∈ R^G`, b-bit asymmetric quantization stores:

```
s_X = (max(X) - min(X)) / (2^b - 1)
z_X = min(X)
X_q = clamp(round((X - z_X) / s_X), 0, 2^b - 1)
```

Keys are grouped per-channel (to handle channel-wise outliers); values are grouped per-token (aligned with the attention weighted-sum structure). An FP16 residual buffer holds the most recent N tokens before they are quantized and committed to the cache.

For the mixed-precision allocation: salient channel pairs (identified by PN-Mix during prefill) are stored at 4-bit; remaining pairs at 2-bit (or 1-bit for very aggressive compression). The resulting average precision across the key cache is approximately 2.26 bits.

## Comparison to Prior Work

| Method | Saliency Criterion | Granularity | Calibration Required |
|--------|-------------------|-------------|---------------------|
| KIVI | Uniform precision | None | No |
| ChanMix | Channel range (key-only) | Individual channel | Yes (offline) |
| Kitty | Average key magnitude | Individual channel | No (online) |
| **PN-Mix** | Query-key pair-norm | **RoPE channel pair** | **No (prefill-adaptive)** |

The key differentiators: PN-Mix considers query-key interactions (not key-only statistics), operates at the RoPE-consistent channel-pair level, and requires no offline calibration.

Numerical comparison (2.25-2.39-bit regime):
- ChanMix at 2.32 bits: PN-Mix at 2.26 bits achieves comparable or better average accuracy
- Kitty at 2.39 bits: PN-Mix at 2.26 bits achieves comparable or better average accuracy
- Below 2.25 bits: PN-Mix degrades gracefully to 1.5 bits; baselines show sharp performance drops

## Reading the Results

Results span three task categories: standard reasoning (math/commonsense), code generation (LiveCodeBench), and AIME (competition-level math). Models tested include DeepSeek-R1-Distill-Llama-8B and Qwen3-8B, generating up to 32K tokens per response.

**The most significant result:** At 1.5-bit average precision, PN-Mix still outperforms baseline methods at 2.25+ bits on several benchmarks. This is the Pareto frontier result — PN-Mix enables aggressive compression without accuracy collapse.

**Memory impact:** At 2.26-bit vs. FP16 (16-bit), KV cache memory is reduced by ~7x. For Qwen3-8B at batch size 32, this enables 8K+ generation lengths that are infeasible in FP16 on a 40GB GPU.

**Throughput:** More aggressive quantization enables larger batch sizes, translating to higher serving throughput. The paper reports large throughput gains in batch-size-limited serving scenarios.

## Key Notes of this Paper

### The Perturbation Decomposition (Equation 4)

The central formula in the paper is the attention-logit perturbation decomposition:

```
ΔLogit = Q̃_{h,t}·K̃_{h,s}^T - Q̃_{h,t}·K̃'_{h,s}^T
       = -Σᵢ ||q^t_i||₂ · ||k^s_i||₂ · (δ^{t,s}_i + cos(φ^{t,s}_i)·ρ^s_i + ρ^s_i·δ^{t,s}_i)
```

**What this says:** The attention logit error caused by quantizing channel pair `i` is **multiplied by** the query-key pair-norm product `||q^t_i||₂ · ||k^s_i||₂`. Think of this as a "lever arm": channel pairs where both the query and key have large norms are like a long lever — even a small angular perturbation (δ) or norm perturbation (ρ) from quantization gets amplified into a large logit error.

**Why prior work was incomplete:** Methods like Kitty use `E[||k||]` alone. But this ignores `||q||`. A channel where keys have large norms but queries have small norms will produce small logit errors — no point allocating high precision there. PN-Mix correctly weights by the product, not the individual magnitudes.

### The Separable Approximation (Equation 5)

```
σ_{h,i} = E_t[||q^t_i||₂] · E_s[||k^s_i||₂]
```

**What this says:** Instead of computing the average of `||q^t_i||₂ · ||k^s_i||₂` over all causal (t,s) pairs (which requires O(T²) computation over the sequence), PN-Mix separately averages query norms over query positions and key norms over key positions, then multiplies. The separability assumption — that query norms and key norms at different positions are approximately independent in the ranking sense — is validated empirically and holds very well in practice (Spearman ρ ≥ 0.996).

**The practical implication:** This means saliency can be computed in O(T) time during prefill, requiring no more memory than a forward pass, and the result is a fixed precision assignment that persists throughout decoding.

## Limitations

- **Key-cache-only:** PN-Mix focuses on the key cache. Value cache quantization remains uniform. A joint key-value mixed-precision scheme could further improve the accuracy-compression tradeoff.
- **Prefill saliency assumption:** The method assumes salient channel pairs during prefill remain salient during decoding. For prompts with unusual structure or very long decoding sequences, this assumption may break down.
- **RoPE-specific:** The RoPE channel-pair structure is central to the method. Non-RoPE architectures (ALiBi, absolute positional encoding) would need a different saliency criterion.
- **Evaluation on reasoning-focused models:** Results focus on DeepSeek-R1-Distill and Qwen3 models. Behavior on non-reasoning models or models with different attention patterns is not fully characterized.
- **Hardware implementation:** The paper reports simulation results for quantization accuracy but does not include fully hardware-optimized kernels for the mixed-precision KV cache layout (4-bit salient + 2-bit remaining).

## Future Work

**Authors suggest:**
- Extending PN-Mix to value cache quantization with a similar attention-aware saliency criterion
- Investigating combined KV cache compression with token eviction and low-rank approximation

**Additional promising directions:**
- **Dynamic saliency update during long decoding:** For extremely long generations (100K+ tokens), periodic saliency recomputation (at checkpoints rather than every step) could improve accuracy without full decoding-time overhead.
- **Saliency-guided quantization hardware design:** Knowing which channel pairs are salient enables specialized memory layouts — salient pairs physically co-located in cache for efficient high-precision reads. This is an architecture design opportunity.
- **Cross-layer saliency sharing:** Salient channel pairs are likely correlated across adjacent transformer layers. A saliency propagation scheme could reduce per-layer computation.
- **Application to speculative decoding:** In speculative decoding, a draft model generates candidate tokens that are verified by the target model. PN-Mix's KV cache compression could enable larger batch sizes for the verifier, improving speculation efficiency.

## Implications for Edge / On-Device Deployment

KV cache quantization is one of the most practically important techniques for on-device LLM deployment. PN-Mix's contributions here are substantial:

1. **7x KV cache memory reduction.** At 2.26-bit vs FP16, the KV cache shrinks by ~7x. For a 1–3B SLM on a device with 8–16GB RAM, this is the difference between supporting 4K context and supporting 28K context — potentially enabling on-device reasoning models.

2. **Graceful degradation to 1.5 bits.** The ability to compress to 1.5-bit average precision while maintaining usable accuracy is critical for the most memory-constrained devices (wearables, microcontrollers with a larger neural accelerator). Most prior methods collapse at this precision.

3. **No offline calibration.** On-device deployment often means diverse user inputs that differ from any calibration set. PN-Mix's prefill-adaptive saliency is perfectly suited to this setting — it automatically adapts to whatever the user's input looks like.

4. **Single prefill computation.** The O(T) saliency computation during prefill is negligible overhead. There's no battery impact from ongoing saliency management during generation.

The combination of aggressive compression, adaptive saliency, and graceful degradation makes PN-Mix directly applicable to production SLM deployments on edge hardware.

## Links
[Original Paper](https://arxiv.org/abs/2610.19605)
