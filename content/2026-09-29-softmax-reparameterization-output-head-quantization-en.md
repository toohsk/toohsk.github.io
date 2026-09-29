Title: Softmax Reparameterization for Output-Head Quantization
Date: 2026-09-29
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Softmax reparameterization exploits an algebraic symmetry of the output head to find an equivalent weight matrix that quantizes with far less prediction distortion, reducing KL divergence by up to 93% and cutting generation latency 10.8% on Phi-4-mini.

## Why This Paper Matters

As small language models (SLMs) adopt ever-larger multilingual vocabularies — Llama grew 6× to 202K tokens, Phi grew 3.9×, SmolLM grew 2.6× — the output projection (the matrix that maps hidden states to next-token logits) has become a major memory and compute bottleneck. In a 3–4B parameter model, the output head alone contains roughly 0.4–0.7 billion weights, or 10–17% of total model size. At single-token decoding, the entire head must be read from memory on every step — making it strictly memory-bandwidth-limited.

The natural solution is to quantize the output head to W4 or W2. But practitioners and quantization libraries (including vLLM) historically leave the head at full precision because it directly determines next-token probabilities with no subsequent learned layer to absorb errors. Quantizing it carelessly introduces KL divergence against the full-precision model — distorting the probability distribution the model was trained to produce.

This paper's core insight is that **the head can be quantized without changing the full-precision distribution**, and that different members of the resulting "equivalence class" of functionally identical heads can have dramatically different quantization error. The method searches that class cheaply to find the member that best survives quantization.

## Core Technical Contribution

### The Algebraic Symmetry

Softmax depends only on **relative logits**, not their absolute values. If you subtract the same vector from every row of the output weight matrix $W$, the resulting logit differences (and therefore the softmax probabilities) are unchanged:

$$W_a = W - \mathbf{1} a^\top \quad \Rightarrow \quad p_{W_a}(h) = p_W(h)$$

This is an exact identity — not an approximation. The "shared row component" $a^\top h$ cancels in every exponent ratio. Any $a$ you choose gives a new weight matrix that produces **identical outputs in full precision**.

But when you quantize, different members of this family produce different quantization residuals $E_a = \mathcal{B}(W_a) - W_a$. The quantizer sees different weight distributions, and the rounding errors land in different directions relative to the vocabulary-row structure.

### Fidelity-Selected Reparameterization

The paper restricts the search to a one-dimensional family parametrized by a scalar $t$:

$$W_t = W - t \cdot \mathbf{1} \mu^\top, \quad \mu = \frac{1}{V} W^\top \mathbf{1}$$

Here $\mu$ is the mean of all vocabulary rows. At $t=0$ you get the original head; at $t=1$ you get ordinary mean-centering (which minimizes the Frobenius norm of the head).

For a 14-point grid over $t \in [-2, 8]$, the method:
1. Quantizes $W_t$ with the chosen quantizer (RTN, AW-MSE, or full-Hessian GPTQ)
2. Evaluates KL divergence against the full-precision source model on a held-out validation set
3. Selects $t^* = \arg\min_t L^\mathcal{B}_{\text{val}}(t)$

The validation-selected coefficient on Phi-4-mini turns out to be $t=4$, well away from the mean-centering value $t=1$. This demonstrates that minimizing weight norm ≠ minimizing prediction distortion after quantization.

### Rank-One Correction for Soft-Capping

Gemma-style models apply an elementwise `tanh` soft-cap to logits before softmax, breaking the linear shift symmetry. The paper handles this with a rank-one correction: the shift is restored before the nonlinearity and reapplied after, maintaining the exact equivalence to machine precision ($\text{KL} \approx 7 \times 10^{-10}$).

## Comparison to Prior Work

Prior quantization work (QuIP, SqueezeLLM, GPTQ) focuses on minimizing weight reconstruction error or Hessian-weighted error. Channel scaling and rotation-based methods (QuaRot, LLM.int8()) address activation outliers. Output Embedding Centering does apply mean-centering, but during **pretraining**, not as a post-training step, and uses a fixed $t=1$.

The key difference here is: this method treats the output head as living in an **additive softmax equivalence class** and searches that class by post-quantization **predictive fidelity** rather than weight-space geometry. The matched residual analysis in the paper confirms this distinction: the selected representative can have higher logit reconstruction error than baseline while having lower Fisher-weighted cost, because the error is redirected into directions that matter less to the probability distribution.

Baselines compared:
- $t=0$ (original head): no reparameterization
- $t=1$ (mean-centering): the natural default  
- RTN, AW-MSE, full-Hessian GPTQ as base quantizers

## Reading the Results

**Table 1: W4 KL (before → after reparameterization)**

| Model | RTN | AW-MSE | GPTQ |
|---|---|---|---|
| Phi-4-mini | 1.23 → 0.351 | 0.936 → 0.256 | 0.158 → 0.059 |
| Gemma 3 | 0.049 → 0.046 | 0.041 → 0.038 | 0.033 → 0.033 |
| Qwen3.5 | 0.021 → 0.019 | 0.013 → 0.013 | 0.011 → 0.010 |
| BLOOM-1.7B | 1.12 → 0.531 | 0.594 → 0.136 | 0.037 → 0.028 |
| XGLM-1.7B | 2.13 → 0.143 | 0.584 → 0.095 | 0.009 → 0.007 |

The pattern is clear: **gains concentrate where baseline quantization error is already large**. Gemma 3 and Qwen3.5 already quantize well at W4 (low baseline KL), so reparameterization changes little. Phi-4-mini, BLOOM/BLOOMZ, and XGLM have large baseline KL — and reparameterization recovers 71–93% of the distortion.

At W2, the benefit broadens: every model and nearly every quantizer combination improves because W2 distorts all heads substantially.

The **10.8% batch-one generation latency reduction** on Phi-4-mini comes from storing the head in W4 instead of BF16. Reparameterization recovers the fidelity lost by this compression without adding any inference overhead — for shift-compatible heads, the selected $t$ is simply folded into the packed W4 weights.

## Key Notes of This Paper

The central technical observation is the **additive softmax equivalence class**:

$$p_{W - \mathbf{1}a^\top}(h) = p_W(h) \quad \forall a \in \mathbb{R}^d$$

because softmax of $Wh - (a^\top h)\mathbf{1}$ equals softmax of $Wh$ (common additive constant in all exponents cancels). This means there is a **continuous family of weight matrices** all producing identical probability distributions in full precision.

The key formula for the scalar family is:

$$W_t = W - t \cdot \mathbf{1}\mu^\top, \quad \mu = \frac{1}{V}W^\top\mathbf{1}$$

Intuitively: $\mu$ is the "average direction" in embedding space of the vocabulary. Subtracting multiples of it from every row shifts each token's logit by $t(\mu^\top h)$, which is the same scalar for all tokens and therefore cancels in softmax. Choosing $t > 1$ amplifies the centering beyond the natural mean; on Phi, $t=4$ works best because the Phi vocabulary rows have a strong shared component in the direction of $\mu$ that creates correlated quantization error when left in.

The **Fisher-weighted residual** interpretation is also illuminating. Quantization error that projects onto the gradient of the cross-entropy loss is "costly" — it pushes probabilities in directions the model has learned to be important. The reparameterization redirects error into low-Fisher directions: it increases weight-space reconstruction error while decreasing the semantically meaningful component of that error.

## Limitations

- The one-dimensional search (scalar $t$) does not guarantee the global minimum within the full additive equivalence class (arbitrary $a \in \mathbb{R}^d$). Better representatives may exist along other directions.
- For W4, gains only appear where baseline KL is already high. Models with good W4 calibration (Gemma 3, Qwen3.5) see minimal improvement.
- Evaluation uses 128 WikiText calibration articles and 16 validation articles — a specific distribution. The authors transfer frozen coefficients to C4 and OpenWebMath and they generalize, but sensitivity to very different calibration distributions is not fully explored.
- The W2 absolute quality remains poor despite KL reductions. The method reduces distortion but does not rescue extremely aggressive quantization.
- Tied heads (shared input embedding / output projection) require a separate copy to be quantized, with storage implications.

## Future Work

The authors suggest extending the search beyond the scalar family to richer parameterizations of the equivalence class, and exploring synergies with per-channel scaling and affine quantization (already shown to be complementary).

Promising follow-on directions:
- **Learned equivalence search during QAT**: Rather than post-training selection, learning $t$ jointly with quantization-aware training could yield even better-calibrated representatives.
- **Extension to attention key/query heads**: A similar shift symmetry exists in attention logits ($q^\top k$ is shift-invariant in the same sense), and SageAttention has already explored mean-centering there — a validation-selected coefficient could improve attention quantization similarly.
- **Vocabulary-conditioned reparameterization**: Different token families (ASCII vs. CJK vs. code tokens) may benefit from different shifts; a block-wise version of the search could exploit this structure.
- **Application during model distillation**: If a student SLM is being trained to match a teacher, reparameterizing the student's output head throughout training — not just post hoc — could improve token-probability alignment.

## Implications for Edge / On-Device Deployment

This paper has direct practical value for SLM deployment:

1. **Memory reduction with accuracy preservation**: The output head of a 3–4B SLM can be 400–700M parameters. W4 quantization halves that memory footprint; reparameterization ensures the fidelity cost is minimal.

2. **Latency gains**: The reported 10.8% batch-one generation latency reduction (9.4% at batch 16) on Phi-4-mini comes purely from the output head. This is free speedup — no architecture changes, no retraining.

3. **Zero inference overhead**: For shift-compatible heads (the common linear-softmax case), the selected shift is baked into the packed W4 weights at conversion time. Inference sees a standard W4 kernel.

4. **Post-training applicability**: The method needs only a small calibration set (128 articles) and runs a 14-point grid search — minutes of compute. This is practical in any quantization pipeline.

5. **Framework integration**: The method sits on top of existing quantizers (RTN, GPTQ, etc.) and remains complementary to per-channel scaling and affine techniques already used in llama.cpp, GGUF, and similar on-device runtimes.

## Links

[Original Paper](https://arxiv.org/abs/2609.31291)
