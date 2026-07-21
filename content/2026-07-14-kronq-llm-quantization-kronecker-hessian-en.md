Title: KronQ: Better LLM Quantization via Kronecker-Factored Hessian
Date: 2026-07-14
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: KronQ improves post-training quantization by incorporating gradient covariance via a Kronecker-factored Hessian, achieving stable 2-bit quantization on LLaMA-3-70B (7.93 perplexity) where GPTQ and GPTAQ completely fail.

## Why This Paper Matters

Post-training quantization (PTQ) is one of the most practically important techniques for making large language models deployable on memory-constrained hardware. At 2-bit weight-only precision, a 70B parameter model's weights compress from roughly 140 GB (at fp16) to ~17.5 GB — a size that fits on a high-end consumer GPU. But 2-bit quantization is notoriously difficult to do well: the information loss is extreme, and most existing methods fail catastrophically at this precision.

The dominant second-order PTQ method, GPTQ, has a subtle but consequential flaw: it constructs the quantization objective using only **input activation statistics**, implicitly assuming that all output channels of a weight matrix contribute equally to the reconstruction loss. This assumption is wrong. In large transformer layers, different output channels have very different sensitivities to quantization error — some feed into attention heads that heavily influence the loss, others into more peripheral paths. Treating them uniformly wastes the available quantization budget.

KronQ corrects this by introducing the **gradient covariance** into the quantization pipeline, giving a complete picture of how quantization error propagates to the final loss. The formalization comes from the Kronecker-factored approximation to the full Hessian of the quantization loss — the same approximation used in K-FAC for second-order optimization. This is a principled, computationally tractable way to get bidirectional sensitivity information that GPTQ simply lacks.

The payoff is dramatic: at 2-bit weight-only quantization on LLaMA-3-70B, GPTQ and its gradient-aware variant GPTAQ diverge completely (perplexity >2000 on WikiText-2), while KronQ achieves **7.93 perplexity** — usable quality at the most aggressive compression ratio.

## Core Technical Contribution

### The Kronecker-Factored Hessian

The starting point is the layer-wise quantization objective. For a weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$, quantizing to $\hat{W}$ introduces an error $\Delta W = \hat{W} - W$. The resulting increase in the model's loss can be approximated via a second-order Taylor expansion:

$$\delta \mathcal{L} \approx \frac{1}{2} \text{vec}(\Delta W)^\top H \, \text{vec}(\Delta W)$$

where $H$ is the Hessian of the loss with respect to the weights $W$. Computing the full Hessian is infeasible for a matrix with $d_{out} \times d_{in}$ entries, so approximations are necessary.

GPTQ uses only the **input activation Hessian**: it approximates $H \approx \mathbf{X}\mathbf{X}^\top \otimes I_{d_{out}}$, where $\mathbf{X}$ is the input activation matrix. This is the Kronecker product of the input covariance with an identity matrix — effectively assuming the output dimension has no structure worth capturing.

KronQ instead uses the **Kronecker-factored approximation**:

$$H \approx \mathbf{G}^\top \mathbf{G} \otimes \mathbf{X}\mathbf{X}^\top$$

where $\mathbf{G} \in \mathbb{R}^{N \times d_{out}}$ is the matrix of output-side gradients collected over calibration data ($N$ samples). This is the K-FAC approximation: the Hessian factors as the outer product of an output-side factor ($\mathbf{G}^\top \mathbf{G}$, the gradient covariance) and an input-side factor ($\mathbf{X}\mathbf{X}^\top$, the activation covariance). The full $H$ captures how quantization errors interact across both dimensions.

### Contribution 1: Bidirectional Incoherence Processing

Once you have the Kronecker-factored Hessian, the quantization quality depends on how "aligned" the weight matrix is with its Hessian eigenvectors. Large weight magnitudes aligned with large Hessian eigenvalues produce large quantization loss; these need careful handling.

Prior work (QuIP, QuaRot, SpinQuant) applies **random rotation** to the input side of the weight matrix to reduce alignment between high-magnitude weights and high-curvature directions. This "incoherence" processing spreads the error more uniformly.

KronQ extends this to **both dimensions**:
- **Input-side rotation** $Q_X$: derived from the eigenvectors of $\mathbf{X}\mathbf{X}^\top$ (or a random rotation)
- **Output-side rotation** $Q_G$: derived from the eigenvectors of $\mathbf{G}^\top \mathbf{G}$

The rotated weight is $\tilde{W} = Q_G W Q_X^\top$. Quantizing $\tilde{W}$ and then transforming back gives an effective quantizer that treats both input and output dimensions uniformly with respect to the Hessian structure. The output-side rotation is the key innovation — it requires the gradient covariance that GPTQ never computes.

The mathematical intuition: after rotation, the quantization error $\Delta \tilde{W}$ has approximately independent entries with similar magnitudes. The Kronecker-factored Hessian for the rotated system is approximately $I \otimes I$ (identity), meaning all entries contribute equally to the loss. This is the "ideal" scenario where uniform quantization is optimal.

### Contribution 2: Gradient-Aware Mixed-Precision Sensitivity

Not all layers should be quantized to the same bit-width. Mixed-precision quantization allocates more bits to sensitive layers and fewer bits to robust layers, staying within an average bit-width budget.

The sensitivity of layer $l$ to quantization is measured by how much the loss increases when that layer is quantized. KronQ derives a sensitivity metric from the Kronecker-factored Hessian traces:

$$\text{sensitivity}(l) = \frac{1}{d_{in} d_{out}} \text{tr}(\mathbf{G}_l^\top \mathbf{G}_l) \cdot \text{tr}(\mathbf{X}_l \mathbf{X}_l^\top)$$

This is the product of the gradient trace (how much the loss cares about this layer's outputs) and the activation trace (how large the inputs to this layer are). The product approximates the expected quantization loss for that layer under uniform quantization.

GPTQ-based sensitivity metrics use only $\text{tr}(\mathbf{X}_l \mathbf{X}_l^\top)$, ignoring how much the loss cares about this layer's outputs. Layers with large activations but small gradient flow (because they feed into low-sensitivity paths) get over-protected; layers with small activations but high gradient sensitivity get under-protected. KronQ's joint metric corrects both errors.

Given sensitivity scores, bit-width allocation is a constrained optimization: assign each layer to a bit-width from a discrete set $\{2, 3, 4, 8\}$ to minimize total sensitivity while satisfying an average bit-width budget.

## Comparison to Prior Work

| Method | Input-side info | Output-side info | Mixed precision | 2-bit LLaMA-3-70B PPL |
|--------|----------------|-----------------|----------------|----------------------|
| GPTQ | ✓ (activation) | ✗ | ✗ | >2000 (diverges) |
| GPTAQ | ✓ | Partial (per-column gradient) | ✗ | >2000 (diverges) |
| QuIP# | ✓ (rotated) | ✗ | ✗ | ~9–10 |
| QuaRot | ✓ (rotated) | ✗ | ✗ | ~9–10 |
| **KronQ** | ✓ (rotated) | ✓ (Kronecker gradient) | ✓ | **7.93** |

The baseline comparison is stark. GPTQ's assumption of equal output-channel sensitivity is so wrong at 2-bit precision that the model is completely unusable. Methods that add input-side rotation (QuIP#, QuaRot) improve substantially but still don't capture output-side sensitivity. KronQ's bidirectional treatment achieves another step change in quality.

GPTAQ is the closest prior work — it attempts to incorporate gradient information but does so at the per-column level rather than through the full Kronecker factorization. The per-column approximation misses the cross-channel structure captured by $\mathbf{G}^\top \mathbf{G}$.

## Reading the Results

**2-bit weight-only quantization on LLaMA-3-70B**: This is the headline result. WikiText-2 perplexity of 7.93 vs >2000 for GPTQ and GPTAQ is not a marginal improvement — it's the difference between a functional model and a broken one. A perplexity of 7.93 is close to the fp16 baseline (~6.x perplexity for LLaMA-3-70B), indicating minimal degradation despite halving the bit-width below what most methods can handle.

**3-bit and 4-bit results**: At higher bit-widths, all methods produce functional models, but KronQ achieves consistently lower perplexity across the LLaMA-3 family (8B, 13B, 70B). The gains are largest at smaller models and lower bit-widths, where gradient covariance matters more because the quantization noise is larger relative to the model's capacity.

**Mixed-precision allocation**: KronQ's sensitivity metric identifies attention projection layers and the first MLP layer as the highest-sensitivity components in LLaMA-3 architectures — they receive 4-bit allocation while other layers receive 2-bit, staying within a 2.x average bit budget while protecting critical capacity.

**Computational overhead**: Collecting gradient statistics requires one forward-backward pass through the calibration data (128–512 samples in the experiments). This is roughly 2× the calibration cost of GPTQ, which requires only forward passes. The overhead is a one-time cost at quantization time and does not affect inference speed.

## Key Notes of This Paper

### The K-FAC Approximation and Why It's Tractable

The full Hessian $H \in \mathbb{R}^{d_{in}d_{out} \times d_{in}d_{out}}$ is intractable for large weight matrices — for a 4096×4096 linear layer, it has $4096^4 \approx 2.8 \times 10^{14}$ entries. The Kronecker-factored approximation $H \approx A \otimes B$ where $A \in \mathbb{R}^{d_{out} \times d_{out}}$ and $B \in \mathbb{R}^{d_{in} \times d_{in}}$ reduces storage to $d_{out}^2 + d_{in}^2$ entries and enables efficient matrix-vector products via the Kronecker product identity:

$$(A \otimes B) \, \text{vec}(X) = \text{vec}(B^\top X A)$$

The rotation matrices $Q_G$ and $Q_X$ are the eigenvector matrices of $A = \mathbf{G}^\top \mathbf{G}$ and $B = \mathbf{X}\mathbf{X}^\top$ respectively. After rotating $W \mapsto Q_G W Q_X^\top$, the approximate Hessian for the rotated system becomes $\Lambda_G \otimes \Lambda_X$ where $\Lambda_G$ and $\Lambda_X$ are diagonal eigenvalue matrices. The quantization problem in the rotated basis has a diagonal Hessian (separable across weight entries), making it exactly solvable with GPTQ-style column-wise quantization.

Crucially, the full bidirectional rotation ensures the Hessian in the rotated basis has **minimum spread** — the eigenvalues are as uniform as possible — which minimizes the worst-case quantization error under any fixed quantizer.

### Why Gradient Covariance Changes Everything at 2-bit

At 4-bit quantization, each weight has 16 possible values, and the quantization error per entry is small enough that ignoring output-channel sensitivity doesn't cause catastrophic degradation. At 2-bit, each weight has only 4 possible values (typically {-1.5, -0.5, 0.5, 1.5} after scaling). A single sensitive output channel carrying a quantization error of one step size can propagate a non-trivial loss increase through the network.

The gradient covariance $\mathbf{G}^\top \mathbf{G}$ tells KronQ which output channels are sensitive: a large eigenvalue corresponds to an output direction that strongly influences the loss. By rotating to align with these eigenvectors, KronQ ensures that quantization error is spread as uniformly as possible across output channels in terms of loss impact — the channel that costs the most to misbehave gets the least quantization noise allocated to it.

GPTQ with only input-side information treats all output channels identically, which can leave highly sensitive channels exposed to large quantization errors at 2-bit precision. That exposure is what causes the divergence GPTQ exhibits on LLaMA-3-70B.

## Limitations

1. **Calibration data requirement**: KronQ requires one forward-backward pass through calibration data (128–512 samples). This is twice the cost of GPTQ-style calibration and requires access to gradient computation, which may be non-trivial in some deployment pipelines.

2. **Memory for gradient matrices**: Storing $\mathbf{G}^\top \mathbf{G}$ requires $O(d_{out}^2)$ memory per layer. For the largest projection matrices in 70B models ($d_{out}$ up to 8192), this is ~270 MB per layer. The full calibration process requires materializing these matrices for all layers simultaneously, imposing a working memory overhead beyond just holding the weights.

3. **Limited evaluation scope**: The paper focuses on weight-only quantization (W-only PTQ). The interaction of KronQ's bidirectional rotation with weight-activation quantization (W+A PTQ, relevant for INT8 deployment) is not studied.

4. **Architecture specificity**: Experiments are on LLaMA-3 family models. Transformer variants with different layer structures (MoE, linear attention, Mamba-style SSM) may have different sensitivity profiles where the gradient covariance behaves differently.

5. **No finetuning comparison**: KronQ is compared to other PTQ methods. A comparison to quantization-aware training (QAT) would contextualize the remaining gap to trainable compression methods.

## Future Work

**From the authors:**
- Extension to activation quantization (INT8 weight+activation deployment)
- Hardware-specific kernel implementation to realize the theoretical speed gains from 2-bit weight storage

**Additional promising directions:**
- **Layer-wise adaptive calibration**: The current method uses the same calibration data for all layers. Adaptive sampling strategies that provide more calibration signal for high-sensitivity layers could improve precision further.
- **Composition with distillation**: KronQ's sensitivity metric could guide layer-wise knowledge distillation, where layers identified as high-sensitivity are distilled from a larger teacher while low-sensitivity layers are directly quantized.
- **Cross-architecture transfer**: If sensitivity metrics are reproducible across checkpoints of the same architecture (different training stages, different datasets), calibration data could be shared, reducing per-model overhead.
- **Quantized fine-tuning**: Using the KronQ quantization as a starting point for parameter-efficient fine-tuning (e.g., LoRA on the quantized model) could recover accuracy lost at extreme bit-widths while maintaining the memory advantage.
- **Extension to SSMs and MoE**: The gradient covariance approach should generalize to Mamba-style state space models and Mixture-of-Experts layers, where the sensitivity landscape differs significantly from dense transformers.

## Implications for Edge / On-Device Deployment

**2-bit quantization enables a new class of deployments.** A 70B parameter model at 2-bit weight storage requires ~17.5 GB for the weights alone. With the full model fitting in a 24 GB GPU (RTX 3090/4090) or two 16 GB GPUs, inference that previously required multi-GPU data-center hardware becomes feasible on consumer hardware or a powerful workstation.

**For smaller SLMs, the implications are even more direct.** Applying KronQ to a 7B model at 2-bit yields ~1.75 GB weight storage — small enough for mobile GPUs, high-end smartphones (e.g., devices with 8–12 GB LPDDR5 unified memory), or embedded AI accelerators with on-chip SRAM in the gigabyte range. If KronQ's perplexity advantage over GPTQ holds at the 7B scale (which the paper's results suggest), this could be the quantization recipe that makes truly capable on-device LLMs practical.

**The gradient covariance overhead is a one-time cost.** Once the quantized model is produced, inference requires no gradient computation. The 2× calibration overhead versus GPTQ is a deployment preparation cost, not a per-inference cost. For a model deployed millions of times, this is negligible.

**Mixed-precision allocation maps naturally to heterogeneous hardware.** SoC designs for mobile AI often have tiered memory: fast on-chip SRAM (small) and slower off-chip DRAM (large). KronQ's sensitivity metric can guide a strategy where the highest-sensitivity layers are stored in on-chip memory (at 4-bit) while the bulk of parameters live in DRAM at 2-bit, minimizing bandwidth pressure for the layers that matter most.

## Links

[Original Paper](https://arxiv.org/abs/2607.07964)
