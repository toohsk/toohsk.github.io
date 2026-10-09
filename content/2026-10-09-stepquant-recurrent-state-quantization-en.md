Title: STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization
Date: 2026-10-09
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: STEPQuant introduces spatial-temporal post-training quantization for hybrid linear attention models, achieving FP32-level accuracy at 6-bit with over 5x recurrent state compression and up to 68.7% total memory reduction in production serving.

## Why This Paper Matters

Modern LLMs are increasingly hybrid: models like Qwen3.8-27B and Kimi-Linear-48B-A3B-Instruct combine standard softmax attention (with a KV cache that grows linearly with sequence length) with **linear attention** (which compresses past tokens into a fixed-size recurrent state). In theory, this solves KV cache memory scaling — the recurrent state stays constant regardless of context length.

In practice, a new bottleneck emerges. **Recurrent states must persist separately for every concurrent request during serving.** In SGLang's official Qwen3.8-27B deployment, the FP32 state pool exceeds the BF16 weight memory footprint at just 70 concurrent requests. Unlike KV cache, which has well-understood quantization methods, recurrent state quantization causes severe accuracy degradation because quantization errors propagate through successive state updates.

This paper's key insight is that not all quantization errors are equal — they differ in both *when* they accumulate (temporal lifetime) and *where* they occur (spatial position within the state matrix). Understanding these two dimensions enables a much more principled quantization strategy.

## Core Technical Contribution

### The Gated Delta-Rule State Update

STEPQuant targets hybrid models using Gated DeltaNet (GDN) or Kimi Delta Attention (KDA), both following the gated Delta-rule state update:

$$S_t = A_t S_{t-1} + B_t$$

where:

$$A_t = (I - \beta_t k_t k_t^\top) D_t, \quad B_t = \beta_t k_t v_t^\top$$

Here $D_t$ is the retention gate controlling how much past memory to keep, $\beta_t \in [0,1]$ is the write strength, and $k_t, v_t, q_t$ are the current token's key, value, and query. The head output is $y_t = S_t^\top q_t$. The state matrix has $d_k \times d_v$ elements — fixed regardless of sequence length, but persistent per request.

### Temporal Dimension: Lifetime-aware Bit Allocation

For quantized decoding, the state update becomes:

$$X_t = A_t \hat{S}_{t-1} + B_t, \quad \hat{y}_t = X_t^\top q_t, \quad \hat{S}_t = \mathcal{Q}_t(X_t)$$

**Proposition 1 (Error Propagation):** The accumulated error $E_t = \hat{S}_t - S_t$ follows:

$$E_t = A_t E_{t-1} + \varepsilon_t$$

where $\varepsilon_t$ is the fresh quantization error at step $t$. Critically, $\|A_t\|_2 \leq \|D_t\|_2 \leq 1$, so errors do decay — but for heads where the gate $D_t \approx I$ (strong memory retention), errors persist for many decoding steps. Empirically, Spearman correlation $\rho_S \approx 0.80$ confirms that heads with longer gate half-lives accumulate larger quantization errors.

The paper defines a **lifetime weight** for each quantization unit $u$:

$$L_u = \sum_{j=0}^{H-1} \exp(2j\ell_u)$$

where $\ell_u$ is the mean log-retention rate. The bit allocation problem then becomes:

$$\min_{\{b_u \in \mathcal{B}_{\bar{b}}\}} \sum_u L_u d_u(b_u) \quad \text{s.t.} \quad \sum_u n_u b_u \leq \bar{b} \sum_u n_u$$

This jointly considers quantization error magnitude and how long that error persists through state transitions. The highest-risk units are kept as FP16 sparse pivots.

### Spatial Dimension: Key-Row-Aware Dual-Axis Fitting

The readout error is:

$$\Delta y_t = E_{t-1}^\top A_t^\top q_t = \sum_i (A_t^\top q_t)_i E_{t-1,i,:}^\top$$

This shows that error in different key rows contributes differently to the readout error. The paper defines a **row-impact score**:

$$\omega_i = \mathbb{E}_\text{cal}[g_{t,i}^2], \quad g_t = A_t^\top q_t$$

Additionally, state magnitudes vary substantially along both key rows and value columns — two distinct axes of non-uniformity. To address this, the paper proposes **Key-Row-Aware Dual-Axis Fitting**: separate scales for key rows and value columns. Row scales account for both current magnitude and $\omega_i$ (high-impact rows get more weight), while column scales are fitted with extra weight on columns in high-impact rows.

## Comparison to Prior Work

Previous work on recurrent state compression either applied uniform quantization (which fails at 4–6 bit), or used generic post-training quantization methods designed for weight matrices (not suitable for dynamic, propagating state errors). STEPQuant is the first work to analyze the temporal and spatial structure of recurrent state quantization error explicitly and design a method around it.

**Baselines and results on Qwen3.8-27B and Kimi-Linear-48B-A3B-Instruct:**

| Method | 6-bit accuracy | 4-bit accuracy |
|--------|---------------|---------------|
| Uniform INT8 | Strong baseline | Significant drop |
| Uniform INT6 | Noticeable degradation | Severe drop |
| STEPQuant 6-bit | **Matches FP32** | — |
| STEPQuant 4-bit | — | **Outperforms INT8** |

The key comparison: STEPQuant 4-bit outperforms uniform INT8 across 13 generation tasks while using half the memory.

## Reading the Results

**Accuracy:** Across 7 long-generation and 6 short-generation tasks, 6-bit STEPQuant tracks FP32-state accuracy within 0–1% on both Qwen and Kimi. The 4-bit configuration nearly matches FP32 on Qwen and maintains a modest gap on Kimi. This is evaluated as mean accuracy across 13 diverse benchmarks.

**Memory:** STEPQuant at 6-bit compresses recurrent states by 5.03× (Qwen) and 5.08× (Kimi), reducing **total serving memory (weights + states) by 68.7% and 53.7%** respectively. The 68.7% figure is particularly striking — this is total GPU memory, not just state memory.

**Throughput:** Integrated into SGLang with optimized GPU kernels, 6-bit STEPQuant achieves up to **2.91× faster state updates**, directly reducing per-token latency for recurrent layers.

**Concurrency:** At 70 concurrent requests on Qwen, the FP32 state pool alone exceeded weight memory. With STEPQuant, that same concurrency level uses a fraction of the previous state memory, enabling much denser serving.

## Key Notes of This Paper

### The core insight: error lifetime as a first-class quantity

Most quantization literature focuses on static weight sensitivity. STEPQuant introduces **error lifetime** as a new dimension of quantization sensitivity. The insight is that a 1% quantization error in a head with gate half-life of 1000 steps is qualitatively more dangerous than the same error in a head with half-life of 10 steps — the former will compound through hundreds of subsequent updates.

The lifetime weight $L_u$ elegantly captures this: when $\ell_u \to 0$ (no memory retention), $L_u \to H$ (just the sum, flat contribution); when $\ell_u$ is large (strong retention), $L_u$ grows exponentially. Under a fixed bit budget, this naturally drives high-precision allocation to heads that remember for a long time.

### The 2D structure of state matrices

Weight matrices in transformers have relatively uniform structure; recurrent state matrices do not. The Delta-rule update creates two kinds of non-uniformity simultaneously:

1. **Key-row variation**: Different rows are accessed with different query vectors, so rows aligned with common query directions have higher impact ($\omega_i$)
2. **Column variation**: The state stores value-space information, which has its own channel-wise magnitude variation

A single scale axis (row-only or column-only) cannot fit both simultaneously as the state evolves. The dual-axis approach handles both without requiring per-element scales.

### Error propagation is both a risk and a mechanism

The Delta-rule's error propagation formula has a useful property: the factor $(I - \beta_t k_t k_t^\top)$ from the Delta update partially reduces existing errors along the current key direction. This means that in domains where queries frequently align with recently updated state regions, error accumulation is naturally suppressed — STEPQuant doesn't fight this mechanism, it identifies and works with it.

## Limitations

1. **Fixed calibration**: Bit allocations are determined offline via calibration data and fixed across requests. If the serving distribution shifts significantly from calibration (domain shift), the allocations may be suboptimal.
2. **FP16 pivot overhead**: The sparse FP16 pivots for highest-risk units add a small amount of overhead compared to fully quantized states.
3. **Hybrid models only**: The method is designed specifically for gated Delta-rule recurrent states. It doesn't directly apply to purely recurrent architectures (Mamba-style SSMs) or pure transformers.
4. **Very long sequences**: Theoretical error propagation analysis under extremely long sequences (100K+ tokens) is not fully characterized.

## Future Work

**Authors' directions:**
- Combining with weight quantization (GPTQ, AWQ) for additional compression
- Extending to chunked parallel decoding scenarios
- Lighter-weight calibration procedures

**Promising follow-on research:**
- **Adaptive calibration**: Online updating of bit allocations based on observed request statistics
- **Cross-architecture generalization**: Applying lifetime-aware analysis to RWKV, Mamba, GLA, and other recurrent architectures with different state update structures
- **Error correction**: Incorporating a lightweight correction module that compensates for quantization error at each step
- **Joint KV+state quantization**: Unified framework for models with both KV cache and recurrent states
- **Training-aware quantization**: Incorporating quantization awareness during pretraining for hybrid models to reduce post-training sensitivity

## Implications for Edge / On-Device Deployment

STEPQuant's most direct impact is on **hybrid model serving at scale** — both data center and edge.

For edge deployment specifically, the challenges are more severe: smartphones have 8–16 GB RAM shared with the OS and other apps, and NPU memory bandwidth is far lower than data center GPUs. If hybrid models (linear attention + softmax attention) become the standard small model architecture — which seems likely given their long-context efficiency — then recurrent state memory becomes a critical bottleneck for on-device inference.

The **5× state compression** means a device that could previously handle 1–2 concurrent inference sessions might handle 5–10. The **2.91× state update speedup** directly reduces per-token generation latency on the recurrent layers. Combined with weight quantization, STEPQuant could enable 3B–7B hybrid models to run on mid-range smartphones with practical latency.

For battery-constrained edge deployment, the memory bandwidth reduction (fewer state bytes to load and store per token) also directly reduces energy consumption — a significant consideration for always-on personal AI assistants.

## Links

[Original Paper](https://huggingface.co/papers/2609.38169)
