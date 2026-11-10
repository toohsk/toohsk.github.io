Title: LowRAM: 26.9x KV Cache Compression by Predicting Values from Keys, Composable with Quantization
Date: 2026-11-10
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: LowRAM achieves near-lossless 4x KV cache compression by predicting values from keys within each attention head, pairs with low-rank key compression, and composes with quantization and eviction to reach 26.9x compression at 94.5% baseline performance.

## Why This Paper Matters

KV cache memory grows linearly with context length and batch size, capping throughput. At long contexts, the KV cache dominates activation memory — a 7B model at 128K context requires ~16GB for the KV cache alone.

Existing post-training methods exploit three largely orthogonal axes of redundancy: eviction (discard unimportant tokens), quantization (lower precision), and low-rank decomposition (approximate KV matrices). LowRAM discovers a **new axis entirely orthogonal to all three**: within each attention head, values are substantially predictable from their corresponding keys.

Keys and values are computed from the same token representation. While they serve different functions (keys for routing attention, values for computing outputs), their shared origin creates a predictable structural relationship. Prior work mostly ignored this, treating K and V as independent compression targets.

## Core Technical Contribution

### Source-Target Analysis: What Predicts What?

Before proposing a method, the paper rigorously evaluates predictive directions. For each layer, train the same predictor architecture to reconstruct either $K_L$ or $V_L$ from various source representations: $K_{L-1}$, $V_{L-1}$, $KV_{L-1}$, $V_L$, $K_L$.

**Key finding**: Reconstructing current-layer values from current-layer keys ($K_L \to V_L$) achieves the highest recovered downstream performance across all configurations.

**Critical secondary finding**: Lower reconstruction MSE does NOT reliably correspond to better downstream performance. Source-target pairs with similar MSE can have significantly different recovered performance. This invalidates using MSE alone to design compression schemes.

### Value Cache Amortisation

For each attention head, fit a lightweight per-head network at test time:

$$f_{\theta_{\ell,h}}(k) = \sigma(kW^{(1)}_{\ell,h} + b^{(1)}_{\ell,h})W^{(2)}_{\ell,h} + b^{(2)}_{\ell,h} + kW^{\mathrm{lin}}_{\ell,h}$$

A 2-layer MLP with a learned linear residual branch. During prefill, adapt parameters by minimizing value reconstruction loss:

$$\mathcal{L}(\theta) = \frac{1}{T}\|f_\theta(K) - V\|_F^2$$

After fitting, rank tokens by reconstruction error and **store residual corrections only for the tokens hardest to predict**:

$$e_i = \|f_\theta(k_i) - v_i\|_2^2$$

The residual budget fraction $p \in (0,1]$ controls how many residuals to keep. The resulting value cache is: `{network parameters, sparse residuals for top-p hardest tokens}`.

At decoding: $\hat{V} = f_\theta(K) + \text{sparse residuals}$.

### Asymmetric Pipeline: LowRAM

Values are predicted from keys; keys still need compression. For keys, low-rank decomposition is better (key energy concentrates in a small spectral subspace). LowRAM pairs value amortisation with **xKV** (Chang et al., 2026): shares a single SVD basis across contiguous layer groups, exploiting alignment between dominant key-cache subspaces of adjacent layers.

**The asymmetric insight**: keys and values are structurally different objects — different optimal compression strategies. This is validated empirically: applying the same predictor strategy to keys yields worse performance than to values.

### Practical Efficiency Improvements

1. **Meta-learned initialization (Slim Attention Prior)**: Pre-learn a "good starting point" for the MLP across many contexts via meta-learning. Dramatically reduces prefill overhead introduced by test-time fitting.

2. **Selective reconstruction**: At each decoding step, reconstruct only a selected subset of KV pairs. Achieves up to **5x higher peak decode throughput** at 4x compression. This exploits the fact that not all KV pairs affect each decoding step equally.

3. **Pre-RoPE inversion**: Before compression, invert RoPE on keys; reapply at attention time. Pre-RoPE representations are more amenable to compression (empirically shown; theoretically, rotation destroys the smooth spectral structure).

## Comparison to Prior Work

Mistral-7B on LongBench contexts > 10K tokens:

| Method | Compression | Performance (% baseline) |
|--------|------------|-------------------------|
| Eviction only | 4× | ~87% |
| Quantization only | 4× | ~89% |
| LowRAM | 4× | ~96% (near-lossless) |
| **LowRAM-EQ** | **26.9×** | **94.5%** |

LowRAM-EQ composes LowRAM with eviction and quantization. The key result: 26.9x compression retaining 94.5% of baseline on long-context tasks. This exceeds what any individual method achieves at equivalent compression ratios.

Results hold across three model families (Mistral, Llama, Qwen), confirming this is a general technique, not a model-specific optimization.

## Reading the Results

**Near-lossless at 4x**: 96% baseline performance at 4x compression means LowRAM alone is essentially transparent at moderate compression. For many applications, there's no perceptible quality difference.

**5x throughput at 4x compression**: This seems counterintuitive — compression reduces memory but throughput increases by more than the compression ratio. The reason: LLM decoding is memory-bandwidth bound. Reducing KV cache size reduces memory bandwidth pressure, directly increasing throughput.

**26.9x composability**: The composition result demonstrates that LowRAM doesn't interfere with quantization and eviction — the three methods exploit orthogonal redundancies. This is the practically important property: existing deployments using quantization can add LowRAM for additional gains without redesigning the pipeline.

**Smaller compression budget for 90% performance**: The paper's Figure 1 shows the right panel: maximum compression while retaining 90% baseline. LowRAM-EQ retains 90% at ~26.9x, while eviction alone hits this only at ~4x. That's a 7x improvement in maximum useful compression ratio.

## Key Notes of This Paper

### Why Values Are Predictable from Keys (Structural Analysis)

Both K and V are linear projections of the same token representation $h$:
$$K = hW_K, \quad V = hW_V$$

The key-to-value relationship is therefore:
$$V \approx K W_K^{-1} W_V = K M$$

where $M$ is a fixed linear map. This means a linear model is already a reasonable predictor. The 2-layer MLP adds non-linearity to capture the residuals where this linear relationship breaks down.

**Why not just use $M$?** The attention mechanism applies RoPE to keys (position-dependent rotation), breaking the simple linear relationship. Pre-RoPE inversion restores it, which is why pre-RoPE representations are more compressible.

### Residual Sparsity and Error Distribution

The method stores residuals only for the $p$ fraction of tokens with highest reconstruction error. The effectiveness depends on the distribution of reconstruction errors:

- If errors are uniform (all tokens equally hard), no sparse selection is possible
- If errors are heavy-tailed (most tokens easy, few hard), sparse selection is highly efficient

The empirical finding that 4x near-lossless compression is achievable suggests the error distribution is indeed heavy-tailed — most tokens are well-predicted by the MLP, and a small fraction are genuinely hard to predict.

### Composability Argument

LowRAM targets within-head key-to-value predictability. Eviction targets token-level importance. Quantization targets numerical precision. These three axes are geometrically orthogonal — exploiting one doesn't diminish the gains from others. This is why composition reaches 26.9x rather than the product of individual compression rates being reduced.

## Limitations

- **Prefill overhead**: Test-time MLP fitting adds latency. Meta-learning reduces but doesn't eliminate this. Single-query settings pay the full cost
- **Most effective for long contexts**: The 10K+ context focus means short-context deployments see limited benefit
- **Implementation complexity**: Three composable methods require careful integration; hyperparameter tuning (residual budget $p$, rank for xKV, bit-width for quantization) adds engineering cost
- **Dependent on key quality**: If key compression via xKV is lossy, value prediction quality degrades (values are predicted from compressed keys)

## Future Work

**Authors' direction**: Deeper integration with other low-rank methods, optimization for multi-query inference (where prefill cost is amortized across many generations).

**Additional promising directions**:
1. **SLM-specific KV compression**: Small models (1–4B) often have different attention patterns than 7B+ models — the value predictability may differ, warranting SLM-specific analysis
2. **Dynamic residual budget**: Adapt $p$ per-layer or per-head based on observed error distributions, rather than a fixed budget across all heads
3. **Training-time alignment**: Train models with explicit awareness of the K→V predictability structure, potentially making the relationship even more exploitable at inference
4. **Streaming compression**: For streaming inference scenarios (live ASR, real-time dialogue), adapt the predictor continuously rather than fitting once at prefill

## Implications for Edge / On-Device Deployment

**Memory math for smartphones**:
- Standard 7B model, 128K context, BF16: ~16GB KV cache
- With LowRAM-EQ (26.9x): ~600MB
- iPhone 16 Pro has 8GB RAM shared between OS, app, and model — 600MB KV cache is tractable; 16GB is not

**Throughput matters for battery life**: 5x throughput improvement directly translates to 5x fewer tokens per second of wall time, which means shorter inference bursts and proportionally lower power consumption for mobile SLMs.

**Composability is the killer feature for edge**: Edge deployments typically already use INT4/INT8 quantization. Adding LowRAM on top of existing quantization (the -EQ variant) multiplies compression without redesigning the deployment stack.

**Deployment path for SLMs**:
1. Start with quantized SLM (e.g., INT4 Qwen 1.5B)
2. Add LowRAM for KV cache compression
3. Add token eviction for ultra-long contexts
4. Result: a small model capable of handling much longer contexts with far less RAM

## Links

[Original Paper](https://arxiv.org/abs/2611.07025)
