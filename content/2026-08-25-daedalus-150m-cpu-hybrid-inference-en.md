Title: Daedalus-150M: A Convolution-Attention Hybrid Designed for CPU Inference
Date: 2026-08-25
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Daedalus-150M is a 150M-parameter hybrid model with 12 convolution + 6 attention blocks designed from the ground up for single-user CPU inference, achieving 1.76× faster decoding at 2048-token context than a parameter-matched all-attention model.

## Why This Paper Matters

Most small language models are built like large ones and then squeezed onto a CPU. Daedalus-150M inverts the process: the authors **fixed the deployment target first** — one user, one token at a time, 4-bit weights, ordinary CPU — and designed the architecture to match it.

This matters because CPUs impose three constraints that invert the usual GPU optimization targets:

1. **No batching.** Every decoded token requires streaming the entire model's weights through the memory hierarchy. Throughput is governed by bytes-read-per-token, not arithmetic throughput.
2. **Memory bandwidth is the bottleneck, not compute.** A modern CPU can issue far more arithmetic than it can feed with operands. Architectures that do more work over fewer bytes win.
3. **The KV cache is a growing tax.** In a standard all-attention decoder, each new token re-reads the keys and values from every prior token in every layer. This cost is linear in context length and paid every token. On a GPU serving many users simultaneously, this is tolerable. On a CPU serving one user at batch size 1, it dominates long-context generation.

The third constraint is the design lever Daedalus pulls. If most layers use a constant-size recurrent state instead of a growing KV cache, decoding cost becomes nearly flat with context length — and the advantage compounds exactly where users feel latency: long conversations and documents.

## Core Technical Contribution

### Architecture: 18 Blocks, 6 Attention, 12 Convolution

Daedalus-150M has 160.49M parameters arranged as 18 blocks with `d_model = 768`, feed-forward inner dimension 2048, vocabulary 49,152, and context 2048. The block layout is:

```
C C C C A C C A C A C A C A C C A C
```

where A = full attention (at indices 4, 7, 9, 11, 13, 16) and C = short convolution. Six attention layers provide long-range retrieval; twelve convolution layers provide a constant-cost recurrence.

### The Short-Convolution Block

Each convolution block computes:

$$B, C, x = \text{in\_proj}(u) \quad \text{(split three ways)}$$

$$y = \text{depthwise\_conv1d}(B \odot x)$$

$$\text{out} = \text{out\_proj}(C \odot y)$$

The convolution uses kernel length $L = 3$ and is **depthwise** (one filter per channel). Its recurrent state is exactly $L - 1 = 2$ timesteps wide, regardless of context length. The gating terms $B$ and $C$ supply input-dependent modulation that a fixed kernel alone cannot provide.

**The crux:** decoding through a convolution block costs the same at token 2000 as at token 2. An attention block does not.

### Why Exactly Six Attention Layers?

Pure convolution loses the ability to do precise long-range retrieval — what attention uniquely provides. The paper retains six full-attention layers, spread throughout the depth to distribute retrieval capacity across representation levels. The ratio is the core design trade-off:

- More attention layers = better retrieval, larger cache, higher decode cost
- Fewer attention layers = smaller cache, potentially losing recall ability

The paper settles on 6/18 (33% attention) based on a memory-traffic argument for batch-size-1 CPU decoding, then validates it empirically with a pre-registered experiment.

### Supporting Architectural Choices

**Grouped-Query Attention (GQA):** 4 KV heads for 12 query heads, reducing the cache on the 6 attention layers by a further factor of 3. The motivation here is bytes-per-token, not GPU memory capacity.

**Tied embeddings:** The embedding matrix at 49,152 × 768 holds 37.7M parameters — 23% of the model. Tying input and output projections removes an entire copy from both the model file and memory traffic.

**Narrow feed-forward:** Inner dimension 2048 (2.67× `d_model`) instead of the conventional 4×, shifting parameters into depth and attention layers and away from the widest, most bandwidth-hungry tensors.

### Cost Model: What the Architecture Should Buy

At batch size 1, the bytes read per generated token are:

$$M(t) = W + \underbrace{2 \cdot L_A \cdot h_{kv} \cdot d_h \cdot b}_{\kappa} \cdot t$$

where $W$ is quantized weight bytes, $L_A$ is the number of attention layers, $h_{kv}$ is KV heads, $d_h$ is head dimension, $b$ is bytes per cache element, and $t$ is context depth.

For Daedalus: $L_A = 6$, $h_{kv} = 4$, $d_h = 64$, $b = 2$ → $\kappa_{\text{hyb}} = 6{,}144$ B/token

For the dense all-attention twin ($L_A = 24$, $h_{kv} = 2$): $\kappa_{\text{dense}} = 12{,}288$ B/token

The model predicts a 1.17× speedup at 2048 tokens. The measured speedup is 1.76×. The gap is attributed to attention's cache traversal being latency-bound (dependent softmax over a cache that may exceed last-level cache) rather than simply bandwidth-bound — and the hybrid having 18 vs. 24 layers, so per-layer fixed costs are paid 25% less often.

### Training

- **Data:** 59.9B tokens from a 10-source English mixture of 16.93B unique tokens (~3.5 epochs), capped at 4 epochs per source
- **Optimizer:** 2D weight matrices use Muon; embeddings, norms, biases use AdamW
- **LR schedule:** Warmup → stable → linear decay to zero (not cosine; full decay is more sample-efficient)

## Comparison to Prior Work

The paper pre-registered a five-task benchmark bar of **42.20** (the strongest of four beatable peers: GPT-2 124M, Pythia-160M, OPT-125M, GPT-neo-125M, MobileLLM-125M) before training. All peers were re-scored on the same harness to avoid comparison artifacts.

| Model | Tokens | 5-Task Mean |
|---|---|---|
| GPT-2 124M | ~300B | ~42.0 |
| Pythia-160M | ~300B | ~42.1 |
| MobileLLM-125M | 1T | ~42.2 |
| **Daedalus-150M (full)** | **59.9B** | **47.31** |

Daedalus clears the bar by 5.11 points while training on 3–6× less data. A 2T-token peer remains 3.9 points ahead — an acknowledged trade-off the authors accepted in exchange for decode speed.

The ablation pair (hybrid vs. dense twin, both 5B tokens) gives the hybrid a 0.81% win on validation bits-per-byte and a tie on downstream tasks — confirming that the quality gap between architectures is small while the speed gap is large.

## Reading the Results

**CPU decode speed** (4-bit weights, 8 threads, 128-token generation after priming context $t$):

| Context depth | Hybrid vs. dense twin | Hybrid vs. external peer (135M) |
|---|---|---|
| 0 tokens | ~1.0× | ~1.0× |
| 512 tokens | growing | growing |
| 2048 tokens | **1.76×** | **2.08×** |

The *shape* of the result is the thesis. At empty context, the hybrid has no advantage because its advantage *is* the KV cache it doesn't keep — and an empty context has no cache. The advantage grows monotonically and predictably with depth.

**Artifact size:** 95.56 MiB (Q4_0) vs. dense twin's 101.62 MiB — 6.3% smaller despite ~equal parameter count, due to tensor shape effects.

**Memory footprint:** At 2048-token context, the full resident set (model + KV cache ≈ 12.6 MB) fits under 128 MB — small enough to run alongside an application.

## Key Notes of this Paper

The convolution block's core equations deserve careful interpretation:

$$B, C, x = \text{in\_proj}(u)$$

The input projection splits into three channels: a **gating signal** $B$ (input gate), a **gating signal** $C$ (output gate), and the **content signal** $x$ to be convolved. This is distinct from a plain conv layer — the gates make the operation input-dependent.

$$y = \text{depthwise\_conv1d}(B \odot x)$$

Hadamard product $B \odot x$ gates the content before convolution. Because it's depthwise with kernel $L=3$, the only memory required is the last 2 timesteps of $B \odot x$ per channel — a **constant 2-vector per channel** regardless of how long the conversation has been.

$$\text{out} = \text{out\_proj}(C \odot y)$$

Output gating with $C$ selects which features of the convolution result to propagate. This three-way split (in → gate-conv-gate → out) is what gives the block expressivity without increasing state size.

The key insight: the entire state of a convolution block during decoding is a fixed-size buffer of 2 timesteps per channel, which can be held in L1 cache. An attention block's state grows without bound and eventually spills to DRAM.

## Limitations

The paper is admirably transparent about its failures:

- **Dead channels:** ~47.9% of short-convolution channels contribute nothing to output. This is stable and represents ~13.6M inert parameters (8.5% waste). Structural pruning was tested but blocked by the inference runtime's shape-checking.
- **No quantization-aware training (QAT):** QAT failed on the first step (non-finite loss) and was disabled. The released model carries a ~6% perplexity penalty from post-training quantization.
- **Oversized vocabulary:** 49,152 tokens inherited from a cancelled distillation plan. Optimal for ~150M would be 24–32k. The embedding table wastes ~13M parameters.
- **Single seed, English only:** No confidence intervals; the 0.81% ablation margin cannot be interpreted as statistically significant.
- **Training mixture drift:** The realized data mixture drifts from the target due to epoch capping and a training restart.

## Future Work

**Authors suggest:**
- Diagnosing the QAT failure and running it from the finished checkpoint
- Addressing dead channels at initialization rather than retrofitting
- A depth ablation (18 layers × 768 vs. 24 × 640) that was designed but unrun
- Multi-seed replication of the ablation
- A retrieval-sensitive evaluation to find the minimum viable attention fraction

**Additional promising directions:**
- **Adaptive attention ratio:** The current 6/18 ratio was chosen analytically. Learned layer-type selection (e.g., through differentiable architecture search) might find better ratios per task family.
- **Better convolution kernels:** The measured speedup exceeds the bandwidth model, suggesting the convolution path has room to improve with optimized CPU kernels — which might reveal even larger advantages.
- **Longer context training:** The model is trained to 2048 tokens; the bandwidth model predicts the advantage grows past that. Testing at 4096+ would quantify the extrapolation.
- **Combining with speculative decoding:** A constant-state convolution block could serve as a highly efficient draft model, since its decoding cost doesn't grow with the draft length.

## Implications for Edge / On-Device Deployment

Daedalus-150M is one of the most practically motivated SLM papers in recent memory precisely because it starts from deployment constraints rather than ending there:

- **95.56 MiB Q4_0 model + ~12.6 MB KV cache < 128 MB total** — fits within the memory budget of embedded applications on modern smartphones
- **1.76–2.08× faster decoding** at realistic conversation lengths means noticeably more responsive assistants on CPU-only devices
- **8-thread saturation at a lower core count** means the model is less likely to monopolize a phone's CPU cores
- The architecture is **compatible with stock llama.cpp binaries** — no custom operators, no kernel patches needed to ship

The broader lesson is architectural: for single-user CPU deployments, replacing two-thirds of attention layers with short convolutions is "free" in quality and pays a large dividend in latency. This principle generalizes beyond this specific model size.

## Links

[Original Paper](https://arxiv.org/abs/2608.20210)
