Title: Soofi S 30B-A3B: Frontier Performance at 3B Active Parameters via Mamba-MoE Hybrid
Date: 2026-07-14
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Soofi S 30B-A3B achieves dense 14–27B model performance while activating only 3B parameters per token through a Mamba-MoE hybrid architecture, delivering 8–9× the decode throughput of dense models at long context — with full data transparency and open weights.

## Why This Paper Matters

The economics of language model deployment are dominated by two costs: **memory bandwidth** (how fast you can stream weights from memory to the compute unit) and **KV cache memory** (how much context you can hold in flight). Both grow with parameter count and sequence length, and both are why large models are expensive to serve.

Most efficiency work focuses on reducing parameter count outright — smaller models, pruned models, quantized models. But there's another axis: a model can have a large **total** parameter count (expensive to store, but you only pay the storage cost once) while activating only a small **fraction** of those parameters per forward pass. Mixture-of-Experts (MoE) architectures exploit this by routing each token through only a subset of expert layers.

Soofi S takes this a step further with a **hybrid Mamba-MoE** design: of its 52 layers, 23 use Mamba-2 sequence mixing (which maintains a fixed-size recurrent state instead of a growing KV cache), 23 use sparse MoE feed-forward layers (only activating 3.2B of 31.6B parameters per token), and only 6 use standard GQA attention (which does maintain a KV cache). The result is a model where:

- The **compute cost per token** corresponds to a ~3B parameter model
- The **memory cost** of weights corresponds to a 31.6B parameter model (but you only pay this once per deployment)
- The **KV cache** stays near-constant as context grows (because only 6 of 52 layers use attention)

The last point is what distinguishes Soofi S most sharply from dense transformers. At 40K context length with batch size 32, Soofi S achieves **8–9× the aggregate decode throughput per GPU** of dense 14–24B models. This isn't a marginal efficiency gain — it's an architectural property that makes long-context, high-concurrency serving economically viable.

The paper's second contribution is equally important: complete training transparency. Unlike most open-weight releases that disclose weights but hide data recipes, Soofi S releases the full pretraining corpus statistics (per-source, per-language token counts), reproducible data construction scripts, the complete learning rate schedule, and all hyperparameters. It is the first sovereign German-English model to reach frontier capability at this level of openness.

## Core Technical Contribution

### The Hybrid Mamba-MoE Architecture

Soofi S adopts the Nemotron 3 Nano reference architecture without modification. The 52-layer network interleaves:

- **23 Mamba-2 layers**: Sequence mixing via a structured state space model. Mamba-2 maintains a fixed-dimensional recurrent state $h_t \in \mathbb{R}^d$ that is updated with each new token via a selective mechanism:

  $$h_t = A_t h_{t-1} + B_t x_t, \quad y_t = C_t h_t$$

  Unlike attention, where the context representation grows with sequence length, the Mamba-2 state is always $d$-dimensional regardless of sequence length. This makes the memory footprint for sequence state **constant** in context length.

- **23 Granular MoE layers**: Sparse feed-forward layers where each token is routed to a small subset of experts. With ~31.6B total parameters and ~3.2B active per token, roughly 10% of the parameter budget is active at any time. The granular (fine-grained) MoE design uses many small experts rather than a few large ones, which gives the router more granular control and reduces the variance of expert utilization.

- **6 GQA attention layers**: Standard grouped-query attention with a KV cache, distributed sparsely through the network depth. These are the only layers that require storing per-token key-value pairs. By having only 6 such layers out of 52, the KV cache grows at roughly 6/52 ≈ 11.5% of the rate of a fully attention-based model of similar depth.

### Why KV Cache Dominates Long-Context Serving

For a dense transformer with $L$ layers, each with key and value projections of dimension $d_{kv}$, the KV cache per sequence is:

$$\text{KV cache size} = 2 \cdot L \cdot d_{kv} \cdot T \cdot \text{bytes per element}$$

where $T$ is the sequence length. This grows **linearly with context length** — serving a request with 100K tokens requires 100× the KV cache memory of a 1K token request. For a large model (e.g., Llama-3-70B with 80 layers and 8192 key/value dimensions), a single 100K-token sequence in fp16 requires:

$$2 \times 80 \times 8192 \times 100{,}000 \times 2 \approx 262 \text{ GB}$$

This is why long-context serving is so expensive for dense transformers: the KV cache alone can exceed the weight memory. Multi-query attention and GQA reduce $d_{kv}$ but the linear scaling with context length remains.

Soofi S changes the denominator: with only 6 GQA layers instead of 52, the KV cache growth rate is reduced by ~9×. At 40K context, the KV cache is 9× smaller than for a dense model of similar depth. This directly translates to the 8–9× throughput advantage measured in the paper.

### The Three-Phase Training Curriculum

Soofi S is trained on a 26.68 trillion token corpus in three phases:

1. **Phase 1: Diverse pretraining** (~20T tokens): A broad German-English mixture with extensive web crawl data, code, and multilingual content. The German-English ratio is deliberately up-weighted to prioritize German language quality.

2. **Phase 2: High-quality annealing** (~5T tokens + 1.58T constant LR): A curated, higher-quality subset. The Warmup-Stable-Decay (WSD) learning rate schedule is used — the decay phase is where the model consolidates its training signal.

3. **Phase 3: Long-context extension** (~100B tokens): Training with 1M-token sequences using context parallelism. This enables the model's 1M-token context window, which is practically valuable for document-level processing.

### Serving Efficiency Under the WSD Schedule

The paper uses a WSD (Warmup-Stable-Decay) learning rate schedule across all phases. The choice is deliberate: the stable plateau allows the model to train at full learning rate for a long period (which gives more opportunity for the Mamba and MoE components to specialize), while the decay phase refines the representations. The minus_sqrt decay shape (learning rate proportional to $1/\sqrt{t}$ during decay) is empirically observed to give better final performance than linear decay.

## Comparison to Prior Work

Soofi S is compared against 16 open-weight models in the 3B–70B active parameter range, plus European sovereign models:

| Model | Active params | KV cache growth | 40K-context TPS/GPU | English | German |
|-------|-------------|----------------|---------------------|---------|--------|
| Llama-3.1-14B (dense) | 14B | Linear | ~1× (baseline) | ~83 | ~71 |
| Llama-3.1-27B (dense) | 27B | Linear | ~0.6× | ~87 | ~76 |
| Qwen3-30B-A3B (MoE) | 3B active | Linear (full attn) | ~2–3× | ~86 | ~78 |
| **Soofi S 30B-A3B** | 3.2B active | **Near-constant** | **8–9×** | **~90** | **~83** |
| OLMo 3 32B (dense) | 32B | Linear | <1× | ~85 | ~77 |

The key differentiator versus other MoE models (like Qwen3-30B-A3B) is the Mamba-2 layers: pure MoE models still use full attention for sequence mixing, giving linear KV cache growth. Soofi S's hybrid design achieves near-constant cache because the Mamba layers handle most sequence mixing without any cache.

European sovereign model comparison: models like BLOOM, Occiglot, and Bafög-Family are substantially outperformed on German benchmarks. Soofi S is the first European sovereign model to match frontier international models on English while achieving state-of-the-art German performance.

## Reading the Results

**Capability Index** (average of Code, GSM8K, GPQA-Diamond, English, German benchmarks, normalized to best model per group):
- Soofi S achieves the highest Capability Index among fully open models, ahead of OLMo 3 32B and Apertus 70B
- It matches dense 14–27B models on aggregate performance while activating 4–9× fewer parameters

**Long-context serving efficiency**:
- At 40K context, batch 32: 8–9× aggregate decode TPS/GPU versus dense 14–24B models
- The throughput advantage is **scale-invariant with context**: as context grows from 4K to 256K tokens, Soofi S maintains near-constant decode TPS while dense models degrade roughly linearly

**German language performance**: Best-in-class on German benchmarks including regional knowledge, outperforming all European sovereign models and matching international frontier models.

**Code performance**: Best code aggregate in both English and German among 17 open base models. Code generation benefits from Mamba's long-range dependency modeling within files.

## Key Notes of This Paper

### How Mamba-2 Achieves Constant-State Sequence Mixing

Mamba-2 is a structured state space model (SSM) with a selective mechanism. The state equation is:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

where $\bar{A}$ and $\bar{B}$ are input-dependent (selective) transition matrices computed from the input $x_t$ via learned projections. The key property: the state $h_t \in \mathbb{R}^{N \times d}$ has fixed dimension regardless of sequence length $t$. No matter how many tokens have been processed, the "memory" is always a fixed-size matrix.

The selectivity mechanism is critical: $\bar{A}$ and $\bar{B}$ depend on the current input, allowing the model to "decide" what to remember and what to forget at each step. This is in contrast to earlier fixed-coefficient SSMs (like S4) that apply the same dynamics regardless of content. The learned selectivity is what gives Mamba-2 language modeling capability comparable to attention.

**The efficiency insight**: generating the next token with Mamba-2 requires only a constant-time state update $(h_t \leftarrow \bar{A}h_{t-1} + \bar{B}x_t)$ regardless of how many tokens have been generated. With attention, generating the next token requires attending over all previous tokens (O(T) computation and O(T) memory for the KV cache). At T=100K, this difference is enormous.

### Granular MoE and the Expert Routing Mechanism

The MoE layers use a gating network (router) that assigns each input token to a subset of experts:

$$y = \sum_{i \in \text{TopK}(g(x))} g_i(x) \cdot E_i(x)$$

where $g(x)$ is the routing score, $\text{TopK}$ selects the top-$k$ experts by score, and $E_i$ is the $i$-th expert network. "Granular" MoE means each expert is a small FFN (smaller than in traditional 2-expert or 4-expert MoE designs), and the model uses a larger number of experts per layer with $k$ typically 2–4.

More experts with smaller size means:
1. The router has finer-grained control over which circuits activate for a given input
2. Expert utilization is more uniform (less "expert collapse" where a few experts dominate)
3. The effective parameter count scales more smoothly

The shared expert variant (used in Soofi S, following the Nemotron 3 Nano design) adds an always-active expert per layer that handles common patterns, with routed experts handling specialized content. This prevents the routing mechanism from having to route every token through experts for basic functionality.

### The Hybrid Design Tradeoff

The hybrid Mamba-MoE design involves engineering tradeoffs:

**Mamba-2 for sequence mixing**: Handles long-range token dependencies efficiently but with bounded context (the state has finite capacity). For tasks requiring exact recall of distant tokens, the finite state is a limitation relative to full attention.

**GQA attention for 6 layers**: Provides exact, unbounded attention for the layers where it matters most. The placement of the 6 GQA layers through the network depth (rather than at the top or bottom only) is a design choice from Nemotron 3 Nano that the authors adopt without modification.

**MoE for feed-forward capacity**: Provides large total parameter capacity (31.6B) at low per-token compute cost (3.2B active). The all-to-all communication for expert routing is the bottleneck that motivates the specific interconnect topology used in training.

## Limitations

1. **KV cache not fully eliminated**: The 6 GQA layers still require KV cache storage that grows with sequence length. At very long contexts (>1M tokens), this becomes a bottleneck again, though at a much smaller scale than full-attention models.

2. **Expert routing communication overhead**: During training and inference, all-to-all communication between expert-parallel ranks introduces latency proportional to batch size and expert count. This is the primary reason granular MoE is harder to deploy than dense models.

3. **No instruction tuning**: The paper presents only the base (pretrained) model. Instruction-following capability requires additional supervised fine-tuning (SFT) and alignment training, which may alter the efficiency profile.

4. **German/English focus**: Strong bilingual performance in German and English comes at the cost of capacity for other languages. Multilingual use cases beyond these two languages may see degraded performance.

5. **Mamba-2 context limitations**: The fixed-state Mamba-2 architecture has finite capacity for long-range recall. For tasks requiring verbatim retrieval from very long contexts, the 6 attention layers may not be sufficient compensation.

## Future Work

**From the authors:**
- Supervised fine-tuning and alignment (instruction-tuned variants)
- Extension to other European languages
- Evaluation of Soofi S's architecture for multilingual code generation

**Additional promising directions:**
- **On-device deployment of the 3B-active-parameter inference path**: The per-token compute matches a 3B dense model. With quantization (4-bit weights), the storage cost is ~16GB, potentially deployable on consumer hardware with unified memory (M-series Macs, high-end mobile APUs).
- **Extending the Mamba-MoE hybrid to dedicated SLM sizes**: The architectural principles (Mamba for sequence mixing, sparse MoE for capacity, minimal attention) could be applied to models with total parameter counts of 3–7B, where the active parameters would be in the 500M–1B range — genuinely embedded-deployable.
- **KV-cache-free inference modes**: In autoregressive generation where the context is a fixed prompt (batch inference scenarios), the Mamba-2 state could be precomputed and reused, potentially eliminating even the 6-layer KV cache overhead.
- **Sparse activation measurement across hardware platforms**: The architectural efficiency claims are measured in throughput (TPS/GPU). Translating these to power efficiency (tokens per joule) on edge hardware would be more directly relevant to on-device deployment decisions.

## Implications for Edge / On-Device Deployment

**The 3B active parameter footprint is the key metric for on-device feasibility.** Inference latency and power consumption for autoregressive generation are dominated by compute per token, which scales with **active** parameters, not total parameters. Soofi S's 3.2B active parameters give it the inference profile of a 3B model despite 31.6B total parameters.

**At 4-bit quantization, weight storage drops to ~16GB.** This is within range of premium workstations, high-memory mobile workstations, and edge inference servers. While not yet smartphone-class, this represents a substantial improvement over the 70GB+ required for dense frontier models.

**The constant KV cache is decisive for multi-turn on-device inference.** On-device assistants often maintain long conversation histories. With a full-attention model, a conversation of 10,000 tokens requires 10× the memory of a 1,000-token conversation. Soofi S's near-constant KV cache growth means memory usage is essentially independent of conversation history length — a critical property for sustained on-device deployment.

**The architectural insight generalizes.** The engineering principle demonstrated by Soofi S — use SSM layers for the bulk of sequence mixing, reserve full attention for only a small fraction of layers — can be applied to purpose-built SLMs. A model with 7B total parameters following this design might have 700M–1B active parameters per token, which puts it firmly in the range of today's on-device AI chips.

## Links

[Original Paper](https://arxiv.org/abs/2607.09424)
