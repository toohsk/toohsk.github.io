Title: Muse Glimmer 30B: An On-Device Agentic Model
Date: 2026-10-22
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Muse Glimmer is a 30B open-source model distilled from a frontier teacher and quantized to ~4-bit, designed to run full agentic workflows — multi-step planning, tool use, and failure recovery — entirely on consumer hardware with 32 GB VRAM.

## Why This Paper Matters

Autonomous AI agents — assistants that can browse, write code, send emails, and orchestrate multi-step tasks — currently require cloud-hosted frontier models. This creates two hard constraints that conflict with each other:

**Privacy.** An on-device agent operating on personal calendars, messages, and documents should not need to send that data to a remote server. Users in healthcare, legal, and personal contexts have genuine reason to require local execution.

**Capability.** On-device models have historically been too weak for reliable agentic behavior. A model that fails at sequential tool invocation, cannot maintain state across steps, or cannot recover from tool errors is not useful for agentic tasks regardless of its benchmark scores.

The gap between "capable enough to be an agent" and "small enough to run locally" is the exact problem this paper addresses. Prior work has shown that general-purpose small models (the Gemma 3, Qwen 2.5 class) can be quantized to run locally, but they were trained as general assistants and lack the specialization for sustained agentic execution. Muse Glimmer attacks this gap directly: a model purpose-built for local agentic deployment, distilled from a frontier teacher, and co-designed with its quantization and speculative decoding pipeline from the start.

## Core Technical Contribution

### Architecture: Dense beats MoE under memory constraints

The first and most significant design decision is the choice of dense over mixture-of-experts (MoE) architecture.

The team ran scaling ladder experiments comparing a dense baseline against two MoE variants (1:8 and 1:16 expert sparsity) under a fixed 32 GB VRAM constraint. The key insight is that under a **constant memory ceiling**, MoE architectures require far more tokens per parameter (TPP) to achieve equivalent loss — 2,800 TPP for 1:8 and 6,000 TPP for 1:16, versus 550 TPP for dense. Given their compute budget, the dense model delivered 2–2.5× better compute efficiency than MoE. This is a deliberate tradeoff: MoE is efficient at inference when memory is abundant, but when total memory is the binding constraint, a dense model wins.

**Language backbone:**
- 52 transformer layers, hidden dimension 6,656 (~25.16B parameters excluding embeddings)
- Vocabulary of 202,048 tokens (~1.34B embedding parameters)
- Grouped-query attention (GQA): 32 query heads, 2 KV heads (16:1 ratio), head dimension 128
- Sliding window attention: 3 local-attention layers (window 2048) + 1 global-attention layer, repeating
- SwiGLU FFN with intermediate dimension 19,968 (~3× hidden dimension)
- RoPE with θ=500,000 on local layers only; global layers inherit positional signal from local layers

The sliding window + GQA combination is critical for agentic deployment. Long-horizon agentic sessions accumulate large KV caches (tool outputs, reasoning traces, history). Constraining KV growth to 2 heads per layer × head dimension 128, combined with the 2048-token sliding window, bounds cache size during multi-step execution.

The team evaluated Multi-Head Latent Attention (MLA, as in DeepSeek-V2) as an alternative KV compression mechanism. Both tested configurations — matching KV head count (halving cache size) and matching total KV footprint (doubling head count) — underperformed GQA on long-context benchmarks (RULER), so MLA was dropped in favor of standard GQA.

**Perception encoder:**
- ViT-G/14 with 50 layers, width 1536, ~1.8B parameters (frozen throughout training)
- Processes images at native resolution via NaViT-style variable-resolution encoding
- 2×2 pixel-shuffle spatial downsampling → 2-layer MLP adapter → linear projection into LM embedding space
- 448×448 image yields 256 visual tokens; maximum 4096 tokens per image

### Training pipeline

Training proceeds in 5 stages: pre-training, mid-training phase 1, mid-training phase 2, SFT, and RL.

**Pre-training via logit distillation:**
The base model is trained by KL divergence against top-256 logits from Muse Spark (the frontier teacher). The distillation objective is KL-only for 95% of tokens; the final 5% transitions to cross-entropy on ground-truth targets. This transition is important: top-k logit distillation has a truncation bias (the excluded probability mass is redistributed), and switching to CE at the end corrects for this accumulated bias. The data spans web text, code, mathematics, multilingual text, and images.

At the end of pre-training, checkpoint averaging ("souping") is applied: the final checkpoint is averaged with 3 prior checkpoints spaced 600 steps apart using uniform weighting. This improves perplexity across all evaluation sets vs. the single final checkpoint.

**Mid-training phase 1 (domain infusion):**
Starting from the souped checkpoint, adds targeted synthetic data including reasoning traces and structured problem-solving demonstrations at a lower learning rate, balancing gains on synthetic objectives against regression on organic perplexity.

**Mid-training phase 2 (agentic capabilities):**
Extends context from 32K to 128K tokens; uses CE loss (teacher logits are unreliable at this distributional shift). Adds tool-use trajectories and agentic demonstrations from a synthetic data pipeline.

**Post-training:** SFT → on-policy distillation → multi-objective RL. On-policy distillation is notable: the student generates rollouts and the teacher provides per-token KL targets on the student's own distribution, resolving train-inference distribution shift and improving downstream RL outcomes.

### Quantization

BF16 weights occupy 55.7 GB. Target: under 20 GB for the LM weights (leaving room for KV cache, draft model, and perception encoder within 24/32 GB VRAM).

The team evaluated 7+ quantization schemes: GPTQ, AWQ, NVFP4/MXFP8 mixed formats, Q4KM, and two calibration-driven dynamic allocation methods (Greedy-KL and EvoPress). The key selection metric is ΔNLL (change in negative log-likelihood vs. BF16).

**EvoPress** uses evolutionary search over the Q4K/Q5K/Q6K space with per-tensor precision assignments. With 418 independently optimized weight tensors (query/key/value/gate exposed separately for each layer), it finds a profile with 51% Q4K, 130% Q5K, 237% Q6K tensors at group size 32, totaling 19.65 GB — the lowest validation loss of any tested method.

**Q4KM** (a modified llama.cpp K-quant scheme with group size 128) provides the smallest memory footprint, selected for 24 GB VRAM deployment.

EvoPress is deployed for 32 GB; Q4KM for 24 GB. Neither shows significant increases in chain-of-thought length vs BF16 (relevant because prior work has shown aggressive quantization can pathologically increase reasoning trace length).

### Speculative Decoding

DFlash speculative decoding is used to improve throughput. The drafter is a lightweight block-diffusion model that predicts a full block of marginal token probabilities in a single forward pass, generating a draft block via independent sampling. Speculative decoding is especially valuable on edge hardware where batch size is 1 and the memory bandwidth bottleneck dominates over compute.

## Comparison to Prior Work

| Model | Size | Agentic benchmarks | Quantization target |
|---|---|---|---|
| Muse Glimmer | 30B | SOTA in size class | 4-bit (EvoPress / Q4KM) |
| Gemma4-31B | 31B | Below Muse Glimmer | Not optimized for on-device |
| Qwen3.6-27B | 27B | Below Muse Glimmer | Not optimized for on-device |

The paper compares against Gemma4-31B and Qwen3.6-27B across tool use, multi-step planning, agentic task completion, multilingual reasoning, and long-context memory. Muse Glimmer leads across these dimensions at high reasoning strength, with comparative results shown for low/medium/high reasoning modes.

The key distinction from general-purpose models of similar size is specialization: the training data, mid-training phases, post-training pipeline, and safety training are all oriented toward agentic reliability rather than generalist benchmark performance.

## Reading the Results

The paper's primary result is the capability scorecard in Table 2, comparing Muse Glimmer (high reasoning) against Gemma4-31B and Qwen3.6-27B. Bold entries indicate best-in-class per benchmark.

The most meaningful results cluster around:
- **Agentic task completion**: Performance on popular agentic scaffolds (TaskWeaver, smolagents, etc.) where the model must plan, call tools, and recover from failures across many steps
- **Tool use / function calling**: Schema-precise tool invocation; failures here compound across agentic steps
- **Long-context memory** (RULER benchmark): Critical for session-length agentic tasks; validates the sliding-window + GQA design

The quantization results (Table 3) are significant: ΔNLL tracks downstream accuracy reasonably well, and neither deployment configuration shows meaningful quality degradation on agentic benchmarks vs BF16. This is non-trivial — prior work has shown that 4-bit quantization can hurt reasoning quality and inflate chain-of-thought length, but the calibrated schemes here avoid this.

## Key Notes of This Paper

### Dense vs. MoE under memory constraint

The core architectural insight is formalized through scaling law experiments. The team fitted power-law curves of bits-per-byte vs. GPU-hours for dense, 1:8 MoE, and 1:16 MoE configurations, all constrained to fit within 32 GB VRAM. The dense model needs only 550 TPP (tokens per parameter) to reach its target bits-per-byte, while 1:8 MoE needs 2,800 TPP and 1:16 MoE needs 6,000 TPP — 5× and 11× more data-compute respectively. Under a fixed compute budget, dense trains to lower loss.

The intuition: MoE's efficiency advantage comes from **activating fewer parameters per token** at inference, which is valuable when the model is larger than what any given token needs. But when the total model size is hard-constrained by VRAM, you can fit a larger dense model than an MoE model with equivalent total parameter count, and the larger dense model wins.

### KL+CE training schedule for distillation

The training objective switches from pure KL divergence (against teacher top-256 logits) to cross-entropy (against ground-truth tokens) at 95% of pre-training. This is a principled correction for a known distillation artifact: when only top-k logits are available from the teacher, the excluded probability mass is reallocated (most implementations put it on pad tokens or redistribute uniformly), introducing a bias. At the very end of training, using CE against ground truth corrects this accumulated bias and improves downstream evaluation.

The formula for the distillation loss during the first 95% of training is:

$$\mathcal{L}_{\text{distill}} = \text{KL}(p_{\text{teacher}}^{(k)} \| p_{\text{student}})$$

where $p_{\text{teacher}}^{(k)}$ is the renormalized distribution over the top-k teacher logits. The KL divergence measures how much information is lost when the student approximates the teacher's token-level distributions.

In the final 5%, this is replaced by:

$$\mathcal{L}_{\text{CE}} = -\sum_t \log p_{\text{student}}(y_t)$$

The transition ensures the student's probability mass over the full vocabulary converges correctly, not just over the top-k supported tokens.

### On-policy distillation resolves distribution shift

Between SFT and RL, on-policy distillation is inserted: the student generates its own trajectories, and the teacher evaluates the student's token distributions on those trajectories, providing KL-based supervision on the student's own output space. This resolves the classic exposure bias problem (training on teacher outputs ≠ generating from own distribution) and consistently improves subsequent RL outcomes. The key formula is:

$$\mathcal{L}_{\text{OPD}} = \mathbb{E}_{x \sim p_{\text{student}}} \left[ \text{KL}(p_{\text{teacher}}(\cdot | x_{<t}) \| p_{\text{student}}(\cdot | x_{<t})) \right]$$

where the expectation is over the student's own generated prefixes $x$, not teacher-generated demonstrations.

## Limitations

**32 GB VRAM is not truly mobile.** A device with 32 GB of dedicated GPU memory is a high-end desktop or professional laptop — not a smartphone. The paper targets consumer hardware, but the practical floor is still enthusiast-class. Running the 24 GB Q4KM variant narrows this somewhat, but mobile SoCs (Apple M4 Pro, Snapdragon X Elite) with unified memory are not directly addressed.

**Multi-image and video are limited.** The perception encoder handles interleaved image inputs well, but the paper explicitly notes the model is not optimized for video. Sequential visual perception across many frames is not a strength.

**Agentic data pipeline requires frontier compute.** The synthetic data generation pipeline uses Muse Spark (a frontier model) as the generator. Reproducing this pipeline without access to a frontier teacher is not feasible — the barrier to building equivalent on-device training infrastructure is high.

**Safety behaviors are trained, not architecturally enforced.** Resistance to prompt injection and privacy awareness are behavioral properties trained via RL penalties, not structural guarantees. Sufficiently adversarial inputs may still elicit unsafe behavior.

## Future Work

The paper suggests several directions:

**Video-native agents.** Extending the perception encoder to handle temporal video sequences natively would enable agents that can observe and reason about screen recordings, camera feeds, and video files — important for computer-use agents.

**Sub-16 GB deployments.** The 24 GB variant is already a step toward broader hardware compatibility. Further architectural and quantization research toward 8–12 GB targets (covering most consumer laptops and current-gen mobile) is the natural next step.

**Memory-efficient long-context.** For very long agentic sessions (>128K tokens), more aggressive KV compression (learned compression, retrieval-augmented memory) remains open. MLA was evaluated and rejected here, but the design space is not exhausted.

**Privacy-preserving on-device RL.** The RL pipeline requires cloud compute for teacher rollouts. Developing RL methods that improve the on-device model using only local feedback (without a cloud teacher) would enable ongoing improvement while preserving privacy guarantees.

**Collaborative edge-cloud architectures.** Papers like PAAC explore using device SLMs for sensitive-span masking before cloud reasoning. Muse Glimmer class models could serve as more capable local agents in such hybrid systems, handling more tasks locally and escalating only when truly necessary.

## Implications for Edge / On-Device Deployment

**The dense-over-MoE finding is directly actionable.** For developers targeting fixed VRAM budgets (the actual constraint for most deployment scenarios), MoE's inference efficiency advantage is negated by the memory constraint. Dense architectures of equivalent memory footprint outperform MoE under a fixed training compute budget. This should inform architecture decisions for any team building models for memory-constrained inference.

**EvoPress quantization is worth adopting.** The evolutionary search approach to mixed-precision assignment consistently outperforms fixed-schedule quantization (GPTQ, AWQ, Q4KM) in terms of quality per byte. The code is available and the methodology is transferable to other models.

**The KV cache is the binding constraint for long sessions.** The design choices — 2 KV heads, sliding window attention, 128-dim head — are all oriented toward bounding KV cache growth. For production agentic systems with multi-hour sessions, KV cache management is as important as model weight size.

**Speculative decoding at batch size 1 is essential.** On consumer hardware, batch size is typically 1. Memory bandwidth is the bottleneck, not compute. Speculative decoding addresses exactly this bottleneck — the draft model (small, cheap) generates candidate tokens, and the main model verifies in parallel. For on-device deployment, speculative decoding should be considered table stakes.

## Links

[Original Paper](https://arxiv.org/abs/2610.24628)
