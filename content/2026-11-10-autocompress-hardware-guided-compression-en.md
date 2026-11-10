Title: AutoCompress: Hardware-Guided LLM Compression That Optimizes for Latency, Not Parameter Count
Date: 2026-11-10
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: AutoCompress uses differentiable architecture search with serving-engine cost models to produce compressed LLMs that directly meet latency, throughput, and memory targets — outperforming Minitron by up to 6.57 points on 8 benchmarks while using 2–3.3x fewer search tokens.

## Why This Paper Matters

The fundamental flaw in existing LLM compression workflows: **parameter count is a poor proxy for deployment cost.**

Two models with identical parameter counts (3.61B total, 1.26B active) can have 35% different latency on an RTX 5070. Two models meeting the same H100 throughput target can differ by 2.2x in parameter count. Practitioners manually translate deployment constraints into parameter budgets — a trial-and-error process that scales poorly and misses hardware-specific optima.

AutoCompress closes this loop: deployment constraints (latency limit, throughput requirement, memory budget) become direct objectives in the architecture search, not post-hoc filters.

## Core Technical Contribution

### Budget-Aware Differentiable Architecture Search

Differentiable search over global and per-layer architecture dimensions using Gumbel-softmax:

$$y_c = \frac{\exp((\kappa_t z_c + g_c)/\tau_t)}{\sum_j \exp((\kappa_t z_j + g_j)/\tau_t)}$$

$z_c$ = learned logit for choice $c$, $g_c$ = Gumbel noise, $\kappa_t$ = logit scale (increases over training), $\tau_t$ = temperature (decreases over training). Increasing $\kappa_t$ and decreasing $\tau_t$ progressively sharpens the soft distribution toward discrete choices — an annealing schedule.

**Joint objective**:
$$\mathcal{L} = \mathcal{L}_{\mathrm{KD}} + \alpha \sum_r \left[\max(0, \rho_r - 1) + \max(0, 1-\delta-\rho_r)\right]$$

$\mathcal{L}_{\mathrm{KD}}$ = forward KL divergence from parent's token distribution (logit distillation). For each constrained resource $r$, $\rho_r = \hat{\text{cost}}_r / \text{budget}_r$. The penalty penalizes both over-budget ($\rho_r > 1$) and under-budget ($\rho_r < 1-\delta$) configurations, with $\delta = 0.02$ creating a zero-penalty band between 98–100% of each cap.

### Heterogeneous Search Space

Unlike Minitron (uniform dimensions across layers), AutoCompress searches both:
- **Global dimensions**: hidden size, Mamba head count (shared across layers)
- **Per-layer dimensions**: expert counts, routed and shared FFN widths, layer removal

Layer removal is handled naturally within the search space: setting FFN width to zero removes the layer. This enables the optimizer to discover architectures where some layers are aggressively pruned while others are preserved at high capacity — matching the actual importance profile of different layers.

### Serving-Engine Cost Models

**Latency model** (additive across layers):
$$\hat{T}(a \mid \mathcal{D}) = T_0(d \mid \mathcal{D}) + \sum_{\ell=1}^L t_\ell(a_\ell; d \mid \mathcal{D})$$

Layer-wise profiling via Nsight Systems: a single run with ~10 different layer configurations in one model populates multiple timing table entries. Profiling is done inside the target serving engine — capturing deployment-specific kernels and execution behavior.

**Memory model**:
$$\hat{M}(a \mid \mathcal{D}) = W + C + P + R + \epsilon_\mathcal{D}$$

$W$ = weight memory, $C$ = KV/Mamba caches, $P$ = prefill workspace, $R$ = runtime buffers. $\epsilon_\mathcal{D}$ = empirically measured residual for CUDA context, graph allocations, etc.

### Four-Stage Pipeline

1. **Cost model construction**: Profile layer configurations in the target serving environment
2. **Supernet initialization**: Sort parent components by activation-based importance; permute weights so highest-ranked come first (enabling shared-weight subsets)
3. **Joint search**: Train architecture router + supernet with distillation + resource penalties
4. **Extraction and recovery**: Select highest-logit architecture; initialize from parent weights; recover quality via logit distillation

## Comparison to Prior Work

Three deployment scenarios against deployment-adapted Minitron:

| Scenario | Hardware | Improvement over Minitron |
|---------|---------|--------------------------|
| Local long-context coding | RTX 5070 | +6.57 points (8 benchmarks avg) |
| Batched datacenter serving | RTX 5080 | +2.35 points |
| Latency-sensitive game NPC | H100 | +0.57 points |

AutoCompress also uses 2–3.3x fewer search tokens.

**Scalability case study**: Compressed Nemotron 3.5 Super from 120B → 74B parameters, enabling BF16 serving at 128K context, batch size 16, within 178GB on a single B200. Only 2.85-point decrease in mean benchmark score.

## Reading the Results

**6.57 points on RTX 5070**: This is a substantial gap. The RTX 5070 scenario (local long-context coding) is where heterogeneous architecture search helps most — consumer GPUs have different memory bandwidth, kernel behavior, and cache dynamics than data center GPUs, and Minitron's uniform search misses hardware-specific optima.

**0.57 points on H100**: The gap is smaller because H100s are extremely efficient and regular, leaving less room for architecture-specific optimization. Even so, AutoCompress matches or exceeds the baseline at the same constraints.

**35% latency gap at equal parameters (Figure 1)**: The paper's motivating example. A narrow, deep architecture has 35% higher latency than a wider, shallower one at identical parameter counts on RTX 5070. This directly demonstrates why parameter budgets are inadequate specifications.

## Key Notes of This Paper

### Soft Architecture Masks and Gradient Flow

Each router gate produces a weighted mask over importance-sorted components. For a choice between widths 64 and 128:
- First 64 channels: mask weight 1 (always retained)
- Remaining channels: mask weight $y_{128}$ (between 0 and 1 during search)

Gradients flow through $y_{128}$ to update the router logit $z_{128}$. Meanwhile, the supernet weights adapt to both architectures simultaneously (since each batch is split across independently sampled configurations).

The Gumbel noise provides stochasticity — it prevents the optimizer from collapsing too quickly to a suboptimal configuration, analogous to simulated annealing. The annealing schedule (increasing $\kappa_t$, decreasing $\tau_t$) transitions from exploration to exploitation.

### Why Heterogeneous Architecture Matters

Different layers in a transformer have different roles and importances:
- Early layers: basic token processing, relatively uniform importance
- Middle layers: complex reasoning, often high importance
- Later layers: output projection, importance varies by task

Forcing uniform dimension across layers (as in Minitron) requires choosing a single compression rate that works everywhere, which either over-compresses important layers or under-compresses unimportant ones. Per-layer search can allocate capacity where it matters.

### Cost Model Accuracy Trade-offs

The additive latency model has known limitations: architectural changes affect kernel selection and cache behavior, breaking the additivity assumption. The paper validates estimates on held-out complete architectures and shows the accuracy is sufficient for guiding search, even if not for precise latency prediction.

The serving-engine-specific profiling is crucial: the same model on vLLM vs. TensorRT-LLM can have different optimal architectures, because kernel implementations differ.

## Limitations

- **Serving-engine specificity**: Cost models are specific to the target serving engine. New engines require re-profiling. Switching deployment stacks may invalidate the compression
- **GPU access required for profiling**: Can't profile without target hardware. Cloud deployment teams without on-premise hardware face a chicken-and-egg problem
- **Mamba + MoE architecture assumption**: The search space is defined for Nemotron's hybrid Mamba/attention/MoE architecture. Standard dense transformers require different search space definitions
- **Supernet training cost**: While lower than full training, the search is not free (~1–2x standard fine-tuning cost based on description of "2–3.3x fewer search tokens" vs. Minitron)

## Future Work

**Authors' direction**: Extending to more deployment scenarios, improving cost model accuracy.

**Additional promising directions**:
1. **Mobile SoC cost models**: CoreML, QNN, TFLite, ONNX Runtime — each has different kernel behavior. Building serving-engine cost models for Apple Neural Engine and Qualcomm NPU would enable AutoCompress to optimize directly for smartphone deployment
2. **Energy efficiency as a constraint**: Battery-powered devices care more about energy per token than throughput. Adding energy consumption to the objective function (measurable via Intel RAPL or hardware power meters) would create edge-deployment-aware compression
3. **Post-deployment adaptation**: Use telemetry from deployed models (actual latency distributions, batch sizes) to continuously refine cost models and re-trigger compression — a closed-loop compression system
4. **Cross-engine portability**: Learn a hardware-agnostic latency representation that generalizes across GPU generations, reducing per-hardware profiling cost

## Implications for Edge / On-Device Deployment

**Game NPC as the edge proxy**: The "in-game character" deployment scenario (strict latency budget, shared GPU with rendering) is functionally identical to smartphone AI deployment (latency target, shared memory with OS and apps). AutoCompress already demonstrates this scenario works.

**Parameter count failure at mobile scale**: Mobile SoCs (Apple A18, Qualcomm Snapdragon 8 Elite) have NPUs with specific compute patterns, shared memory architectures, and cache behaviors that differ significantly from server GPUs. Parameter-count-based compression budgets are even less predictive on mobile than on data center GPUs.

**Heterogeneous architecture for mobile**: Mobile inference engines often have different efficiency profiles for different layer types — attention vs. FFN vs. normalization. Allowing per-layer size variation could enable significant efficiency gains on mobile that uniform architectures miss.

**The pipeline for SLM deployment**:
1. Profile target mobile SoC in target framework (CoreML, QNN)
2. Run AutoCompress with mobile serving constraints
3. Extract and distill the hardware-optimal SLM
4. Result: a model that is explicitly designed to meet mobile latency/memory targets, not one that happens to be small enough to fit

## Links

[Original Paper](https://arxiv.org/abs/2611.08290)
