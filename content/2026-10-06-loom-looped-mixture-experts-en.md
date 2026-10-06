Title: LOOM: Scaling Looped Mixture-of-Experts Beyond Two Loops
Date: 2026-10-06
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: LOOM identifies two failure modes of naively stacking transformer loops — hidden-state variance explosion and expert selection collapse — and fixes both to enable stable scaling to 9-12 loops, turning a 700M-parameter model into an effective 1.7B-FLOP model without increasing parameter count.

## Why This Paper Matters

There are two ways to make a language model more capable: give it more parameters, or give it more computation per token. Most research focuses on the first axis. Looped Transformers explore the second — by repeatedly passing the same token sequence through the same (weight-shared) transformer block multiple times, the model effectively has a deeper computational graph than its parameter count suggests.

This matters enormously for on-device and edge deployment. A 700M parameter model with 5 loops processes each token at roughly the computational depth of a 3.5B parameter non-looped model, but only occupies ~1.4 GB of memory at bf16 (5 loops don't multiply parameter storage). The FLOP cost of inference is higher, but modern AI accelerators — including NPUs on mobile chips — are compute-bound, not memory-bound for small batch sizes. On-device inference often bottlenecks on memory bandwidth when loading weights, not on arithmetic throughput.

The catch: looping naively doesn't work beyond two iterations. Prior work consistently observed that gains saturate quickly and often reverse — the extra compute from loops 3, 4, 5+ bought little improvement or actively degraded the model. The reason was unknown.

LOOM's contribution is diagnosing *why* looping fails and engineering specific fixes, unlocking stable scaling to 9-12 loops. The diagnosis and solution are both genuinely interesting.

## Core Technical Contribution

### The Two Failure Modes

LOOM identifies two distinct pathologies that emerge when you naively apply more than two loops:

**Failure Mode 1: Hidden-state variance explosion (curse of depth)**

Each residual update in a transformer can be written schematically as:
$$h^{(l+1)} = h^{(l)} + f(h^{(l)})$$

where $f$ represents an attention or MLP sublayer. In a looped transformer, this update is applied repeatedly with the same $f$ (weight sharing). If the residual contributions $f(h^{(l)})$ add coherently in the same direction, the hidden-state norm $\|h^{(l)}\|$ grows with each loop iteration. After $L$ loops, variance has accumulated $L$ times. This makes representations "drift" — the internal state after loop 5 looks nothing like after loop 2, even though the same weights are applied.

In a standard (non-looped) transformer, each layer has independent weights, so there's no systematic accumulation — different layers' residual contributions tend to cancel partially. Weight sharing removes this cancellation.

**Failure Mode 2: Expert selection collapse**

In Mixture-of-Experts (MoE) layers, a routing network selects which experts process each token. In a looped MoE, the same router is applied at every loop iteration. Once the router "decides" that a given token should go to expert 3, it makes the same decision at loop 2, loop 3, loop 4, ..., because the token representation (and thus the router input) has been processed by the same experts each time and converges to a pattern that consistently activates those same experts.

Result: extra loops add computation (FLOP cost), but not computational diversity — you're running the same experts multiple times on increasingly similar inputs. This is why perplexity stops improving and can even get worse.

### LOOM's Solutions

LOOM is built on a single unifying principle: **each loop should contribute new computation while keeping the recurrent state stable**.

**Fix 1: Residual scaling to bound variance**

To prevent hidden-state variance from growing with each loop, LOOM scales the residual update at each iteration:

$$h^{(l+1)} = h^{(l)} + \frac{1}{\sqrt{L}} \cdot f(h^{(l)})$$

By scaling each residual contribution by $1/\sqrt{L}$ (where $L$ is the number of loops), the total variance accumulated after $L$ loops is bounded regardless of how many loops are applied. The factor $1/\sqrt{L}$ ensures that $L$ independent Gaussian contributions each of variance $\sigma^2/L$ sum to total variance $\sigma^2$ — constant regardless of $L$.

This is analogous to the $1/\sqrt{d}$ scaling in attention softmax: both control variance to prevent exponential blow-up with depth.

**Fix 2: Input re-injection at every loop**

Even with variance control, hidden states can drift away from the input's information content. LOOM re-injects the original input embedding $e_0$ at the start of every loop:

$$h^{(l+1)}_{input} = h^{(l)}_{output} + e_0$$

This provides a "reset signal" — regardless of how the hidden state has evolved through prior loops, it always receives a fresh injection of the original input information. This stabilizes recurrence by preventing the model from losing track of the input through accumulated computation.

The intuition is similar to why skip connections in ResNets work: preserving direct access to the input prevents degradation in deep networks.

**Fix 3: Per-loop routers for expert diversity**

To fix expert selection collapse, LOOM introduces separate router networks for each loop iteration, while the expert weights themselves remain shared:

- **Shared expert weights**: expert parameters are the same across loops (weight sharing, parameter efficiency)
- **Per-loop routers**: each loop $l$ has its own router $g^{(l)}$, which selects experts independently

With per-loop routers, the same expert can be selected at different loops for different reasons. A token might activate experts {1, 4} at loop 1 (based on early-loop representations) and experts {2, 7} at loop 3 (based on loop-3 representations), giving genuinely different computational paths at each depth.

**Fix 4: Looping Residual**

To leverage the computational history from earlier loops, LOOM adds a "Looping Residual" connection that carries outputs from earlier loop iterations forward:

$$h^{(l+1)} = h^{(l)} + \text{LoopingResidual}(h^{(0)}, h^{(1)}, ..., h^{(l)})$$

This allows later loop iterations to access and build on the results of earlier processing, rather than treating each loop as processing from a fresh state. It's analogous to dense connections in DenseNet but across temporal (loop) depth rather than spatial depth.

## Comparison to Prior Work

Prior looped transformer work:
- **Universal Transformers**: first to explore weight sharing with positional encoding per step — demonstrated feasibility but saturated quickly
- **DEQ (Deep Equilibrium Models)**: solve for a fixed point iteratively — mathematically elegant but computationally expensive for training
- **Prior looped MoE**: settled on 2 loops as practical maximum; gains degraded beyond that

LOOM's quantitative comparisons (700M model, near-iso-FLOP):

| Configuration | Perplexity | Zero-shot accuracy | Loops |
|--------------|-----------|-------------------|-------|
| Non-looped baseline | 18.36 | 38.84% | 1 |
| Naive looping (no LOOM fixes) | ~18.x (degrading) | ~38% (degrading) | 3+ |
| **LOOM** | **16.54** | **39.53%** | **5** |

For the 1.7B model without FLOP matching (60B tokens, maximum loops):

| Configuration | Perplexity | Zero-shot accuracy | Loops |
|--------------|-----------|-------------------|-------|
| Non-looped baseline | 9.62 | 42.4% | 1 |
| **LOOM** | **7.77** | **47.7%** | **9** |

The 1.7B model's perplexity improvement from 9.62 to 7.77 is substantial — roughly equivalent to a 40-50% parameter count increase in a standard non-looped model.

## Reading the Results

**The iso-FLOP comparison is the most informative**: At equal FLOPs (a fair comparison that accounts for the fact that more loops means more compute), the 700M LOOM model with 5 loops reduces perplexity from 18.36 to 16.54 — an 11% improvement. This means that for a fixed computational budget at inference time, looped MoE is better than non-looped MoE at the same parameter count.

**700M 5-loop vs 1.7B 9-loop**: The more dramatic result comes without FLOP matching. A 1.7B parameter model using 9 loops achieves perplexity 7.77 and 47.7% zero-shot accuracy — competitive with what a standard ~3B parameter model would achieve. The parameter savings are significant for on-device storage.

**Scaling to 9-12 loops with stability**: The key technical achievement is that LOOM doesn't just improve at 5 loops but maintains stable performance improvements up to 9-12 loops depending on model size. Prior work couldn't go beyond 2. The ablations show that each LOOM fix contributes independently — removing any one of them causes performance degradation at higher loop counts.

**Zero-shot accuracy gains** (42.4% → 47.7% for the 1.7B model) span a range of tasks. This suggests the improvement is not narrow/overfitted but reflects genuinely improved general language understanding from the deeper effective computation.

## Key Notes of This Paper

### Why Variance Scaling Works: The Random Walk Analogy

Consider the hidden state after $L$ loop iterations of a simplified linear residual:
$$h^{(L)} = h^{(0)} + \sum_{l=0}^{L-1} \Delta_l$$

where $\Delta_l = f(h^{(l)})$ is the residual contribution at loop $l$. If we model $\Delta_l$ as roughly independent (an approximation that works early in training when representations vary across loops), then:
$$\text{Var}(h^{(L)}) \approx \text{Var}(h^{(0)}) + L \cdot \text{Var}(\Delta)$$

Variance grows linearly with $L$. After 9 loops, the hidden-state variance is roughly 9× larger than after 1 loop. This means the effective "scale" of information in the hidden state drifts upward, eventually overwhelming the normalization layers and causing representations to behave erratically.

With $1/\sqrt{L}$ scaling:
$$h^{(L)} = h^{(0)} + \frac{1}{\sqrt{L}}\sum_{l=0}^{L-1} \Delta_l$$
$$\text{Var}(h^{(L)}) \approx \text{Var}(h^{(0)}) + L \cdot \frac{1}{L}\text{Var}(\Delta) = \text{Var}(h^{(0)}) + \text{Var}(\Delta)$$

Now variance growth is bounded: after 9 loops, the variance is at most $\text{Var}(h^{(0)}) + \text{Var}(\Delta)$ regardless of $L$. The hidden state remains on the same "scale" as the original representation.

### Expert Selection Collapse: Why Shared Routers Fail

Consider a token $t$ with representation $h_t^{(1)}$ after loop 1. The shared router computes:
$$\text{route}^{(1)}(t) = \arg\max_e \langle g_\theta(h_t^{(1)}), e_k \rangle$$

Suppose expert $e_3$ is selected. Now $h_t^{(2)} = \text{Expert}_3(h_t^{(1)})$. The next routing decision is:
$$\text{route}^{(2)}(t) = \arg\max_e \langle g_\theta(h_t^{(2)}), e_k \rangle$$

Since $h_t^{(2)}$ is the output of $\text{Expert}_3$ applied to $h_t^{(1)}$, and since $\text{Expert}_3$ is trained to produce representations that route well, the router tends to select expert $e_3$ again (or the same small subset). This is self-reinforcing: the representation evolves in a direction that the router has learned to associate with that expert cluster.

With per-loop routers $g_\theta^{(l)}$, each loop has learned a different routing function from the same representation space. The diversity of routing decisions comes from learning different "notions of relevance" at different processing depths, analogous to how early layers of a standard deep network detect different features than late layers.

### Input Re-injection as an Anchor

The input re-injection fixes a subtle but important failure mode: catastrophic forgetting *within a forward pass*. After many loops, the hidden state $h^{(L)}$ can become dominated by the accumulated residual updates and "forget" the original input tokens. The model then processes its own previous computation rather than the input.

This is the recurrent analog of the vanishing input problem in RNNs: the hidden state diverges from the input sequence as the number of recurrence steps grows.

Re-injecting $e_0$ at each loop acts as a "hard reset" of the input channel — no matter how many loops have run, the model always has access to the original token embeddings. This is especially important for tasks requiring precise character-level or token-level processing (like arithmetic) where small differences in the input encoding matter.

## Limitations

1. **Full paper not in open HTML/Markdown**: The paper is available only as PDF, so precise formulation details for some components (exact residual scaling implementation, Looping Residual architecture specifics) require reading the PDF directly.

2. **Increased inference FLOPs**: Each additional loop multiplies inference compute by $1 + 1/L_{base}$ approximately. A 9-loop model requires 9× the FLOPs of the non-looped equivalent. This is beneficial on memory-bandwidth-limited hardware but increases latency on compute-limited hardware.

3. **Training cost**: Looped architectures require backpropagating through all $L$ loop iterations, which multiplies the training memory requirement by roughly $L$ (for activation checkpointing) or $L^2$ (without). Training a 9-loop 1.7B model is more expensive than training a 1.7B standard model.

4. **Limited scale validation**: Experiments go up to 1.7B parameters. Whether LOOM's techniques generalize to 7B+ parameter MoE models (common scales for production SLMs) is not yet demonstrated.

5. **Architectural constraint**: LOOM requires Mixture-of-Experts. The per-loop router mechanism is specific to MoE layers; applying it to dense transformers would require a different diversity mechanism.

## Future Work

**From the authors:**
- LOOM is available at [github.com/hed-ucas/LOOM](https://github.com/hed-ucas/LOOM) — scaling experiments to larger models are likely upcoming.
- Exploration of optimal loop allocation across transformer layers.

**Additional promising directions:**

- **Adaptive looping**: Rather than a fixed number of loops, allow the model to decide how many loops each token sequence needs (like Adaptive Computation Time). Easy inputs exit early; hard inputs get more loops. This could give large quality gains with minimal average latency increase.

- **Mixture of loop counts**: Train a model that can operate at variable loop counts — useful for edge deployment where the power budget varies. Fewer loops on battery-saver mode, more loops when plugged in.

- **LOOM combined with distillation**: Use LOOM's deeper effective computation as a student architecture in knowledge distillation (like Latent-MOPD). The student uses fewer parameters than the teacher but achieves comparable depth via looping.

- **Attention-efficient variants**: LOOM's per-loop routers work at the MoE level. Combining with linear attention or state-space models (which have constant-memory recurrence) could make the full looping pipeline memory-efficient for long sequences.

- **Specialization of loops**: Train per-loop routers with an explicit diversity regularization that forces loops to specialize in different subtasks (e.g., loop 1 = entity recognition, loop 2 = logical inference). This could make the computational allocation more interpretable.

## Implications for Edge / On-Device Deployment

**Memory-efficient deep computation**: The core value proposition for edge deployment is that a 700M parameter LOOM model provides the computational depth of a model several times larger, while fitting in ~1.4 GB (bf16). Modern mobile NPUs (Apple Neural Engine, Qualcomm Hexagon, MediaTek APU) can saturate their FLOPs capacity with small models — the bottleneck is often loading the weights from DRAM. LOOM trades weight memory for compute cycles: exactly the right trade-off for these accelerators.

**Power envelope flexibility**: If the per-loop design allows early exit (a natural extension), the same model can run at different quality/latency/power tradeoffs. A voice assistant on low battery could use 3 loops; a complex reasoning task could use 9.

**Scale mismatch addressed**: On-device models are typically 100M–1.7B parameters. Standard 1.7B models struggle on complex multi-step reasoning. LOOM's 1.7B-parameter, 9-loop model achieving 47.7% zero-shot accuracy represents a meaningful step toward reasoning-capable on-device SLMs.

**Quantization compatibility**: Looped models have fewer unique weight tensors than standard deep models (by design). This may make them more amenable to quantization: a single expert's weights are quantized once, but used many times across loops, so quantization error is amortized differently. Whether LOOM models are more or less quantization-friendly than equivalent-FLOP standard models is an open and practically important question.

## Links

[Original Paper](https://arxiv.org/abs/2610.01153)
