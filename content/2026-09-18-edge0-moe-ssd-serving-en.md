Title: Edge0: Serving a 35B MoE from SSD with Trained Routing Prediction on Consumer Hardware
Date: 2026-09-18
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Edge0 is a streaming MoE inference engine that runs a 35B model at ~20 tok/s on a 24GB Mac mini M4 Pro using only 3GiB of active memory, by combining SSD-offloaded expert weights, a trained prerouter that predicts expert routing one token ahead to overlap disk reads with compute, and an unmerged recovery LoRA to compensate for int4 quantization and routing approximation losses.

## Why This Paper Matters

Running a frontier-scale language model locally has been, until now, a problem with no satisfying answer. A 35B Mixture-of-Experts (MoE) model at 4-bit quantization weighs 19.5 GB — on a 24 GB consumer machine, that leaves almost nothing for the operating system, KV cache, or anything else the user is running. The standard responses — quantize further or put it in a data center — both fail: going below 4 bits erodes quality unrecoverably, and "put it in the cloud" defeats the purpose of local deployment (privacy, latency, offline use, cost).

MoE architectures make the problem intriguing. At inference time, a 35B MoE only *activates* about 3B parameters per token — the experts for that token are the only ones that need to be computed. But those 19.5 GB of weight still need to be *stored* somewhere. And that's the insight this paper builds on: **the weights only need to be available when they're needed, not all at once**.

The genuine difficulty is a data dependency: to know which experts to load from SSD for layer N+1, you need the output of layer N — which hasn't been computed yet. Naive SSD offloading stalls for one disk read per layer per token. On an Apple M4 Pro with NVMe, that stall is real.

Edge0 breaks this dependency with a trained predictor and shows the whole system — SSD streaming, prediction, quantization recovery — can work together at practical quality and speed on a $600 Mac mini.

## Core Technical Contribution

Edge0's design has three interlocking pieces:

### 1. SSD Streaming Expert Layers

Expert weights live on disk as int4 per-layer stacked safetensors. A bounded **streaming pool** (backed by the OS page cache with LRU eviction) holds the currently-needed expert weight tensors in memory. The key design is a **staged double-buffer**: at any decode step, fixed slots hold the predicted experts for the current token's forward pass, while the next token's predicted experts are being pre-fetched from disk in parallel.

Four executor paths handle different scenarios — `exact` (baseline), `staged` (decode workhorse), `hot` (LRU-resident hot experts), and `whole-layer` (prefill) — all computing the same math so they can switch per-layer without changing outputs.

An **incremental stack** mechanism (`incr_stack`) means only changed experts need to be swapped between steps, avoiding a full 9-tensor rebuild per layer per step.

### 2. The Prerouter: Prediction Is the Routing

The critical innovation is treating the predictor's output **as the actual routing decision**, not just a hint. Because the staged expert set equals the routed set by definition, there's nothing to drop — the coverage-vs-quality tradeoff that other pre-gating schemes handle at runtime is eliminated.

**Architecture**: Each prerouter head is a small MLP per layer:
```
input → [fc1 → erf-gelu → fc2] + linear_residual → routing_logits
```

The input feature vector concatenates:
- The hidden state from this layer (dim = d_model)
- Two one-hot vectors of size E each: the current token's actual routing and the previous token's routing

So the input dimensionality is `d_model + 2E`. For the 35B tier: `2048 + 2×256 = 2560`, with a hidden width of 512. For the 8B tier: `1536 + 2×128 = 1792`.

The linear residual path is warm-started from the **next layer's router weight**, so the head begins training from "just apply the next router to this hidden state" and the MLP learns the correction on top.

**The timing trick**: The head owned by layer N runs at token *t* on layer N's post-attention norm output and predicts layer N+1's routing at token *t+1*. This one-token lead means the SSD reads for the next token's experts can be issued while the current token is being computed — overlapping disk latency with GPU compute.

**Scale**: 33 prerouter heads in the 35B tier (covering layers 6–38), 16 in the 8B tier (layers 7–22). The computation is a single flush per decode step predicting all layers at once.

### 3. Recovery LoRA: Unmerged by Design

The served model is: `y = W_int4(x) + (α/r) · B·A·x`

The LoRA adapter compensates for two quality-degrading operations simultaneously:
1. **int4 quantization** of the base weights (GPTQ/AWQ-style, group-64)
2. **Routing replacement** — using prerouter logits instead of the real router

The key finding that makes this work is about what happens if you try to *merge* the LoRA into the int4 base and re-quantize: **most of the effect is destroyed**. LoRA deltas (RMS ~10⁻³) sit below the 4-bit group quantization step size, so requantization erases them. Specifically:
- On a dense projection: only **2%** of the weight-level delta survives
- On an attention projection: only **34%** survives
- At the logits level: only **18%** survives

Serving the LoRA **unmerged** as a parallel delta costs only 42 MB of adapter weights and has no measurable effect on decode speed. The int4 base is read-only; retraining a tier means swapping two files.

### Training Recipe

Three phases, all on the dequantized bf16 reconstruction of the int4 checkpoint (training on what is served), with the base frozen throughout:

1. **Phase 1 — Distill the heads**: Train only the prerouter heads to predict next-layer routing (loss = imitate the next layer's true router). Heads first because SFT signals later would drown the tiny head gradients.

2. **Phase 2 — SFT on the student path**: Attach LoRA to attention, linear-attention, and shared-expert projections (not routed experts — those stream and must stay replaceable). Train with prerouter routing active over ~2M rows of teacher-generated text.

3. **Phase 3 — On-policy distillation**: The Phase 2 checkpoint generates its own tokens; the fp16 base teacher scores them. Loss is **reverse KL** (mode-seeking) applied to the teacher's top-k tokens, with a tail term for the remaining mass. Converges in ~10% of Phase 2's data (~200K rows).

## Comparison to Prior Work

| System | Memory | Decode (tok/s) | Notes |
|--------|--------|----------------|-------|
| Naive resident mlx-lm (same 35B model) | 18.2 GiB | 3.9 | Whole model in RAM |
| Edge0 K=4 (35B) | 2.9 GiB | 20.4 | SSD streaming, prerouter |
| On-demand streaming (no prerouter, K=4) | ~1.7 GiB | ~3.5 | Just SSD, no prefetch |

The prerouter speedup is +80–84% over on-demand streaming across K=2, 4, 8 configurations.

**Related work this improves upon**:
- **llama.cpp `--cpu-moe`**: moves experts to CPU/system RAM, doesn't address SSD offloading or prefetching
- **PowerInfer**: splits neurons into hot/cold across GPU+CPU — weights still need to fit in RAM+VRAM combined
- **Mixtral-offloading / MoE-Infinity**: expert caching by popularity/reuse — still requires the full model footprint in system memory; no trained predictor
- **FlexGen**: extends to SSD for weights+KV together, but no routing prediction to hide disk latency
- **Pre-gated MoE**: selects next block's experts within the same token, but requires per-layer synchronization (30–100ms per step overhead) that costs more than it saves in a streaming engine; Edge0 leads by a full token instead

## Reading the Results

**Throughput**: 20.4 tok/s for a 35B model on a Mac mini M4 Pro 24GB is practically fast — enough for real interactive use. Compare this to the naive resident approach at 3.9 tok/s, which is *also* nearly unusable because the 18.2 GiB weight residency leaves almost nothing for the KV cache.

**Memory**: The 2.9 GiB active memory figure means Edge0 leaves ~21 GB of the 24GB machine for the KV cache, OS, and other processes. This is a qualitative difference from the 18.2 GiB resident approach, which makes the machine nearly unusable for other tasks.

**Quality gap**: 3.9 points mean benchmark gap (35B) against fp16 is small but not zero. The distribution matters — AIME (math reasoning) is -6.1 points for 35B and -10.0 for 8B. Factual recall and general reasoning are mostly within 5 points. Long-chain reasoning is where the approximations hurt most.

**Prerouter mechanics**: At K=4, the prerouter reduces blocked time waiting on expert loads from 244.0 ms to 101.9 ms per step — a 2.4× reduction in blocking. The gain is "conservation, not prediction quality": both arms read nearly the same bytes, but the prerouter reads them earlier, in fewer and larger (colder) loads that are no longer on the critical path.

## Key Notes of This Paper

### The "Prediction Is the Routing" Formalism

The central insight is elegantly simple. A conventional staged-decode system would:
1. Compute layer N's output
2. Run the router to get expert indices for layer N+1
3. Fetch those experts from SSD
4. Compute layer N+1

Steps 3 and 4 are sequential — you can't start 3 until you have 2. Edge0 replaces step 2 with a prediction made one token earlier:

```
Token t-1: layer N runs → prerouter head predicts layer N+1's routing for token t
Token t:   layer N+1 fetches those experts (during layer N's compute) → uses them
```

The prediction is consumed as the actual routing. The coverage error (predicting the wrong expert) is paid once during training — the recovery LoRA and SFT on the student path handle it — rather than at runtime through fallback loads or dropped tokens.

### Why Unmerged LoRA Works Here

The arithmetic is clear: int4 group-64 quantization has a step size of roughly `2 * max_val / 16 per group`. LoRA delta magnitudes (RMS ~10⁻³ for fine-tuning) are well below this step for dense layers. Merging and re-quantizing maps most delta values to zero. The parallel delta formula:

```
y = W_int4(x) + (α/r) · B·A·x
```

computes both terms in bf16 and adds them before the output leaves the layer — the LoRA correction is applied in floating point, never quantized. The cost is 42 MB of extra data (two low-rank matrices) and a second matmul — negligible at decode throughput (batch size 1).

### Incremental Stacking

Each MoE layer's experts are represented as stacked tensors (gate/up/down weight matrices, each with scales and biases = 9 tensors). Rebuilding all 9 stacked tensors from scratch each step costs 44 ms for a 40-layer model — comparable to the compute itself. The `incr_stack` mechanism tracks which slots changed between steps and rewrites only those tensors in place. Since adjacent tokens share ~25% of expert assignments, ~75% of slots are the same and don't need rebuilding.

### Prerouter Head Input: Why Two One-Hots?

The input feature appends two one-hot vectors of size E (number of experts) to the hidden state:
- **Current token's routing**: which experts this layer actually routed to
- **Previous token's routing**: which experts the previous token used

This gives the head two types of information: the current "topic" (via current routing) and routing continuity (many adjacent tokens use similar experts). The two-hot augmentation is cheap (2E dimensions vs d_model >> 2E) but gives the MLP direct access to routing history without requiring it to extract this from the hidden state.

## Limitations

- **No concurrent requests**: Edge0 serves one request at a time, FIFO-serialized. Batching would change the expert working set in ways not modeled here.
- **Graph-building floor**: 44 ms per step of CPU-side graph building is a floor that doesn't improve with faster storage — it's the streaming engine's overhead, not a disk bottleneck.
- **Expert reuse is low**: Adjacent tokens agree on only ~25% of a layer's experts. This means ~75% of prefetched experts are loaded once and paid for twice (in unreclaimable residency). The mechanism wins because disk latency is high, not because prefetching is efficient.
- **Quality loss in reasoning**: Long-chain math reasoning (AIME) suffers the most — -6.1 pts (35B) and -10.0 pts (8B). The routing approximation introduces errors that compound across the many reasoning steps.
- **Prerouter gains diminish on fast storage**: The speedup is proportional to the cold-read time that's moved off the critical path. On fast NVMe or with a hot page cache, the gain shrinks.
- **Feature drift unaddressed**: Prerouter heads are trained on the base router's selections but at inference consume their own predictions. This distributional shift is compensated by the recovery LoRA but not directly corrected.

## Future Work

**Authors suggest**:
- Moving tensor assembly off the main thread (currently 62.5 ms of synchronous work at K=8)
- Eliminating the duplicate bundle+stacked-tensor view in memory (~0.45 GiB savings at K=8)
- Concurrency and batching support at the serving layer

**Additional promising directions**:
- **Adaptive K selection**: The prerouter head outputs a full distribution over experts — this could enable dynamic top-k selection (use fewer experts for "easy" tokens, more for "hard" ones) to trade quality for speed dynamically
- **Prerouter distillation without base router**: Could the prerouter heads be trained without access to the original fp16 router? This would make Edge0 applicable to models where the original router isn't accessible
- **SSD-aware quantization**: Current int4 quantization is designed for DRAM serving; a quantization scheme that co-designs the group structure with SSD page boundaries could improve read efficiency
- **Cross-layer expert correlation**: The current prerouter uses one-layer lookahead; predicting N+2 or N+3 layers ahead with a deeper predictor could hide more disk latency on slow storage
- **Integration with flash attention**: For long contexts, KV cache can also overflow to SSD; combining Edge0's expert streaming with KV offloading (e.g., Fathom-style sparse KV reads) could handle both memory bottlenecks together

## Implications for Edge / On-Device Deployment

This paper shifts the practical definition of "local AI" in a meaningful way:

**What changes now**: A 35B-class model — which was previously data-center-only — can run at interactive speeds on a $600 consumer machine, with enough headroom for the OS and KV cache. The machine doesn't need to be dedicated to AI inference.

**The memory footprint story**: 3 GiB of active GPU/unified memory to serve a 35B model is the key number. A modern iPhone 16 Pro has 8 GB RAM; future versions will have more. The active memory budget Edge0 establishes for 35B inference is already compatible with mid-tier consumer hardware in 2–3 years.

**SSD as the weight tier**: This paper establishes SSD as a viable weight storage tier — not RAM, not SSD-as-RAM, but SSD with trained prefetching. The NVMe bandwidth on modern phones (similar to MacBook NVMe) makes this increasingly viable on mobile.

**The recovery LoRA paradigm**: The unmerged LoRA approach could generalize: any heavily quantized model degraded by quantization + approximation could potentially be recovered with a small adapter served in parallel. This decouples the base model (compressed for size) from quality (LoRA for accuracy) in a way that's easy to update.

**Practical tradeoffs**: The single-user, no-batching limitation makes Edge0 a personal inference engine rather than a multi-user server. This is appropriate for on-device: a phone or laptop running inference for one user at a time is exactly the use case.

## Links

[Original Paper](https://arxiv.org/abs/2609.18063)
