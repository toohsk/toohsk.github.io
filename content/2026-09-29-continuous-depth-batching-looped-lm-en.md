Title: Depth-Adaptive Inference of Looped Language Models via Continuous Depth Batching
Date: 2026-09-29
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Continuous Depth Batching (CDB) enables the first practically efficient system for depth-adaptive inference in looped small language models like Ouro 1.4B and Huginn 3.5B, achieving up to 99% of the theoretical maximum speedup from adaptive early-exit computation.

## Why This Paper Matters

Looped language models (looped LMs) share a single block of transformer layers across multiple "loop" iterations rather than stacking independent layers. This weight sharing naturally supports **depth-adaptive inference**: if a token is "easy," the model can exit after fewer loops; if it's "hard," it gets more compute. The result is a model that allocates compute proportional to token difficulty rather than applying uniform depth to every token regardless of need.

This is a compelling property for small language models on resource-constrained devices. A 1.4B or 3.5B looped model can in principle execute like a shallower model for routine text while matching a deeper model's quality for complex reasoning — all within the same parameter budget.

The problem is that this adaptive compute property has been essentially theoretical. **Tokens with different loop counts cannot be processed in a single forward pass.** Standard batching systems like vLLM assume all tokens in a batch travel through identical computation paths. A token exiting after 3 loops and one exiting after 7 loops must be handled separately, creating a batching problem that has prevented depth-adaptive inference from delivering real-world speedups.

This paper introduces Continuous Depth Batching (CDB), the first practical solution to this batching problem, and validates it on two looped SLMs: Ouro 1.4B and Huginn 3.5B.

## Core Technical Contribution

### Continuous Depth Batching (CDB)

CDB's key idea is to **form new batches between loop steps** rather than requiring all tokens to share the same loop count throughout a request. Instead of the standard single forward pass, CDB inserts a scheduling decision point at each loop boundary:

1. **Per-step batch formation**: After each loop iteration, the scheduler collects all tokens that have finished their current iteration, regardless of where they started. Tokens that are still looping continue; tokens that should exit are drained from the batch.

2. **Exit prediction**: CDB predicts which tokens will exit after each step, using this to prepare the next batch **asynchronously** — filling the compute pipeline without waiting for the current loop to complete. This hides scheduling overhead behind computation.

3. **Looped KV-caching**: Standard KV cache management assumes fixed layer counts per request. CDB manages a specialized cache that tracks KV states across loop iterations, allowing the model to attend to its own previous-loop computations (important for recurrent-style models like Huginn).

4. **Non-looped layer scheduling**: Real looped LMs have non-looped components — token embedding, the LM head, and sometimes unshared transformer blocks at the boundaries. CDB explicitly schedules when these are executed relative to the loop steps, separating them from the looped core.

### Architecture-Dependency Finding

The paper's experimental evaluation reveals an important architectural constraint: the benefit of CDB **depends heavily on how much of the model is inside the recurrent loop**.

- **Huginn 3.5B** has a small fraction of parameters outside the loop, making it well-suited for depth-adaptive inference with CDB.
- **Ouro 1.4B** (and similar models with large non-looped components) face heavier scheduling overhead because the non-looped layers (embedding, LM head) must be re-executed whenever any token enters or exits the system, even if the looped core is efficiently batched.

The key metric introduced is the fraction of total compute inside the loop. CDB approaches its theoretical maximum when this fraction is high (≥ 95%). When large non-looped layers dominate, the benefit of adaptive looping shrinks because these layers cannot be skipped.

## Comparison to Prior Work

**Standard vLLM / continuous batching** handles variable-length sequences by processing them in a single forward pass with attention masking. It assumes uniform depth and has no mechanism for per-token loop-count variation.

**Adaptive computation in standard transformers** (Universal Transformers, PonderNet, etc.) has been studied but treated as a theoretical property rather than an efficiently deployable one. These approaches generally required custom, non-batched inference.

**Speculative decoding** achieves speedups by predicting future tokens with a draft model, but operates orthogonally — it doesn't exploit per-token compute adaptivity within a single model's layers.

CDB is the first system that:
1. Handles heterogeneous loop counts within a batch in production-style serving
2. Eliminates batching dead time via asynchronous exit prediction
3. Works with real looped SLMs that have complex KV-cache requirements (Huginn's recurrent memory)

## Reading the Results

The paper reports that CDB achieves **up to 99% of the estimated maximum speedup** from depth-adaptive inference on Ouro 1.4B and Huginn 3.5B.

The "estimated maximum speedup" is a theoretical bound: if every token exits optimally early (based on oracle exit decisions), how fast could the model run? Achieving 99% of this bound means the scheduling overhead of CDB is essentially negligible — the speedup is limited by model architecture, not by the system.

Key results:
- **Huginn 3.5B** (fully looped architecture) sees larger speedups than Ouro 1.4B, confirming the architectural dependency finding.
- Exit behavior varies substantially across tokens and prompts — easy tokens consistently exit early while reasoning-intensive tokens use full depth. This validates the premise that adaptive compute is genuinely useful.
- The **looped KV-cache management** adds minimal overhead because the batch formation at each loop step amortizes the cost across many tokens.

The practical implication is clear: for a fully looped SLM like Huginn, CDB can deliver the compute savings of depth-adaptive inference that the model architecture promises, without requiring custom per-token serial execution.

## Key Notes of This Paper

The scheduling problem CDB solves is fundamentally a **producer-consumer batching problem** at the loop level. At each loop boundary:
- Tokens that exit are "consumed" (removed from the active batch, passed to the LM head)
- New tokens that enter a batch are "produced" (passed into the loop)
- The batch for the next loop step is assembled from survivors of the current step plus any newly admitted tokens

The asynchronous exit prediction works as follows: while loop step $k$ is executing, the scheduler predicts which tokens will exit after step $k$, and pre-assembles the batch for step $k+1$. When step $k$ finishes, the scheduler validates the prediction and either uses the pre-assembled batch or falls back to synchronous assembly. In practice, exit behavior is predictable enough that asynchronous pre-assembly is almost always correct, virtually eliminating scheduling stalls.

The **looped KV-cache** is distinct from standard KV cache in that a single token may contribute multiple sets of keys/values — one per loop iteration — that the model should attend to. Managing this without exploding memory requires careful tracking of which loop-iteration KV entries are still needed by in-flight requests.

## Limitations

- CDB's efficiency strongly depends on the fraction of compute inside the loop. Models with large non-looped embedding or LM head layers (common in current designs) see limited speedup because those components create unavoidable per-token overhead.
- Exit decisions must be made at each loop boundary — the exit criteria are model-specific and require training-time integration. Not all models support meaningful early exit.
- The system adds implementation complexity over standard continuous batching, requiring specialized KV-cache management and asynchronous prediction infrastructure.
- The paper evaluates on two specific looped SLMs (Ouro 1.4B, Huginn 3.5B). Generalization to other looped architectures (Universal Transformers, ACT-based models) is asserted but not extensively tested.
- Throughput vs. latency tradeoffs under mixed prompts with very different exit profiles are not deeply analyzed.

## Future Work

The paper suggests that future looped LM architectures should be designed with the CDB constraint in mind: **maximizing the fraction of compute inside the recurrent core** is the key design target for depth-adaptive inference efficiency.

Promising research directions:
- **Trainable exit mechanisms**: Current exit criteria are often threshold-based on a learned exit gate. Training the exit gate specifically to maximize CDB efficiency (by encouraging predictable, batch-compatible exit patterns) could further close the gap to the theoretical maximum.
- **Hybrid looped-non-looped scheduling**: For models with non-trivial non-looped components, batching the non-looped layers across multiple loop boundaries (rather than per-token) could reduce their overhead.
- **CDB for speculative decoding**: Looped LMs are natural small draft models for speculative decoding (their fast early-exit path as the draft, their full depth as the verifier). CDB could enable a unified system where the same model serves as both draft and verifier with dynamic depth allocation.
- **Multi-modal looped models**: The depth-adaptive principle could be especially valuable in vision-language models, where image tokens and text tokens have very different computational requirements.

## Implications for Edge / On-Device Deployment

Depth-adaptive inference through CDB has significant implications for SLM deployment on constrained hardware:

1. **Dynamic compute budget**: A looped SLM can adapt its FLOPs to the difficulty of each inference task. A simple question answers quickly (few loops); a complex reasoning task uses full depth. This is more efficient than having separate small and large models.

2. **Reduced average latency**: If 60–70% of tokens in a typical workload exit early, the average per-token compute drops substantially — directly reducing generation time on mobile/embedded hardware.

3. **Single model for variable quality tiers**: On-device applications that need fast responses for some queries and high quality for others can use a single looped model with dynamic exit thresholds, instead of maintaining multiple model checkpoints.

4. **Battery and thermal management**: Variable compute means variable power draw. A looped model running CDB can modulate its energy consumption based on available thermal headroom — reducing power when the device is hot without fully degrading output quality.

5. **Architecture guidance**: The paper's finding that "fully looped architectures are best suited to depth-adaptive inference" is actionable guidance for SLM designers targeting on-device deployment. New models should minimize non-looped parameter count to maximize the efficiency gain from CDB.

## Links

[Original Paper](https://arxiv.org/abs/2608.09444)
