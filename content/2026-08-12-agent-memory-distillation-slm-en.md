Title: Agent Memory Distillation: Bridging the Gap Between Large Teachers and Small LLM Agents
Date: 2026-08-12
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: AMD is a training-free framework that transfers hierarchical memory distilled from a large teacher agent (GPT-5-mini) to 4B–8B student models, achieving +27.2%p on AppWorld and enabling students to surpass their teacher—without a single gradient update.

## Why This Paper Matters

Tool-augmented agents — systems that call APIs, execute code, and orchestrate multi-app workflows — represent one of the most practically valuable frontiers in applied AI. But deploying capable agents on resource-constrained hardware requires small language models (SLMs). The central problem: **SLMs are bad at generating successful trajectories**, which means they have little material to learn from.

Conventional memory-augmented agent systems (ExpeL, AWM, MemP) work by extracting insights from a model's own past successes and feeding them back at inference. This works well for large models, which succeed often. For a 4B model tackling multi-app tasks with 20+ interaction turns, zero-shot success rates are so low that the memory bank is dominated by failure traces. Reflection on failure has limits; without a bank of successful examples, structural patterns never emerge.

The naive solution — just hand the small model its teacher's memory — fails for a subtle reason: **the capability gap makes teacher knowledge illegible**. A teacher-generated workflow that says "authenticate first, then open the playlist" assumes the student can independently figure out the authentication subtask. It usually cannot. This is the agentic analog of the classic teacher-student gap problem in knowledge distillation.

AMD (Agent Memory Distillation) is the first systematic treatment of this problem. It proposes hierarchical memory construction that bridges the gap between a capable teacher's experience and a small student's comprehension ceiling.

## Core Technical Contribution

### The Three-Level Memory Hierarchy

AMD constructs three complementary memory types from the teacher's successful trajectories, each targeting a different level of task granularity.

**Workflow Memory** — task-level abstraction. For each successful teacher trajectory $\tau_i^T$, the teacher model generates a verbalized insight covering: the apps/tools involved, key preconditions, decision rules, validation cues, and common failure patterns to avoid. Critically, all runtime-specific values (IDs, emails, file paths) are replaced with typed placeholders (`<ID>`, `<EMAIL>`, `<FILE_PATH>`), making the insight applicable to future tasks with different concrete inputs. Each Workflow entry is stored as $m_i^{wf} = (q_i, \text{ins}_i)$ — a natural-language query paired with the insight — encoded as a dense vector.

**Subtask Memory** — intermediate behavioral examples. Each teacher trajectory is decomposed into semantically coherent segments (e.g., "authenticate with service", "execute sequence of related API calls"). The segmentation is performed by the teacher LLM, optionally guided by rule-based heuristics. Each segment $e_{i,k}$ stores the concrete tool calls and observations from the teacher, labeled with a short description $d_{i,k}$. The entry $m_{i,k}^{st} = (\ell_{i,k}, d_{i,k}, e_{i,k})$ shows the student exactly *how* to execute a particular class of subtask, not just *what* to do.

**Function Memory** — per-function calling conventions. Individual tool invocations are extracted from successful trajectories. Each record $m_{i,j}^{fn} = (f_{i,j}, E_{i,j}, \text{doc}(f_{i,j}))$ stores the function name, a concrete execution example (with calling context, arguments, and observations), and the API documentation. Unlike Workflow and Subtask memories — retrieved by semantic similarity — Function memory is indexed by function name and retrieved reactively only when that function fails.

### Two Injection Modes

**Proactive Injection** (Workflow + Subtask): At task start, before the student issues any tool call:
1. The task instruction queries $\mathcal{M}^{wf}$ by cosine similarity; top-1 insight is prepended to the system prompt.
2. The student decomposes the task into up to 6 subtask labels; each label queries $\mathcal{M}^{st}$; top-1 matching segment per subtask (with deduplication) is appended to the system prompt.

**Reactive Injection** (Function): When a tool call returns an error, the failing function name looks up $\mathcal{M}^{fn}$. Matching records are ranked by cosine similarity between the task instruction and each record's stored context, and the top examples are appended to the error message. This avoids context bloat during successful execution.

### Formal Objective

AMD maximizes the student's expected task success without gradient updates:

$$\max_{\mathcal{M}} \; \mathbb{E}_{s \sim \mathcal{S}} \left[ R\!\left(\pi^S(s; \mathcal{M})\right) \right]$$

where $\mathcal{M} = \mathcal{M}^{wf} \cup \mathcal{M}^{st} \cup \mathcal{M}^{fn}$ and $R$ is the task success reward. The memory bank is built once from teacher trajectories $\mathcal{D}^T_+$ (successful runs only) and reused indefinitely.

## Comparison to Prior Work

| Method | Memory Granularity | Training-Free | Cross-Model Transfer |
|---|---|---|---|
| ReasoningBank | Task-level (flat) | ✓ | Limited |
| MemP | Hierarchical (same model) | ✓ | ✗ |
| SASM | Subtask-level | ✓ | Limited |
| **AMD** | **3-level (WF + ST + FN)** | **✓** | **✓** |

The key differentiator: **AMD explicitly designs for the capability gap**. Prior methods assume memory generated by model X is legible to model X. AMD assumes memory generated by a large teacher may be illegible to a small student, and structures the memory hierarchy to overcome that.

Baselines on AppWorld:
- ReasoningBank actually *hurts* Qwen3-4B: 14.88% → 10.71% (flat teacher memory as noise)
- MemP and SASM: inconsistent, sometimes negative
- AMD: 14.88% → 49.40% (+34.52%p)

## Reading the Results

**AppWorld (168 tasks, multi-app, Python API)** — the most demanding benchmark:

| Student Model | Zero-shot | AMD | Gain |
|---|---|---|---|
| Qwen3-4B | 14.88% | 49.40% | +34.52%p |
| Qwen3-8B | ~35% | 51.79% | ~+17%p |
| Gemma4-E4B | — | 54.17% | — |
| Llama3.1-8B | — | — | — |
| Teacher (GPT-5-mini) | — | 50.00% | — |

The Gemma4-E4B and Qwen3-8B students **surpass the teacher** at 54.17% and 51.79% respectively. This is not model confusion — the authors interpret it as the student re-instantiating teacher decision patterns under its own inductive biases, potentially discovering more generalizable strategies.

**BFCL V3 (200 tasks, structured function calling)**: Average +11.2%p. Three of four students outperform the teacher (teacher avg: 38.39%; Qwen3-8B: 40.96%).

**ToolSandbox (129 scenarios, stateful conversational tool-use)**: Average +3.4%p. Smallest gain, as expected — the benchmark involves LLM-simulated user interactions that are harder to anticipate via static memory.

**Interaction efficiency**: AMD doesn't just improve accuracy — it makes students *faster*. On AppWorld, Qwen3-4B drops from 23.8 interaction turns (zero-shot) to 14.9 (AMD), vs. teacher's 10.1. The student executes more like the teacher, following more direct paths through the task.

**Ablation — which memory type matters most?**

Adding memory types incrementally on AppWorld:
- WF alone: meaningful gains
- WF + ST: largest incremental jump (+25.0%p for Qwen3-4B over WF alone) — **Subtask memory is the key driver**
- WF + ST + FN: additional gains, smaller in magnitude

**Effect of student model size** (Qwen3 family, GPT-5-mini teacher):
- 1.7B: 21.43% — limited capacity to utilize memory
- 4B: 49.40% — **peak relative gain (+34.52%p)**
- 8B: 51.79%
- 14B: 52.68% — marginal improvement; already near teacher ceiling

The sweet spot is around 4B: large enough to comprehend and apply hierarchical memory, small enough to benefit substantially.

## Key Notes of this Paper

### Why k=1 Retrieval is Optimal

One of the most practically significant findings: retrieving only 1 memory entry per type (k=1) outperforms retrieving more (k>1). For larger models, additional context is managed well. For SLMs, more retrieved examples add noise that exceeds the model's ability to synthesize — they get confused by competing examples rather than benefiting from broader coverage. This fundamentally limits how much context injection can help a small model, and it's a strong argument for memory quality over quantity in SLM-facing retrieval systems.

### The Teacher-Student Compatibility Paradox

A counterintuitive finding: for the weaker 4B student, the moderately capable teacher (GPT-5-mini, 50% AppWorld accuracy) outperforms the much stronger teacher (GPT-5.5, 91% AppWorld accuracy) in transfer quality. The stronger teacher's memories are written from a perspective too far above the student's comprehension level. The teacher solves problems so fluently that their trajectories omit the explanatory scaffolding the student needs. Middle-capability teachers, whose solving process is closer to the student's horizon, generate more legible memories. This mirrors intermediate teacher assistant findings in conventional KD literature (Mirzadeh et al., 2020).

### Memory Hierarchy Mirrors Human Skill Acquisition

The three-level hierarchy maps naturally onto how humans learn complex procedural skills: first understand the overall process (Workflow), then study concrete examples of each phase (Subtask), then learn specific tool/API usage details (Function). This psychological alignment may partly explain why the hierarchy works better than flat or same-granularity approaches.

### Reactive vs. Proactive Memory: The Error-Driven Design

The choice to inject Function memory *only on failure* is architecturally important. Injecting all function documentation proactively would bloat the context (many tools are never called in a given task) and would likely overwhelm the SLM's in-context processing capacity. The reactive design keeps the primary context lean and delivers targeted help at the precise moment of failure.

## Limitations

1. **Upfront teacher execution cost**: Building $\mathcal{M}$ requires running GPT-5-mini (or another capable teacher) on the full training task set. This is a one-time API cost, but it is non-trivial at scale.

2. **1.7B boundary**: At 1.7B parameters, even with AMD, AppWorld accuracy stays at 21.43%. There appears to be a floor below which in-context memory injection simply cannot compensate for limited model capacity.

3. **Teacher-student compatibility is brittle at the extremes**: Very small students need moderately capable teachers; very large teachers generate legible memories only for larger students. Selecting the right teacher-student pairing requires empirical tuning.

4. **Benchmark scope**: Evaluation covers three tool-use benchmarks only. Generalization to code generation, mathematical reasoning, or multi-modal tasks is not demonstrated.

5. **Static memory bank**: Once built, the memory bank does not update. A student that develops new successful strategies cannot contribute back to $\mathcal{M}$.

## Future Work

**Authors' directions**:
- Combining AMD with parameter updates (fine-tuning on teacher-memory-augmented trajectories)
- Memory compression to reduce bank size
- Extending to harder agentic tasks with longer horizons

**Additional promising directions**:

**Dynamic memory evolution**: A hybrid system where the student's own occasional successes are inserted back into the memory bank could gradually reduce teacher dependency. Reinforcement-based filtering — keeping only student memories that demonstrably improve outcomes — would maintain quality.

**Multi-teacher ensembles**: Using memories from teachers of varying capability could address the compatibility paradox: moderate-capability teacher memories for lower-granularity concepts, high-capability teacher memories for high-level workflow patterns.

**Memory compression and indexing for edge**: Current memory banks may be large (one entry per successful teacher trajectory, across many tasks). Techniques like memory clustering, summarization hierarchies, or learned memory compression could make AMD deployable on memory-constrained devices.

**Cross-domain transfer**: Can a memory bank built from one domain (e.g., e-commerce task automation) transfer to a different domain (e.g., medical record queries)? The abstracted placeholder format in Workflow memory suggests cross-domain reuse may be feasible.

## Implications for Edge / On-Device Deployment

AMD's architectural split — **teacher-side memory generation in the cloud, student-side inference on device** — is precisely the pattern needed for practical SLM deployment.

In the deployment pipeline:
- **Phase 1 (offline, cloud)**: Run the teacher agent on representative tasks; build $\mathcal{M}^{wf}$, $\mathcal{M}^{st}$, $\mathcal{M}^{fn}$; encode all entries; ship the memory bank to the device.
- **Phase 2 (online, device)**: Student (4B–8B) runs locally; retrieves relevant memory by vector similarity at task start and on tool failures; no cloud calls needed during inference.

This enables capable agentic behavior on devices without real-time LLM API access — essential for offline or low-connectivity environments (industrial automation, healthcare, edge IoT).

The key practical challenge is memory bank size. A full AppWorld-scale bank with embeddings may exceed RAM budgets on constrained devices. Future work on memory quantization, sparse indexing, and on-device FAISS or similar retrieval infrastructure will be needed to realize this deployment pattern at scale.

## Links

[Original Paper](https://arxiv.org/abs/2608.07169)
