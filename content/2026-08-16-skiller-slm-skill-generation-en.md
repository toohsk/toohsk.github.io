Title: SKILLER: Language-Level Reinforcement Learning for Reusable Skill Extraction in Small Language Models
Date: 2026-08-16
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: SKILLER solves the model-mismatch bottleneck that makes frontier-model agent skills fail on small LMs, by using a frontier model as actor+critic in a natural-language RL loop, achieving up to 20.4 percentage-point gains on Qwen3.5-9B and enabling a 4B model to outperform an unoptimized 9B on software engineering tasks.

## Why This Paper Matters

Modern agent harnesses — Claude Code, OpenCode, OpenClaw — rely on "agent skills": reusable, structured packages of procedural knowledge that constrain an LLM's behavior space for repeatable, high-quality task execution. But there's a catch: **skills crafted for frontier models like Claude Opus or GPT-5.x don't transfer to compact 4B–9B models**. They induce cognitive overload, argument hallucination, and cascading task failure.

This creates a cost problem. Running frontier models at inference time for every agent task is prohibitively expensive at scale. Small open-source models deployable on consumer GPUs (Qwen3.5, Gemma 4) represent a compelling alternative — but only if their skills are specifically designed for their constrained capabilities.

The fundamental insight: **skills for large models carry implicit assumptions** (broad context tolerance, robust self-correction, complex branch navigation) that compact models simply cannot meet. The problem isn't the model's size per se — it's the skill-model mismatch.

SKILLER addresses this directly: a framework that automatically generates skills specifically tailored to the behavioral constraints of any target small model.

## Core Technical Contribution

SKILLER is a **natural-language-driven reinforcement learning framework** that treats the textual skill itself as the policy to optimize. No neural weight updates. No gradient computation. All RL signals propagate through structured natural language.

### The RL Formulation

Let $\mathbf{x}$ be a task instance (instruction + tools + runtime inputs + output contract). A frozen compact model $\pi$ executes tasks under skill $\mathcal{K}_i$ at optimization step $i$, inducing the effective conditional policy:

$$\pi_{\mathcal{K}_i}(a_t \mid h_t, \mathbf{x}) \triangleq \pi(a_t \mid h_t, \mathbf{x}, \mathcal{K}_i)$$

The skill acts as a behavioral modifier prepended to context — not a fine-tuned weight, but a text document that reshapes the model's action distribution at inference time.

**Environment transition:**
$$(\tau_i, r_i, \mathbf{v}_i) = \mathcal{E}(\mathbf{x}, \mathcal{K}_i; \pi)$$

This runs the task through the small model and returns: execution trajectory $\tau_i$, scalar reward $r_i \in [0,1]$ (task success or test pass rate), and verifier diagnostics $\mathbf{v}_i$ (per-test outcomes, error messages).

**Optimization objective:**
$$\mathcal{K}_{\mathbf{x}}^* \in \arg\max_{\mathcal{K} \in \mathbb{K}} J_{\mathbf{x}}(\mathcal{K}), \quad J_{\mathbf{x}}(\mathcal{K}) = \mathbb{E}_{(\tau,r,\mathbf{v}) \sim p_{\mathcal{E},\pi}(\cdot \mid \mathbf{x},\mathcal{K})} [r]$$

### The Four Core Components

**1. State quadruple:**
$$\mathbf{s}_i = (\mathbf{x}, \tau_i, \tau^*, \mathbf{v}_i)$$

The critical element here is $\tau^*$ — a reference trajectory showing a successful execution path. Comparing the actual trajectory $\tau_i$ with $\tau^*$ allows the system to identify not just *whether* execution failed, but *where it first diverged* from a successful strategy. This causal localization is key.

**2. Critic** (frontier model $\mathcal{C}_\phi$):
$$\mathbf{g}_i = \mathcal{C}_\phi(\mathbf{s}_i, r_i, \mathcal{K}_i, \mathcal{M}_i)$$

The critic receives both the scalar reward (did it succeed?) and the verifier diagnostics (why did it fail?). It systematically distinguishes between: missing procedural guidance, tool misuse, output-contract violations, and non-actionable infrastructure failures. Output is concrete, localized editing instructions — not a request to rewrite the whole skill.

**3. Replay Memory** $\mathcal{M}_i$: A compact textual history (not raw token transitions) containing:
- Failure signatures with verifier diagnostics
- Critic summary history
- Accepted edits and their observed outcomes

This prevents repeated failures, protects effective behavior, and provides evidence for rollback decisions when a modification causes regression.

**4. Actor** (frontier model $\mathcal{A}_\theta$):
$$\Delta_i = \mathcal{A}_\theta(\mathbf{x}, \mathcal{K}_i, \tau^*, \mathbf{g}_i, \mathcal{M}_i), \quad \mathcal{K}_{i+1} = \operatorname{Apply}(\mathcal{K}_i, \Delta_i)$$

The actor applies only four bounded edit operations: **Insert, Replace, Create, Delete**. This constraint is deliberate — it prevents unconstrained rewrites that could discard previously effective content. Crucially, the actor can create **task-local helper scripts** that offload complex procedural reasoning from natural language into deterministic code, sidestepping the context-overload problem entirely.

### Progressive Skill Disclosure

To prevent compact models from being overwhelmed by long textual contexts, SKILLER uses a progressive disclosure mechanism that reveals skill content incrementally during execution.

The full optimization produces a composed policy: $\mathcal{K}_I = \operatorname{Apply}(\cdots\operatorname{Apply}(\mathcal{K}_0, \Delta_0),\ldots,\Delta_{I-1})$ over $I=5$ steps.

## Comparison to Prior Work

| Method | Core Approach | Key Weakness |
|--------|--------------|--------------|
| AutoSkill | Lifelong self-evolution from interaction history | Shallow prompt rewriting; doesn't address model mismatch |
| EvoSkill | Multi-agent automated skill discovery | Optimized for frontier models; fails on compact models |
| SkillX | Automatic skill knowledge base construction | Unconstrained generation causes textual bloat; highest token cost |
| Manus | Closed-source commercial skill generator | Verbose domain context triggers cognitive overload on small models |
| Human-authored | Expert-written procedural instructions | Implicitly assumes large-model reasoning capacities |
| **SKILLER** | **NL-RL with target model as environment** | — (proposed method) |

**Cost comparison (Qwen3.5-9B, all 5 benchmarks):**
- AutoSkill: lowest cost, smallest performance gain
- SkillX: highest token consumption and monetary cost
- SKILLER: optimal cost-performance ratio; substantially cheaper than SkillX with commanding performance advantage

Human-authored skills and Manus-generated skills actually *degrade* performance versus no-skill baseline on several benchmarks. This directly validates the model-mismatch hypothesis: skills assuming strong reasoning capacity become counterproductive when applied to compact models.

## Reading the Results

### Main Results (Table 1, 5 benchmarks, 3-run average)

**For Qwen3.5-9B:**
- Absolute improvement range: **+4.3 to +20.4 percentage points** over the best baseline

**For Qwen3.5-4B:**
- Absolute improvement range: **+1.8 to +13.3 percentage points** over the best baseline

**Most significant result:** On SWE-Skills-Bench (complex software engineering tasks), Qwen3.5-**4B** with SKILLER skills achieves a higher pass rate than Qwen3.5-**9B** with human-authored skills, AutoSkill, EvoSkill, SkillX, or Manus skills. A well-optimized behavioral constraint is worth more than doubling parameter count.

**Benchmark-specific insights:**

- **SWE-Skills-Bench**: Largest gains, continuous improvement over all 5 optimization steps. Software engineering strictly penalizes unverified file edits, out-of-order tool calls, ungrounded argument fabrication — exactly the failure modes SKILLER's critic-actor loop targets.
- **SkillLearnBench**: Rapid convergence in 1–2 steps. When bottlenecks are output formatting or basic tool routing, errors are quickly isolated and resolved.
- **GAIA**: Matches best baseline. Multi-hop retrieval success depends partly on factual knowledge SKILLER can't inject — but it still regulates the search process effectively.

**Zero-shot transfer (Table 2):** Skills generated on half of GAIA and EarthBench instances transfer to the held-out half. This demonstrates extraction of genuinely reusable procedural rules, not surface-level overfitting. For Qwen3.5-4B, SKILLER achieves highest accuracy on GAIA zero-shot; Manus skills, which slightly lead on EarthBench, **fall below the no-skill baseline on GAIA**, showing a fundamental trade-off where verbose domain context collapses 4B reasoning on multi-hop tasks.

### Structural Analysis (Table 3, SkillsBench)

SKILLER-generated skills have:
- **Lower word count** than AutoSkill/EvoSkill/SkillX (concise, targeted instructions)
- **Low TF-IDF cosine similarity** between tasks (matching human-authored skill diversity — no boilerplate reuse)
- **Highest script count and LOC** — complex procedures are offloaded to deterministic code

This reveals a deliberate paradigm shift: minimizing prompt verbosity while maximizing code-level execution capability, perfectly accommodating compact models' limited context windows.

## Key notes of this Paper

### Language-Level Policy Iteration

The deepest insight in SKILLER is the reformulation of skill optimization as RL where the "policy" lives in text space, not weight space. Standard RL updates $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$. SKILLER instead performs:

$$\mathcal{K}_{i+1} \leftarrow \operatorname{Apply}(\mathcal{K}_i, \mathcal{A}_\theta(\ldots))$$

This works because:
1. **The optimization target** (skill quality) is evaluable without gradients — we just run the task and check if it passes
2. **The optimization direction** (what to change) is diagnosable by a frontier model given rich execution evidence ($\tau_i$, $\tau^*$, $\mathbf{v}_i$)
3. **The optimization step** (how to change) is bounded to prevent destructive rewrites

### The Dual-Signal Critic Design

Most prior work gives a critic only one of: (a) a scalar reward, or (b) natural language feedback. SKILLER gives it both simultaneously: $r_i$ tells the critic *how bad* the failure was; $\mathbf{v}_i$ (verifier diagnostics) tells it *what specifically went wrong*. This combination enables causal attribution rather than correlation-based pattern matching.

The state quadruple $\mathbf{s}_i = (\mathbf{x}, \tau_i, \tau^*, \mathbf{v}_i)$ is carefully designed: by pairing the failed trajectory $\tau_i$ with a successful reference $\tau^*$, the critic can identify the precise action where execution diverged. This is analogous to counterfactual reasoning — "if step 3 had been different, subsequent steps would have succeeded."

### Script Externalization as Cognitive Offloading

The finding that SKILLER generates the most LOC reveals a key strategy: instead of trying to teach the small model to perform complex multi-step procedures through natural language, SKILLER moves the complexity into executable scripts. The small model's job becomes simpler: call the right script with the right arguments. This exploits what small models *are* good at (following precise, short instructions) while sidestepping what they're bad at (maintaining coherence across long instruction chains).

## Limitations

1. **Frontier model dependency at generation time**: Offline skill generation requires access to a strong frontier model (GPT-5.4 in experiments). The generation is one-time, but not zero-cost.
2. **Narrow evaluation coverage**: Only tested on Qwen3.5-9B and Qwen3.5-4B. Generalization to other architectures (Llama, Gemma 4, Mistral) is unverified.
3. **Factual knowledge ceiling**: On multi-hop reasoning tasks (GAIA), SKILLER can regulate *how* the model searches but cannot supply missing factual knowledge the model doesn't have.
4. **Fixed 5-step schedule**: The RL schedule is fixed at 5 iterations for all tasks. Adaptive step allocation based on task complexity could improve efficiency.
5. **Replay memory scaling**: Long-horizon skill optimization could accumulate excessive memory. Pruning strategies are not discussed.
6. **Task-local scope**: Each skill is optimized per-task-type. Cross-task skill transfer and library-level co-optimization are left to future work.

## Future Work

**Authors' suggested directions:**
- Batch optimization across multiple task instances in parallel
- Multi-task skill sharing and cross-domain generalization

**Additional promising directions:**

1. **Self-distillation for critic/actor**: Currently GPT-5.4 plays both actor and critic. Could an intermediate-size model (e.g., a fine-tuned 30B) learn to perform this optimization, reducing the frontier model dependency? "Skills optimized by small models for small models" is an attractive recursive structure.

2. **Continuous skill library evolution**: Rather than task-local optimization, build a skill library that grows as the agent accumulates experience. When new error patterns emerge in production, trigger automatic skill patching using SKILLER's framework — a live feedback loop between deployment and skill maintenance.

3. **Error taxonomy formalization**: SWE-Skills-Bench errors cluster into "argument hallucination," "out-of-order tool calls," "output contract violations." Formalizing a taxonomy of compact model failure modes could enable more targeted, efficient critic prompting.

4. **Optimal stopping for RL steps**: Learning when to stop iterating (early stopping based on reward plateau or critic confidence) rather than running a fixed 5 steps could significantly reduce generation costs for simpler tasks.

5. **Verification of skill transferability across model versions**: As Qwen3.5 evolves to Qwen3.6, do optimized skills remain valid? Understanding the robustness of skills across model versions is critical for production deployment.

## Implications for Edge / On-Device Deployment

SKILLER's results carry a direct and important message for edge AI: **the limiting factor for SLMs in agentic tasks is not parameter count — it is the mismatch between skill design and model capability**.

**Practical takeaways:**

- **One-time optimization cost, zero-cost inference**: Skills are generated offline using frontier models (expensive, but once). At inference time, only the small on-device model runs. This amortizes the optimization cost across all future task executions — exactly the right trade-off for deployment at scale on constrained hardware.

- **4B models doing 9B-level work**: The demonstration that Qwen3.5-4B + SKILLER outperforms Qwen3.5-9B + human skills on SWE-Skills-Bench suggests that mobile-class models (which typically support 4B–7B at acceptable latency) can handle significantly more complex tasks than currently assumed.

- **Specialization through skills, not fine-tuning**: SKILLER provides an alternative to on-device fine-tuning for specialization. No training data collection, no compute for gradient updates, no risk of catastrophic forgetting — just skill generation.

- **Privacy-sensitive applications**: Medical, legal, and financial use cases where data cannot leave the device can benefit from SLMs with SKILLER-generated skills that approach frontier model quality on structured tasks.

- **Task-specific embedded deployment**: Industrial robotics, automotive infotainment, smart-home controllers — scenarios with constrained hardware and predictable task distributions are ideal for pre-generating optimized skills offline and deploying them statically.

The broader implication: as SLMs improve in raw capability, the bottleneck for edge deployment shifts from "can the model understand the task?" to "does the skill guide the model reliably?" SKILLER is an early answer to the second question.

## Links

[Original Paper](https://arxiv.org/abs/2608.10538)
