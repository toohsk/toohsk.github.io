Title: SKILLER: Teaching Small Language Models to Follow Skills via Language-Level Reinforcement Learning
Date: 2026-08-15
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: SKILLER proposes a natural-language-driven RL framework that automatically generates executor-specific skills for small LVLMs, enabling 4B–9B models to rival closed-source frontier performance by treating the compact model's agent loop as the RL environment and propagating all signals through text.

## Why This Paper Matters

Agent harnesses — systems where a language model orchestrates tools, files, and APIs to complete multi-step tasks — are becoming a dominant deployment paradigm. Products like Claude Code, Codex, and OpenClaw all rely on **agent skills**: packaged procedural instructions that constrain a model's behavior for repeatable, high-quality task execution.

The problem is economic. Today, agent harnesses require frontier-scale models (GPT-5, Claude Opus) to interpret and execute complex skill documents reliably. Running a frontier model for every token of every task at scale is expensive. The obvious fix — swap the frontier model for a smaller, cheaper open-source model — breaks down immediately. Skills that work for a 100B+ model fail catastrophically when handed to a 4B or 9B model.

**Why does this happen?** This is the paper's central observation: the *model-mismatch problem*. Skills authored by humans or generated for frontier models implicitly assume:

- Expansive context tolerance (the model can hold 10,000 tokens of procedure in working memory)
- Robust error recovery (the model can self-correct from ambiguous instructions)
- Broad reasoning capacity (the model can infer intent from high-level abstractions)

Small models have none of these. Injecting a frontier-optimized skill into a 4B model causes argument hallucination, skipped verification steps, and cascading failures. The result is worse task performance than running the small model *without* any skill at all.

SKILLER's thesis: **the mismatch is fixable, but only by generating skills tailored specifically to the execution characteristics of the target small model.** This is the gap the paper fills.

## Core Technical Contribution

SKILLER is a **natural-language policy optimization** framework. Instead of tuning neural weights, it treats the *text of the skill document* as the optimizable variable, and runs an RL-flavored optimization loop using:

- **The small model's agent execution as the environment** (observing what actually breaks)
- **A frontier model as actor and critic** (diagnosing failures and rewriting the skill)
- **Natural language as the sole communication channel** (no gradient propagation, no weight updates)

### Formal Setup

Let $\mathbf{x}$ be a target task instance (instructions, tools, input/output contract). A frozen compact model $\pi$ executes the task conditioned on a skill $\mathcal{K}_i$:

$$\pi_{\mathcal{K}_i}(a_t \mid h_t, \mathbf{x}) \triangleq \pi(a_t \mid h_t, \mathbf{x}, \mathcal{K}_i)$$

The environment returns an execution trajectory $\tau_i$, a scalar reward $r_i \in [0,1]$, and verifier diagnostics $\mathbf{v}_i$ (per-test outcomes, error messages):

$$(τ_i, r_i, \mathbf{v}_i) = \mathcal{E}(\mathbf{x}, \mathcal{K}_i; \pi)$$

The objective is to find a skill that maximizes expected verifier reward:

$$\mathcal{K}_{\mathbf{x}}^* \in \arg\max_{\mathcal{K} \in \mathbb{K}} \mathbb{E}_{(\tau,r,\mathbf{v}) \sim p_{\mathcal{E},\pi}(\cdot \mid \mathbf{x}, \mathcal{K})} [r]$$

Rather than differentiating through the model or environment, SKILLER explores the skill space $\mathbb{K}$ via verifier-grounded natural-language updates.

### The Four-Component Loop

**1. State & Reward**

After each rollout, the system constructs a structured state quadruple:

$$\mathbf{s}_i = (\mathbf{x}, \tau_i, \tau^*, \mathbf{v}_i)$$

Where $\tau^*$ is a *reference trajectory* showing a successful execution (privileged, offline-only — never given to the small model at runtime). Pairing the actual trajectory with the reference makes it possible to pinpoint exactly where the small model's execution first diverged from a successful strategy.

**2. Critic**

A frontier-model critic $\mathcal{C}_\phi$ receives the state, reward, current skill, and replay memory, and returns *natural-language skill modification suggestions*:

$$\mathbf{g}_i = \mathcal{C}_\phi(\mathbf{s}_i, r_i, \mathcal{K}_i, \mathcal{M}_i)$$

The critic's job is not to rewrite the skill — it's to *diagnose causally*: distinguish between missing procedural guidance, tool misuse, output-contract violations, and infrastructure failures. It identifies what already works and issues concrete, localized repair instructions to the actor.

**3. Replay Memory**

$\mathcal{M}_i$ is a textual history that stores:
- Failure signatures with verifier diagnostics
- Critic-summary history across steps
- Accepted edits and their observed outcomes

This discourages repeated failure modes and provides evidence for rolling back regressions.

**4. Actor**

A frontier-model actor $\mathcal{A}_\theta$ implements bounded edits to the skill artifact:

$$\Delta_i = \mathcal{A}_\theta(\mathbf{x}, \mathcal{K}_i, \tau^*, \mathbf{g}_i, \mathcal{M}_i)$$
$$\mathcal{K}_{i+1} = \text{Apply}(\mathcal{K}_i, \Delta_i)$$

The four allowed edit operations are: **Insert, Replace, Create, Delete**. Crucially, the actor may also *synthesize task-local helper scripts* — offloading complex procedural reasoning from natural language into deterministic external code that the small model simply calls.

### Progressive Skill Disclosure

A key design choice: the skill document is presented to the small model under a *progressive disclosure* mechanism. Rather than dumping the entire skill at step 0, the mechanism gates which sections are revealed based on execution progress. This directly addresses the context-window overload problem.

### Batch Optimization

SKILLER also supports batch optimization over a set of instances $\mathcal{B} = \{\mathbf{x}^{(n)}\}_{n=1}^N$. Skills can be updated independently per instance or tied across related instances by aggregating evidence before a single update. The empirical batch objective is $\hat{J}_i(\mathcal{B}) = \frac{1}{N}\sum_{n=1}^N r_i^{(n)}$.

## Comparison to Prior Work

### What came before

| Method | Approach | Problem |
|--------|----------|---------|
| Human-authored skills | Experts write instructions for frontier models | Assumes broad reasoning; induces cognitive overload in SLMs |
| Manus (closed-source) | Frontier model generates skills | Optimizes for frontier model; skills degrade SLM performance |
| AutoSkill | Rewrites prompts based on execution history | Shallow: doesn't address model-mismatch at a structural level |
| EvoSkill | Evolutionary skill discovery for multi-agent systems | Not targeted to compact model execution |
| SkillX | Automatically constructs skill knowledge bases | Excessive output token consumption; not executor-specific |

The key gap that all prior methods share: **they generate skills for what a strong model can do, not for what the target compact model will actually do.**

### SKILLER's structural difference

SKILLER treats the compact model's *actual runtime behavior* as the ground truth the skill must accommodate. The optimization loop is closed around real execution, not around a proxy objective.

## Reading the Results

SKILLER was evaluated on five benchmarks using Qwen3.5-9B and Qwen3.5-4B:

- **SkillsBench**: 26 single-skill tasks across diverse domains
- **SWE-Skills-Bench**: 117 software engineering instances (10 high-difficulty skills)
- **SkillLearnBench**: 100 continual skill generation instances
- **GAIA**: 165 multi-step information-seeking tasks
- **EarthBench**: 248 Earth-science and data-processing workflow samples

**Main results** (vs. best baseline):

- Qwen3.5-9B gains: +4.3 to +20.4 percentage points over best baseline
- Qwen3.5-4B gains: +1.8 to +13.3 percentage points
- On SkillsBench single-skill tasks: Qwen3.5-9B with SKILLER matches closed-source frontier model performance

**The most striking finding**: On SWE-Skills-Bench, **Qwen3.5-4B equipped with SKILLER surpasses Qwen3.5-9B with human-authored skills, EvoSkill, SkillX, AutoSkill, or Manus-generated skills.** A 4B model with the right skill beats a 9B model with the wrong skill. The skill quality matters more than the parameter count, at least for structured tasks.

**Learning dynamics**: On SWE-Skills-Bench (complex procedural domain), performance improves across all 5 RL iterations — early iterations fix coarse failures, later ones inject finer constraints. On SkillLearnBench (output contract / basic tool routing), both models converge within 2 iterations. The optimization pace tracks domain complexity.

**Zero-shot transfer**: Skills generated on a held-out half of GAIA/EarthBench transfer effectively to the unseen half. The optimization loop extracts genuinely reusable procedural patterns rather than overfitting to specific instances.

**Structural analysis** (on SkillsBench): SKILLER skills are *shorter* and more *diverse* than baseline outputs (lower TF-IDF similarity, matching human-authored diversity). They also contain *more scripts and more code*. This reflects the key design principle: offload complex reasoning from natural language into deterministic tools.

**Cost**: SKILLER is more token-efficient than SkillX, which generates verbose outputs without environment grounding. SKILLER's targeted updates minimize unnecessary generation.

## Key Notes of This Paper

### The Optimized Policy Is Text, Not Weights

This is the central inversion. Standard RL optimizes $\theta$ (neural parameters) via gradient descent through differentiable computations. SKILLER optimizes $\mathcal{K}$ (a text document) via natural-language edits through a non-differentiable environment. The "gradient" is the critic's diagnosis; the "weight update" is the actor's bounded edit.

This makes SKILLER compatible with any black-box compact model — no architecture assumptions, no fine-tuning infrastructure.

### State Quadruple $\mathbf{s}_i = (\mathbf{x}, \tau_i, \tau^*, \mathbf{v}_i)$

The reference trajectory $\tau^*$ is the key that enables causal diagnosis. Without it, the critic can only observe that the model failed. With it, the critic can locate the *earliest divergence point* between the actual and successful execution — the precise moment the skill failed to prevent an error. This transforms "the model failed" into "the skill lacked a mandatory verification step at line 7 of the workflow."

### Bounded Actor Edits

The four operations (Insert, Replace, Create, Delete) are *localized* by design. The actor cannot rewrite the entire skill in one step. This constraint:
- Prevents regression of previously working behavior
- Forces the actor to make precise, diagnosable changes
- Keeps the replay memory meaningful (each delta is traceable)

### Progressive Skill Disclosure

Small models have smaller effective context windows than they nominally support — attention dilution means a 4B model may effectively lose track of content from earlier in a long context. Progressive disclosure bypasses this by gating skill content to the relevant phase of execution.

### Skill-as-Code Offloading

SKILLER-generated skills contain significantly more scripts than baseline methods. The intuition: if a procedure is complex enough to cause hallucinations in a small model (generate random file paths, perform multi-step computation), encode it as executable code that the model simply calls. The model only needs to correctly invoke a function; it doesn't need to reason through the computation.

## Limitations

**1. Offline skill generation cost**: The SKILLER optimization phase uses a frontier model (GPT-5.4) as both actor and critic. This is expensive at generation time, even if cheap at inference time.

**2. Reference trajectory dependency**: The framework requires a successful reference trajectory $\tau^*$ for each task. Obtaining this may require a frontier model to solve the task first — adding setup cost and creating a dependency on frontier model availability during the offline phase.

**3. Narrow model coverage**: All experiments use Qwen3.5-9B and Qwen3.5-4B. Transferability of SKILLER-generated skills to other model families (Gemma, Llama, Phi) is not characterized.

**4. Task structure assumption**: The framework assumes tasks have official verifiers providing structured feedback ($\mathbf{v}_i$). Applying SKILLER to tasks without executable verifiers (e.g., open-ended writing) would require alternative reward signals.

**5. Single-task skill scope**: While zero-shot transfer is demonstrated within GAIA and EarthBench task families, cross-domain skill generalization (a skill learned on SWE tasks transferring to Earth science) is not studied.

## Future Work

**Authors' directions** (implied from paper structure):
- Scaling to even more compact models (sub-4B / edge-class hardware)
- Reducing frontier-model dependency in the actor/critic (using open-source reasoning models)
- Continual online skill refinement as new failure modes emerge in production

**Additional promising directions**:

**Hierarchical skill compression**: SKILLER generates task-specific skills. A follow-on system could identify common procedural patterns across tasks and factor them into reusable sub-skills, further reducing the generation cost of new skills.

**Small-model critic bootstrapping**: Rather than relying on a frontier model as critic, one could investigate whether a medium-scale model (30B–70B) fine-tuned on SKILLER's critic outputs can serve as a cheaper replacement, enabling the entire pipeline to be run without API calls to closed-source systems.

**Skill portfolio transfer**: Learning a skill for task A often generates partial knowledge useful for task B. A retrieval-augmented skill generation approach could seed SKILLER's initial $\mathcal{K}_0$ with transferred fragments, accelerating convergence.

**Quantized model support**: Compact models deployed on edge hardware are typically quantized (INT4/INT8). SKILLER's executor-specific approach would naturally accommodate quantized models as environments — the framework doesn't assume any specific execution precision.

**Multi-model ensembling at inference**: SKILLER generates skills targeting a single executor. Future work could generate an ensemble of executor-specific skills for multiple compact models and route incoming tasks to the most appropriate executor-skill pair based on task characteristics.

## Implications for Edge / On-Device Deployment

SKILLER's implications for on-device AI are significant and direct:

**Skills are zero-inference-overhead artifacts**: Once generated offline, a SKILLER skill is a plain text document. Adding a skill to a small model's context costs exactly the skill's token count — no additional model, no retrieval system, no serving infrastructure.

**Consumer GPU deployment**: The paper targets Qwen3.5-4B and Qwen3.5-9B, which run on consumer-grade GPUs (a 4B model fits in ~8GB VRAM at FP16). SKILLER enables these models to approach frontier task performance without any hardware upgrade.

**Offline generation, online efficiency**: The expensive frontier-model computation happens once, offline. Deployed at scale, a skill generated for $2–10 in frontier API costs can be reused millions of times on cheap local hardware.

**No fine-tuning required**: Deploying SKILLER-enabled agents requires no fine-tuning, no LoRA adaptation, no model weight changes. This is critical for edge deployment where storage and update bandwidth are constrained.

**Skill portability**: Because skills are text, they can be version-controlled, audited, and updated independently of the model. This enables a deployment model where the on-device model is frozen and skills are updated OTA as task requirements evolve.

**Direct path to smartphone deployment**: A Qwen3.5-4B model with SKILLER-generated skills could realistically run agent tasks (file management, API calls, data extraction) on a high-end smartphone. The paper's SWE-Skills-Bench results — where a 4B model with SKILLER outperforms a 9B model without — suggest that well-crafted skills could enable phone-class models to handle tasks previously requiring cloud inference.

## Links

[Original Paper](https://arxiv.org/abs/2608.10538)
