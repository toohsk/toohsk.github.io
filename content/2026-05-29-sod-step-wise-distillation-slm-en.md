Title: SOD: Step-wise On-Policy Distillation for Small Language Model Agents
Date: 2026-05-29
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: SOD fixes a critical failure mode in distilling agentic tool-use capabilities into small language models — cascading divergence from erroneous tool calls — by adaptively down-weighting teacher supervision at steps where the student has already deviated, enabling a 0.6B model to reach 26.13% on AIME 2025.

## Why This Paper Matters

Building small language models that can reason with external tools — running code, querying search engines, executing calculations — is one of the hardest open problems in the practical deployment of AI. The difficulty is not purely about model capacity; it is fundamentally about **temporal credit assignment in long-horizon interactions**.

When a small model calls a tool incorrectly at step 5 of a 20-step reasoning chain, the environment state diverges from what it would have been under a correct call. Every subsequent step operates in a different context than the teacher expected. If we naively apply knowledge distillation — asking the teacher "what should come next?" at each step — the teacher is answering for a world the student never created.

This cascading divergence problem is qualitatively different from single-turn generation tasks. In those tasks, a distillation mistake at token $t$ affects token $t+1$ and gradually fades. In tool-integrated reasoning (TIR), a routing mistake at step 5 changes the **observations** that feed into steps 6–20, creating an entirely different problem state. The teacher's advice for those later steps is not merely slightly off — it may be **logically incoherent** given where the student actually ended up.

Prior work on on-policy distillation (OPD) showed promise in single-turn settings but had not confronted this fundamental issue in agentic, multi-step environments. This paper identifies and fixes it.

## Core Technical Contribution

### The Failure Mode: Cascading Divergence

SOD begins with a careful empirical diagnosis of why naive OPD fails for TIR. In OPD:

1. The **student** generates a trajectory (tool calls + reasoning steps) on its own
2. At each step, the **teacher** computes what it would predict given the student's trajectory so far
3. A KL divergence loss forces the student toward the teacher's distribution at each step

The paper shows that when the student makes a tool call error at step $t$, the step-level divergence $D_t = KL(\pi_{\text{teacher}} || \pi_{\text{student}})$ at step $t$ is high. But crucially, at step $t+1$, the teacher still tries to provide guidance — now for a trajectory that has gone off the rails. The teacher's prediction is conditioned on a **different trajectory prefix** than what the student experienced, because the student received different tool observations.

This creates a vicious cycle: high divergence at step $t$ → misleading teacher signal at step $t+1$ → the student learns incorrectly for step $t+1$ → higher divergence at step $t+2$ → and so on.

In experiments, the paper shows that the average step-level divergence **increases monotonically** across steps in naive OPD, confirming the cascade hypothesis.

### The SOD Solution: Adaptive Step-Level Reweighting

SOD introduces a conceptually clean fix: **weight each step's distillation loss inversely with the divergence at that step**.

The loss function is:

$$\mathcal{L}_{\text{SOD}} = \sum_{t=1}^{T} w(D_t) \cdot \mathcal{L}_{\text{KL}}^{(t)}(\pi_{\text{student}} \| \pi_{\text{teacher}})$$

where:
- $T$ is the total number of steps in the trajectory
- $D_t$ is the KL divergence at step $t$: how far apart are the teacher and student distributions at this step
- $w(D_t)$ is a **decreasing** function of $D_t$ — when divergence is high, we trust the teacher less
- $\mathcal{L}_{\text{KL}}^{(t)}$ is the standard token-level KL loss at step $t$

The weight function $w(\cdot)$ is designed to be:
- $w(D_t) \approx 1$ when $D_t \approx 0$ (student and teacher aligned → trust teacher fully)
- $w(D_t) \to 0$ as $D_t \to \infty$ (student has diverged → ignore teacher's advice for this step)

A natural choice is $w(D_t) = \exp(-\alpha D_t)$ for some temperature $\alpha$, though the paper explores several variants.

The intuition is powerful: **when the student is in a state the teacher recognizes, the teacher's advice is reliable and should be weighted fully. When the student is in a state the teacher has never seen (because the student made earlier errors), the teacher's advice is unreliable and should be down-weighted.**

This is analogous to how a human expert guides a student: if the student is following the expected approach, detailed step-by-step guidance is helpful. But if the student has gone off on a completely different path, the expert's expected-path advice is useless or harmful — better to let the student explore rather than confuse them with irrelevant instruction.

### Step-Level vs. Trajectory-Level Approaches

An alternative to SOD would be to simply reject trajectories with high overall divergence (trajectory-level filtering). The paper argues against this for two reasons:

1. **Sample inefficiency**: At small model scales (0.6B–3B), most trajectories will have at least some divergence. Trajectory-level rejection discards the majority of training signal.

2. **Mixed-quality trajectories**: A trajectory may have correct steps 1–4 and 12–20 with only steps 5–11 being divergent. Discarding the entire trajectory wastes the useful signal from the correct steps.

SOD's step-level approach retains the learning signal from aligned steps while discarding misleading signals from divergent steps — using all available data more efficiently.

## Comparison to Prior Work

| Method | 0.6B AIME 2025 | 1.5B Math Avg | Approach |
|--------|---------------|--------------|---------|
| GRPO (RL) | ~12% | ~42% | Sparse outcome rewards |
| OPD (naive) | ~18% | ~48% | Dense token supervision, no divergence handling |
| **SOD** | **26.13%** | **~58%** | Step-wise adaptive reweighting |
| Teacher (GPT-4 class) | ~65% | ~78% | N/A |

Baselines include:
- **GRPO (Group Relative Policy Optimization)**: Pure RL with outcome rewards. Trains on whether final answers are correct. Provides only sparse signal.
- **SFT (Supervised Fine-Tuning)**: Training directly on teacher trajectories, ignoring the student's actual states. Offline distillation, no on-policy correction.
- **OPD (naive)**: On-policy distillation without the adaptive reweighting — the ablation that confirms the divergence cascade is the key problem.
- **OPD + trajectory rejection**: Trajectory-level filtering. Less data-efficient than SOD.

SOD achieves **up to 20.86% improvement over the second-best baseline** on the joint benchmark across math, science, and code tasks.

## Reading the Results

The 26.13% on AIME 2025 for a 0.6B model deserves context. AIME (American Invitational Mathematics Examination) is a competition math benchmark that requires multi-step algebraic and combinatorial reasoning. The questions are hard enough that most humans who attempt them score below 50%.

At 0.6B parameters:
- GPT-2 (1.5B, older) scores near 0% on AIME
- Recent 0.5B–1B SLMs without tool use typically score 3–8%
- SOD's 0.6B student with tool-integrated reasoning reaches **26.13%**

This is not just an improvement over baselines — it represents a qualitative regime shift. The model has learned to **use a calculator and code interpreter** as external tools to decompose hard problems into solvable sub-computations, rather than trying to do all arithmetic in its weights. The tools compensate for the limited capacity of the small model.

The benchmark suite (math, science, code) tests different forms of tool-integrated reasoning:
- **Math**: Using calculators and symbolic math tools
- **Science**: Using code to run simulations or retrieve formulas
- **Code**: Writing and executing code for algorithmic tasks

SOD shows consistent improvement across all three, suggesting the step-wise reweighting is solving a general failure mode, not one that's specific to a single task type.

## Key Notes of This Paper

### Step-Level Divergence Measurement

The divergence $D_t$ at step $t$ must be computed **during training**, not post-hoc. Concretely:

At each step $t$ of the student's on-policy trajectory:
1. The student generates its prediction distribution $\pi_{\text{student}}(a_t | s_{1:t-1}, a_{1:t-1})$
2. The teacher is queried on the student's context to produce $\pi_{\text{teacher}}(a_t | s_{1:t-1}^{\text{teacher}}, a_{1:t-1})$

Note the crucial detail: the teacher sees the **student's trajectory prefix** up to step $t$, not the teacher's own trajectory. This is what makes it "on-policy" — the teacher evaluates what it would do in the student's actual situation.

$D_t = \text{KL}(\pi_{\text{teacher}} \| \pi_{\text{student}})$ measured at the action distribution level (not token level), then aggregated.

### Stability of Adaptive Weighting

A potential concern: if the weight function $w(D_t)$ drops to near-zero for most steps (because the student diverges early), will training collapse due to insufficient signal? The paper shows this does not happen in practice because:

1. **Early steps are usually aligned**: At step $t=1$ (the first reasoning step), the student and teacher start from the same prompt. Divergence is small, and full teacher supervision applies.
2. **Gradual divergence**: Divergence typically grows slowly, meaning many early steps receive near-full supervision.
3. **Outcome signal as floor**: Even when step-level weights drop to near-zero, the final outcome signal (answer correctness) provides a floor of training signal to prevent complete collapse.

The combination creates a stable training dynamic: dense supervision when aligned, sparse outcome signal when diverged.

## Limitations

1. **Teacher cost**: SOD requires querying a powerful teacher model on every student trajectory. At 0.6B student scale, the teacher is typically a 7B–70B model. This makes training substantially more expensive than pure RL (GRPO) which only evaluates final answers.

2. **Tool environment assumptions**: SOD assumes deterministic tool execution (a calculator always gives the right answer given the right input). In environments with stochastic tools or noisy observations, the divergence measurement becomes more complex.

3. **Curriculum not explored**: The paper trains on a fixed difficulty distribution. It's unclear whether curriculum learning (starting with easy tool-use tasks and progressing to hard ones) would improve the learning dynamics for very small models.

4. **Limited to reasoning tasks**: SOD is evaluated on math/science/code. Transfer to open-ended conversational tasks (where "correct" tool use is harder to define) is not demonstrated.

## Future Work

**Authors' suggested directions:**
- Extending SOD to multi-agent settings where multiple SLMs collaborate on a task
- Applying step-level divergence reweighting to reinforcement learning (not just distillation)
- Investigating how SOD interacts with model merging techniques

**Promising follow-on directions:**
- **Online teacher distillation**: Rather than using a static teacher, continuously update the teacher based on what the student has learned, maintaining a constant divergence gap
- **SOD for multimodal tools**: Applying the same framework to tool-integrated reasoning with image processing tools, databases, or web browsers
- **Divergence-guided data curation**: Using the step-level divergence signal not just at training time but to select which trajectories to include in a static distillation dataset
- **Lightweight student-teacher co-training**: Training the teacher and student jointly, where the teacher learns to provide advice that's actually learnable by the student's capacity level

## Implications for Edge / On-Device Deployment

SOD's most exciting implication is what it enables for **autonomous on-device AI agents**:

1. **Calculator + code = near-infinite reasoning depth**: A 0.6B model cannot store all mathematical knowledge in its weights. But a 0.6B model that can reliably call a calculator for arithmetic and a Python interpreter for algorithms can solve problems far beyond its "intrinsic" capability. SOD makes this tool-use reliability achievable at sub-1B parameter scales.

2. **Energy-efficient agentic AI**: Tool-calling shifts expensive computation to deterministic, energy-efficient operations (running Python, querying a database) rather than probabilistic neural inference. A model that takes 10 tool calls instead of generating 1000 tokens can be significantly more energy-efficient overall.

3. **Privacy-preserving local reasoning**: Math homework help, coding assistance, scientific calculations — all use cases that could run on-device without sending sensitive data to the cloud, if the SLM is powerful enough to use local tools reliably.

4. **Foundation for on-device autonomous agents**: Smartphones already have native calculators, code execution sandboxes (Pythonista, Scriptable), and local databases. A SOD-trained 0.6B model integrated with these local tools represents a credible path toward genuinely useful autonomous on-device agents.

## Links

[Original Paper](https://hf.co/papers/2605.07725)
