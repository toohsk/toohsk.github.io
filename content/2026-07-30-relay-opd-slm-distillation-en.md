Title: Pass the Baton: Trajectory-Relayed On-Policy Distillation for Small Language Models
Date: 2026-07-30
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Relay-OPD fixes the "prefix failure" problem in on-policy distillation by having a teacher model briefly take over at critical divergence points, improving small model reasoning by +5.73% over standard OPD while cutting training trajectory length by 50%.

## Why This Paper Matters

Knowledge distillation is the central technique for making small language models that can reason. The goal: take a capable but expensive teacher model and transfer its capabilities into a compact student. For mathematical reasoning — arguably the hardest capability to compress — this requires not just pattern matching but genuine step-by-step problem solving.

On-policy distillation (OPD) was designed to address this: let the student generate its own reasoning trajectories, then use the teacher to supervise those trajectories token-by-token. This grounds the supervision in the student's actual failure modes rather than an idealized teacher-generated trace.

But OPD has a fatal flaw that becomes especially acute in long-horizon reasoning: **prefix failure**. Once a student commits to the wrong reasoning direction — a wrong lemma at step 3 of a 20-step proof, a wrong decomposition early in an algebra problem — every subsequent generation builds on that mistake. The teacher, asked to supervise later steps of this derailed trajectory, is forced to provide guidance from a position it never expected to reach. That guidance is unreliable at best and actively misleading at worst.

This paper quantifies the problem and proposes an elegant solution: at the precise moments when the teacher would redirect the reasoning but the student would barrel ahead, hand the baton to the teacher for a few steps. The resulting method — **Relay-OPD** — improves average accuracy on 8 math benchmarks by +5.73% over OPD for a 1.7B student and reduces training trajectory length by over 50%.

## Core Technical Contribution

### The Prefix Failure Problem

When a student generates a trajectory y = (y₁, ..., y_N), OPD trains on the reverse-KL advantage at each position:

$$A_t^\text{OPD} = \log\pi_T(y_t \mid h_t) - \log\pi_{\bar\theta}(y_t \mid h_t)$$

where h_t = (x, y_{<t}) is the current prefix and π_T, π_̄θ are teacher and old student policies.

The sign of this advantage tells the student whether to increase or decrease the probability of the sampled token. When the student reasons correctly, the teacher agrees and the advantage is near zero. When the student starts going wrong, the teacher assigns lower probability to the student's tokens — the advantage is negative and the student learns to backtrack.

But here's the problem: by the time the student is 5 wrong steps into a misdirected chain, the teacher's supervision at step 6 is conditioned on a prefix neither the teacher nor the student "would normally be at." The teacher might recommend a reflection token (*But*, *Wait*, *However*) that makes no sense in the context the student has now drifted into. Training on that signal harms rather than helps.

### The Handoff Trigger

The key insight is that **teacher-student divergence in reasoning direction is observable without external supervision**. Specifically:

$$\phi(h) = \mathbf{1}\bigl[a^T(h) \in \mathcal{R}\bigr] \cdot \mathbf{1}\bigl[\mathcal{K}_S(h) \cap \mathcal{R} = \varnothing\bigr]$$

where:
- R is the set of **reflection tokens** — words like *But*, *Wait*, *However*, *No* that signal a reasoning redirect
- a^T(h) is the teacher's top-1 next token (argmax of π_T(·|h))
- K_S(h) is the student's top-K support set (top-K tokens by probability under π_̄θ(·|h))

φ(h) = 1 means: the teacher wants to redirect the reasoning (its top token is a reflection), but the student would not redirect (none of its K most probable tokens are reflections). This is a label-free signal that prefix failure is occurring.

The trigger is intentionally asymmetric: it fires only when the teacher *wants* to redirect and the student *does not*. If the student also recognizes the need to redirect, there is no failure requiring intervention.

### Relay Trajectory Construction

When the trigger fires, Relay-OPD inserts a teacher leg into the trajectory:
1. The teacher generates a short corrective segment — by default, a reflection token followed by L=3 reasoning paragraphs (~70 tokens)
2. Control returns to the student, which continues from the corrected prefix
3. A relay budget (M=2, L=3) caps the maximum number of takeovers and length of each teacher segment

The relay budget is critical. A teacher that takes over too frequently or too long pulls the trajectory far from the student's own policy — the resulting training signal becomes closer to offline distillation (training on teacher-generated data) than on-policy learning. The limited budget ensures the trajectory remains anchored to the student's distribution while strategically fixing the failure points.

### Efficient Implementation via Speculative Decoding

Alternating between two models during generation would require constant model switching, creating massive scheduling overhead. The authors cleverly avoid this by recasting Relay-OPD as a **state-switched speculative decoding process**.

The student acts as the draft model and the teacher as the target model. At each position, the decoding state s_t ∈ {S, T, ⊥} determines who is generating:

$$s_{t+1} = \begin{cases} \mathsf{T} & \text{if } s_t=\mathsf{S},\ \phi(h_{t+1})=1,\ j_t < M \\ \mathsf{S} & \text{if } s_t=\mathsf{T},\ \ell_t=L,\ j_t < M \\ \bot & \text{if } s_t=\mathsf{T},\ \ell_t=L,\ j_t=M \\ s_t & \text{otherwise} \end{cases}$$

In student legs, every draft token is accepted (the student accepts its own output unconditionally). In teacher legs, standard speculative rejection sampling runs against the teacher. The teacher logits computed during verification simultaneously provide the trigger criterion φ — no additional forward pass required.

This is exact: teacher legs are distributionally identical to the teacher generating from the corrected prefix, and student legs are ordinary student sampling. The two-model interleaving is replicated by a single engine.

The optimization objective across the entire relay trajectory z = (z₁, ..., z_N) is:

$$\mathcal{L}_\text{Relay} = -\mathbb{E}_{x,z}\!\left[\frac{1}{N}\sum_{t=1}^{N}\min\!\bigl(\rho_t A_t^\text{Relay},\, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t^\text{Relay}\bigr)\right]$$

This is a clipped PPO-style objective applied to the relay trajectory, with the advantage defined as the log-ratio between teacher and old student probabilities at each generated token.

## Comparison to Prior Work

**Baselines tested:**
- SFT: standard supervised fine-tuning
- KD: token-level offline knowledge distillation
- GRPO: outcome-reward RL
- OPD: standard on-policy distillation
- TRD: offline trajectory rewriting by the teacher
- FastOPD: fixed-length truncation of student rollouts
- SKD: token-level mixing of teacher and student generation

**Results for Qwen3-1.7B-Non-Thinking (teacher: Qwen3-4B-Instruct-2507):**

| Method | AIME 24 | AIME 25 | AIME 26 | MATH | AMC | Olympiad | HMMT-F | HMMT-N | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Student | 12.60 | 9.58 | 7.40 | 71.95 | 47.89 | 38.54 | 6.34 | 4.38 | 24.84 |
| OPD | 35.83 | 25.52 | 23.33 | 85.70 | 70.08 | 55.27 | 20.08 | 14.06 | 41.23 |
| FastOPD | 42.29 | 30.42 | 26.35 | 87.95 | 74.30 | 58.16 | 23.58 | 20.73 | 45.47 |
| **Relay-OPD** | **42.71** | **32.81** | **30.52** | **89.50** | **76.88** | **58.79** | **24.72** | 19.79 | **46.96** |

**Results for Qwen3-0.6B-Non-Thinking:**

| Method | Avg | Train Len |
|---|---|---|
| OPD | 28.03 | 6900 tokens |
| FastOPD | 30.42 | 3302 tokens |
| **Relay-OPD** | **31.04** | **2490 tokens** |

Key comparisons:
- vs OPD: +5.73% avg (1.7B), +3.01% (0.6B)
- vs FastOPD (best prior trajectory intervention): +1.49% (1.7B), +0.62% (0.6B)
- Training trajectory length: 50.7% reduction vs OPD for 1.7B; 63.9% for 0.6B

The failure modes of competing methods are informative:
- **TRD** (offline rewriting) actually hurts — 30.69 vs 41.23 for 1.7B. Rewriting artifacts make trajectories look unnatural and harm learning
- **SKD** (token-level mixing) barely improves over OPD — it uses generic distributional disagreement as the mixing signal, not reasoning-direction divergence
- **FastOPD** (fixed truncation) helps but cannot demonstrate *how to recover* from a failed reasoning prefix — it just discards late supervision

## Reading the Results

The numbers on AIME benchmarks are striking. AIME (American Invitational Mathematics Examination) problems are competitive-math problems that even top-tier models struggle with; a 1.7B model reaching 32.81% on AIME 2025 and 30.52% on AIME 2026 via distillation from a 4B teacher is a strong result.

The most important number is not the absolute accuracy but the **trajectory efficiency**: Relay-OPD reaches its best checkpoint at step 35 for 1.7B, versus step 55 for OPD. It does so with average rollout length 2,296 tokens vs OPD's 4,658. The student is learning faster from shorter, better-curated trajectories.

The training dynamics reveal why: teacher token ratio starts at ~13% early in training but drops to 2–3% after ~20 steps. As the student improves, it needs less correction. The intervention is self-limiting — the better the student gets, the less the teacher intervenes, and the more on-policy the training becomes. This is the correct behavior for an adaptive distillation method.

The higher **policy entropy** of Relay-OPD versus OPD and FastOPD is also meaningful: teacher intervention at failure points exposes the student to different trajectory prefixes than it would have generated alone, increasing exploration of the reasoning space.

## Key Notes of This Paper

### The Handoff Trigger in Detail

The trigger criterion φ(h) is designed to be:
- **Label-free**: no ground truth labels, verifier calls, or reward model queries required
- **Causally correct**: it fires only when the teacher would change direction, not generically whenever teacher and student disagree
- **Symmetric in design**: K in K_S(h) controls the sensitivity. Large K (student needs most of its top tokens to not be reflection tokens) makes the trigger more conservative; small K makes it hair-trigger

The choice of reflection tokens ℛ = {But, Wait, However, No, Actually, ...} captures a specific semantic category: tokens that signal "I was wrong, let me reconsider." These are the linguistic markers of reasoning self-correction in chain-of-thought models trained with RL. They serve as a proxy for "the teacher thinks the current reasoning direction should change."

### The Optimization Objective

Equation 9 is the PPO-style clipped loss applied to relay trajectories:

$$\mathcal{L}_\text{Relay} = -\mathbb{E}\!\left[\frac{1}{N}\sum_t \min\!\bigl(\rho_t A_t,\, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\bigr)\right]$$

where A_t^Relay = log π_T(z_t | h_t^z) - log π_̄θ(z_t | h_t^z) and ρ_t = π_θ(z_t | h_t^z) / π_̄θ(z_t | h_t^z).

The clipping is critical: without it, teacher legs (where z_t is generated by the teacher) could produce arbitrarily large advantages for tokens the student initially assigns very low probability. Clipping at 1 ± ε limits the magnitude of each per-token update, preventing the student from overcommitting to the teacher's correction.

The key insight is that the entire relay trajectory — both student legs and teacher legs — participates in optimization. Teacher legs are not just scaffolding; the student is directly trained to reproduce them. This is what makes "baton passing" work: the student learns not just to avoid the failed prefix, but to generate the corrective reasoning step that follows it.

### Speculative Decoding as a Unification Tool

The use of speculative decoding to unify student and teacher generation is elegant. In standard speculative decoding:
- Student (draft model) generates a candidate token
- Teacher (target model) verifies with acceptance probability α_t = min(1, π_T(a^S_t) / π_̄θ(a^S_t))
- On rejection, resample from the residual: q_t(v) ∝ [π_T(v) - π_̄θ(v)]₊

In student legs, the target is the student itself (π^tgt = π_̄θ), so α_t = 1 always — every draft is accepted, reducing to ordinary sampling. In teacher legs, standard speculative rejection sampling runs. The state machine switches between these modes based on φ(h). This means the authors got teacher-intervention trajectories "for free" within an existing speculative decoding framework.

## Limitations

- **Math-only evaluation**: All experiments are on mathematical reasoning benchmarks. Whether the handoff trigger generalizes to code, scientific reasoning, or multi-step tool use is unknown
- **Reflection token vocabulary is heuristic**: The ℛ set is manually specified and may not capture all reasoning-redirect signals, or may capture some non-redirect uses of these tokens
- **4B teacher required**: The method needs simultaneous access to both teacher and student during training. For very memory-constrained pipelines (single consumer GPU), keeping a 4B teacher loaded alongside a 1.7B student in float32 may be impractical
- **Single training epoch**: All methods train for 1 epoch; longer training dynamics are not explored
- **No ablation of reflection token vocabulary**: The sensitivity to the exact composition of ℛ is not tested

## Future Work

Authors' suggested directions:
- Extending to non-math domains and agentic tool-use tasks
- Exploring oracle-label-free evaluation for harder reasoning problems

Promising directions this work opens:
- **Progressive relay budget reduction**: Start with a high M, L budget early in training (high intervention when student is weak) and anneal toward lower budget as the student improves — implementing the self-limiting behavior more explicitly
- **Cross-scale trigger calibration**: The threshold K in the trigger criterion could be adapted based on the student's current policy entropy, creating a fully adaptive intervention rate
- **Relay-OPD for code generation**: The reflection-token paradigm has a natural analogue in code — comments like `# Error: this approach won't work` could serve as code-domain handoff triggers
- **Multi-teacher relay**: Different teachers could take over at different failure types — a math specialist for arithmetic failures, a logic specialist for deduction failures

## Implications for Edge / On-Device Deployment

The implications for small model deployment are significant:

1. **50% shorter training trajectories directly translates to 50% less compute** for an online distillation step. If distillation is done periodically on-device to adapt to user behavior, this halves the adaptation cost.

2. **Relay-OPD does not require labels during training**. The handoff trigger φ(h) uses only teacher and student logits — no ground-truth answer, no verifier, no reward model. This makes it applicable in domains where labeled data is expensive or unavailable.

3. **The method is compatible with speculative decoding at inference time**. A student-teacher system trained with Relay-OPD naturally produces a student that can work alongside the teacher in a speculative decoding setup at deployment — the student's ability to recognize and recover from failure prefixes makes it a better draft model.

4. **The training efficiency gains are especially valuable for 0.6B models**: For the smaller student (0.6B), training trajectory length drops from 6,900 to 2,490 tokens — a 63.9% reduction. At the sub-1B scale, every token of training compute saved directly extends the feasibility of on-device continuous learning.

## Links

[Original Paper](https://arxiv.org/abs/2607.26057)
