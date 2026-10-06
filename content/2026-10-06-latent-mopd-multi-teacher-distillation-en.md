Title: Latent-MOPD: Training One Small Model to Beat Its Specialist Teachers
Date: 2026-10-06
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Latent-MOPD trains a 1.5B student to surpass its domain-specialist teachers by supervising both token predictions and internal hidden states from multiple frozen teachers, using a crossfade schedule and domain-pure batch organization.

## Why This Paper Matters

Knowledge distillation is the primary route to capable small language models: take a large, expensive model (or several) and transfer its competence into a smaller, deployable student. But most distillation methods have a ceiling: the student learns *what* the teacher predicts (token distributions), but not *how* the teacher computes those predictions — the intermediate reasoning states buried in the hidden layers.

This matters especially when the goal is to combine multiple specialists. Modern practice produces domain-specific RL-tuned models: one optimized for mathematical reasoning, one for code, one for logic. Each is excellent in its domain. Running all three at once is three times the inference cost, three times the memory, and complex to orchestrate on-device. The natural solution is to distill all three into one compact model.

The challenge is genuinely hard. Averaging teacher output distributions (uniform mixing) dilutes each specialist's confident signal. Naively mixing representation supervision from multiple teachers causes training collapse. And learning from your own generated responses (on-policy distillation) rather than fixed training data is crucial for capability transfer but adds further complexity.

Latent-MOPD solves all three problems with a single, principled method: route each student rollout to its specialist teacher, then supervise both the token predictions *and* the hidden states of that teacher — while carefully organizing when and how each type of supervision is applied.

The result: a 1.5B parameter student that surpasses the per-benchmark best of its three 1.5B specialist teachers on 5 of 9 benchmarks, trained in only 62 gradient steps.

## Core Technical Contribution

### The Setup: On-Policy Multi-Teacher Distillation

The framework starts from on-policy distillation (OPD): rather than training on a fixed dataset, the student generates its own responses, and the teacher provides supervision on those self-generated responses. This prevents the student from being penalized for reasonable responses that happen to differ from teacher demonstrations.

For $N$ frozen teachers $\{\pi_{T_i}\}_{i=1}^N$, each prompt $x$ carries a domain label $d$ selecting which specialist evaluates the student's response. The student samples $\hat{y} \sim \pi_\theta(\cdot | x)$, and the selected specialist $\pi_{T_d}$ evaluates it.

### The Token Channel: Domain-Routed MOPD

Averaging teacher distributions is problematic because specialists disagree on out-of-domain tokens. If the math teacher is uncertain about code tokens and the code teacher is uncertain about math tokens, averaging washes out the confident signal from whichever teacher is appropriate.

MOPD solves this by hard routing: each prompt goes to exactly one teacher, determined by domain label. The student learns from the teacher-to-student log-probability ratio over the student's own top-$k$ candidates:

$$\mathcal{L}_{OPD}^{route} = -\frac{1}{|M|}\sum_{t \in M}\sum_{v \in V_{k,t}} \text{sg}[A_{t,v}] \log p_{\theta,t}(v)$$

where $V_{k,t}$ is the top-$k$ support of the student's fixed scoring distribution at position $t$, $A_{t,v}$ are advantages computed from teacher-to-student log-probability ratios (normalized by domain standard deviation), and $\text{sg}$ is stop-gradient. $k=16$ in the experiments.

The key insight: the student learns to assign higher probability to tokens that the relevant teacher favors more than the base student does — a relative, domain-specific correction rather than an absolute imitation.

### The Representation Channel: Late-Layer Alignment

Beyond token predictions, each teacher's hidden states encode *how* it processes the input — intermediate reasoning steps, feature extractions, and computation patterns that influence but don't appear in the final distribution.

For a selected layer pair, the representation loss at token position $t$ is:

$$\ell^{route}_{rep,t} = \frac{\lambda}{d_T}\left\|\nu\left(g_\psi(z^S_t)\right) - \text{sg}\left[\nu(z^{T_d}_t)\right]\right\|_2^2$$

where:
- $z^S_t$ = student hidden state at position $t$ (from the selected layer)
- $z^{T_d}_t$ = teacher hidden state for the active specialist $d$
- $g_\psi: \mathbb{R}^{d_S} \to \mathbb{R}^{d_T}$ = projection (identity when same architecture family; shared trainable linear map when teacher/student have different hidden widths)
- $\nu(z) = z / \max(\|z\|_2, \epsilon)$ = L2 normalization
- $\text{sg}$ = stop-gradient on the teacher target

**Which layers to align?** This is architecture-dependent. For same-family teachers (same architecture, different RL fine-tuning), Centered Kernel Alignment (CKA) shows high correspondence through most layers with divergence near the output. The paper aligns the **last three layers** — where the teachers have developed their most domain-specific representations, and where alignment signal is richest.

For cross-family teachers (different scale, e.g., 7B teachers distilling into a 1.5B student), CKA shows a "middle-layer trough" of poor correspondence. Only the **final layer** is aligned, since it plays a common functional role (feeding directly into the LM head) regardless of architecture.

**Storage efficiency**: Selecting 3 of 28 layers requires 89.3% less storage than aligning all layers — 0.28 GiB vs 2.63 GiB for 16,384 positions in BF16.

### The Crossfade Schedule

A critical discovery: sustained representation supervision causes problems. In cross-family settings, keeping the representation loss active throughout training degrades below the starting point. The solution is a **crossfade**: start with representation supervision dominant, then gradually shift to token-only supervision on a per-teacher schedule.

At global step $s$, with $u$ completed updates for the currently active teacher and window $T_w$:

$$\mathcal{L}^{(s)} = \alpha(u)\,\mathcal{L}_{OPD}^{route} + \beta(u)\,\mathcal{L}_{rep}^{route}$$
$$\alpha(u) = \min\!\left(1, \frac{u}{T_w}\right), \quad \beta(u) = \max\!\left(0, 1 - \frac{u}{T_w}\right)$$

The representation weight $\beta$ starts at 1 and decays to 0 over $T_w$ per-teacher steps; the token weight $\alpha$ ramps up simultaneously. Crucially, each teacher has its own counter — the crossfade clock resets when the domain rotates, so each teacher's representation supervision follows its own independent schedule.

$T_w = 10$ steps same-family, $T_w = 7$ cross-family, over 62 total training steps.

### Domain-Pure Batch Organization

Perhaps the most surprising finding: mixing samples from different domains within a single optimizer update causes catastrophic collapse when using representation supervision. An experiment with same-family all-layer representation-only supervision:
- **Domain-pure batches** (32 samples, all from one domain): stable training, reaches 91.7 on MATH-500
- **Interleaved batches** (same samples, mixed domains per update): collapses to 9.8 at step 30 (vs. 84.1 baseline)

The intuition is that different teachers impose conflicting gradient directions on the shared student backbone when mixed in one update. Domain-pure batches ensure each update receives coherent supervision from a single specialist's representation targets.

## Comparison to Prior Work

| Method | Representation supervision | Multi-teacher | On-policy | Same-family Norm |
|--------|---------------------------|---------------|-----------|-----------------|
| Uniform averaging | ✗ | ✓ (averaged) | ✓ | 0.xx |
| MOPD-style (token-only) | ✗ | ✓ (routed) | ✓ | 0.90 |
| OPRD-style (all layers, rep-only) | ✓ (all 28) | adapted | ✓ | 0.64 |
| Rep-only (last 1) | ✓ (last 1) | adapted | ✓ | 0.64 |
| Rep-only (last 3) | ✓ (last 3) | adapted | ✓ | 0.73 |
| Latent-MOPD (OPRD-style, const.) | ✓ (all 28, joint) | ✓ | ✓ | 0.94 |
| **Latent-MOPD (default, crossfade)** | ✓ (last 3, crossfade) | ✓ | ✓ | **1.05** |

The Norm metric measures progress relative to the base student (0) and the per-benchmark best teacher envelope (1). A score of 1.05 means the student exceeds the best-teacher envelope, outperforming the strongest individual teacher on most benchmarks despite having the same parameter count.

### Cross-family results (7B teachers → 1.5B student)

| Benchmark | MOPD-style | Latent-MOPD | Teacher best |
|-----------|-----------|-------------|-------------|
| AIME24 | 36.5 | **39.4** | ~40+ |
| MBPP | 55.4 | **57.9** | — |
| GYM (logic) | 29.2 | **34.1** | — |
| BBH | 45.8 | **49.5** | — |

Norm rises from 0.16 (MOPD-style) to 0.26 (Latent-MOPD) in the cross-family setting — the absolute gains are smaller because the 7B teacher has a hidden width 2.3× the student, but representation alignment still adds consistent value.

## Reading the Results

**The 5-of-9 surpass is the headline**: The same-family student outperforms the per-benchmark best among its three teachers on BBH, MuSR, MBPP, MBPP+, and Minerva — spanning all three domains. The student has the same parameter count as each teacher (1.5B) but covers all domains from a single model. This is not averaging the teachers but genuinely exceeding them.

**Minerva: 33.5 → 36.3 over MOPD-style**. Minerva requires multi-step mathematical reasoning including physics and quantitative reasoning. The representation supervision specifically helps here — the student learns not just what answer the teacher predicts but the intermediate computational pattern used to arrive at it.

**GYM (logic): 29.2 → 34.1** in cross-family settings, a 17% relative gain over token-only distillation. Logic tasks may benefit particularly from representation alignment because logical inference requires specific attention patterns that manifest in hidden states but not necessarily in the output distribution.

**62 training steps**: The entire distillation runs in 62 gradient steps — processing 1,984 prompt presentations, generating 7,936 responses. This is remarkably data-efficient. The explanation is the combination of on-policy generation (student is always learning from its own current ability level) and representation supervision (richer signal per update than token distributions alone).

**General capability retention**: On three benchmarks not in the training pool (ARC-Challenge, WinoGrande, HellaSwag), the student stays within 1.1 points of the base — no catastrophic forgetting of general abilities.

## Key Notes of This Paper

### The Crossfade Formula in Depth

The crossfade schedule deserves careful attention because it reveals something important about what representation supervision does and doesn't provide.

$$\alpha(u) = \min\!\left(1, \frac{u}{T_w}\right), \quad \beta(u) = \max\!\left(0, 1 - \frac{u}{T_w}\right)$$

At $u=0$: $\alpha=0, \beta=1$ — pure representation supervision, no token prediction guidance.
At $u=T_w$: $\alpha=1, \beta=0$ — pure token supervision, representation loss disabled.
At $u=T_w/2$: $\alpha=0.5, \beta=0.5$ — equal weighting.

The key insight from LastOPD (which Latent-MOPD extends to multi-teacher): representation supervision is most valuable early in training, when the student's representations are far from the teacher's and the signal to "move toward" the teacher's internal computation is strongest. Later, when the student has adopted the teacher's computational patterns, the token-level signal is cleaner to follow because the representations are already aligned.

The per-teacher clock means the schedule resets when domains rotate. Under a 3-domain rotation with $T_w=10$: domains cycle math → code → logic → math → ... Each time the math teacher becomes active, the crossfade restarts for math's representation, regardless of where code and logic are in their own schedules. This prevents one teacher from having its representation supervision "exhausted" while another still relies on it heavily.

**Why domain-pure updates are necessary at the representation level**: Consider two teachers $T_{math}$ and $T_{code}$ with hidden states $z^{T_{math}}$ and $T_{code}$. A student batch mixed from both domains produces gradients:
$$g_{math} = \nabla_\theta \ell(z^S, z^{T_{math}})$$
$$g_{code} = \nabla_\theta \ell(z^S, z^{T_{code}})$$
These gradients pull the student's representations in different directions. At the token level, this tension is manageable because token distributions are compact (softmax over vocabulary). At the representation level, hidden states live in high-dimensional space and the gradient conflict can be severe — the student is simultaneously trying to produce the math teacher's layer-28 representation for one input and the code teacher's layer-28 representation for another. With token supervision, routing isolates teachers; with representation supervision, routing is insufficient — batch isolation is required.

### The CKA Diagnostic

The paper uses Centered Kernel Alignment (CKA) to determine which layers to align. CKA measures representational similarity between two sets of hidden states on the same inputs:

$$\text{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

where $X, Y$ are matrices of hidden states from two models on the same inputs. CKA=1 means the representations are identical up to rotation; CKA=0 means orthogonal.

For same-family teachers (shared architecture, different RL fine-tuning): CKA is high (>0.8) through most layers and drops in the last 4-5 layers. This means the teachers have developed similar low-level representations but diverged in their domain-specific computations near the output — exactly the layers where alignment signal is richest.

For cross-family teachers (different scale and training): CKA shows a "middle-layer trough" — similarity drops in intermediate depths and recovers at the final layer. This suggests the models learn different intermediate representations but converge to similar final-layer abstractions that feed the output head. Only aligning the final layer exploits this structure without imposing alignment where there's no natural correspondence.

## Limitations

1. **Domain coverage**: Only three domains (math, code, logic) are tested. Real-world deployment needs covering many more domains — whether the domain-pure batch organization scales to 10+ domains without memory or scheduling issues is unknown.

2. **Teacher availability at training time**: All teachers must be loaded simultaneously to provide routing and supervision for their respective domains. For 3 × 1.5B teachers, this requires ~9B parameters resident during training. Scaling to larger teachers (7B each → 21B total) significantly increases hardware requirements.

3. **Fixed training budget**: All experiments use exactly 62 gradient steps. The paper doesn't explore whether longer training or annealing strategies further improve performance, or whether there's a risk of overfitting to the training domains.

4. **Parameter count parity**: The student always has the same parameter count as each individual teacher. The case of using distillation to *compress* — producing a 0.5B or 1B student from 7B teachers — is not explored.

5. **No verifier rewards**: The training uses no verifier (outcome reward) signals — only the teacher's distributional and representational guidance. Combining Latent-MOPD with RL reward signals might yield further gains.

## Future Work

**From the authors:**
- Extending the method to broader multi-teacher setups beyond three specialists
- Investigating longer training runs and their effect on the crossfade dynamics

**Additional promising directions:**

- **Compressed SLMs from larger specialists**: The most direct application for on-device deployment is using Latent-MOPD to distill a 7B or 13B specialist into a 1B student. The cross-family results suggest this is feasible; scaling experiments are the natural next step.

- **Dynamic domain routing**: The current implementation uses fixed domain labels. A learned router (like a mixture-of-experts gating network) could dynamically allocate prompts to specialists, potentially enabling unsupervised multi-domain distillation without domain labels.

- **Representation supervision as a warm-start**: The crossfade design suggests representation supervision is most useful early. Could Latent-MOPD's first phase be used as a fast warm-start for conventional RL fine-tuning, seeding the student with better-shaped representations before task-specific optimization?

- **Scaling to more teachers**: With 3 teachers, domain-pure updates allocate ~20 gradient steps per teacher. With 10 teachers and the same total budget, each teacher gets 6 steps — is that enough? Adaptive budgeting where more update steps go to teachers where the student is weakest could maintain effectiveness.

- **Combination with quantization**: If the goal is edge deployment, the distilled 1.5B student will likely be quantized. Does the representation alignment survive 4-bit or 2-bit quantization, and does distillation from a representation-supervision target improve robustness to quantization?

## Implications for Edge / On-Device Deployment

**Multi-domain capability in a single compact model**: The central promise for on-device deployment is eliminating the need to switch between multiple specialist models. A smartphone assistant handling math tutoring, code completion, and logical reasoning currently must either use three separate models (infeasible on-device) or accept degraded per-domain quality from a generalist. Latent-MOPD's 1.5B student beats individual specialists while occupying only ~3 GB at bf16 — well within the memory envelope of modern mobile devices (8–12 GB LPDDR5).

**62-step distillation is practical**: Creating a specialized student model takes 62 gradient steps over ~2,000 prompt presentations. This opens the possibility of personalized on-device specialization: a base SLM shipped on-device could be fine-tuned post-deployment using Latent-MOPD with cloud-based specialist teachers, without requiring the end device to host those large teachers during the 62-step distillation process (which could run on-cloud and push only the delta to the device).

**No inference overhead**: The projection $g_\psi$ and the teacher models are used only at training time. The deployed student has standard transformer architecture with no additional inference components. The SLM runs at full speed post-deployment.

**Combination with quantization and pruning**: A 1.5B student quantized to 4-bit requires ~750 MB. At 2-bit (if methods like KronQ are applied), ~375 MB. These sizes are compatible with neural processor units (NPUs) on flagship mobile chips. The representation supervision during distillation may help the student develop more quantization-friendly representations — a hypothesis worth testing.

## Links

[Original Paper](https://arxiv.org/abs/2610.02381)
