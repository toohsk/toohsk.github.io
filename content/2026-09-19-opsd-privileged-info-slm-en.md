Title: What Does Privileged Information Actually Add to On-Policy Distillation for Small Language Models?
Date: 2026-09-19
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: A systematic study using Qwen3-1.7B and SmolLM3-3B reveals that the primary driver of improvement in on-policy self-distillation is cross-mode transfer (thinking-enabled teacher → direct-response student), with privileged reasoning references adding only modest additional benefit — and that switching from direct-response to thinking-enabled student rollouts actually hurts performance.

## Why This Paper Matters

On-policy self-distillation (OPSD) has emerged as a powerful technique for improving language model capabilities without external data. The core idea: train a "teacher" version of the model on the same task but with access to privileged information — a worked solution, a complete reasoning trace, a polished answer — and use its token distributions to supervise the "student" version on a harder task.

For small language models specifically, OPSD is particularly attractive because it offers a path to significant capability improvement without requiring larger models or expensive new data collection. The question driving this paper is sharp and underappreciated: **how much of the OPSD benefit actually comes from the privileged reference, and how much comes from the distillation process itself?**

This matters for several reasons:

1. **If the privileged reference is responsible for most gains**, then the bottleneck is reference quality — better worked solutions, more complete traces, higher-quality polished answers. Research should focus on obtaining better references.

2. **If distillation itself is responsible**, then the bottleneck is the training dynamics — the teacher-student gap, the rollout strategy, the KL weighting. Research should focus on optimization.

3. **If the student's rollout format matters more than the reference**, then choices made about how to sample student trajectories during training could dominate everything else.

The paper's findings are surprising and important on all three dimensions.

## Core Technical Contribution

### AMPLE-Math: A Controlled Distillation Benchmark

The central methodological contribution is **AMPLE-Math**, a suite of 5,319 mathematical problems designed specifically to disentangle the effects of privileged references from the effects of distillation mechanics.

The key design decision: each problem has **six reasoning views** that all share the same final answer but differ in how the reasoning is presented:

1. **Answer only** — just the final numerical result, no reasoning
2. **Direct response** — standard solution presentation
3. **Scratchpad** — informal working notes, intermediate steps
4. **Chain-of-thought** — structured step-by-step reasoning
5. **Polished solution** — edited, clean presentation with clear logical flow
6. **Complete trace** — full thinking-enabled reasoning including revision and self-correction

By comparing these views under identical experimental conditions (same student model, same training setup, same evaluation), the authors can isolate what each type of privileged information contributes above and beyond baseline distillation.

### The Matched Reference-Free Baseline

The paper introduces a crucial control: **reference-free distillation (RFD)**, where the teacher is the same thinking-enabled model but without any privileged reference at generation time. The teacher generates what it would naturally predict given only the problem, and its distributions supervise the student.

RFD captures the signal that comes purely from the teacher-student **mode mismatch**: the teacher uses extended reasoning (thinking-enabled, System 2), while the student produces direct responses (System 1). The cross-mode transfer from thinking to direct-response provides a natural form of implicit supervision even without explicit references.

By comparing each reference type against this RFD baseline, the paper can cleanly measure the **marginal contribution** of each reference type above and beyond the mode-transfer baseline.

## Comparison to Prior Work

Previous OPSD studies evaluated their methods as holistic systems — teacher with privileged reference versus no distillation, or distillation versus RL alone. This made it impossible to attribute gains to specific components.

AMPLE-Math enables the first controlled ablation of reference type at fixed training conditions:

| Reference Type | Qwen3-1.7B improvement | SmolLM3-3B improvement |
|---|---|---|
| No distillation (RL only) | baseline | baseline |
| Reference-free distillation (RFD) | ~most gains | ~substantial gains |
| Answer only | ≈ RFD | ≈ RFD |
| Chain-of-thought | slightly > RFD | slightly > RFD |
| Polished solution | strongest gain (Qwen) | moderate gain |
| Complete trace | ≈ polished (Qwen) | +2pp over RFD at step 50 |

The key finding: **reference-free distillation accounts for the majority of improvement** in both model families across both in-domain and external benchmarks (including holdout problems not seen during OPSD training).

## Reading the Results

**The most important finding is about what happens when you change the student rollout format.** At the same training checkpoint, replacing short direct-response student rollouts with long thinking-enabled student rollouts **turns gains into losses** in both Qwen3-1.7B and SmolLM3-3B.

This is striking: same model, same teacher, same reference quality — just a different format for sampling student trajectories — and the outcome flips from improvement to degradation. This implies that the format mismatch between thinking-enabled teacher and direct-response student is **load-bearing for the improvement**. The teacher's thinking traces provide a form of implicit curriculum that helps the student organize its direct-response outputs; when the student also uses thinking traces, this curriculum signal is lost (or inverted).

**The polished solution is the strongest explicit reference for Qwen.** Among the reference types, a polished solution — an edited, clean proof rather than a raw trace — provides the most marginal benefit over RFD for Qwen3-1.7B. This suggests that presentation quality matters more than reasoning depth for this model family at this scale.

**Complete traces help SmolLM3-3B more than Qwen3-1.7B.** SmolLM3-3B shows a measurable +2pp improvement from complete traces at step 50 of training that Qwen models do not show to the same degree. This may reflect SmolLM3-3B's different architecture and pretraining, where extended traces provide genuinely novel reasoning structure rather than just refining existing capabilities.

## Key Notes of This Paper

### The Cross-Mode Transfer Hypothesis

The paper's theoretical framework for explaining its results is the **cross-mode transfer hypothesis**: OPSD improves access to existing reasoning capabilities through parameters shared by direct-response and thinking-enabled inference.

Formally, let $\theta$ be the model parameters. The thinking-enabled mode $\pi_{\theta}^{\text{think}}$ and the direct-response mode $\pi_{\theta}^{\text{direct}}$ both use the same $\theta$ but differ in their system prompt and generation style.

The teacher supervises the student with:
$$\mathcal{L}_{\text{OPD}} = \mathbb{E}_{x \sim \pi_{\theta}^{\text{direct}}} \left[ \text{KL}(\pi_{\theta_T}^{\text{think}}(\cdot|x) \| \pi_{\theta_S}^{\text{direct}}(\cdot|x)) \right]$$

The key insight: the KL between the thinking-enabled teacher and the direct-response student is **asymmetrically informative**. The teacher's thinking mode has access to solution pathways that the student's direct mode has not yet learned to shortcut. The KL loss gradually teaches the student to compress thinking paths into direct responses — a form of **implicit distillation of reasoning shortcuts**.

When the student also uses thinking-enabled rollouts, this asymmetry disappears. The teacher and student are now both in thinking mode, and the KL loss measures only distributional differences within the same reasoning style. The cross-mode compression signal is gone.

### The Matched Loss Intervention

The paper includes a subtle but important "matched loss intervention" experiment in Qwen: manually changing the token-level supervision weight on specific reasoning tokens while leaving everything else identical.

Strikingly, the authors find that **significant changes to token-level supervision can leave student behavior largely unchanged** — the aggregate accuracy shifts very little even when specific token probabilities are substantially altered.

This challenges the common assumption that dense token-level KL loss translates directly into dense behavioral supervision. Much of the KL signal may be absorbed into irrelevant parts of the distribution (rare tokens, discourse markers) rather than the tokens that drive correct reasoning.

## Limitations

1. **Mathematics domain only.** AMPLE-Math covers mathematical reasoning. Whether the cross-mode transfer hypothesis generalizes to code, factual QA, or instruction following is not studied.

2. **Two model families.** Qwen3-1.7B and SmolLM3-3B may not represent the full diversity of SLM behavior under OPSD. Models trained with different pretraining recipes may show different reference sensitivities.

3. **Fixed teacher throughout training.** The teacher is frozen at a single checkpoint. Whether the marginal contribution of references changes if the teacher is also updated during student training is not explored.

4. **Step 50 horizon.** The SmolLM3-3B result (+2pp from complete traces) is reported at training step 50. Whether this advantage persists at convergence is unclear.

## Future Work

**Suggested by the authors:**
- Extending the reasoning view decomposition to code synthesis and instruction following to test generality
- Studying how reference type sensitivity changes across model scales (e.g., does polished solution remain the strongest reference type at 7B?)
- Developing reference quality metrics that predict marginal OPSD benefit

**Additional promising directions:**
- **Reference-free distillation as a standalone training paradigm:** If RFD accounts for most gains, it may be worth developing it as a first-class training method that does not require reference generation pipelines at all.
- **Dynamic mode switching during student training:** Rather than fixing the student rollout format (direct-response), could adaptive switching between formats during training capture benefits from both?
- **Probing what the cross-mode transfer signal teaches:** Mechanistic interpretability tools could identify which attention heads or MLP layers are modified by RFD, giving concrete insight into what "compressed reasoning shortcuts" look like in weights.

## Implications for Edge / On-Device Deployment

1. **Reference generation is not the bottleneck.** For practitioners fine-tuning SLMs for on-device use, this paper's message is operationally liberating: you do not need to invest heavily in high-quality reference generation pipelines. Reference-free distillation — cheap to run, requiring no external data or annotation — captures most of the available gain.

2. **Rollout format selection is critical.** When setting up an OPSD training pipeline for edge deployment, the student rollout format (direct-response vs. thinking-enabled) is a more impactful design choice than reference quality. The direct-response format is also preferable for inference efficiency: thinking-enabled responses generate more tokens, which is a latency and power concern on device.

3. **1.7B and 3B models benefit from OPSD.** The study validates that even very small models (1.7B, 3B) receive measurable improvement from OPSD, suggesting that on-device model training can leverage this method without requiring teacher models that are orders-of-magnitude larger.

## Links

[Original Paper](https://arxiv.org/abs/2609.20612)
