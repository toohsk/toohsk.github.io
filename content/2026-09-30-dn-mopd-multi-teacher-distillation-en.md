Title: Beyond Teacher Assignment: Domain-Normalized Multi-Teacher On-Policy Distillation
Date: 2026-09-30
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: DN-MOPD fixes a critical flaw in multi-teacher distillation where domain feedback imbalance causes the student to over-learn easy signals while ignoring hard ones, recovering most of the lost mathematics gain by rescaling each domain's feedback by its measured spread.

## Why This Paper Matters

Multi-teacher distillation is one of the most promising techniques for building capable small language models. The idea is elegant: reinforcement learning can specialize a large language model into multiple expert teachers — one excellent at mathematics, one at code, one at instruction following. Then, instead of training a student on a single generalist teacher, you use all the specialists simultaneously, routing each prompt to its relevant expert and training the student on the aggregated feedback.

This should work better than any single teacher. A math specialist knows more about math than a generalist; a code specialist knows more about code. The student should absorb all of these advantages simultaneously.

But in practice, it doesn't work as expected. This paper diagnoses why.

**The core insight:** different domains produce feedback signals with wildly different statistical properties. Instruction-following feedback (from prompts like "write a haiku about autumn") is highly variable — the teacher can have widely divergent opinions across similar prompts. Mathematics feedback is tightly concentrated — the teacher's token-level log-probabilities cluster around a narrow range of values. When you sum these signals together to train a student, the high-variance domain dominates. The student spends its gradient budget learning to satisfy the instruction-following specialist while effectively ignoring the math specialist's harder, narrower signal.

This is not a routing problem. You can perfectly identify which teacher is relevant to each prompt. The problem is that correct routing, by itself, says nothing about how strongly each teacher's feedback should influence the shared student.

## Core Technical Contribution

### Multi-Teacher On-Policy Distillation (MOPD)

MOPD operates as follows. Given a set of domain-specialized teacher models {π_T^d} where d ∈ {math, code, instruction, ...}, and a student model π_θ:

1. The student generates a response y to a prompt x from domain d: y ~ π_θ(· | x)
2. The relevant specialist π_T^d provides token-level feedback: for each token y_t, compute the log-probability log π_T^d(y_t | x, y_{<t})
3. The advantage at each token is: A_t = log π_T^d(y_t | x, y_{<t}) - log π_̄θ(y_t | x, y_{<t})
4. The student is updated via a KL-regularized policy gradient, weighted by these advantages

In standard MOPD, the advantages from all domains are summed directly into a single optimization objective. There is no normalization: a math advantage of 0.3 and an instruction-following advantage of 0.3 are treated as equivalent signals.

### The Feedback Imbalance Problem

The paper's key empirical finding: instruction-following advantages have a standard deviation several times larger than math advantages. Across a batch of mixed-domain prompts, the gradient is dominated by the high-variance domain even when math prompts are equally represented.

This explains why MOPD students fail to surpass single-teacher students on mathematics despite having access to a math specialist. The math signal is being diluted by the sheer volume of the instruction-following signal's spread.

### Domain-Normalized MOPD (DN-MOPD)

The fix is conceptually simple but requires careful implementation. DN-MOPD rescales each domain's advantages by the measured standard deviation of that domain's feedback:

$$A_t^{\text{DN}} = \frac{A_t^d}{\sigma_d}$$

where σ_d is the running standard deviation of advantages from domain d, estimated online during training.

The effect: each domain contributes equally to the student's gradient budget in expectation, regardless of the raw variance of its feedback signal. The math specialist's concentrated signal is amplified; the instruction-following specialist's diffuse signal is attenuated. The routing decides *which* teacher teaches; the normalization decides *how much* its feedback counts.

The authors validate that the gain comes specifically from **turning down instruction-following feedback** rather than turning up math alone. When they test with fixed domain weights close to the values DN-MOPD measures, performance is comparable — confirming that the normalization is doing the right thing, not accidentally discovering a magic weighting.

### Implementation

DN-MOPD runs on top of existing MOPD infrastructure with minimal overhead:
- Maintain per-domain exponential moving averages of advantage variance
- Divide advantages by the corresponding domain's running standard deviation before computing gradients
- No additional forward passes, no architectural changes, no new hyperparameters beyond the EMA decay rate

The simplicity is notable — this is a statistical correction applied at the gradient level, not a new training algorithm.

## Comparison to Prior Work

**Baselines:**
- SFT: standard supervised fine-tuning on mixed-domain data
- Single-teacher MOPD: train with the best single specialist (math specialist, typically)
- Standard MOPD: multi-teacher on-policy distillation without normalization

**Models tested:** Qwen3.5 at three sizes (the paper doesn't specify exact sizes but implies small, medium, and large variants of Qwen3.5).

**Results (6 public benchmarks, 3 random seeds, 2 answer-length limits):**
- DN-MOPD improves average score over standard MOPD at **every tested model size**
- DN-MOPD recovers most of the mathematics accuracy lost in standard MOPD vs. single-teacher training
- Instruction-following performance is maintained (not sacrificed)
- Results are consistent across all 3 seeds, confirming statistical reliability

The key comparison: MOPD does not beat the best single-teacher setup. DN-MOPD does. This is the result that validates the method — it achieves the original goal of multi-teacher distillation that MOPD failed to deliver.

## Reading the Results

The most important number is not any single benchmark score — it's the fact that **DN-MOPD recovers most of the math gain** without sacrificing instruction following. This resolves what looked like a fundamental tradeoff: you couldn't have both capabilities simultaneously with MOPD.

The controls with fixed domain weights are particularly informative. When the authors manually set weights close to the values DN-MOPD measures (high weight for math, lower weight for instruction following), performance is comparable to DN-MOPD. This is a clean validation: the normalization is simply measuring and applying the right weighting, not discovering some other effect. It also means practitioners who know their domain variance ratios ahead of time could apply fixed weights directly.

The consistency across 3 seeds is significant. Multi-teacher distillation is notoriously sensitive to random initialization and batch ordering. DN-MOPD's reliability suggests the normalization is addressing a systematic structural issue (domain feedback scale), not an optimization quirk.

The two answer-length limits (short and long) test whether the effect is specific to a generation length regime. Consistency across both suggests the feedback imbalance is a fundamental property of the domain specalists' policies, not an artifact of particular generation lengths.

## Key Notes of This Paper

### The Core Formula

The normalization formula is:

$$A_t^{\text{DN}} = \frac{\log \pi_T^d(y_t \mid x, y_{<t}) - \log \pi_{\bar\theta}(y_t \mid x, y_{<t})}{\sigma_d}$$

Think of this as whitening the advantage distribution per domain. Without normalization, the gradients from different domains arrive at different scales. With normalization, each domain contributes approximately unit-variance gradient signal. The student can no longer "accidentally" prioritize one domain simply because that domain's feedback happens to be more spread out.

The denominator σ_d is the key innovation. It must be estimated online because the advantages change throughout training as the student improves. Using a running EMA of the advantage variance gives a stable, low-lag estimate that adapts as the student's policy evolves. Early in training, when the student is far from all teachers, advantages are large. Later, when the student has converged toward the teachers, advantages shrink. The normalization adapts to this throughout training.

### Why This Matters for Multi-Teacher Learning

There's a deeper principle here. Multi-teacher setups are common beyond LLMs — ensemble distillation, multi-task learning, and federated learning all face similar gradient scale mismatches. The MOPD case is a clean instance where the mechanism is identifiable: the teachers were trained on different tasks with different loss landscapes, producing feedback signals at different scales.

DN-MOPD's solution is domain-specific variance normalization. But the underlying principle — measure the statistical properties of each source's gradient contribution and normalize accordingly — generalizes broadly. This paper provides a concrete recipe for one important case.

## Limitations

- **Requires variance estimation**: Online estimation of per-domain advantage variance adds minor computational overhead and introduces the EMA decay rate as a new hyperparameter
- **Domain boundaries must be known**: The method requires knowing which specialist is relevant to each prompt — the routing itself is not learned in this work
- **Evaluated only on Qwen3.5**: Results may not generalize across architectures with different training distributions or teacher specialization methods
- **Three domains tested**: Math, code, and instruction-following. Whether normalization is sufficient for more heterogeneous domain sets (e.g., 10+ specialists) is not explored
- **No analysis of normalization dynamics**: How σ_d evolves during training and whether instabilities arise in early training are not discussed

## Future Work

Authors' directions:
- Extending to larger domain sets beyond the three primary skills tested
- Exploring adaptive routing that co-evolves with normalization

Promising directions this work opens:

- **Cross-architecture transfer**: Does DN-MOPD's measured weighting for a Qwen3.5 setup transfer to other architectures? If σ_d is a property of the task, the weights should be architecture-independent
- **Hierarchical multi-teacher**: As the student improves, some teachers become redundant. A schedule that gradually removes low-utility teachers while maintaining normalization could improve training efficiency
- **Federated multi-teacher**: In privacy-preserving SLM training, different organizations train specialists on their private data. DN-MOPD's normalization approach could enable combining these specialists without sharing raw feedback signals — each organization provides normalized advantages
- **Teacher merging vs. teacher normalization**: DN-MOPD requires maintaining multiple separate teacher models during training. An alternative: first merge the teachers using model merging techniques (e.g., TIES, DARE), then distill from the merged model. Comparing these approaches directly would clarify when multi-teacher setups are necessary

## Implications for Edge / On-Device Deployment

DN-MOPD has direct implications for how SLMs are trained for deployment:

1. **Better multi-skill SLMs**: On-device assistants need to handle math, code, and instructions — all the skills MOPD tried to combine. DN-MOPD finally makes the combination work. The result is a single small model genuinely capable across all three rather than one that excels at instruction following while underperforming on math.

2. **Training efficiency**: The normalization is applied post-hoc to existing MOPD infrastructure. Organizations already using MOPD for SLM training can adopt DN-MOPD with minimal engineering changes. This is not a new architecture — it's a calibration step on an existing training pipeline.

3. **Smaller student models remain viable**: One risk of multi-teacher distillation is that the student needs to be large enough to absorb multiple skill sets. DN-MOPD's recovery of math performance in small student models suggests that the bottleneck was signal quality, not student capacity. This is encouraging for sub-1B SLM training.

4. **Spec-decoding compatibility**: A student trained with DN-MOPD to be proficient in multiple domains is a better speculative decoding draft model — its math drafts are more reliable for a math-specialist verifier, its code drafts more reliable for a code verifier. Domain-balanced capability directly improves speculative decoding acceptance rates for mixed workloads.

## Links

[Original Paper](https://arxiv.org/abs/2609.35347)
[GitHub](https://github.com/LiXin97/DN-MOPD)
[Project Page](https://lixin.ai/DN-MOPD/)
