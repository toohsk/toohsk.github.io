Title: Self-Improvement with Auxiliary Rewards: Learning from Verified Traces and Evaluating Transfer
Date: 2026-11-04
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Auxiliary trajectory scoring signals derived from token-level entropy statistics improve small language model self-improvement beyond binary answer correctness, yielding up to +2.08 percentage points on math reasoning benchmarks and transferring to critique and scientific reasoning tasks without task-specific training.

## Why This Paper Matters

Modern reasoning post-training for language models relies on a seemingly clean signal: the answer is either correct or not. This binary feedback, known as outcome-level or answer-level reward, powers the reinforcement learning pipelines behind models like DeepSeek and QwQ. The elegance of RLVR (Reinforcement Learning with Verifiable Rewards) is that you don't need expensive human annotation — if you can check the answer automatically, you have a training signal.

But answer-level rewards have a fundamental blind spot: **they cannot distinguish among reasoning traces that all reach the correct answer**. If a 1.7B-parameter model generates eight solutions to an AIME problem, some of which are concise and logically tight while others meander with redundant steps before landing on the same answer, a binary reward treats all correct traces identically. The model receives the same training signal regardless of trace quality.

This matters especially for small language models. With limited capacity, the model is more sensitive to the quality of training examples. Reinforcing low-quality but correct traces can teach the model to adopt inefficient reasoning patterns. And in self-improvement settings — where the model trains on its own outputs without access to reference answers — this problem compounds: the model bootstraps its own reasoning quality, so the quality of feedback determines the ceiling of improvement.

This paper asks directly: **when answer-level rewards cannot distinguish successful solutions, can auxiliary trajectory preferences improve learning and downstream transfer?**

## Core Technical Contribution

### The Insight: Token-Level Entropy Encodes Reasoning Quality

The key observation underlying this work comes from prior research on information density in language models: successful reasoning trajectories tend to show **locally smooth but globally non-uniform information flow**. That is, a good reasoning trace doesn't spike wildly in token-by-token entropy (local smoothness), but its entropy does vary meaningfully across different phases of the solution (global variation). Poor traces either exhibit local noise (uncertainty fluctuates erratically step by step) or stay completely flat (the model is either always confident or always uncertain throughout).

Building on this, the paper introduces two auxiliary scoring metrics derived entirely from the model's own token-level predictive distributions — no external annotator needed.

### SC: Self-Certainty

Self-Certainty (SC) scores a response by the **mean predictive confidence** of its tokens. Specifically, using the top-25 renormalized probability distributions at each position, it summarizes confidence across the full response. Traces where the model is consistently confident are preferred. This captures trace quality in terms of decisiveness: a model that "knows what it's doing" produces consistently high-probability tokens.

### UID-Conj: Uniform Information Density Conjunction

UID-Conj is the more novel signal. It measures two properties of the entropy trajectory — a sequence of per-position entropy values computed from the model's predictive distribution along the response:

**Global variation (G):** The response's entropy trajectory is divided into 32 relative-position bins. The variance of bin-averaged entropy values measures how much the information density changes across phases of the response. High variance means different reasoning stages have meaningfully different certainty levels — consistent with structured problem-solving.

**Local smoothness (S):** The negative mean absolute difference between adjacent bins measures how abruptly entropy changes from one bin to the next. High smoothness means the entropy profile transitions gradually, without erratic spikes.

Both features are transformed into empirical midrank quantiles within each prompt's response group (online) or via empirical CDFs on a calibration split (offline):

$$u_i = Q_x(G_i), \quad v_i = Q_x(S_i)$$

The UID-Conj score is the **product of these quantile ranks**:

$$q_i^{\text{Conj}} = u_i \cdot v_i$$

This product requires a trace to rank highly on **both** dimensions to receive a high score — it's a conjunction in the logical sense. A trace that varies globally but is locally noisy gets penalized on v; one that is smooth but flat gets penalized on u. Only traces that are simultaneously globally varied and locally smooth receive high scores.

Intuitively, this selects for reasoning traces that exhibit structured phase transitions (different sections at different certainty levels) with smooth within-section progression — the hallmark of organized, deliberate problem-solving.

### Online TTRL Integration

In the online setting, the paper uses TTRL (Test-Time Reinforcement Learning), which adapts the model on test problems without reference answers by using majority voting as a proxy for correctness. The reward for each response is:

$$R_i = 2g_i - 1 + \lambda \cdot \text{clip}(q_i - \bar{q}_x, -\delta, \delta)$$

where $g_i = 1$ if the response agrees with the majority vote, $\bar{q}_x$ is the mean auxiliary score among majority-consistent responses, and the clip bounds the auxiliary adjustment to $\pm \lambda\delta = \pm 0.05$. This preserves the dominant answer-level ordering while adding fine-grained differentiation among responses that all support the majority answer. The small $\lambda = 0.1$ ensures the auxiliary signal supplements rather than overrides the correctness signal.

Advantages are computed via GRPO (Group Relative Policy Optimization) on the combined rewards.

### Offline Extensions

**DPO preferences:** For each prompt, the highest-scored correct response is paired against the lowest-scored correct response. Both are correct — the preference reflects trace quality, not answer correctness. This enables Direct Preference Optimization without needing incorrect responses.

**Weighted SFT:** All correct responses are retained, but softmax-weighted by auxiliary score with temperature $\alpha$:

$$w_i = m_x \cdot \frac{\exp(\alpha q_i)}{\sum_{j \in \mathcal{C}_x} \exp(\alpha q_j)}$$

The mean weight within each prompt remains 1, so the total effective sample size is preserved; only the emphasis shifts toward higher-scoring traces.

## Comparison to Prior Work

The closest prior work is **Intuitor**, which uses SC as the sole reward in GRPO and reports generalization from math to out-of-domain tasks. This paper's key extension is studying whether entropy-trajectory structure (UID-Conj) adds signal beyond confidence (SC), and evaluating transfer to critiquing and scientific reasoning — not just math.

Compared to **process reward models**, this approach requires no step-level annotation. Compared to **length preference** methods, it focuses on information-theoretic trace structure rather than output length.

Baselines:
- Base Qwen3-1.7B (no adaptation): 25.51% macro-average on the math transfer benchmarks
- TTRL (unmodified, majority-vote reward only): ~25.5% (slight improvement)
- TTRL + SC: 27.54% (+2.03pp over base)
- TTRL + UID-Conj: 27.59% (+2.08pp over base)
- Offline uniform SFT (correct responses, equal weight): baseline
- Offline weighted SFT + UID-Conj: +1.93pp over uniform (95% CI [0.44, 3.44])

## Reading the Results

The headline numbers are modest: 1.93–2.08 percentage points of improvement. For math competition problems like AIME, this is not negligible — these benchmarks are designed to discriminate between models at the frontier, and consistent gains with tight confidence intervals are meaningful.

More important than the magnitude is the **transferability**. The model is adapted on AIME2026 problems — but UID-Conj improves performance on HMMT2026, MinervaMath, CriticBench, and LabBench2 without any task-specific training. This suggests the auxiliary signal is not just teaching the model to pattern-match AIME solutions; it is improving the underlying quality of reasoning traces.

The Qwen3-4B results reveal an interesting failure mode of standard TTRL: the model over-specializes to AIME2026 (accuracy rises from 55.33% to 60.00%) but performance on HMMT2026 and MinervaMath drops — a classic trade-off between in-distribution and out-of-distribution performance. UID-Conj mitigates this: the model's AIME2026 accuracy is preserved at the base level, but HMMT2026 and MinervaMath improve to 13.94% and 31.40%. This "less is more" result is the strongest evidence that UID-Conj is regularizing the training distribution, not just memorizing.

Critique transfer (CriticBench +1.24pp verdict accuracy, +1.58pp incorrect-class F1) suggests that well-structured mathematical reasoning — as selected by UID-Conj — transfers to the meta-task of evaluating the correctness of others' reasoning.

## Key Notes of This Paper

### The UID-Conj Formula in Detail

The entropy trajectory idea deserves careful unpacking. For a response $y$ of length $T$ tokens, define the per-token entropy:

$$H_t = -\sum_{v \in \text{top-25}} p_t(v) \log p_t(v)$$

where $p_t$ is the renormalized top-25 distribution at position $t$. The trajectory is then binned into 32 equal-length segments, with each bin's value being the mean entropy of its tokens.

**Global variation G:** Population variance of these 32 bin values. High G means the response has phases with meaningfully different entropy levels — e.g., a setup phase at high entropy followed by a confident computation phase. This matches the intuition that good reasoning has distinguishable structure.

**Local smoothness S:** Negative mean absolute difference between adjacent bins, i.e., $S = -\frac{1}{31}\sum_{k=1}^{31} |H_{k+1} - H_k|$, where $H_k$ is the bin mean. High S (near zero) means entropy transitions gradually, without sudden jumps. Sudden jumps indicate the model oscillating between confident and uncertain in unpredictable ways.

The product $q = u \cdot v$ of quantile ranks enforces that **both conditions must hold**. If a trace scores in the 90th percentile on G but the 20th percentile on S, its UID-Conj score is 0.90 × 0.20 = 0.18 — far lower than a trace scoring 0.70 × 0.70 = 0.49. This geometric mean structure penalizes traces that excel on one dimension while failing on the other.

### Why This Works at Training Time (But Not Always at Selection Time)

A subtle finding: UID-Conj's advantages as a **training signal** are stronger than its advantages as an **inference-time selector**. At selection time (picking the best of N responses), SC outperforms UID-Conj by an independent judge's assessment. But UID-Conj produces better-trained models. The paper hypothesizes that UID-Conj selects for less redundant responses — responses the judge rates lower on validity but which contain less noise for the model to learn from. Training on focused, non-redundant correct traces may build better generalization than training on verbose-but-valid ones.

## Limitations

- **Task dependence**: SC is better for trace quality and instruction following; UID-Conj is better for math training and transfer. No single auxiliary signal dominates across all settings. Practitioners need to choose based on the target task.
- **Scale of improvement**: 1–2 percentage points are statistically significant but practically small. The method is a supplement to, not a replacement for, more data or larger models.
- **Domain coverage**: All experiments use math and science benchmarks. Whether auxiliary trajectory preferences transfer to code generation, dialogue, or multi-modal tasks is unknown.
- **Model family**: Only Qwen3-1.7B and Qwen3-4B are studied. The relationship between model size/family and the efficacy of entropy-based scoring remains unexplored.
- **Limited adaptation data**: The online setting uses only 30 AIME2026 problems — extremely low-data. Performance under more problems, or different curriculum ordering, is not evaluated.
- **Theoretical foundation**: The UID hypothesis (smooth, varied entropy = good reasoning) is an empirical observation, not a proven property. It may not hold for domains with fundamentally different reasoning structures.

## Future Work

The authors suggest studying auxiliary preferences for a broader range of tasks and exploring combinations of multiple auxiliary signals. Several additional directions seem promising:

**Curriculum design via auxiliary scoring**: If UID-Conj selects training examples that build better generalization, applying it to pre-training data selection (not just self-generated traces) could improve base model quality before any RLVR fine-tuning.

**Adaptive λ scheduling**: The paper uses a fixed λ=0.1 throughout training. An adaptive schedule that increases reliance on auxiliary signals as the model's answer-level performance saturates could extract more signal from trace quality differences.

**Cross-model auxiliary scoring**: The auxiliary scores are computed from the model being trained. Using a frozen, stronger model to score traces (teacher-scored UID-Conj) could provide a more reliable quality signal — especially for smaller models whose own entropy signals may be noisier.

**Combining UID-Conj with process reward models**: Process rewards provide step-level feedback; UID-Conj provides trajectory-level quality assessment. Integrating both could give fine-grained and holistic supervision simultaneously.

**Extending to code generation**: Code correctness is verifiable (tests pass or fail), making it another RLVR domain. Code traces have natural phase structure (problem analysis → algorithm design → implementation → debugging) that might map well onto the UID-Conj entropy pattern.

## Implications for Edge / On-Device Deployment

This paper is directly relevant to the practical challenge of improving small models for edge deployment without requiring more data, more compute, or a larger model.

The core value proposition: **given a fixed small model and a fixed number of training problems, auxiliary trajectory scoring extracts more learning signal from existing correct traces**. This is valuable when:

- **Data is scarce**: Edge applications often operate in specialized domains (medical, industrial, legal) where high-quality training problems are expensive or private. Getting more out of 30 problems, as this paper demonstrates, is practically significant.
- **Model size is fixed**: Hardware constraints on smartphones, wearables, or embedded systems impose hard memory and compute budgets. If the model must be 1.7B parameters, improving its reasoning quality via training signal refinement — rather than scaling — is the only path.
- **Self-improvement without cloud dependency**: TTRL without reference answers means the model can potentially improve on its own test-time experience. For edge devices that process private data locally, this enables adaptation without sending data to a server.
- **Transfer reduces fine-tuning overhead**: The result that AIME math adaptation transfers to CriticBench and LabBench2 suggests that improving mathematical reasoning quality generalizes. For resource-constrained devices, a single fine-tuning pass that improves multiple downstream tasks is far more efficient than task-specific fine-tuning.

The 2.08pp improvement on a 1.7B model corresponds to a qualitative difference in reasoning reliability for competition-level problems. At this model size, such gains from better training signal quality — rather than more parameters — represent exactly the kind of efficiency improvement needed for capable on-device reasoning.

## Links

[Original Paper](https://arxiv.org/abs/2610.41296)
[HuggingFace Paper Page](https://huggingface.co/papers/2610.41296)
