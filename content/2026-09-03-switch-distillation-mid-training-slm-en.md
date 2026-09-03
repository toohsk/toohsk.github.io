Title: Switch Distillation: Teaching Small LMs to Reason Without Forgetting Facts
Date: 2026-09-03
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Meta's Switch Distillation fixes a previously unknown failure mode of knowledge distillation during mid-training — it boosts reasoning by 1.61–1.71× while preserving 96.7–96.8% of factual recall by routing each token to either KD loss or cross-entropy based on teacher confidence.

## Why This Paper Matters

Training smaller language models (SLMs) from stronger teacher models via knowledge distillation (KD) is one of the most promising paths to capable, deployable AI at the edge. But there's a catch almost nobody knew about: **KD behaves fundamentally differently depending on which stage of training you apply it in**.

Pre-training distillation is well understood — a student learns from scratch while a teacher guides it, and both reasoning and factual recall improve together. But modern LLM training pipelines increasingly use a middle phase called **mid-training**: after pre-training ends, the model continues learning on curated, domain-specific corpora (math, code, instruction-following data) before fine-tuning. This phase is where the model acquires much of its specialized capability.

This paper reveals a critical, previously undiagnosed failure mode: **applying standard forward KL distillation during mid-training helps reasoning but actively slows factual recall acquisition**. The student ends up a better reasoner but a worse fact-knower compared to just training with cross-entropy.

The stakes are high. Mid-training pipelines are now standard. Any team distilling a smaller model at this stage using naive KD is unknowingly trading away factual accuracy.

## Core Technical Contribution

### The Problem: Stage-Dependent Distillation Behavior

Through controlled experiments, the authors show that:

- During **pre-training**: forward KL distillation simultaneously improves reasoning **and** factual recall vs. standard next-token prediction (NTP)
- During **mid-training**: forward KL distillation improves reasoning but **degrades** factual recall acquisition relative to NTP

This is a stark reversal. The same algorithm, applied one training phase later, produces opposite effects on one of the two key capabilities.

### Root Cause: Teacher Confidence Asymmetry

The authors trace this to two asymmetries that converge during mid-training:

**Asymmetry 1 — Teacher confidence by data domain:**
- On *procedural / reasoning* data (math proofs, code, multi-step logic): the teacher is highly confident. Given a reasoning chain, there are relatively few valid next steps. Teacher entropy is low.
- On *knowledge-intensive / factual* data (named entities, dates, definitions): the teacher distributes probability more broadly. Many plausible-looking tokens could follow. Teacher entropy is high.

**Asymmetry 2 — Student's evolving knowledge state:**
- Mid-training students already have some world knowledge from pre-training. They acquire **low-entropy factual associations** (specific names, numbers, facts) *early* in mid-training.
- When the teacher provides a wide, uncertain distribution over factual tokens, it actually competes with the precise, narrow signal the student is naturally learning from the data.

When you apply forward KL distillation uniformly to all tokens, you're asking the student to match a teacher that is:
- **Sharp and correct** on reasoning tokens → helpful signal
- **Broad and uncertain** on factual tokens → noisy, disruptive signal that interferes with natural factual learning

### Switch Distillation

The fix is elegant: **route each token to the appropriate loss based on teacher confidence**.

Formally, for each token $t$ in the sequence, Switch Distillation computes:

$$\mathcal{L}_{\text{switch}}(t) = \begin{cases} D_{\text{KL}}\!\left(p_T(\cdot \mid x_{<t}) \;\|\; p_S(\cdot \mid x_{<t})\right) & \text{if } H\!\left(p_T(\cdot \mid x_{<t})\right) < \tau \\ -\log p_S(x_t \mid x_{<t}) & \text{otherwise} \end{cases}$$

Where:
- $p_T$ = teacher's predicted distribution over the vocabulary at position $t$
- $p_S$ = student's predicted distribution
- $H(p_T)$ = the teacher's predictive entropy: $-\sum_v p_T(v) \log p_T(v)$
- $\tau$ = an entropy threshold (a single hyperparameter)
- The fallback is standard **cross-entropy** (NTP) using the ground-truth next token

**What each term does:**
- $D_{\text{KL}}(p_T \| p_S)$: Forward KL divergence. Forces the student toward wherever the teacher assigns probability. Zero-forcing: if teacher gives token $v$ high probability, student must also give it high probability. This works well when the teacher's signal is sharp.
- $H(p_T)$: Teacher predictive entropy. Low entropy → teacher is concentrating probability on few tokens, confident about what comes next. High entropy → teacher is uncertain, spreading probability widely.
- The routing signal $H(p_T) < \tau$: a binary gate that switches between "trust the teacher" and "trust the data".

The entire mechanism adds negligible computation: entropy of the teacher's output logits is computed as a byproduct of the forward pass.

## Comparison to Prior Work

The paper benchmarks against the main distillation objectives:

| Method | Reasoning | Knowledge & Commonsense | Factual Recall |
|--------|-----------|------------------------|----------------|
| NTP (baseline) | 1.00× | 1.00× | 1.00× |
| Forward KL | 1.55×↑ | 1.10×↑ | **0.88×↓** |
| Reverse KL | 1.38×↑ | 1.05×↑ | 0.94×↓ |
| JSD | 1.42×↑ | 1.07×↑ | 0.91×↓ |
| **Switch Distillation** | **1.61–1.71×↑** | **1.13–1.19×↑** | **96.7–96.8%** |

All prior distillation methods hurt factual recall during mid-training. Switch Distillation is the first to improve reasoning without sacrificing it.

Baselines include forward KL, reverse KL (mode-covering), and Jensen-Shannon divergence (symmetric, less extreme than either). The gains hold across multiple teacher sizes.

## Reading the Results

**The 1.61–1.71× reasoning gain** means that on math/logic benchmarks (likely MATH, GSM8K, or similar), a student trained with Switch Distillation solves roughly 60–70% more problems correctly than a student trained from the same data with just NTP. This is substantial.

**The 96.7–96.8% factual recall** means the student retains nearly all of what pure NTP would have taught it about facts. Forward KL alone drops this to 88%, a significant regression on tasks like trivia, entity extraction, and factual QA.

**The post-training results** are critical for practitioners: after SFT and RLHF fine-tuning, Switch Distillation still maintains 1.25–1.32× reasoning and 1.13–1.20× knowledge/commonsense gains vs. NTP, while the factual recall gap closes completely. This means the mid-training benefit **survives** the fine-tuning phase, making it a durable investment.

## Key Notes of This Paper

### The Entropy Routing Signal

The teacher's predictive entropy $H(p_T)$ is the central mechanism, so it's worth understanding concretely.

**Low entropy scenario (reasoning data):** Imagine the model is mid-way through a math proof: `"Therefore, x = 4 + ..."`. The teacher distributes high probability on `"3"` or `"5"` (integers that complete simple arithmetic), near-zero on most other tokens. $H(p_T)$ is small. The student should trust this signal — the teacher's distribution encodes mathematical structure worth learning.

**High entropy scenario (factual data):** Imagine the text reads: `"The author of this novel was ..."`. The teacher spreads probability across dozens of author names. From the teacher's perspective, many are plausible given the context. $H(p_T)$ is large. Asking the student to match this uncertain distribution teaches it to be uncertain about facts — the opposite of what we want.

### Why Forward KL Is Particularly Harmful Here

Forward KL $D_{\text{KL}}(p_T \| p_S)$ is "zero-avoiding": if $p_T(v) > 0$, the student is penalized for giving $p_S(v) = 0$. When the teacher spreads probability across 50 names, the student is penalized for not assigning nonzero probability to all 50. This forces the student to maintain broad uncertainty over facts — fighting against the sharp, low-entropy factual associations it's naturally learning from seeing specific names appear in specific contexts.

Reverse KL and JSD are somewhat less extreme but still suffer from the same fundamental issue of over-weighting uncertain teacher signals during mid-training.

### The Threshold τ

The paper reports that Switch Distillation is **not highly sensitive** to the threshold τ across a reasonable range. This is important for practical adoption — it means the method doesn't require extensive hyperparameter search. A practitioner can use a conservative threshold (lean toward KD) or a liberal one (lean toward NTP) without large performance swings.

## Limitations

**Single architecture focus:** The experiments use a specific model family and teacher-student size combinations. Whether the entropy threshold τ generalizes across very different architectures (e.g., MoE vs. dense, different tokenizers) is not tested.

**Threshold selection:** While the paper reports low sensitivity, τ is still a hyperparameter. Practitioners need at least a small validation set to tune it; the right threshold may shift across different mid-training corpora with different ratios of reasoning vs. factual content.

**Binary routing:** The switch is hard: either full KD or full NTP. A soft interpolation (weight KD by $1 - H(p_T)/H_{\max}$) might recover even more factual recall while maintaining reasoning gains, but is not explored.

**Pre-training stage:** The analysis of when distillation is harmful is focused on mid-training. Whether early vs. late pre-training shows analogous asymmetries is left open.

**Corpus dependence:** The factual recall vs. reasoning trade-off is measured on a specific mid-training corpus. Corpora with different domain mixes (more code, less encyclopedic text) might show different asymmetry profiles.

## Future Work

**Authors' suggested directions:**
- Extend Switch Distillation to fine-tuning and RLHF stages, where similar teacher-confidence asymmetries may appear
- Explore soft interpolation instead of hard switching
- Study the relationship between mid-training corpus composition and the teacher confidence asymmetry profile

**Additional promising directions:**

*Adaptive thresholds:* Rather than a single global τ, learn a per-domain or per-token-type threshold. A threshold trained on a small held-out validation set could dynamically adjust to the factual vs. procedural ratio of different data sources.

*Teacher-student co-training:* The analysis reveals that teacher and student evolve differently during mid-training. Could we co-train teacher and student in a curriculum that deliberately sequences reasoning-heavy data first and factual data second, aligning their confidence profiles more closely?

*Self-supervised routing:* Instead of relying on teacher entropy, can the student itself estimate which tokens are "factual" vs. "procedural" — e.g., by token type classification — and self-route without a teacher forward pass?

*Cross-stage distillation:* This paper focuses on distilling from a teacher that is already post-trained. What happens when the teacher is itself a mid-training checkpoint? Can knowledge be transferred across students at the same training stage?

## Implications for Edge / On-Device Deployment

Switch Distillation directly impacts how practitioners build SLMs for real-world deployment:

**Better capability balance:** On-device SLMs need both reasoning (to follow instructions, do multi-step tasks) and factual recall (to answer questions correctly, fill in domain-specific knowledge). Standard mid-training KD was quietly hurting one at the expense of the other. Switch Distillation gives you both — critical for general-purpose edge assistants.

**Mid-training is already in the pipeline:** Most production-grade small models (Phi, Gemma, Qwen, SmolLM) already use some form of mid-training. Swapping in Switch Distillation requires minimal engineering change (just adding entropy computation on the teacher's logits and a routing branch) with significant upside.

**Factual recall matters for on-device:** Edge models are often used in contexts with no retrieval augmentation — a mobile assistant needs to "know things" from memory, not from a cloud search. The 88% factual recall of naive KD vs. 96.7% for Switch Distillation is a meaningful gap in user-facing accuracy.

**Post-training durability:** The gains survive fine-tuning, meaning you don't need to change your SFT or RLHF pipeline. You distill better during mid-training, then fine-tune as usual, and deploy a model that is sharper across all axes.

## Links

[Original Paper](https://arxiv.org/abs/2609.01532) | [GitHub](https://github.com/facebookresearch/midtraining-distillation)
