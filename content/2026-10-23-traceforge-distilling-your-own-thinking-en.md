Title: TraceForge: Beating the Teacher by Synthesizing Family-Native Reasoning Traces
Date: 2026-10-23
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: TraceForge trains a generator to synthesize reasoning traces in the student's own model family style from any teacher's answer—bypassing the moat of closed-source reasoning trace restrictions—and achieves +3.7/+7.1 points on HumanEval+/MBPP+ over the same teacher's real traces, because same-family traces reduce catastrophic forgetting during SLM training.

## Why This Paper Matters

Knowledge distillation from large frontier models into small language models (SLMs) depends on one thing above all: access to the teacher's *reasoning*, not just its final answers. A short final answer tells the student what to conclude; a detailed chain-of-thought shows how to get there. The best distillation pipelines—those behind models like DeepSeek-R1—have consistently relied on rich teacher reasoning traces.

The problem: Frontier model providers (Anthropic, OpenAI) now return answers only. The reasoning trace—what the model actually thought—is either hidden or summarized beyond usefulness. The stated justifications are safety and IP protection. The effective result is a moat. You can still distill from open-source models, but the open-source models are behind the frontier, so the ceiling you're distilling toward is lower.

This paper removes the moat entirely, legally. The key claim: **you don't need the teacher's reasoning trace. You can generate a better one yourself.**

## Core Technical Contribution

### The Problem Framing

Let a conversation have five elements: system prompt $S$, user turns $U$, the reasoning trace $R$, tool calls and responses $T$, and the final answer $A$. Frontier models now supply $S, U, A, T$ but not $R$.

TraceForge trains a generator to model:

$$P(R \mid S, U, A, T)$$

That is: given everything except the reasoning trace, produce a reasoning trace that would naturally lead to the given answer, written in the style of the student's own model family.

Crucially, the teacher only appears at inference time, supplying $A$ (and possibly $T$). During training, the generator never sees the teacher at all—it trains entirely on the student family's own data.

### Stage 1: Supervised Generator Training

The generator is a model from the student's own family (Qwen-family in this paper). It is fine-tuned on ~10,000 conversations collected from Qwen3.8-Max, where all five elements $S, U, R, A, T$ are present.

Training objective: maximize $\log P(R \mid S, U, A, T)$, with the loss falling only on $R$. The answer $A$ is part of the input (not the output), so the model learns to write a derivation that *ends at* $A$ rather than restate it. Two important properties follow:

- The generator learns the *form* of its own family's reasoning traces—their rhythm, their hedging style, how they set up sub-problems, when they write code—from real examples, not from a prompt.
- Because the reasoning traces trained on were written for the family's own answers, when the generator is asked to write a trace for a closed-source teacher's answer, it is doing genuine inference, not memorizing.

After this stage, an independent judge (GPT-6.1-SOL) scores generated traces at 0.54 vs. real traces at 0.66. The gap motivates the adversarial stage.

### Stage 2: Critic Training

A critic model—at least as large as the generator—is trained to discriminate real reasoning traces from generated ones. For each of the 10,000 samples, the generator produced one matched negative; the critic is trained until it assigns the positive at least 0.2 higher probability than the negative.

Key lesson from the paper: **the critic must be at least as large as the generator**. A smaller critic fits shortcuts within a few hundred steps, collapses, and gives the generator nothing useful to learn against.

### Stage 3: Adversarial Co-Training

Five rounds of online co-training:

1. Generator produces 5,000 traces
2. Critic scores them against real traces
3. Traces that fool the critic → critic training data (hard negatives)
4. Traces that don't → preference pairs for the generator (DPO-style)
5. Held-out set re-scored by GPT-6.1-SOL

After five rounds: generator alone scores 0.68 (vs. 0.66 for real traces). Best-of-4 with critic filtering reaches 0.82.

### At Inference (Distillation Pipeline)

Given a teacher's answer $A$ to a prompt, the generator produces a family-native reasoning trace $R'$. The student is then trained on $(S, U, R', A, T)$ pairs via standard SFT. The teacher contributed only its answer; the reasoning style is entirely the student's family.

## Comparison to Prior Work

Five corpora were compared, all using the same 5,000 prompts and the same base model (Qwen3.5-9B-Base):

| Corpus | Reasoning Trace Source | Answer Source | HumanEval+ | MBPP+ |
|--------|----------------------|---------------|-----------|-------|
| S1 | Fable 5 (real) | Fable 5 | 64.63 | 52.38 |
| **S2** | **TraceForge (family-native)** | **Fable 5** | **68.29** | **59.52** |
| S3 | Qwen3.8-Max (real) | Qwen3.8-Max | 46.95 | 24.60 |
| S4 | TraceInversion (generic synth) | GPT-6.1-SOL | 3.66 | 6.61 |
| S5 | TraceForge (family-native) | GPT-6.1-SOL | 59.15 | 60.05 |

S1 is the textbook approach—use the frontier model's real traces. **S2 beats S1 by 3.66 and 7.14 points** despite never seeing Fable 5's actual reasoning. The family-native traces are *better training signal* than the real teacher traces.

S3 is the most important control: Qwen3.8-Max is a stronger model than the student, from the same family. It should be a near-perfect teacher. It isn't—it trails by 18 and 28 points. Teacher capability is necessary but not sufficient; trace *style compatibility* matters too.

S4 (TraceInversion) is the previous state of the art for trace synthesis from closed teachers. It generates generic, numbered-step traces with a recognizable synthetic feel. The results (3.66 / 6.61) confirm that surface-level plausibility does not translate to training signal.

S5 closes the argument: same GPT-6.1-SOL answers as S4, TraceForge traces instead of TraceInversion. The gap is 55 and 53 points—entirely attributable to the trace synthesizer.

## Reading the Results

**The training dynamics tell the deeper story.** Table 2 reports every checkpoint:

- S2 and S5 rise monotonically—stable, smooth convergence. The student never needs to "unlearn" its own habits to absorb the training data.
- S1 is unstable: peaks at 66.46 at step 600, drops to 33.54 at step 1200 (verified in two re-runs), recovers to 64.63. A 33-point oscillation on a benchmark is not noise; it's the model fitting a foreign style, overshooting, and recovering.
- S3 reaches 48.78 at step 300 and degrades from there. The in-family model Qwen3.8-Max writes traces in a style the student can absorb—but the capability ceiling is lower, and the final performance reflects it.
- S4 collapses after step 900. The model was fitting a corrupted signal from the start; by the end, outputs are dead loops and repetition.

**The core empirical finding**: Same-family traces reduce catastrophic forgetting. The student treats family-native traces as data it already knows how to process. The adaptation cost is lower, and the capability from the teacher's stronger answers comes through cleanly.

## Key Notes of This Paper

### The Generator's Objective

$$\mathcal{L}_{\text{gen}} = -\log P_\theta(R \mid S, U, A, T)$$

The design choice that makes this work: **A is an input, not an output**. The generator can't just reproduce $A$ in a different voice; it must construct a chain of reasoning that *arrives* at $A$. This forces the model to learn the logical structure of the family's reasoning, not just its surface statistics.

Compare to a model trained to generate $(R, A)$ jointly from $(S, U)$: that model learns to correlate style with correctness, but doesn't have the answer anchoring the trace. The anchor is what makes the generated trace useful for teaching—it has to actually justify the right answer.

### The Adversarial Loop Formula

The adversarial co-training can be seen as iterative preference optimization. At round $r$, define a binary reward: $\text{reward}(R') = 1$ if the critic assigns the generated trace higher than the real one, else $0$. The DPO-style update pushes the generator toward:

$$\max_\theta \mathbb{E}\Bigl[\text{reward}(R') \cdot \log P_\theta(R' \mid S,U,A,T) - \beta \, \text{KL}[P_\theta \| P_{\text{ref}}]\Bigr]$$

This is a standard RLHF objective with a binary reward. What's unusual: the reward model (critic) is being trained simultaneously and updated with hard negatives from the generator's failures. The loop terminates not when the reward is maximized but when the judge score plateaus—a practical stopping criterion that avoids overfitting to an increasingly-fitted critic.

### Why Smaller Critics Fail

A smaller critic has lower representational capacity than the generator. It quickly learns a few surface statistics (trace length, sentence structure, certain phrases) and uses those as shortcuts. From that point, the generator learns to fool those shortcuts rather than actually improve quality. The resulting "improved" traces are stylistically correct by the critic's shallow standards but semantically poor. Lesson: for adversarial training to work, the discriminator must be a genuine challenge.

## Limitations

- Evaluated only on the Qwen family. Family-native traces might work less well if the student family has less distinctive stylistic patterns, or if the generator's training data was small.
- The 10,000 conversation training set is domain-specific (Claude Code trajectories in this case). Generalization to math, scientific reasoning, or multilingual tasks is untested.
- The benefit appears strongly on coding benchmarks; knowledge-based tasks (ARC, BoolQ) show no measurable difference between S1 and S2. The technique is most useful when reasoning style matters.
- The generator inference is an extra compute cost at data-generation time. For large-scale distillation datasets this is non-trivial.

## Future Work

- **Multi-family generalization**: Can a single generator model handle multiple student families (Qwen, Llama, Mistral) by conditioning on the family identity? A cross-family generator would unlock even broader distillation pipelines.
- **Reward-model distillation**: The critic trained here is effectively a reasoning-trace quality model. Distilling it into a lightweight verifier could accelerate best-of-N selection at inference.
- **Math and science traces**: The technique hasn't been tested on MATH, olympiad problems, or scientific reasoning. If the family-native benefit extends there, it could close the gap between SLM and frontier model math capabilities.
- **Self-improvement loop**: Use the student's own improved traces as new data to fine-tune the generator. This would create a closed loop where better traces produce better students, which produce better traces.

## Implications for Edge / On-Device Deployment

TraceForge directly raises the ceiling for on-device SLMs:

1. **Access to frontier-model capability without frontier-model size**: The student (9B) can absorb the reasoning capability of a frontier model (Fable 5, GPT-6.1-SOL) even when that model hides its reasoning. For on-device SLMs targeting coding tasks, this could mean reaching frontier-adjacent capability in a 1–3B model with the right distillation data.
2. **Stable training**: The monotonic convergence of S2 and S5 means fewer GPU hours to reach peak performance. When compute is scarce (as it is for fine-tuning on-device SLMs), training stability is a real cost.
3. **No API policy risk**: S4 (TraceInversion) relied on term-of-service violations. TraceForge is explicitly designed to stay within legal bounds—important for any product deployment where provenance must be auditable.
4. **Data flywheel**: An on-device model could, in principle, generate family-native traces for new user queries (using a small amount of cloud compute for the teacher answer), accumulate them as a personalized dataset, and periodically fine-tune the on-device model. TraceForge makes the trace-generation step viable.

## Links

[Original Paper](https://arxiv.org/abs/2610.29680)
