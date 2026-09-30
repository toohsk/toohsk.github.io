Title: Knowing When Thinking Is Not Enough: Teaching Small Reasoning Models to Reason Beyond Their Parametric Knowledge
Date: 2026-09-30
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: FlyBy diagnoses two distinct failure modes in small reasoning models — execution bottlenecks where more thinking helps, and knowledge bottlenecks where it doesn't — then trains 4B/8B models to selectively query stronger models at knowledge bottlenecks, surpassing Qwen3-14B performance at 2.7× lower serving cost.

## Why This Paper Matters

Test-time compute scaling has emerged as one of the most powerful techniques for improving small language model reasoning. The intuition: if a model can't solve a problem in one pass, let it think longer — generate more reasoning tokens, explore multiple solution paths, self-correct. This works remarkably well for many tasks.

But "think longer" is not always the right operation. This paper asks a harder question: **when does additional thinking help, and when does it not?**

The answer has significant practical consequences. If you're deploying a small reasoning model and it's struggling with a math problem, the naive approach — give it more compute, let it generate more tokens — wastes resources when the problem isn't a reasoning failure at all. The model might be failing not because it can't *execute* the reasoning correctly, but because it simply lacks the *knowledge* that would make the correct reasoning path reachable.

The paper formalizes this distinction with empirical interventions across two model families at multiple scales: **execution bottlenecks** (where more thinking and self-reflection can recover the correct answer) versus **knowledge bottlenecks** (where no amount of thinking helps because the relevant facts aren't in the model's parametric memory). Armed with this taxonomy, the authors build **FlyBy**, a selective querying framework that solves the knowledge bottleneck case the right way — by asking for help.

The numbers are striking: FlyBy-4B achieves 45.96% pass@8 on 1,158 hard problems across six benchmarks, surpassing Qwen3-14B (41.64%) while costing 2.7× less to serve.

## Core Technical Contribution

### Diagnosing the Two Failure Regimes

The first contribution is diagnostic. The authors intervene at intermediate reasoning states — mid-trajectory, with the model partially through a problem — and ask: what happens when we force the model to reconsider from this state?

Two distinct outcomes emerge:

**Execution bottlenecks:** The correct solution path *is* reachable from the current reasoning state. Self-refinement (forcing a reflection token, asking the model to reconsider) can recover it. The model "knows" the answer in some sense — it just made an execution error (wrong calculation, wrong application of a known formula). More thinking helps.

**Knowledge bottlenecks:** The correct solution path is *not* reachable from the current reasoning state. No amount of self-refinement changes the outcome. The model consistently fails because the relevant external information (a specific formula, a domain fact, a theorem from outside its training distribution) is not in its parametric memory. More thinking does not help.

A key empirical finding: self-refinement "largely consolidates probability mass onto solutions already reachable from the current state, rather than making new ones reachable." This is the precise characterization of why test-time scaling has limits: it can improve execution quality but cannot expand the knowledge frontier.

### FlyBy: The Selective Querying Framework

FlyBy's design follows directly from the diagnosis:

1. The small reasoning model (sRM) attacks the problem first
2. After some reasoning, it diagnoses its own state: is this an execution bottleneck or a knowledge bottleneck?
3. At a knowledge bottleneck, it formulates a targeted query to a stronger model (the "oracle"), asking specifically for the missing knowledge
4. The stronger model's response is incorporated and the sRM continues reasoning

The oracle is not asked to solve the whole problem — it's asked to supply the missing knowledge. This preserves the on-device model's reasoning agency while surgically addressing the specific gap.

### Training Procedure

Training FlyBy has two stages:

**Stage 1: Supervised Fine-Tuning (SFT) for bootstrapping.** The authors construct training examples where a small model reasons, hits a knowledge bottleneck (identified by comparing to ground truth and oracle), formulates a query, receives oracle information, and successfully completes the problem. SFT teaches the model the basic behavior pattern: reason → diagnose → query → continue.

A key design decision is the **multi-depth query action**: the model can issue queries at different levels of specificity (asking for a hint, asking for a formula, asking for a direct fact). SFT bootstraps all these depths.

**Stage 2: Cost-Aware Reinforcement Learning.** SFT teaches the behavior but not the calibration. RL trains three orthogonal decisions:
- **Whether to query**: Don't query for execution bottlenecks (waste of tokens); do query for knowledge bottlenecks
- **What to ask**: Formulate the most informative, targeted query (broad queries waste the oracle's bandwidth; narrow queries get exactly what's needed)
- **How much to spend**: Each query has a cost (latency, tokens). RL optimizes for accuracy per unit cost, learning when the marginal benefit of querying justifies the cost

The reward function balances problem-solving accuracy against query cost, driving the model toward efficient querying behavior.

### Architecture and Scales

FlyBy is trained on Qwen3-4B and Qwen3-8B base models. The oracle is a larger model (not specified as a single fixed oracle — the framework is oracle-agnostic).

The system treats oracle queries as a finite-rate resource: the model can issue multiple queries per problem, but each incurs a cost. The RL training shapes how often and how efficiently the model uses this resource.

## Comparison to Prior Work

**Baselines on 1,158 hard problems across 6 benchmarks:**

| Method | pass@1 | pass@8 | Serving Cost |
|---|---|---|---|
| Qwen3-8B | 15.31% | — | 1× |
| FlyBy-4B | 16.85% | 45.96% | ~0.37× |
| Qwen3-14B | — | 41.64% | ~2.7× |
| FlyBy-8B | — | 51.81% | ~1× |

Key comparisons:
- **FlyBy-4B vs Qwen3-14B (pass@8):** 45.96% vs 41.64% — a smaller model with selective querying outperforms a model 3.5× its size
- **FlyBy-4B vs Qwen3-8B (pass@1):** 16.85% vs 15.31% — also wins at single-generation evaluation, where there's no ensemble benefit
- **FlyBy-8B (pass@8):** 51.81% — scaling the base model further improves results

The serving cost comparison (2.7× lower for FlyBy-4B vs Qwen3-14B) reflects token generation cost during deployment. FlyBy-4B incurs additional oracle query cost, but this is factored into the comparison — the net cost is still substantially lower.

The 6 benchmarks are described as "hard problems." This is the relevant regime: on easy problems, execution bottlenecks are more prevalent and self-refinement helps more. FlyBy's selective querying is most valuable precisely when the problems are hard enough to surface genuine knowledge bottlenecks.

## Reading the Results

The most striking result is FlyBy-4B's pass@8 superiority over Qwen3-14B. pass@8 measures whether any of 8 independent generations is correct — it reflects the model's "coverage" of the solution space more than single-shot accuracy. FlyBy-4B at 45.96% coverage outperforms Qwen3-14B at 41.64% despite being 3.5× smaller.

This matters for practical deployment in two ways:
1. **Best-of-N sampling** (a common inference strategy for high-stakes queries) benefits from a higher pass@k ceiling
2. **Policy diversity** is preserved: FlyBy's model explores different problem approaches across the 8 samples, rather than repeatedly hitting the same knowledge bottleneck

The pass@1 result (FlyBy-4B at 16.85% vs Qwen3-8B at 15.31%) is arguably even more impressive. pass@1 is the real-world scenario — the user asks a question and gets one response. Winning pass@1 means FlyBy-4B actually outperforms a model twice its size in the single-shot scenario. The selective querying improves not just coverage but first-shot quality.

The scaling behavior (FlyBy-8B at 51.81% pass@8) shows the method scales cleanly with base model size. The 4B→8B jump is substantial: +5.85 points. This suggests the bottleneck diagnosis and querying mechanism benefit from stronger base reasoning capability in the sRM.

## Key Notes of This Paper

### The Two Bottleneck Taxonomy

The paper's core conceptual contribution is formalizing the execution/knowledge bottleneck distinction empirically. The key idea: at intermediate reasoning state h (after some reasoning steps y_{<t}), define the **reachable solution set** R(h) as the set of correct solutions that the model can generate from h in expectation across multiple continuations.

- If h is an execution bottleneck: R(h) contains the ground truth, but self-correction (forcing a reflection token) increases the probability of reaching it. The model "knows" but stumbled.
- If h is a knowledge bottleneck: R(h) does NOT contain the ground truth, regardless of how many continuations are sampled. The model doesn't know, and no amount of reflection changes this.

This isn't just a diagnostic distinction — it's the basis for the entire FlyBy intervention. The system identifies which type of bottleneck has been reached before deciding whether to query.

### The Cost-Aware RL Formulation

The RL objective is roughly:

$$\mathcal{J}(\theta) = \mathbb{E}\left[\text{Accuracy}(y, y^*) - \lambda \cdot C_{\text{query}}\right]$$

where λ is a cost weight and C_query is the cost of queries issued during the reasoning trajectory.

This is a cost-sensitive policy optimization problem. The challenge: the model must learn a binary decision (query or not) and a content decision (what to ask), both under cost pressure. The SFT bootstrap is critical here — without it, RL would face an enormous exploration problem (the model needs to first discover that querying can help, which is rare to stumble upon randomly).

The multi-depth query action is an important design choice. Single-depth querying (always ask the same type of question) doesn't leverage the different types of knowledge gaps. A model that can ask "what is the formula for X?" versus "is it true that Y follows from Z?" versus "what's the approach for this class of problem?" can be much more efficient with oracle bandwidth.

### Why Selective Querying Over Just Using a Bigger Model

The paper implicitly argues against the "just use Qwen3-14B" approach for all queries. The argument:

1. **Cost at scale**: 2.7× lower serving cost is not marginal. For high-volume deployments, this is the difference between a viable and non-viable product
2. **Targeted queries are efficient**: FlyBy doesn't outsource the whole problem to the oracle — it only asks for the specific knowledge gap. The oracle does much less work than solving the problem from scratch
3. **Privacy and latency**: On-device execution of the sRM handles most computation locally; only targeted queries leave the device. This has privacy and latency advantages
4. **The sRM still reasons**: FlyBy's sRM is not a dispatcher — it reasons independently and queries strategically. This means its reasoning capability is actually exercised and improved through training

## Limitations

- **Oracle availability required**: FlyBy requires access to a stronger model at inference time. In truly air-gapped or offline deployments, there is no oracle to query
- **Bottleneck detection may be noisy**: The model must correctly identify knowledge vs. execution bottlenecks at test time. Misclassification (querying during an execution bottleneck, or not querying during a knowledge bottleneck) wastes resources or leaves performance on the table
- **Evaluated on math benchmarks only**: The 6 benchmarks are mathematical reasoning tasks. Whether knowledge bottlenecks exist in the same form for code, scientific reasoning, or factual question answering is not established
- **Oracle cost not fully analyzed**: The paper establishes 2.7× lower cost vs Qwen3-14B for all queries, but the marginal cost of oracle queries varies by problem difficulty — the distribution of query rates is not fully characterized
- **Query quality not robustly analyzed**: How sensitive is performance to the quality of the formulated queries? If the sRM asks poorly-targeted questions, the oracle's answers may not resolve the bottleneck

## Future Work

Authors' suggested directions:
- Extending to agentic tasks beyond mathematical reasoning
- Exploring multi-round query dialogues (iterative clarification with the oracle)
- Oracle-free variants where the "oracle" is a retrieval system rather than a stronger LLM

Promising directions this work opens:

- **Knowledge bottleneck detection as a standalone capability**: Training a lightweight classifier that predicts bottleneck type before committing to a full reasoning trace — this could enable very early exit for problems requiring oracle assistance
- **Federated knowledge querying**: Different sRMs on different devices may have complementary knowledge. A peer-to-peer query network could let devices query each other instead of a centralized oracle, reducing latency and single-point bottlenecks
- **Retrieval-augmented knowledge filling**: Replace the oracle with a RAG system — at a knowledge bottleneck, retrieve relevant documents rather than querying a stronger model. This removes the oracle dependency entirely for fact-based knowledge gaps
- **FlyBy for continual learning**: Each oracle query reveals a knowledge gap. These gaps could be logged and used to construct targeted SFT examples for periodic on-device fine-tuning, gradually reducing the frequency of oracle queries as the sRM learns from its own bottlenecks

## Implications for Edge / On-Device Deployment

FlyBy is designed with deployment economics in mind, and the implications are substantial:

1. **Most reasoning happens locally**: FlyBy's sRM handles the full reasoning trajectory on-device. Oracle queries are the exception, not the rule. For latency-sensitive applications (voice assistants, real-time problem solving), the fast local path handles most queries without network round-trips.

2. **Graceful degradation**: If oracle connectivity is lost, FlyBy degrades to standard sRM reasoning — slower progress on knowledge bottlenecks, but not a complete failure. This is better than fully server-dependent architectures.

3. **Cost model matches deployment reality**: On-device inference costs are roughly proportional to the number of tokens generated by the on-device model. Oracle query costs are proportional to the query size and the oracle's token generation. FlyBy's cost-aware RL directly optimizes this real deployment cost function — it's not a proxy.

4. **4B models with selective knowledge querying can outperform 14B models** that run entirely locally or in the cloud. For edge deployments where 14B is too large to run locally, FlyBy opens up a new point on the accuracy-cost Pareto frontier: a 4B model on-device with targeted cloud queries beats a 14B cloud model on raw accuracy while being substantially cheaper to serve.

5. **Privacy-preserving deployment**: Since the sRM handles most reasoning locally, sensitive problem context stays on-device. Only the specific knowledge query (often a short factual question stripped of sensitive context) leaves the device. This is a significantly better privacy posture than routing all queries to a server-side model.

## Links

[Original Paper](https://arxiv.org/abs/2609.34327)
[GitHub](https://github.com/tally0818/FlyBy)
