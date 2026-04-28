Title: The Cognitive Penalty: When More Reasoning Breaks Edge-Native SLMs
Date: 2026-04-28
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: A rigorous ablation on Qwen-3.5-9B reveals that System 2 (extended chain-of-thought) reasoning catastrophically degrades edge-native SLMs used for DAO governance — a 26.7% cognitive collapse rate and 17x latency penalty vs. System 1 direct inference.

## Why This Paper Matters

The dominant assumption in the LLM literature since chain-of-thought was introduced is monotonically simple: **more reasoning = better answers**. Give the model space to think, and it will think better. This paper directly challenges that assumption in the specific and increasingly practical context of **edge-native SLMs operating under Byzantine Fault Tolerance (BFT) constraints**.

Decentralized Autonomous Organizations (DAOs) manage billions of dollars in on-chain assets. They are increasingly exploring Small Language Models as **constitutional firewalls** — autonomous agents deployed at governance checkpoints to vet proposals and detect semantic social engineering (manipulation through persuasively written proposals that appear legitimate but subtly violate the DAO's stated principles). The edge-native aspect matters: such firewalls must run locally on validators without cloud round-trips to prevent latency attacks and preserve decentralization.

Two questions naturally arise:
1. Should these governance SLMs use System 1 (fast autoregressive) or System 2 (extended deliberative reasoning) inference?
2. Is "thinking harder" about adversarial proposals safer?

The counterintuitive finding: **System 2 reasoning is catastrophically worse on every safety metric in this adversarial setting**.

## Core Technical Contribution

The paper introduces **Sentinel-Bench**, an 840-inference empirical framework that performs a **strict intra-model ablation** on Qwen-3.5-9B. The design is carefully controlled:

- **Same model, same weights**: The ablation toggles only the inference-time compute mode, keeping all parameters frozen. This isolates the effect of reasoning depth from model capacity.
- **Adversarial dataset**: The Optimism DAO governance dataset, containing real and synthetically generated adversarial proposals designed to manipulate the model into incorrect verdicts.
- **Two modes under test**:
  - **System 1**: Standard autoregressive generation — the model generates a verdict directly from the proposal text without extended internal reasoning.
  - **System 2**: Extended latent reasoning — the model is allowed (or prompted) to reason through the proposal step by step before rendering a verdict.

### Key Metrics

- **Adversarial Robustness**: Does the model correctly identify adversarial proposals?
- **Juridical Consistency**: Given the same input twice, does the model render the same verdict?
- **State Finality Time**: How long until the model commits to a verdict?
- **Reasoning Non-Convergence Rate**: How often does extended reasoning fail to converge to any verdict?

## Comparison to Prior Work

Prior work on reasoning in adversarial settings largely assumed that more compute improves robustness. Studies on chain-of-thought, tree-of-thought, and inference-time scaling showed improvements on math, coding, and logical reasoning tasks. The key difference here is the **adversarial governance setting**: the input is crafted to exploit the model's reasoning process, not just test its knowledge.

Previous adversarial ML research focused on input perturbations (adversarial examples in vision) or prompt injection (jailbreaks). Sentinel-Bench introduces a third attack surface: **exploiting the model's reasoning process itself** — adversarial proposals that trigger extended deliberation but ultimately fail to converge, creating a denial-of-consensus attack.

## Reading the Results

The results are stark and deserve careful attention:

| Metric | System 1 | System 2 |
|---|---|---|
| Adversarial Robustness | **100%** | Degraded |
| Juridical Consistency | **100%** | 72.6% |
| Reasoning Non-Convergence | **0%** | **26.7%** |
| State Finality Time | **<13 seconds** | ~220 seconds (17x) |

**The 26.7% Reasoning Non-Convergence (cognitive collapse) rate** is the central finding. More than 1 in 4 adversarial proposals caused System 2 reasoning to fail to reach a definitive verdict. From a BFT standpoint, this is fatal: if 1/3+ of validator nodes fail to converge, consensus breaks.

**The 72.6% juridical consistency** means that even when System 2 does reach a verdict, it reaches the same verdict only 72.6% of the time when the same proposal is presented again. A governance system that would rule differently on the same proposal depending on timing is not trustworthy.

**The 17x latency overhead** introduces a critical vulnerability: **Governance Extractable Value (GEV)**. If an attacker knows that adversarial proposals trigger 220-second deliberations vs. 13-second decisions for normal proposals, they can:
1. Time other governance actions to coincide with deliberation windows
2. Submit multiple adversarial proposals to occupy validator compute, creating a denial-of-service on governance

**Reasoning-Induced Sycophancy** (observed in 1.5% of adversarial trials): In some cases, extended reasoning causes the model to generate very long internal monologues (averaging 25,750 characters) that rationalize *accepting* adversarial proposals it should have rejected. The model "talks itself into" the wrong answer through elaboration — the opposite of what more reasoning should achieve.

## Key Notes of This Paper

### The Sentinel-Bench Evaluation Protocol

The 840 inference trials are structured to enable rigorous comparison:
- Equal numbers of benign and adversarial proposals
- Multiple runs per proposal per mode (enabling consistency measurement)
- Frozen weights throughout (pure inference-time ablation)

The key experimental design principle: by keeping weights fixed and only toggling the reasoning mode, the paper provides a clean measurement of **the causal effect of System 2 reasoning** rather than conflating it with model size or training differences.

### The Cognitive Collapse Mechanism

Cognitive collapse (Reasoning Non-Convergence) occurs when extended reasoning cycles through inconsistent partial arguments without converging. Formally, if we denote the model's internal reasoning state as a sequence $R_1, R_2, \ldots, R_T$, collapse occurs when:

$$\nexists T^* : P(\text{verdict} \mid R_{T^*}) > \tau$$

for any reasonable threshold $\tau$ within the token budget. The model generates reasoning that perpetually reweighs the adversarial proposal's arguments without ever reaching a sufficient confidence margin for a verdict.

The adversarial proposals are designed to present balanced-seeming arguments on both sides. System 1 (intuitive inference) pattern-matches to known good/bad governance patterns and commits quickly. System 2 (deliberative reasoning) attempts to reason through the arguments, gets pulled between them, and may never resolve.

The intuition: **adversarial social engineering is specifically crafted to exploit deliberative reasoning** by constructing plausible arguments for the wrong conclusion. Fast intuitive processing, which relies on pattern matching from training rather than in-context deliberation, is less susceptible to this manipulation.

### BFT Constraints and Why They Make This Worse

Byzantine Fault Tolerance requires that for a system with $f$ faulty nodes, there must be at least $2f+1$ honest nodes that agree. In standard BFT protocols, "faulty" means Byzantine actors. The paper extends this to include **cognitively faulty** nodes — nodes that fail to produce a verdict due to reasoning non-convergence.

If 26.7% of nodes experience cognitive collapse on adversarial proposals, and adversarial actors time proposals to maximize this exposure, the effective BFT fault tolerance is severely degraded. The paper frames this as a **systemic governance vulnerability** that only manifests under System 2 reasoning.

## Limitations

The authors are appropriately candid:

1. **Single model**: The ablation is performed on Qwen-3.5-9B only. Whether the phenomenon generalizes across model families (Llama, Phi, Gemma) is not tested.

2. **Single dataset**: The Optimism DAO adversarial dataset may not represent all DAO governance contexts. Different DAOs (DeFi protocols, NFT platforms, infrastructure DAOs) may have different adversarial proposal patterns.

3. **Static token budget**: System 2 is tested with a fixed token budget for reasoning. Adaptive budgets (stopping early if confidence is reached) might reduce the cognitive collapse rate.

4. **No fine-tuning for the task**: Neither System 1 nor System 2 mode is fine-tuned specifically for DAO governance. A model fine-tuned on governance data might behave differently.

5. **Limited adversarial diversity**: The adversarial proposals are from one source. Truly adaptive adversaries who can learn from the model's responses might find different attack vectors.

## Future Work

**From the authors:**
- Extending Sentinel-Bench to multiple model families and DAO datasets
- Exploring hybrid approaches: System 1 for initial screening, System 2 only when confidence is low
- Formal analysis of GEV attack surfaces in BFT governance systems

**Additional promising directions:**
- **Calibrated reasoning activation**: Train a lightweight classifier to decide *when* System 2 reasoning helps vs. hurts, creating a meta-controller that selectively engages deliberation
- **Adversarially robust chain-of-thought**: Training specifically on adversarial governance datasets with System 2 reasoning enabled to make the model's deliberation process more robust
- **Multi-agent validation**: Rather than relying on a single model's consistency, implement a fast consensus protocol across multiple System 1 instances to detect and handle adversarial proposals collectively

## Implications for Edge / On-Device Deployment

The paper's finding has direct implications beyond DAO governance:

- **Any BFT edge deployment** that requires consensus between nodes should prefer System 1 inference for latency and consistency. The 17x overhead of System 2 is incompatible with real-time consensus protocols.

- **Adversarial input detection** should run in System 1 mode first. If an input triggers extended deliberation, that itself may be a signal that the input is adversarially crafted.

- **Edge SLM resource planning**: System 1's <13 second finality enables predictable compute budgets. System 2's 220-second worst case makes resource planning impossible on constrained hardware.

- **The "just reason more" fallacy**: Developers building edge governance agents should resist the intuition that extended reasoning improves safety. In adversarial settings under BFT constraints, the opposite is empirically true.

The broader lesson: **inference-time compute scaling has context-dependent safety properties**. It helps in benign reasoning tasks but actively creates attack surfaces in adversarial governance scenarios.

## Links

[Original Paper](https://hf.co/papers/2604.16913)
