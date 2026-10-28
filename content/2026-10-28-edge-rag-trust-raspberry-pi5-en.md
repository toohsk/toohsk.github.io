Title: Trust Mechanisms for Offline Generative AI at the Edge
Date: 2026-10-28
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: A new framework for running Qwen2.5 3B/7B models fully offline on Raspberry Pi 5 with signed knowledge packs and retrieval-score gating achieves near-perfect gap detection while exposing hard limits of on-device trust.

## Why This Paper Matters

Most generative AI deployments assume you have the internet. But an enormous fraction of IoT devices don't: agricultural sensors, industrial PLCs, hospital equipment in shielded rooms, smart meters, and border infrastructure all operate in intermittent, restricted, or fully disconnected environments. When connectivity is unavailable, today's LLM-powered assistants simply stop working.

The harder problem isn't just running inference offline — it's trusting the output. A cloud-connected model can (in principle) be updated with current knowledge and monitored for misbehavior. An air-gapped device running stale weights and a local knowledge base has no such backstop. What happens when the knowledge base is outdated? What if it's been tampered with? What if the model confidently answers questions it shouldn't?

This paper is one of the first to tackle offline generative AI at the edge as a **trust engineering problem**, not just an inference efficiency problem. The authors study Qwen2.5 models (0.5B, 3B, 7B) running fully local RAG on a Raspberry Pi 5 with no network access, and ask: what combination of mechanisms can make the system appropriately reliable — answering when it knows, abstaining when it doesn't, and resisting knowledge base tampering?

The Raspberry Pi 5 is not a toy target. At ~$80, it represents the upper end of deployable IoT hardware that still fits in constrained budgets. Getting useful language model behavior on it — and making it trustworthy — has real implications for the huge number of deployments where cloud connectivity is a non-starter.

## Core Technical Contribution

### The Connectivity-Regime Framework (C0–C4)

The paper's first contribution is a taxonomy of five connectivity regimes:

- **C0**: Permanent air gap — no connectivity ever
- **C1**: Occasional connectivity — e.g., monthly maintenance window
- **C2**: Intermittent connectivity — sporadic, unpredictable
- **C3**: Restricted connectivity — specific ports/protocols only
- **C4**: Always-connected — standard cloud deployments

This framework is important because it forces designers to think about which trust mechanisms are even possible given the connectivity regime. A digital signature can be verified offline. An NLI-based verifier can run locally. But a live model update, a cloud-side output audit, or a real-time knowledge check all require connectivity.

### Signed, Versioned Knowledge Packs

The update channel for disconnected regimes is a key design decision. The authors propose **signed, versioned knowledge packs**: curated sets of embedded passages with a cryptographic signature from a trusted authority. This means:

1. Knowledge updates arrive as discrete, versioned packages (not streaming crawls)
2. Each pack is signed — the device can verify integrity before accepting it
3. Versioning enables rollback if a pack causes problems

The empirical result is striking: verifying a signed pack takes approximately **4.5 ms** on the Pi 5, versus about **29 minutes** to embed its passages (i.e., build the local vector index). Signature verification is essentially free; it's the embedding computation that costs.

However, the paper also shows that cryptographic signing alone is insufficient. Out of nine tested update attacks, signature and hash checks alone accepted six. Only layered verification (signature + content validation + output monitoring) rejected them all. And critically: **a validly signed poisoned pack passes every pack check**. If the signing authority is compromised, or if an attacker can inject at the source, cryptographic trust fails completely.

### Retrieval-Score Gate for Knowledge Gap Detection

A major challenge in edge RAG is knowing when the local knowledge base doesn't contain the answer. The paper proposes an **empirically calibrated retrieval-score gate**: a threshold on the retrieval score (cosine similarity between query and retrieved passages) that predicts whether the model has sufficient knowledge to answer.

Results:
- **AUROC 0.97** for detecting "out-of-scope" questions (where the knowledge base genuinely lacks relevant information)
- **AUROC 0.68** for detecting "in-topic unanswerable" questions (where the knowledge base discusses the topic but the specific question is unanswerable from the evidence)

The first result is excellent — the gate is nearly perfect for questions on topics the knowledge base doesn't cover. The second result reveals a harder problem: when the knowledge base has related but not directly relevant passages, the model often retrieves them with medium-high scores and generates confident-sounding wrong answers. The gap between 0.97 and 0.68 is a fundamental limitation of retrieval-score-based gating.

### Deterministic Output Validator

For malformed outputs — outputs that don't match expected format or structure — the authors deploy a **deterministic output validator** that converts them to explicit abstentions. Result: zero observed loss of correct answers, while all malformed outputs become explicit "I don't know" responses.

The NLI-based verifier (which checks whether the generated answer is entailed by the retrieved passages) adds little improvement in practice, suggesting that the bottleneck is not entailment failure but rather the model generating plausible-sounding answers when it shouldn't.

## Comparison to Prior Work

The closest prior work includes:

| System | Device | Model Size | Connectivity | Trust Mechanisms |
|--------|--------|-----------|--------------|-----------------|
| This paper | Raspberry Pi 5 | 0.5B–7B | Air-gapped | Signed packs + retrieval gate + output validator |
| mllm-NPU | Snapdragon | 7B | Online optional | None studied |
| FlexInfer | Mobile GPU | 7B–13B | Online | None studied |
| TZ-LLM | Mobile ARM | 7B | Online | Secure enclave |
| Standard cloud RAG | Server | 70B+ | Required | Cloud-side monitoring |

No prior work specifically addresses the combination of (a) fully offline operation, (b) a structured knowledge update channel with integrity guarantees, and (c) empirical calibration of refusal/abstention mechanisms for edge SLMs.

The Qwen2.5 model family is used as the backbone. The 3B model — a true SLM — is the most practically interesting result: it achieves **the same gold-answer containment rate as the 7B model** on SQuAD 2.0, while running at less than half the latency and estimated energy.

## Reading the Results

The headline numbers come from 300 SQuAD 2.0 questions evaluated across three model sizes (approximate 0.5B, 3B, 7B) and three context window sizes, with fully local retrieval:

- **3B model gold-answer containment = 7B model gold-answer containment** — a major SLM finding. For this task (factoid QA with good retrieval), scaling from 3B to 7B provides no containment benefit.
- **7B model far more often exceeds token-F1 ≥ 0.5** — the 7B model generates fuller, more complete answers even when both contain the key facts.
- **Five retrieved passages lowers correctness from 75% to 52% vs. three** — more context is not better. With five passages, noise retrieval drowns out the signal. This is a critical practical finding for edge RAG design.
- **Poisoned pack attack**: On 30 previously-correct questions, injecting poisoned content reduced correct answers to zero, with 14 of 30 returning the injected answer verbatim.

The 75%→52% correctness drop with 5 vs. 3 retrieved passages is perhaps the most actionable result: for edge RAG deployment, **use fewer retrieved passages**. The instinct to retrieve more for better coverage is wrong in practice.

## Key Notes of This Paper

### The Retrieval-Score Gate Formulation

The gate computes a retrieval score $s = \max_{i} \text{sim}(q, p_i)$ where $q$ is the query embedding and $p_i$ are the retrieved passage embeddings. A threshold $\tau$ is calibrated on a held-out set:

$$\text{decision}(q) = \begin{cases} \text{answer} & \text{if } s \geq \tau \\ \text{abstain} & \text{if } s < \tau \end{cases}$$

The threshold $\tau$ is chosen to maximize a weighted combination of recall (answering answerable questions) and precision (abstaining on unanswerable ones). The AUROC metric measures how well the score $s$ ranks answerable above unanswerable at all thresholds, which is why AUROC 0.97 for out-of-scope and 0.68 for in-topic-unanswerable represents a genuine capability gap: the score distribution for in-topic unanswerable questions overlaps substantially with answerable ones.

### The Connectivity Regime Impact on Trust Architecture

The C0–C4 framework implies specific trust architectures:

- **C0 (air gap)**: All trust must be pre-loaded. Signed packs are the only update channel. No NLI verifier that uses an external model can be used. The device must carry everything it needs.
- **C1–C2 (occasional/intermittent)**: Signed packs can be delivered during connectivity windows. Pack verification on reconnection. Time-limited signing keys are feasible.
- **C3 (restricted)**: Specific endpoint allowlisting. Differential updates (only changed passages). Bandwidth-constrained embedding update strategies.

The framework forces a design question that practitioners usually skip: **at which regime are you actually operating?** Getting this wrong means either over-engineering trust mechanisms that connectivity would handle or under-engineering them for true disconnected operation.

## Limitations

The authors are explicit about what their system does not solve:

1. **In-topic unanswerable detection** remains poor (AUROC 0.68). The retrieval-score gate cannot distinguish "I have related context but not the answer" from "I have the answer." This is the central unsolved problem.

2. **Signing authority trust is a prerequisite**. Cryptographic integrity verification is only as strong as the authority that signs the packs. A compromised signing authority defeats all pack-level defenses.

3. **None of the evaluated mechanisms establish the truth of evidence**. The system can detect gaps, validate format, and check signatures — but it cannot verify that the knowledge base contains accurate information to begin with.

4. **Model size tested is limited**. The paper tests up to 7B parameters. On a Raspberry Pi 5, larger models are simply not feasible at practical latencies.

5. **SQuAD 2.0 is a narrow benchmark**. Factoid QA with well-formed questions is a favorable setting. Open-ended reasoning, multi-hop questions, and conversational RAG remain untested.

## Future Work

**From the authors:**
- Better calibration methods for in-topic unanswerable detection
- Lightweight NLI verifiers that provide value at edge compute budgets
- Differential embedding updates to reduce the 29-minute embedding cost on pack updates

**Additional promising directions:**
- **Federated knowledge packs**: Multiple edge nodes contributing local observations that get aggregated, signed, and redistributed without central cloud dependency
- **Temporal validity metadata**: Knowledge pack passages annotated with "valid until" timestamps, enabling the gate to route time-sensitive queries to abstention as packs age
- **Adversarial retrieval hardening**: Training retrieval models that are explicitly robust to the kinds of passage poisoning attacks demonstrated in this paper
- **Smaller specialized models**: Instead of general-purpose 3B models, domain-specific models (e.g., medical RAG, industrial maintenance) fine-tuned at 1B–3B for specific edge use cases

## Implications for Edge / On-Device Deployment

This paper delivers several immediately actionable findings for anyone building offline LLM deployments:

1. **3B models are sufficient for factoid QA at the edge**. The 3B→7B jump buys generation quality but not answer containment. For retrieval-grounded QA on a Pi 5 class device, 3B is the right size.

2. **Use 3 retrieved passages, not 5**. The 75%→52% drop at 5 passages is not a model failure — it's a retrieval failure. Fewer passages means less noise.

3. **Deterministic output validation is free and reliable**. Adding a structural validator that converts malformed outputs to explicit abstentions costs nothing in performance and removes a class of failure.

4. **Cryptographic signing of knowledge packs is table stakes, but not sufficient**. Plan for the signing authority to be compromised. Layered verification is required.

5. **Calibrate your retrieval gate on your domain**. The AUROC 0.97 / 0.68 split shows that generic thresholds may not translate. Domain-specific calibration is needed for production deployments.

For IoT operators, the Raspberry Pi 5 results give a concrete baseline: a ~$80 device running Qwen2.5-3B can answer factoid QA from local knowledge at practical latencies with reasonable reliability — but only if you engineer the trust mechanisms carefully.

## Links

[Original Paper](https://huggingface.co/papers/2610.33732)
