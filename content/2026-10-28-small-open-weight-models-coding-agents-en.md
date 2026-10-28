Title: Can Small Open-Weight Models Get Real Coding Work Done?
Date: 2026-10-28
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: An empirical systems experience report following 4,165 real coding-agent sessions reveals that small open-weight models (Qwen3-Coder 30B-A3B MoE, 4-bit quantized) can achieve 80–93% task verification rates as coding workers, but only when serving, protocol, KV caching, and behavioral verification are all engineered correctly.

## Why This Paper Matters

Benchmarks for small language models are everywhere. Papers routinely report that a 3B or 7B model "approaches GPT-4" on some MMLU subset, or that a quantized MoE model "passes 90% of HumanEval." What these benchmarks don't tell you is whether the model can actually finish a real piece of software engineering work — meaning code that passes a behavioral test someone else wrote, delivered through a realistic agent protocol, with the surrounding infrastructure doing its job.

This paper takes a different approach: a retrospective engineering experience report from one developer running small open-weight models as coding agents from July to October 2026. The author worked through distributed inference, cloud GPU serving, local Apple Silicon deployment, agent protocol integration, and ultimately verified coding task studies — with actual hidden behavioral checks, not just the model's own claims of success.

Why does this matter for SLMs? Because if small models can genuinely complete real coding work — not just produce plausible code but produce code that passes independent behavioral verification — then the case for on-device and self-hosted LLM infrastructure gets dramatically stronger. The question isn't "what does it score on benchmarks?" but "can a developer trust it with real work?"

The answer from this paper: **yes, with important qualifications**. The primary tested model — Qwen3-Coder 30B-A3B, a mixture-of-experts model with ~3B active parameters, running as a 4-bit MLX checkpoint on a 32GB Apple M1 Pro — achieves 80–93% verified task success rates as a coding worker. But the path to those numbers reveals that "the model" is only one component.

## Core Technical Contribution

This paper is not a new architecture or training method. It is a **systems engineering account** of what it actually takes to make small open-weight models work as coding agents. The contribution is the evidence and taxonomy.

### The Four Success Levels

The paper's most useful framing is a hierarchy of what counts as success:

1. **Served**: The checkpoint loads and responds to API requests
2. **Coherent text**: It produces natural language output
3. **Parsed and executed tool calls**: Its responses fit the agent's interface format
4. **Verified task completion**: The resulting code passes independent behavioral checks

These are not interchangeable. The paper documents cases where a model passes levels 1–3 while completely failing level 4 — or where infrastructure failures prevent level 3 from being measured at all. The most striking example: 19 of 36 worker artifacts that were rejected solely for renamed parameters (a level-3 interface failure) actually passed the hidden behavioral checks (level-4). The artifacts were functionally correct but rejected by the protocol before the humans even saw them.

This taxonomy is the key insight for anyone building real coding agent infrastructure: **verified outcome is not a property of the model alone**. It's a property of (model × serving × protocol × verification).

### The Engineering Layers

The author worked through five engineering layers in sequence:

1. **Distributed inference** (July): Splitting inference across two laptops via llama.cpp RPC. Result: usable for chat, unusable for coding agents due to latency and session state management.

2. **Single-node serving with reproducible recipes** (August–September): Moving to cloud GPUs (vLLM, SGLang) and later local Apple Silicon (llama.cpp with GGUF, MLX). Key finding: deployment is not portable. The same model required different precision, different kernel configurations, and different runtime versions across serving stacks. One configuration silently fell back to slower kernels because a compiler was absent.

3. **Agent protocol integration**: Model-specific tool-call formatting, turn control, and context behavior differed even across nominally compatible OpenAI-API-compatible endpoints. A compatible API doesn't mean a compatible agent.

4. **Session continuity**: KV cache state outside the model is critical. The paper measures **median TTFT of 69.4s for cold-prefix requests vs. 2.8s for high-reuse requests** (observational, from 85 cold and 280 high-reuse records). This is a 25x difference from cache behavior alone — a confound that benchmark papers never account for.

5. **Verified coding tasks**: The final studies used hidden behavioral checks, task-specific TypeScript fixtures, and a fixed Claude Opus 5.5 parent model. The parent inspected worker artifacts before applying them, and only applied code passed the hidden checks.

### Key Quantitative Results

**First delegation study (45 trials, 3 conditions):**
- Direct coding (Opus 5.5 parent doing everything): **15/15**
- Delegation to Qwen3-Coder 30B-A3B (first run): **12/15**
- Delegation to Qwen3-Coder 30B-A3B (second run): **14/15**

**Second study (45 trials, fixed artifact contract):**
- Direct coding (Opus 5.5): **45/45**
- Delegation to Nemotron Cascade 2 worker: **12/45** (11 worker substitutions — most code was replaced by the parent)
- Delegation to Qwen3-Coder 30B-A3B worker: **0/45** (failed to satisfy the interface contract)

**Cost finding:** In both studies with a direct arm, direct coding had **lower API list-price-equivalent cost per verified success** than delegation. Delegation to small models didn't save money at the success rate achieved.

## Comparison to Prior Work

The paper situates itself relative to SWE-bench and SWE-Lancer (benchmark-oriented evaluations of coding agents) and harness design papers. The key differentiator is:

| Aspect | Benchmark Papers | This Paper |
|--------|----------------|-----------|
| Verification | Model's own claims | Independent hidden behavioral checks |
| Protocol | Controlled lab | Production engineering constraints |
| Model setup | Clean serving | Real deployment with runtime issues |
| Success definition | Output quality | End-to-end verified outcome |
| Data source | Synthetic tasks | Real developer sessions |

The 4,165 gateway turns come from the author's own real coding sessions, not synthetic assignments. The debugging sessions, integration test results, and configuration logs are real operational artifacts.

On model size: the paper explicitly scopes "small" as practitioner-facing shorthand for "substantially smaller than frontier services in active computation and deployment requirements." The primary worker is 30B total parameters with ~3B active (MoE), 4-bit quantized, on a 32GB Apple M1 Pro. This is meaningfully small for a laptop deployment, though not micro-scale.

## Reading the Results

**The 12/15 and 14/15 numbers** for Qwen3-Coder delegation are the paper's strongest evidence for small model utility. On a 15-task suite, a quantized MoE model with ~3B active parameters achieved 80–93% verified task success rates. These are not toy tasks — they are TypeScript fixtures with hidden behavioral checks.

**The 0/45 for Qwen3-Coder in the second study** is equally important. When the interface contract was stricter (the fixed artifact contract), the model completely failed to satisfy it. Not "failed on some tasks" — failed on all 45. The same model that achieved 12–14/15 in the first study scored 0/45 in the second. The difference was the strictness of the interface validation, not the model's underlying capability.

**The 19/36 diagnostic** is the paper's most theoretically important result: 19 worker artifacts that were rejected for interface violations (renamed parameters) passed the hidden checks unchanged when the rule was bypassed. This means the interface contract was rejecting functionally correct code before behavioral verification could run. When measuring "can small models do real work," the measurement system itself can be the bottleneck.

**The 69.4s vs. 2.8s TTFT** shows that KV cache behavior creates a 25x latency difference independent of the model. For interactive coding use, this makes the difference between "usable" and "unusable," but it's invisible in benchmark papers.

## Key Notes of This Paper

### The Four Success Levels as a Measurement Framework

The four success levels form a lattice where higher levels require lower levels as prerequisites:

$$\text{Verified} \subseteq \text{Executed} \subseteq \text{Coherent} \subseteq \text{Served}$$

The paper's key insight is that most evaluations measure at levels 1–2, some at level 3, almost none at level 4. And level 4 requires a joint property of the entire stack:

$$P(\text{Verified}) = P(\text{Served}) \cdot P(\text{Coherent} | \text{Served}) \cdot P(\text{Executed} | \text{Coherent}) \cdot P(\text{Verified} | \text{Executed})$$

Where each conditional probability is determined by different infrastructure components: serving recipe, model instruction-following, agent protocol implementation, and code correctness. A failure at any level is invisible to measurements upstream of it.

### Interface Contract as a Measurement System

The post-hoc diagnostic finding (19/36 rejected-for-interface passing behavioral checks) formalizes a subtle measurement problem. Define:

- $V_{\text{interface}}$: passes interface validation (format, signatures, imports)
- $V_{\text{behavioral}}$: passes hidden behavioral checks

The paper shows empirically that $V_{\text{interface}} \not\Rightarrow V_{\text{behavioral}}$ and — more importantly — $\neg V_{\text{interface}} \not\Rightarrow \neg V_{\text{behavioral}}$.

The interface contract is a proxy for behavioral correctness. A strict proxy that correctly rejects bad code also incorrectly rejects some good code. The 53% false negative rate (19 of 36 rejected artifacts were behaviorally correct) shows the proxy has high precision but significant false positives on renamed parameters. For evaluation purposes, this means the measured success rate is a lower bound on actual capability.

### KV Cache and Session Continuity

The 69.4s / 2.8s TTFT split (cold-prefix vs. high-reuse) illustrates a property that benchmark papers systematically ignore: **a served model's performance depends heavily on its cache state**, which depends on session history. For on-device and self-hosted SLM deployment:

- Cold starts (new sessions, evicted caches) are 25x slower
- Warm starts (repeated context, high-reuse prefixes) approach optimal throughput
- Scheduling and batching decisions that affect cache utilization change the effective performance envelope dramatically

This is the practical gap between "the model is fast" and "coding sessions are fast."

## Limitations

The author is careful about the limits of what the evidence establishes:

1. **Private records**: The operational logs, configurations, and task results are not publicly available. Readers cannot independently verify counts.

2. **Observational, not experimental design**: The 4,165 gateway turns come from real work, not controlled assignments. Different models handled different workloads at different times.

3. **No matched benchmark**: The distributed-inference detour was abandoned without a paired benchmark against single-node serving, so the performance difference is qualitative.

4. **One developer**: All observations come from one developer's workflow on one hardware configuration. Generalization to different setups is uncertain.

5. **Post-hoc diagnostic caveat**: The 19/36 diagnostic finding explicitly excludes those 19 cases from official results. They are evidence about the measurement system, not evidence of verified success.

6. **Scope of "small"**: The primary worker is a 30B total-parameter MoE model. Dense 3B, 7B, or 13B models were not the primary subjects of the delegation studies.

## Future Work

**Authors' suggested directions:**
- Better artifact contracts that minimize false rejections while preserving behavioral validation
- More controlled studies with random assignment of tasks to conditions
- Publication of operational logs (pending privacy review)

**Additional directions this work opens:**
- **Calibrated interface contracts**: Designing artifact validation as a statistical estimator of behavioral correctness, with calibrated false-positive rates, rather than a hard binary filter
- **Cache-aware agent scheduling**: Scheduling coding sessions to maximize KV cache reuse, reducing TTFT from ~70s to ~3s for common workflows
- **MoE-specific serving optimization**: The 30B-A3B configuration activates ~3B parameters per token; optimal routing and expert-loading strategies for Apple Silicon are unexplored
- **Mixture-of-experts SLM for coding**: The 3B active parameter count of Qwen3-Coder 30B-A3B raises the question of whether a dense 3B model trained specifically for coding tool-call tasks could achieve similar verified rates with simpler serving infrastructure

## Implications for Edge / On-Device Deployment

This paper is a practitioner's guide for anyone deploying small open-weight coding models on self-hosted or on-device infrastructure:

1. **Serving recipe portability is not guaranteed**. A model that works on vLLM may require different precision, kernels, or context settings on SGLang or llama.cpp. Document recipes explicitly.

2. **KV cache is infrastructure, not a model property**. A 25x TTFT difference from cache state means cache management decisions are first-class performance engineering decisions.

3. **Tool-call protocol compatibility varies even across "compatible" endpoints**. Test agent behavior, not just API compatibility.

4. **Behavioral verification is a separate layer from generation quality**. A model that produces correct-looking code can still fail an independent behavioral check. Plan for this explicitly.

5. **Interface contracts are measurement systems with error rates**. Strict contracts that reject good artifacts reduce measured success rates. Calibrate the contract to the measurement goal.

6. **MoE models at ~3B active parameters are viable coding workers** at 80–93% verified rates on the tested suite — but only with correct serving and protocol engineering. The model is necessary but not sufficient.

For teams evaluating whether small open-weight models can replace frontier API calls in coding pipelines: the answer is "sometimes yes, with substantial infrastructure investment." The model capability gap is real but narrower than benchmark scores suggest; the engineering gap is what actually dominates verified outcomes.

## Links

[Original Paper](https://huggingface.co/papers/2610.34241)
