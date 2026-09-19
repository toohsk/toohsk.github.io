Title: Sample Count Is Not Enough: Inference Scheduling Shapes Energy and Latency for Small Language Models
Date: 2026-09-19
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Using Phi-3-mini and Qwen2.5-1.5B on A100 GPUs, this study shows that serializing the same N=8 candidates across 8 separate calls uses 4.6–4.9x more GPU energy and takes 5.8–6.1x longer than generating all 8 in a single batched call — a critical and under-reported cost variable for on-device SLM deployment.

## Why This Paper Matters

Test-time scaling — the practice of generating multiple candidate responses and selecting among them — has become a standard technique for improving small language model accuracy on reasoning tasks. Self-consistency, best-of-N, and related methods all depend on generating N candidate responses per prompt.

The standard way to describe the compute budget in these methods is N: "we generate 8 candidates." But N tells us nothing about **how** those candidates are generated. Are all 8 generated in a single GPU forward pass as a batch? Or are they generated in 8 separate sequential calls, one candidate at a time?

This paper makes a simple but important observation: **the same N can be executed with vastly different hardware cost depending on the generation schedule.** On A100 GPUs with Phi-3-mini and Qwen2.5-1.5B:

- A single batched call of 8 candidates (1×8 schedule): baseline energy and latency
- Eight serial calls of 1 candidate each (8×1 schedule): **4.64–4.86× more GPU energy, 5.77–6.12× longer P95 latency**

This is not a corner case. Many realistic deployment scenarios generate candidates sequentially: logs must be collected between calls, seeds must be set deterministically, or intermediate results must be analyzed before the next generation. HPC batch job scheduling in particular often leads to serialized execution. The gap is large enough that two systems reporting the same "N=8 test-time scaling" result could differ by nearly 5× in energy and 6× in latency.

## Core Technical Contribution

### Formalizing the Generation Schedule

The paper introduces a clean formalization. Let N be the total number of candidates. A **generation schedule** is:

$$\mathcal{S} = (b_1, b_2, \ldots, b_C), \quad \sum_{c=1}^C b_c = N$$

where C is the number of generation calls and $b_c$ is the batch size for call c. The four schedules studied at N=8 are:

| Schedule | Calls | Candidates per call |
|---|---|---|
| 1×8 | 1 | 8 |
| 2×4 | 2 | 4 |
| 4×2 | 4 | 2 |
| 8×1 | 8 | 1 |

All schedules generate the same total N=8 candidates, use the same sampling parameters (temperature 1.0, top-p 0.95), and apply the same plurality voting for answer selection.

### Energy and Latency Measurement

Energy is measured using NVML (NVIDIA Management Library) cumulative energy counters, capturing gross GPU-device energy over the full inference interval (including prompt processing, prefill, decoding, answer extraction, and voting). Idle energy is not subtracted — the reported values are **gross GPU-device energy**, not isolated computational energy.

This measurement scope is appropriate for on-device deployment analysis: the device is consuming power throughout the inference window whether it is actively computing or waiting between calls.

## Comparison to Prior Work

A notable audit finding: the paper surveyed representative test-time scaling studies and found that **most report only N (candidate count) without reporting calls per query, candidates per call, or measured energy**. The table of reporting practices shows systematic gaps across foundational self-consistency papers, adaptive-budget methods, and repeated-sampling studies.

This paper is the first systematic study of how generation schedule structure affects hardware cost at fixed N, using real small language models on dedicated GPU nodes.

## Reading the Results

**Accuracy scales with N regardless of schedule.** Increasing N from 1 to 8 in batched mode improves accuracy by 8.4pp for Phi-3-mini and 18.4pp for Qwen2.5-1.5B on GSM8K. The accuracy gain is the same regardless of whether this is done in a single 1×8 call or in 8 sequential 1×1 calls (candidates within the same schedule are drawn independently). This confirms that accuracy is determined by candidate count, not by schedule structure.

**Energy is dominated by schedule structure.** At fixed N=8:
- **Energy ratio (8×1 vs 1×8)**: 4.64× (Phi-3-mini), 4.86× (Qwen2.5-1.5B)
- **P95 latency ratio (8×1 vs 1×8)**: 5.77–6.12×

The pattern is consistent across three independently scheduled A100 nodes per model and on V100 GPUs in a short-output (SciQ) workload.

**The mechanism is GPU utilization.** Larger batch sizes keep the GPU cores better utilized. In a 1×8 call, the GPU processes all 8 candidate sequences simultaneously, with each decode step operating on a batch of 8 sequences. In an 8×1 schedule, each call processes a single sequence — the GPU idles between calls, and within each call, the parallel compute capacity is mostly unused.

The average GPU power per call actually decreases at larger batch sizes in some configurations (more work per unit time), but the gross energy and latency are far lower because the GPU runs fewer total seconds.

## Key Notes of This Paper

### The Generation Schedule Formalism

The schedule $\mathcal{S} = (b_1, \ldots, b_C)$ cleanly separates two orthogonal dimensions of inference cost:

1. **Logical token volume**: $T_{\text{input}}^{\text{logical}} = NP_q$ (same for all schedules at fixed N)
2. **Hardware utilization pattern**: determined by $C$ and the $b_c$ values

The "energy waste" in serial execution comes from two sources:
- **KV cache inefficiency**: each serial call re-processes the same prompt from scratch. Prompt tokens are processed C times total, versus once in a batched call.
- **Batch underutilization**: modern GPU tensor cores are designed for large matrix operations. A batch of 1 sequence provides minimal parallelism compared to a batch of 8.

Formally, for a simple roofline model, GPU memory bandwidth utilization scales approximately as:

$$\text{Util} \approx \frac{b \cdot d_{\text{model}}}{b \cdot d_{\text{model}} + \text{overhead}} \quad \text{(memory-bound decode regime)}$$

At small b (1 or 2), the overhead term dominates; at larger b (8+), the useful compute fraction increases substantially.

### Practical Guideline

The paper's practical conclusion: **when candidates are independent and GPU memory allows it, generate all N candidates in a single batched call.**

Independence is guaranteed when candidates are sampled from the same prompt with the same decoding parameters — standard self-consistency and best-of-N setups satisfy this trivially. Memory constraint is the only legitimate reason to split into multiple calls.

### Minimum Reporting Recommendations

The paper proposes that multi-candidate inference experiments should report:
- Candidate count N
- Calls per query C
- Candidates per call $b_c$
- Inference hardware
- Gross GPU-device energy (from power monitoring, not from FLOP estimates)

## Limitations

1. **Single-GPU focus.** The study measures per-GPU energy on dedicated single-GPU jobs. Multi-GPU inference (tensor parallel, pipeline parallel) introduces additional scheduling variables not studied here.

2. **A100 and V100 only.** Consumer edge hardware (mobile SoCs, NPUs, CPU-only devices) has very different memory bandwidth, batch efficiency curves, and power monitoring capabilities. The 5× energy ratio may not transfer directly.

3. **Fixed decoding parameters.** Temperature 1.0 and top-p 0.95 are used throughout. Different sampling parameters (greedy, lower temperature) change response length distributions and may shift the energy ratios.

4. **No study of speculative decoding or continuous batching.** Serving frameworks like vLLM use continuous batching that blurs the boundary between serial and parallel execution. The energy analysis assumes simple single-job inference, not a production serving stack.

## Future Work

**Suggested by the authors:**
- Extending the analysis to adaptive-budget methods where N varies per prompt
- Studying multi-GPU energy profiles with tensor parallelism
- Comparing scheduling effects across different model sizes (1B, 7B, 70B)

**Additional promising directions:**
- **On-device profiling on mobile hardware:** The energy gap between batched and serial execution may be even larger on mobile SoCs, where DRAM bandwidth is a tighter bottleneck. Quantifying this for Snapdragon and Apple Silicon NPUs would directly inform on-device SLM deployment.
- **Combining with KV cache reuse:** If candidates share a common prefix (e.g., multi-turn conversation context), the prompt re-processing overhead in serial calls could be eliminated via prefix KV cache. Studying the interaction of schedule efficiency and prefix caching is a natural extension.
- **Energy-accuracy tradeoffs across N:** The paper shows that batching and N are orthogonal. A follow-up could map the full 2D Pareto frontier: for a given energy budget, what is the optimal (N, schedule) combination?

## Implications for Edge / On-Device Deployment

1. **Batch your candidates.** For on-device inference where memory allows, this paper gives a clear quantitative case for batching all test-time scaling candidates in a single call. A 5× energy reduction is the difference between a usable and an unusable battery-powered application.

2. **Report generation schedules.** Teams benchmarking on-device SLM performance should include generation schedule in their reporting. A result that looks "better" under serial-call evaluation may be far worse in energy than the same N under batched evaluation.

3. **The results validate Phi-3-mini and Qwen2.5-1.5B as test-time scaling targets.** These models show meaningful accuracy gains from N=1 to N=8 (8.4pp and 18.4pp respectively on GSM8K), confirming that test-time scaling is a viable accuracy-compute tradeoff for SLMs small enough for on-device deployment.

4. **Hardware choice matters.** V100 vs A100 shows the same qualitative scheduling effect but different absolute numbers. Teams deploying on specific edge hardware should benchmark the scheduling efficiency curve on their target device before committing to a test-time scaling strategy.

## Links

[Original Paper](https://arxiv.org/abs/2609.19499)
