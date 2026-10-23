Title: PACE: Four-Bit Recurrent State Quantization for Linear Attention Models
Date: 2026-10-23
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: PACE achieves near-lossless 4-bit quantization of linear-attention recurrent states by rounding less frequently and applying column-wise scale groups, enabling ~1.5× more concurrent sequences on the same accelerator without any fine-tuning.

## Why This Paper Matters

Linear-attention language models—models that replace the Transformer's growing KV cache with a fixed-size recurrent state—are among the most promising architectures for on-device and server-side inference. Because the recurrent state doesn't grow with sequence length, memory use is predictable and bounded. That's a fundamental advantage for deployment.

But the predictability cuts both ways. At 64 concurrent sequences, the fp32 recurrent states of a 9B model already occupy 10–41% of the memory shared with weights. You want to quantize those states down to 4 bits. The problem is that every published 4-bit state quantizer loses accuracy—sometimes badly—unless it either raises the average bit width back up or fine-tunes part of the model. Neither is acceptable on-device: fine-tuning costs time and on-edge memory, and higher bit width defeats the purpose.

PACE solves this without fine-tuning, using two insights about where the quantization error actually comes from.

## Core Technical Contribution

### Background: The Gated Delta Rule

Modern linear-attention models use a recurrence where the state $S_t \in \mathbb{R}^{d_k \times d_v}$ evolves as:

$$S_t = a_t \bigl(I - \beta_t k_t k_t^\top\bigr) S_{t-1} + \beta_t k_t v_t^\top$$

where $a_t \in (0,1)$ is a data-dependent per-step decay, $k_t$ is the key vector, $v_t$ is the value vector, and $\beta_t$ is a write strength. This can be rewritten more cleanly as:

$$S_t = a_t S_{t-1} + k_t u_t^\top, \qquad u_t = \beta_t\bigl(v_t - a_t S_{t-1}^\top k_t\bigr)$$

The vector $u_t$ is called the **delta value**—it's the component of $v_t$ that the current state doesn't already explain, scaled by $\beta_t$. Every token writes one outer product $k_t u_t^\top$ into the state.

At inference, the kernel writes $S_t$ to memory after every token. Every prior quantizer quantizes that write, injecting a small error at each step. That error then propagates through subsequent decays.

### Insight 1 — Rounding Period (PACE's "P")

Quantization error is not uniformly costly. In a recurrence with per-head mean decay $a_h$, an error of energy $e_h$ injected at one rounding retains at most $a_h^{2 \, dP}$ of its energy by the next rounding $dP$ tokens later. The total accumulated error across many roundings is bounded by:

$$E_h = \frac{e_h}{1 - a_h^{2\,dP}}$$

This is a standard fixed-point filter result. The key observation: **a longer decode period $dP$ dramatically reduces $E_h$** because more decay has had time to dissipate the error before the next injection. Prior quantizers used $dP = 1$ (round every token). PACE sets $dP = 8$ by default, reusing a **decode buffer** already present in recent serving work (KVBuffer, ReplaySSM, DeltaLog). The buffer stores recent keys and values in full precision and flushes them in one step—so PACE piggybacks the rounding on a memory structure that exists anyway.

Prefill was already fine: chunked kernels process 64–1024 tokens at once and write the state only at chunk boundaries. PACE brings decoding in line with prefill's natural rhythm.

### Insight 2 — Column Structure (PACE's "C")

Once you're rounding less often, the next question is how to pick quantization scale groups. A group-wise quantizer assigns one step size per group of $g$ adjacent entries. If one entry in the group is much larger than the rest, all the small entries get rounded coarsely.

PACE measures the distribution of large entries in the state and finds a clean pattern: **large entries form columns**, meaning they occur at the same value-axis index across every key-axis index. This structure comes directly from the delta value $u_t$: some channels of $u_t$ stay large across many tokens, and the outer product $k_t u_t^\top$ imprints that channel's magnitude into the entire column. Permuting the channels of $u_t$ randomly at every step destroys the column structure—confirming the causal link.

The practical consequence: scale groups should run along the key axis (within one column), not along the value axis (across columns). Grouping across columns mixes a large column into a small one, wasting quantization levels. Key-axis groups keep the large entries within the same group and calibrate the step size to them.

### Insight 3 — Head-Level Precision (PACE's "4/8")

Per equation above, a head with slow decay ($a_h$ close to 1) accumulates quantization error faster because less dissipation occurs between roundings. PACE-4/8 identifies the 25% of heads in each layer with the highest rounding-error retention (the longest-memory heads) and keeps them at 8 bits. The remaining 75% stay at 4 bits. This selection is done once per checkpoint—it's a static property of the model, not a runtime decision.

### Two Variants

- **PACE-4**: All heads at 4 bits throughout. No fine-tuning.
- **PACE-4/8**: 25% of heads per layer at 8 bits, rest at 4 bits. No fine-tuning.

Both use decode period $dP = 8$ and column-aligned scale groups. The Triton kernels PACE provides keep the decoding speed benefit of the decode buffer even after quantization.

## Comparison to Prior Work

| Method | Bits/value | Fine-tuning | Notes |
|--------|-----------|-------------|-------|
| Quamba2 | 8 | No | Static scales |
| Q-Mamba | 4 | Yes (~2% weights) | Fine-tuning required |
| DAMP | ~9.9 | No | 8-bit state + mixed fp16 keys |
| PACE-4 | ~4.1 | No | Beats Q-Mamba at equal memory |
| PACE-4/8 | ~5.3 | No | Competitive with 8-bit DAMP at ~2/3 memory |

DAMP was previously the best no-fine-tuning method. It operates at ~9.9 bits per value (keeping the riskiest key channels in fp16) and rounds every token during decoding. PACE-4/8 matches DAMP's accuracy on three of four tested models while using about two-thirds of the memory. PACE-4 beats Q-Mamba (which requires fine-tuning) at equal memory.

Evaluated on four hybrid models spanning 4B to 35B parameters, using MMLU-Pro, IFEval, and RULER-32k.

## Reading the Results

The Figure 1 description (mean performance on MMLU-Pro, IFEval, RULER-32k vs. bits per value on Qwen3.5-9B) captures the story: PACE sits on the Pareto frontier. At any given memory budget, PACE offers the best accuracy of all no-fine-tuning methods.

The most important number is the memory ratio. At 64 concurrent sequences, PACE-4/8 uses roughly 2/3 of what DAMP uses. That's not a marginal improvement—it means you can serve 50% more concurrent requests on the same hardware, or deploy the same throughput on a smaller, cheaper accelerator.

For on-device inference, even PACE-4 is compelling: it halves state memory compared to fp16 without touching the weights or running any calibration fine-tuning.

## Key Notes of This Paper

### The Error Accumulation Formula

$$E_h = \frac{e_h}{1 - a_h^{2\,dP}}$$

This formula is the theoretical backbone of the paper. Intuition: imagine injecting a small error into a leaky integrator (the recurrent state). The error decays at rate $a_h^2$ per token. After $dP$ tokens, only $a_h^{2\,dP}$ of the original error survives. Errors from successive roundings accumulate, but because each one is smaller by the time the next one arrives, the total is bounded by a geometric series with ratio $a_h^{2\,dP}$.

With $dP = 1$ (round every token), the denominator becomes $1 - a_h^2$, which can be small for heads with $a_h$ close to 1—causing high error. With $dP = 8$, the denominator grows, and $E_h$ shrinks. The insight is that you can spend the buffer's memory (it stores recent keys and values at full precision) to buy a longer rounding period, and that trade is almost always worth it.

### Why Columns Exist in the State

The state is a sum of outer products $k_t u_t^\top$. If channel $j$ of $u_t$ is consistently large across many tokens (i.e., $|u_{t,j}|$ is large for many $t$), then column $j$ of every outer product is large, so column $j$ of the accumulated state $S_t$ tends to be large. This is an empirical fact about how the gated delta rule works in practice—certain value channels are persistently important.

By forming scale groups along the key axis, PACE puts all entries from one column into the same group. Since the column is uniformly large (or uniformly small), the max/RMS ratio within the group is low, meaning no individual entry dominates and every entry gets a reasonable number of quantization levels.

## Limitations

- Evaluated only on hybrid architectures (Transformer + linear-attention layers) from 4B to 35B. Pure recurrent models may have different column structure properties.
- The decode period $dP = 8$ is chosen empirically; optimal $dP$ may vary by model and task.
- The decode buffer adds a small constant memory overhead (stores $dP$ key-value pairs per layer per sequence) that must be counted in the total budget—PACE's memory accounting is explicit about this, but deployments must plan for it.
- Results are on standard academic benchmarks; code-generation and long-document tasks that stress very long context may behave differently.

## Future Work

The authors demonstrate PACE on the gated delta rule specifically. Extensions include:

- **Other recurrent rules**: RWKV, Mamba/SSM variants, GLA—the column structure may or may not hold, but the periodic-rounding framework is architecture-agnostic.
- **Quantization-aware training**: PACE is post-hoc (no training). Combining periodic rounding with light QAT might push below 4 bits while maintaining accuracy.
- **Dynamic period scheduling**: The optimal $dP$ may be query-dependent. A fast per-head uncertainty estimate could trigger a flush early (shorter $dP$) on especially noisy generations.
- **Weight quantization + state quantization**: Separately quantizing weights (GPTQ, AWQ) and states is well-studied; joint optimization could recover additional memory.

## Implications for Edge / On-Device Deployment

PACE is directly actionable for on-device linear-attention models:

1. **No fine-tuning**: The entire pipeline—download model, deploy with PACE—requires no GPU training at the edge. This is the critical property for mobile and embedded deployments where users can't run training loops.
2. **Fixed memory**: Linear-attention's fixed-size state is already ideal for devices with hard RAM limits. PACE halves (PACE-4) or reduces by ~40% (PACE-4/8) the state memory, making deployment feasible on devices with 4–6 GB RAM.
3. **Throughput**: The Triton kernels preserve the decode buffer's speed advantage. On resource-constrained hardware where memory bandwidth is the bottleneck, reducing state size directly translates to faster generation.
4. **Batch serving on NPU/edge accelerators**: For server-side edge inference (e.g., a local-area server running inference for a building's devices), PACE-4/8 enables serving 1.5× more concurrent users on the same accelerator—a direct cost reduction.

## Links

[Original Paper](https://arxiv.org/abs/2610.29545)
