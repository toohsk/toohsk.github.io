Title: Sketchpad: Certified Exact Arithmetic with a 0.8B Vision-Language Model in the Browser
Date: 2026-11-10
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Qwen3.5-0.8B with a LoRA adapter achieves mathematically guaranteed exact arithmetic of arbitrary length running in a browser — the first practical system where a small model can be formally proven to never make an arithmetic mistake.

## Why This Paper Matters

Language model arithmetic accuracy is estimated on samples. Even GPT-6.1 Sol fails 3 of 220 problems (on 300- and 1,000-digit sums). Probabilistic accuracy is fundamentally insufficient for applications that require guarantees.

The core challenge: guaranteeing behavior on infinitely many inputs requires that behavior be fixed by finitely many facts. With chain-of-thought, every step can read all previous steps — Moore's theorem (1956) shows that for any finite test set, there exists a correct decoder that fails on longer sequences. No amount of testing can close this gap.

Sketchpad solves this by architectural constraint: each step reads only a **bounded view** (3 cells) of an external working memory. This reduces the infinite space of possible inputs to a **finite set of 2,446 views**. Certify those 2,446 views and you've certified all inputs of any length. This is verifiable in Lean.

## Core Technical Contribution

### The Sheet Machine Architecture

Three components:

1. **Sheet (state)**: A grid of 32-pixel cells, each blank or a digit, holding all arithmetic state
2. **Model (step execution)**: Qwen3.5-0.8B + rank-16 LoRA adapter (10.2M params = 1.2% of model)
3. **Runtime (view movement)**: A few hundred lines of code that moves the view and writes outputs — but writes NO digit of a result

**The pinned-sheet policy**: Each step receives only `[operation token] [3-cell view] [=] [2 placeholders]` as input. The model outputs one digit, one carry/borrow, and the next operation. Nothing else passes between steps.

Under this policy, the model's output is a function of the view alone: $f(v)$. This is the key invariant that makes certification possible.

### Training on View Tables

The LoRA adapter is trained directly on tables of all possible views:
- Column addition (SUM): 242 views
- Long multiplication (5 operations): 1,553 views  
- Subtraction (SUB): 242 views
- Quotient digit estimation (EST): 400 views
- Estimate decrement (DEC): 9 views
- **Total: 2,446 views**

Training loss: cross-entropy + hinge loss on logit margins + distillation penalty toward base model on plain text. Rows with margin below 2 are replayed. Training stops when all rows have margin ≥ 3.

Without the adapter, Qwen3.5-0.8B passes only 12 of 2,446 views.

## Comparison to Prior Work

| Model | Problems solved (220 total) | Notes |
|-------|---------------------------|-------|
| **Sketchpad (0.8B + LoRA)** | **220/220** | Mathematically certified |
| GPT-6.1 Sol | 217/220 | Strongest baseline |
| Qwen3.5-0.8B (no CoT) | ~12 | Base model |
| Various LLMs with CoT | 193–217 | Probabilistic |

This is not a marginal improvement. A 0.8B parameter model running in a browser **outperforms the strongest available LLM** on this benchmark, with mathematical guarantees.

## Reading the Results

**Browser certification**: All 2,446 views pass on Apple M4 laptop GPU running ONNX Runtime Web on WebGPU in 16-bit precision. Smallest margin: **3.02 logits** (addition view 5+6 with no carry). This tolerates any perturbation smaller than 1.51 logits — meaning quantization noise, floating-point rounding, and runtime variations are all provably within tolerance.

**Comparison (200 pre-registered problems)**: 200/200 exact. The model solved sums of up to 1,000 digits and products of up to 20-digit numbers, all correct.

**Speed**: Independent procedures are batched (4 parallel), yielding 32 forward passes instead of 73 for a typical problem. Measured cost vs. model: median ratio 1.01 (essentially perfect prediction).

**Dialog adapter**: A gating scalar on the adapter allows the model to decide when to invoke the sheet. Stays within 2 points of base model on instruction following and MMLU, but loses 5.8 points on fresh grade-school word problems (learning to invoke the sheet vs. answering conceptually).

## Key Notes of This Paper

### Theorem 1: Exactness on Views

The central certification theorem:

$$c_{t+1} = \mathrm{paint}(c_t, f(\mathrm{view}(c_t)))$$

**Theorem**: If the runtime always presents views from the table ($R_T(I) \subseteq V$) and the model is correct on all views ($m(v) > 0$ for all $v \in V$), then every run of every length is exact.

**Proof sketch**: By induction on $t$. If runs agree at step $t$ and $\mathrm{view}(c_t) \in V$, then $f(\mathrm{view}(c_t)) = T(\mathrm{view}(c_t))$ (by the view certificate), so $c_{t+1}$ is the same in both runs.

**Intuition**: The certification says "if the model handles every possible 3-cell combination correctly, it will handle every arithmetic problem correctly regardless of length." The runtime is a trusted orchestrator that only ever creates inputs the model was certified on.

### Theorem 2: Why Chain-of-Thought Cannot Be Certified

If each step reads all previous steps, there exists a family of decoders — each a correct lookup plus one attention head — where for every finite test set, some member passes every row with positive margins, yet fails on longer runs.

**Intuition**: The attention head writes `4Rk/(e^c + n)` to the wrong logit, where $k$ is the number of events and $n$ is context length. With $e^c = 3M + 5$ (where $M$ is the longest test context), this term exceeds the margin $R$ only when $n > M$. No finite test covers all possible lengths.

This is a constructive proof that chain-of-thought verification is fundamentally undecidable for sequential tasks.

### Theorem 3: Robustness to Perturbation

$$\|z - \tilde{z}\|_\infty \leq \tau \text{ and } z_y - \max_{a \neq y} z_a > 2\tau \implies y = \mathrm{argmax}(\tilde{z})$$

A margin of $> 2\tau$ guarantees that perturbations up to $\tau$ cannot flip the decision. The certification at margin 3.02 covers perturbations up to 1.51 logits in any direction.

**Key implication**: The certificate is computed on the deployment path itself (WebGPU 16-bit), not in a simulator. The guarantee is real, not approximate.

### The Composability Property

Certified procedures compose: if each of addition, subtraction, multiplication, and division is certified, then procedures that use them (GCD, square root, polynomial arithmetic) are also certified, because the runtime controls all data flow between procedures.

## Limitations

- **Procedure coverage**: Only arithmetic operations are certified. No natural language understanding, symbolic reasoning, or domain knowledge
- **Runtime is not formally verified**: The sheet layouts, outer loops, and reference procedures are conventional code — not proved in Lean. The guarantee is relative to this runtime being correct
- **New procedures require re-training and re-certification**: Adding a new arithmetic operation means extending the tables and re-certifying all views
- **Dialog-mode degradation**: The gating mechanism loses 5.8 points on word problems — the model sometimes fails to recognize when to invoke the arithmetic sheet

## Future Work

**Authors' directions**: Extension to algebraic procedures, symbolic runtime for polynomial arithmetic.

**Additional promising directions**:
1. **Full end-to-end verification**: Formalizing the runtime in Lean would create an end-to-end machine-verified arithmetic system — a potential standard for safety-critical computation
2. **Generalizing the view-bounded architecture**: Any algorithm expressible as a finite-state machine operating on external memory admits this certification strategy — sorting, string matching, graph traversal
3. **Quantization safety bounds**: The margin theorem provides a theoretical framework for computing the maximum quantization level that preserves certification — directly applicable to INT4/INT8 deployment planning
4. **LoRA certification as a standard**: The technique of certifying adapter behavior on a finite view set could become a standard for validating domain-specific adapters before deployment in regulated industries

## Implications for Edge / On-Device Deployment

This paper has profound implications for the edge AI ecosystem:

1. **Minimal resource footprint**: 10.2M additional parameters (1.2% of 0.8B model) certified arithmetic. The entire system runs in a laptop browser — smartphones and embedded devices are within reach
2. **WebGPU deployment**: No app installation, no backend server — certified arithmetic directly in web pages. Practical for financial, educational, and embedded calculator applications
3. **Dynamic capability addition via LoRA**: Certified adapters can be loaded/unloaded at runtime, enabling dynamic capability management on edge devices with tight memory budgets
4. **The certification paradigm**: Instead of hoping the model generalizes correctly, you *prove* it. For safety-critical edge applications (medical devices, industrial controllers), this matters enormously — it changes AI from "probably correct" to "provably correct"
5. **Hardware-software co-certification**: The margin framework suggests certifying models directly on target hardware (Apple M4 in this case), accounting for chip-specific numerical behavior

## Links

[Original Paper](https://arxiv.org/abs/2611.04434)
