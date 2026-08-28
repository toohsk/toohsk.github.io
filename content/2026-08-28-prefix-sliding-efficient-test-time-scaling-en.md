Title: 3× Faster Reasoning with No Retraining: Prefix Sliding Caps Memory for On-Device Test-Time Scaling
Date: 2026-08-28
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Prefix Sliding discards unimportant intermediate reasoning tokens while preserving the instruction prefix and recent reasoning window, making existing language models 3× faster without retraining and enabling reinforcement learning to scale reasoning traces to 100k+ tokens — directly enabling long-horizon reasoning on memory-constrained devices.

## Why This Paper Matters

Test-time scaling is one of the most effective techniques for improving language model capability without changing model weights: let the model "think longer" by generating a chain-of-thought reasoning trace before producing the answer. Models like DeepSeek-R1 and OpenAI o1-series demonstrated that longer reasoning traces reliably translate to higher accuracy on hard reasoning tasks.

But test-time scaling has a brutal memory problem: **the entire reasoning trace must be kept in memory via full attention**. Every intermediate reasoning token adds to the KV cache, which grows linearly with reasoning length. A 100,000-token reasoning trace requires a KV cache roughly 100× larger than a 1,000-token trace. This is already expensive in data centers — it is prohibitive on memory-constrained edge hardware.

The key insight of this paper is empirical: **most intermediate reasoning tokens lose importance as the model continues reasoning**. Early scratch work that the model did 50,000 tokens ago is usually not needed for the current prediction. If this is true, then keeping all those tokens in the KV cache is waste.

Based on this insight, the paper proposes **Prefix Sliding**: a simple mechanism that keeps the instruction prefix (always important) and a sliding window of the most recent tokens (currently important), while discarding intermediate reasoning tokens that fall outside this window. This bounds the KV cache size regardless of total reasoning length.

The results are remarkable: **without any training, Prefix Sliding makes existing reasoning models 3× faster while maintaining performance**. With reinforcement learning training, Prefix Sliding enables models to reason coherently over traces that exceed 100,000 tokens — something impossible with full attention at current memory capacities.

This paper matters for SLMs and edge deployment because it provides a path to long-horizon reasoning on devices where memory is the binding constraint.

## Core Technical Contribution

### The Memory Problem in Test-Time Scaling

Standard transformer attention is quadratic in sequence length: generating token $T_{n+1}$ requires attending over all $n$ previous tokens. In practice, this is implemented via the KV cache: each previous token's key and value are stored and reused at every subsequent step. The KV cache memory is:

$$\text{KV cache size} = n_\text{layers} \times 2 \times n_\text{heads} \times d_\text{head} \times L_\text{seq} \times \text{bytes per value}$$

For a typical 7B model with 32 layers, 32 heads, and 128 head dimension at BF16 (2 bytes per value), a 100k token reasoning trace requires approximately:

$$32 \times 2 \times 32 \times 128 \times 100{,}000 \times 2 = 52\text{ GB}$$

This exceeds the VRAM of any current consumer GPU, let alone a mobile device. Long reasoning traces are simply not possible on-device with full attention.

### The Prefix Sliding Mechanism

Prefix Sliding is conceptually simple:

1. **The prefix P** contains the key instructions, tools, and problem statement. It is always kept in the KV cache.
2. **A sliding window W** of the $k$ most recent tokens is kept in the KV cache.
3. **Tokens outside P and W** are discarded from the KV cache.

Formally, at each generation step, the attention mask is:

$$M_t(i) = \begin{cases} 1 & \text{if } i \in P \text{ (prefix tokens)} \\ 1 & \text{if } t - k \leq i \leq t \text{ (recent window)} \\ 0 & \text{otherwise (discarded)} \end{cases}$$

The total KV cache size is bounded by $|P| + k$ regardless of total generation length $t$. This is the key property: **memory usage becomes constant** with respect to reasoning length, rather than linear.

At each step, when the window slides forward (a new token is added and the oldest window token is dropped), the discarded token's KV entries are freed. The prefix KV entries are never freed.

### Why Prefix + Recent Window Works

The design of what to keep is deliberate. Two alternatives that do not work as well:

1. **Vanilla sliding window** (no prefix retention): The model loses access to its instructions and tools. It cannot maintain coherent long-horizon behavior because it forgets what the problem was.

2. **Summarizing intermediate tokens**: Generate a compressed summary of intermediate reasoning when the context grows too large. This is expensive (requires additional generation), lossy (the model must guess what to summarize), and introduces artifacts (the model sees two different "modes" of token — real and summarized).

Prefix Sliding works because:
- **Prefix tokens are always important**: The instructions define the task, the tools define the action space, and the problem statement is the thing the model is trying to solve. These never become less relevant.
- **Recent tokens are always locally important**: The model's current generation is conditioned on its recent computation. The last few thousand tokens represent "what the model is doing right now" and must be retained.
- **Far intermediate tokens are empirically unimportant**: The paper provides empirical evidence that discarding intermediate reasoning tokens (beyond the recent window) causes no accuracy degradation in the no-retraining setting. This validates the core assumption.

### Training with Prefix Sliding (Reinforcement Learning)

Without training, Prefix Sliding achieves 3× speedup with maintained accuracy — but the model was not designed to reason without all its intermediate tokens, so there is a ceiling on how long it can reason coherently.

With RL training, the model learns to reason under the Prefix Sliding constraint from the start. Specifically:
- Rollouts during RL training are generated with Prefix Sliding active
- The model learns to structure its reasoning so that each conclusion can be derived from the prefix and recent window, without relying on distant intermediate steps
- Gradients flow through the constrained computation graph

The RL training enables **scaling to reasoning traces beyond 100,000 tokens** while maintaining coherent reasoning. This is a qualitative capability leap: the model is no longer limited to reasoning traces that fit in memory via full attention.

The paper demonstrates that RL-trained Prefix Sliding **achieves better performance than vanilla RL training** — not just comparable. This suggests that the memory constraint imposed during training acts as a form of regularization, encouraging the model to structure its reasoning more efficiently.

## Comparison to Prior Work

| Method | Memory Bound? | Requires Retraining? | Quality vs. Full Attention |
|---|---|---|---|
| Full Attention | ❌ O(L) | N/A | Baseline |
| Vanilla Sliding Window | ✓ O(k) | No (or yes) | Worse (loses prefix) |
| Summarization | Partial | Yes (for summarizer) | Worse (lossy) |
| **Prefix Sliding (zero-shot)** | **✓ O(\|P\|+k)** | **No** | **Matches full attention** |
| **Prefix Sliding (RL-trained)** | **✓ O(\|P\|+k)** | **Yes (RL)** | **Better than full attention at long reasoning** |

The comparison with vanilla sliding window is the most instructive. The only difference between vanilla sliding window and Prefix Sliding is that Prefix Sliding retains the prefix. This simple addition is the entire explanation for Prefix Sliding's advantage: the model always knows what problem it is solving and what tools it has available.

The comparison with summarization is also important. Summarization intuitively seems like it should work well — compress intermediate reasoning into a short summary and continue. But in practice, this requires generating extra tokens (the summary), introduces lossy compression (the summary may miss important intermediate steps), and confuses the model by presenting two qualitatively different kinds of tokens (real reasoning and summaries). Prefix Sliding avoids all of this by simply discarding tokens rather than replacing them.

## Reading the Results

**3× speedup without retraining**: The speedup comes from reduced KV cache size. Smaller KV cache means:
1. Less memory bandwidth during attention (each forward pass reads fewer cached values)
2. Earlier memory pressure relief (the cache stops growing after reaching |P|+k)
3. Better GPU utilization (less time waiting for memory, more time on compute)

The speedup is not trivially explained by "fewer tokens in attention." Prefix Sliding does not change the total number of tokens *generated* — it changes how many tokens are *attended to* at each step. The 3× speedup reflects the dramatic memory bandwidth savings from attending to |P|+k tokens instead of the full sequence.

**RL-trained Prefix Sliding > vanilla RL training**: This is the most surprising result. One might expect that imposing a memory constraint during training would hurt performance (the model has less information). Instead, it helps. The mechanism is likely that the constraint acts as **curriculum regularization**: the model must develop reasoning strategies that are *locally coherent* (each step follows from recent steps and the prefix) rather than globally arbitrary (any step can reference any earlier step). This structural constraint turns out to improve the quality of long reasoning chains.

**Scaling to 100k+ token reasoning traces**: This is a qualitative threshold. Current state-of-the-art reasoning models top out at roughly 32k-64k token reasoning traces due to memory constraints. Prefix Sliding breaks this ceiling, enabling reasoning at scales that were previously impossible. The practical implication is that Prefix Sliding-trained models can tackle problems that require very long deliberation — multi-step mathematical proofs, complex planning tasks, deep code debugging.

## Key Notes of This Paper

### The Importance Token Decay Observation

The core empirical observation driving Prefix Sliding is that **intermediate reasoning tokens' importance decays with distance**. The paper provides evidence for this through ablation experiments: systematically varying what is kept vs. discarded and measuring the impact on accuracy.

The finding can be understood through an analogy: when solving a complex problem on paper, you write intermediate work. But once you've derived a sub-result and moved to the next step, you don't re-read all the prior algebra — you just remember the result and continue. The intermediate steps have "served their purpose" and become less relevant.

In attention terms: the attention weights that later tokens assign to earlier intermediate tokens decrease as the distance grows. Prefix Sliding makes this explicit as a structural constraint rather than relying on the model to implicitly deweight distant tokens.

### The Prefix Sliding Attention Mask

The attention computation with Prefix Sliding is:

$$\text{Attention}(Q_t, K, V) = \text{softmax}\left(\frac{Q_t K_{[P \cup W_t]}^T}{\sqrt{d}}\right) V_{[P \cup W_t]}$$

where $P$ is the prefix token index set and $W_t = \{t - k, ..., t\}$ is the current window. The mask $[P \cup W_t]$ is the union of these two sets.

This differs from vanilla sliding window ($W_t$ only) and full attention ($\{0, ..., t\}$). The prefix is a "free" addition that adds $|P|$ entries to the attended set but never grows — it is a constant overhead regardless of reasoning length.

The computational complexity per step with Prefix Sliding is $O(|P| + k)$ instead of $O(t)$. For long reasoning traces where $t \gg |P| + k$, this is a dramatic reduction.

### The RL Training Recipe

The RL training uses the same approach as standard RLVR (RL with verifiable rewards) for reasoning models, with one modification: rollouts are generated with Prefix Sliding active. This means:
- The model never sees full attention during RL training (it only sees prefix + recent window)
- The reward is still based on correctness of the final answer
- The policy gradient updates are computed over the constrained computation graph

The model learns to "pre-load" important information into its reasoning structure near the beginning (making it part of what gets retained in the window) and to structure its conclusions locally (derivable from recent steps). This is not an explicitly incentivized behavior — it emerges from the constraint.

## Limitations

- **Prefix size matters**: If the prefix is very large (many tools, long system prompt), the effective remaining window $k$ may be small, limiting the model's recent context. Optimal prefix design becomes important.
- **Task-dependent accuracy**: Prefix Sliding works well for tasks where intermediate reasoning tokens become less important over time (e.g., mathematical reasoning). For tasks requiring explicit back-reference to specific intermediate steps, accuracy may degrade.
- **The window size k is a hyperparameter**: Too small and the model loses recent context; too large and the memory savings are insufficient. The optimal k depends on the reasoning structure of the task and model.
- **RL training cost**: The RL training required to unlock 100k+ token reasoning is not free. The paper does not fully characterize the compute cost relative to standard RL training.
- **Not applicable to prefix-free architectures**: Models without a clear prefix structure (where the "instructions" are interleaved with the problem) may not benefit as much, since the "prefix" to retain is not well-defined.

## Future Work

**Adaptive sliding window**: The current window size k is fixed. An adaptive variant that monitors attention weight distributions and discards tokens when their attention weight falls below a threshold would be more principled. High-attention tokens could be retained even outside the fixed window; low-attention tokens could be discarded even within the window.

**Prefix Sliding for multimodal reasoning**: Visual reasoning tasks (VQA, visual math) often require attending back to the original image tokens throughout reasoning. Prefix Sliding would naturally include image tokens in the "prefix" and provide a framework for multimodal long-horizon reasoning.

**Combining with KV cache compression**: Prefix Sliding discards entire tokens' KV entries. KV cache compression methods (quantization, low-rank approximation) reduce the per-token KV size. These are orthogonal: apply compression to the retained prefix and window tokens for further memory savings.

**Hybrid attention patterns**: Rather than a hard binary (keep/discard), future work could use soft attention masks or hierarchical attention — precise attention for recent tokens, coarse (averaged or summarized) attention for intermediate tokens, and precise attention for the prefix. This would preserve more information than Prefix Sliding while using less memory than full attention.

**Prefix Sliding for agentic tasks**: Long-horizon agentic tasks (coding agents, research agents) involve many tool calls and observations. The prefix naturally corresponds to the initial task specification and available tools, while the window tracks recent actions and observations. This is exactly the structure Prefix Sliding was designed for.

## Implications for Edge / On-Device Deployment

Prefix Sliding directly addresses the primary barrier to deploying long-horizon reasoning on-device:

**Constant memory regardless of reasoning length**: The binding memory constraint for on-device reasoning models is the KV cache growth. Prefix Sliding converts this from O(reasoning_length) to O(prefix + window), which is constant. A device with 4 GB of RAM can now theoretically support arbitrarily long reasoning (bounded only by time and battery), not just reasoning traces that fit in its memory.

**3× speedup enables real-time reasoning**: The 3× speedup from reduced KV cache memory bandwidth makes long-horizon reasoning tractable on devices without dedicated inference accelerators. A reasoning trace that would take 3 minutes on-device becomes a 1-minute operation.

**No retraining required for deployment**: The zero-shot version of Prefix Sliding (no RL training needed) can be applied to any existing reasoning model. Device manufacturers or app developers can take a pretrained reasoning SLM and apply Prefix Sliding at deployment time without any training infrastructure.

**RL-trained variants for best performance**: For applications that specifically need long reasoning (offline problem-solving, complex planning), RL-trained Prefix Sliding models can be trained centrally and deployed to edge devices. The training happens once; deployment benefits are permanent.

**Battery and thermal implications**: Shorter KV cache means less memory access per generation step, which reduces power draw. On battery-powered devices, this translates directly to longer session duration for reasoning-intensive applications.

The most significant implication is that Prefix Sliding breaks the "reasoning quality vs. device capability" trade-off. Previously, devices with limited memory could only run models with short reasoning budgets, limiting their accuracy on hard tasks. With Prefix Sliding, the same device can run models with reasoning traces 10-100× longer at the same memory footprint — translating directly to higher accuracy on hard reasoning tasks.

## Links

[Original Paper](https://huggingface.co/papers/2608.26070)
