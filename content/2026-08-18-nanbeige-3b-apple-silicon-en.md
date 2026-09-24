Title: Nanbeige4.2-3B on Apple Silicon: Five Deployment Bugs and a Chunked-Prefill Fix
Date: 2026-08-18
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: A systematic investigation of Nanbeige4.2-3B deployment on Apple Silicon M2 Max uncovered five independent bugs preventing out-of-box use, plus a chunked-prefill strategy that extends usable context from 4,096 to 11,231 tokens (2.7×) on 32 GiB unified memory.

## Why This Paper Matters

A growing gap exists between SLM papers claiming competitive benchmark scores and models that actually run reliably on target hardware. This paper closes that gap, painfully, for one high-profile model on one of the most important edge-AI platforms: Apple Silicon.

Nanbeige4.2-3B was released with reported strong performance on agentic office-workflow tasks—competitive with or better than Qwen3.5-4B and Qwen3.5-9B. The model uses a Looped Transformer (LT) architecture to achieve greater effective depth without scaling parameter count, a principled approach for parameter-efficient SLMs.

In practice: the released checkpoint cannot be evaluated at all on Apple Silicon (MPS). Five independent bugs prevent it from loading or running correctly. And once those five bugs are fixed, the LT architecture's inherent memory doubling makes it impractical for agentic tasks on 32 GiB unified memory—the context lengths needed for multi-turn agentic traces exceed what naive prefilling can handle.

This paper matters to anyone deploying SLMs on Mac hardware (or any unified-memory device). The bugs found are not Nanbeige-specific pathologies; they represent systematic failure modes that can appear in any model that inherits similar code patterns. The chunked-prefill solution is immediately applicable to any memory-constrained SLM deployment.

## Core Technical Contribution

### The Looped Transformer Memory Problem

Nanbeige4.2-3B's LT architecture processes the input through one stack of L physical transformer layers, then feeds the output back through the same L layers a second time before producing logits:

```
Input → [L layers, pass 1] → [L layers, pass 2] → Logits
         same weights ↑          same weights ↑
```

This doubles effective model depth at fixed parameter count—the key parameter-efficiency benefit. But during prefilling, self-attention materializes an attention score matrix proportional to (seq_len²) per layer per pass:

```
Standard Transformer peak prefill memory: O(L × seq_len²)
Looped Transformer (2 passes):           O(2L × seq_len²)
```

On dedicated GPU hardware with CUDA memory management, this doubling is manageable. On Apple Silicon's unified memory—where the OS, competing processes, and the model share the same physical pool, with no CUDA-style page-out path—it is catastrophic for long agentic traces.

### The Five Bugs

Loading `Nanbeige/Nanbeige4.2-3B` via `AutoModelForCausalLM.from_pretrained(..., device="mps")` fails or silently misbehaves in five independent ways:

**Bug 1: RoPE buffer persistence (most dangerous)**  
The `inv_freq` rotary embedding buffer is silently zeroed on load and never repopulated before the first forward pass. RoPE contributes zero positional information—the model runs but cannot distinguish token order. This produces fluent-looking but positionally-incoherent outputs, making it easy to mistake for a poorly-trained model rather than a broken one.

**Bug 2: RoPE-config dispatch KeyError**  
The custom modeling code's RoPE-type dispatch raises `KeyError` for certain valid config values during model construction. This fires before any forward pass and is not MPS-specific—it blocks loading on any device.

**Bug 3: Cache API sentinel mismatch**  
Calling `model.forward()` directly with `past_key_values=None` calls `DynamicCache.from_legacy_cache(...)`, an API removed in current transformers releases. This crashes at inference time.

**Bug 4: Position-IDs re-trim (MPS-specific)**  
A bug in the custom attention code's position tracking produces a hard crash specifically on MPS devices (does not reproduce on CPU).

**Bug 5: Tied-weights key format**  
An incompatible tied-weights naming convention breaks `save_pretrained()`. A successfully-patched running model cannot be re-serialized without this additional fix.

### Chunked Prefill: The Memory Fix

The core algorithmic contribution is chunked prefilling (CP), which replaces single-shot full-prompt processing with incremental 256-token chunk processing:

```python
cache = DynamicCache()
n_full_chunks = (total_len - 1) // chunk_size  # chunk_size = 256

for i in range(n_full_chunks):
    start, end = i * chunk_size, (i + 1) * chunk_size
    outputs = model(
        input_ids=input_ids[:, start:end],
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(start, end),
    )
    cache = outputs.past_key_values

# Decode the remainder normally
return model.generate(input_ids=remaining_tokens, past_key_values=cache, ...)
```

**Memory complexity change:**

| Method | Peak attention memory |
|--------|-----------------------|
| Naive prefill | O(prompt_len²) per loop iteration |
| Chunked prefill | O(chunk_size × running_total) |

The peak per-step tensor is bounded to `(chunk_size × cumulative_kv_length)` regardless of total prompt length. The tradeoff: sequential sub-calls instead of one large kernel, adding per-chunk overhead.

### System Prompt Regression Fix

The chat template has a conditional that silently discards the model's trained tool-use system prompt whenever a caller supplies any system message:

```jinja2
{% if messages[0]['role'] == 'system' %}
    {{ messages[0]['content'] }}\n\n     {# replaces the default entirely #}
{% else %}
    {{ DEFAULT_TOOL_PROMPT }}            {# no trailing \n\n, directly followed by # Tools #}
{% endif %}
```

The model's tool-calling reliability is calibrated to the exact byte sequence the default branch produces. The two-character difference (`\n\n` vs nothing) between the two branches is enough to break structured multi-tool outputs.

**Fix:** Modify the template to take the default auto-insert path always, then append the caller's system content after the default prompt rather than replacing it.

## Comparison to Prior Work

This paper is not a benchmark comparison against other models—it's a deployment engineering report. The relevant comparison is the same model before vs. after fixes:

| Condition | MCPMark Filesystem (easy) | BFCL Single Tool | Max Context (32 GiB) |
|-----------|--------------------------|------------------|----------------------|
| Unpatched | Cannot evaluate (0%) | Cannot evaluate | — |
| Patched + CP | 3/10 (30%) | Near-perfect | ~11,231 tokens |
| Naive prefill (patched) | Limited to short contexts | Near-perfect | ~4,096 tokens |

For context on where 30% sits: this is the **actual** performance of the model when running correctly, versus the zero that the buggy release achieves. The model card's reported numbers were presumably measured in a controlled environment where these bugs were worked around.

## Reading the Results

### LongBench-Pro memory experiment (50 samples, M2 Max 32 GiB)

| Prompt len | Naive max BS | Chunk max BS | CP time overhead |
|------------|--------------|--------------|-----------------|
| 1,024 | 2 | 4 | +22.8% |
| 2,048 | 1 | 4 | +40.9% |
| 4,096 | fails | 1 | — |
| 8,192 | fails | 1 | — |
| 11,231 | fails | 1 | — |

**The 40.9% overhead at 2,048 tokens** is significant for latency-sensitive applications but represents access to 2× more batch parallelism. For offline batch processing the throughput can still be favorable.

**The 4,096+ failure** is the key result: naive prefilling cannot serve Nanbeige4.2-3B on 32 GiB Apple Silicon for any context a real agentic task would need. Chunked prefilling is not optional—it's required.

### MCPMark failure analysis

Two failure modes dominated the 7 failed tasks:
1. **Path repetition**: Model correctly selects the right tool but generates the same absolute path 21 times, causing timeout
2. **Context accumulation OOM**: Multi-turn tasks grow context over multiple rounds until exceeding memory capacity

Both suggest the model's underlying tool-calling quality (when it functions) is reasonable for simple tasks, but context length management and output repetition control are unsolved problems.

## Key Notes of this Paper

### The O(seq_len²) Attention Memory Formula

Self-attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The `QK^T` matrix has shape `(batch, heads, seq_len, seq_len)` — quadratic in sequence length. During prefilling, this must be materialized in memory.

For a Looped Transformer with N loops, this computation runs N times over the same prompt:

```
Total peak prefill memory ∝ N × seq_len²
```

With seq_len=4096 and N=2, the attention matrix alone requires ~4096² × 2 × num_heads × batch × bytes ≈ several GiB. On unified memory, this competes with GPU kernel state, OS pages, and anything else running on the machine.

**Chunked prefill's bound:**

```
At chunk i: matrix shape = (chunk_size × i×chunk_size)
Peak over all chunks: max(chunk_size × i×chunk_size) = chunk_size × seq_len
```

By trading sequential chunk overhead for bounded peak memory, CP transforms a quadratic ceiling into a linear one. This is a direct application of online algorithms reasoning applied to attention computation.

### Why System Prompt Whitespace Breaks Tool Calling

The model's SFT/RL data was generated exclusively through the auto-insert (default) template path, producing an exact byte sequence including the `{{ DEFAULT_TOOL_PROMPT }}` followed immediately by `\n# Tools`. When a caller supplies a system message, the rendering path appends `\n\n` before the Tools section instead.

The model was never trained on that format. At test time, the `\n\n` separating the system prompt from the tools section doesn't match the training distribution of tool-call examples. For structured generation tasks like tool-calling—where the model must produce exact JSON within `<tool_call>...</tool_call>` tags—even small distributional shifts in the prefix can cause cascading failures in the structured output.

This is not a Nanbeige-specific bug. **Any model whose SFT data was generated through a specific template rendering path, and whose inference template has conditional branches producing different whitespace, will have this vulnerability.** It is worth auditing any SLM's chat template for exactly this pattern before production deployment.

### MPS OOM Memory Corruption

The paper documents a PyTorch/MPS bug: a single caught `RuntimeError: MPS backend out of memory` permanently reduces the serving process's available MPS memory. `torch.mps.empty_cache()` and `gc.collect()` do not reclaim it; only process restart does.

This means: on an eval harness that runs multiple tasks sequentially, one OOM (expected when context grows past the model's limit) poisons all subsequent tasks in the same process. The fix is subprocess isolation—restart a fresh process before each task.

## Limitations

1. **30% success on easy tasks**: Not production-ready for agentic use. The model handles simple single-tool tasks but fails on tasks requiring multi-turn context management or multi-tool sequencing.

2. **Hardware specificity**: All results are on M2 Max 32 GiB. Behavior on M4 Max, M-Ultra, or other unified-memory configurations is unverified.

3. **Speed penalty of chunked prefill**: 22–41% overhead at practical prompt lengths. For interactive applications where first-token latency matters, this is meaningful.

4. **Unresolved MPS OOM persistence**: The process restart workaround is a hack. The underlying PyTorch/MPS memory management issue is not fixed.

5. **Patching via monkeypatching**: All fixes are implemented as sibling-file monkeypatches that shadow the checkpoint's custom modeling code. This requires maintenance whenever the base transformers package or model code updates.

## Future Work

**Authors' suggested directions:**
- Kernel fusion to reduce per-chunk overhead in chunked prefilling
- Evaluation on broader MCPMark task categories

**Promising follow-on research:**

1. **Quantization × Looped Transformers**: Weight-sharing across loops means quantization errors are applied identically in both passes. Does error amplify (coherent noise) or cancel (random noise)? Systematic study of INT4/INT8 quantization on LT architectures is needed.

2. **Adaptive loop depth**: Rather than always applying exactly N loops, dynamically adjust loop count based on prompt complexity or early-exit criteria. Easy prompts might need only 1 loop; complex reasoning tasks use the full 2. This could reduce average inference cost significantly.

3. **Unified-memory-aware SLM architecture search**: Design SLMs from the start for unified-memory constraints (Apple Silicon, Snapdragon NPUs, MediaTek Dimensity). This means minimizing peak activation memory as a first-class design objective alongside parameter count.

4. **Chat template formal specification**: The whitespace-sensitivity discovery argues for a standardized, formally specified chat template format that guarantees consistent byte sequences regardless of caller-provided message structure.

5. **MPS memory recovery mechanisms**: A PyTorch contribution to properly recover MPS memory budget after OOM events, equivalent to CUDA's memory management capabilities.

## Implications for Edge / On-Device Deployment

**The benchmark-to-deployment gap is real and costly.** This paper quantifies it: a model scoring competitively in controlled benchmarks requires 5 bug fixes, an alternative prefilling algorithm, a chat template patch, and subprocess isolation before it can run reliably. **Every SLM targeting consumer hardware should undergo this kind of systematic adversarial deployment testing before release.**

**Unified memory is not "the same as CUDA memory."** The no-page-out property, the shared OS/GPU/model pool, and the memory corruption after OOM are distinct constraints. Architectural decisions (like LT's 2× peak memory) that are benign on H100s can be disabling on M-series Macs or Snapdragon devices.

**Chunked prefill as a general technique:** Any SLM with long system prompts—RAG context, tool definitions, few-shot examples—can benefit from chunked prefilling on memory-constrained devices. The implementation shown here is model-agnostic: it works with any Hugging Face `transformers` model that supports incremental `DynamicCache` updates.

**Tool-calling template hygiene:** Before shipping any SLM with tool-calling capability to production, test it with: (a) no system message, (b) a generic system message, (c) the model's default system message provided explicitly, and (d) both default + caller system message. Each path should produce structurally valid tool calls.

**Practical target-device profiling:** The chunked prefill results show that on 32 GiB Apple Silicon, a 3B LT model can handle up to 11,231-token contexts—but only with the right prefilling strategy and per-task subprocess isolation. Know your deployment target's constraints before promising context lengths.

## Links

[Original Paper](https://arxiv.org/abs/2608.13987)
