Title: CPU-ARC: 16–291x Cheaper LLM Reasoning via Agentic Caches Running on Small Language Models
Date: 2026-11-10
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: CPU-ARC builds executable reasoning caches from frontier LLM traces that run on CPU with small language models, achieving equivalent performance on difficult math/science benchmarks at 16–291x lower invocation cost.

## Why This Paper Matters

LLMs solve every problem from scratch. When a system repeatedly handles the same *class* of problems — math problems with different numbers, the same API query with different payloads — it pays frontier model costs every time, even when the reasoning is identical.

The key insight the paper formalizes: once a problem has been reasoned through, **applying that solution to new instances requires almost no LLM involvement**. The search for a solution and the execution of a solution are separable. CPU-ARC exploits this.

The challenge: caching must handle rephrasing ("Find how many positive integers evenly divide 196" vs. "How many positive whole-number divisors does 196 have?") AND reparameterization (196 → 144). Traditional semantic search handles rephrasing poorly and reparameterization not at all. CPU-ARC handles both by converting reasoning traces to parameterized code.

## Core Technical Contribution

### Two Foundational Observations

**O1 (Execution separation)**: Once reasoned, the execution — the actual computation — can be done with little LLM involvement. Search and execution are decoupled.

**O2 (Reasoning reusability)**: Much of LLM reasoning for one instance is reusable for a whole class of *equirational* problems — problems solvable with identical reasoning steps, varying only in parameters.

### Agentic Reasoning Cache (ARC) Construction

An ARC is built by a *builder model* $\mathcal{B}$ analyzing a *source model* $\mathcal{S}$'s reasoning traces:

1. **Parameterization analysis**: Identify which parts of the input can vary without changing the reasoning
2. **Code generation**: Express the reasoning trace as a composition of *semantic functions*
3. **Async verification**: ARC entries are code — reviewable asynchronously by humans or AI judges
4. **Many-to-one mapping**: Problem instances → natural-language descriptions of semantic functions that solve them

### Semantic Functions

The key data structure: a Python callable referencing everything by **natural-language descriptions and examples** rather than fixed identifiers.

```python
f: Context → dict[str, Any]
```

Runtime primitives:
- `load(ctx, query)`: retrieve a value from layered context memory via vector similarity
- `save(ctx, description, value)`: store a labeled value
- `compute(ctx, expression)`: evaluate a mathematical or string expression
- `call(ctx, function_description)`: discover and invoke the right semantic function by description

**Semantic discovery** is the key: a caller doesn't need to know the callee's name, only what it does. Changes to a dependency's implementation don't require updating callers, as long as the description semantically entails the old one.

This design makes any previously cached computation reusable across any semantically equivalent scenario, regardless of phrasing, structure, or parameter values.

```python
# Standard function — breaks on rephrase
def compute_divisors(n):
    return count_from_factorization(n)

# Semantic function — robust to rephrase
def semantic_fn(ctx):
    n = load(ctx, "the number to analyze")
    factors = compute(ctx, "prime factorization of {n}")
    result = compute(ctx, "count divisors from {factors}")
    save(ctx, "number of positive divisors", result)
```

### Gated Deployment

```python
def serve(request):
    try:
        result = cache.use(request)
        if result is not False:
            return result
    except CacheInvocationError:
        pass
    return llm(request)  # fallback to full model
```

The cache is tried first; misses and errors fall through to the source model. This makes deployment safe — a cache failure never breaks the system.

## Comparison to Prior Work

| Method | Robustness | Hit correctness | Explainability | Cost |
|--------|-----------|----------------|----------------|------|
| Monolithic (prefix caching) | Low | High | Medium | Cheap |
| Semantic similarity search | Medium | Low | Low | Cheap |
| Modular (prior work) | High | Medium | Medium | Expensive |
| **CPU-ARC** | **High** | **High + verifiable** | **Full (code)** | **Cheap** |

CPU-ARC is a hybrid: monolithic in that the whole cache entry is inspectable and certifiable, modular in that individual semantic functions are independently reusable. Prior modular methods required a full generation even on cache hits; ARC's code executes without LLM involvement.

## Reading the Results

Evaluated on difficult math and science benchmarks (MATH-500, OlympiadBench, etc.):

**Cost savings**: ARC invocation costs **16–291x less** than the source model (varies by task complexity and cache hit rate). The 291x figure is for mathematical problems with high equirationality (many instances share the same reasoning).

**Construction cost**: 12–84x the original reasoning cost (one-time). Break-even point: on average ~40 invocations per cache entry.

**Accuracy**: Matches frontier model performance on benchmarks used to measure intelligence. This is the deliberate design choice — test on the hardest cases first.

**Robustness**: ARC correctly handles reparameterized inputs (different numbers, same reasoning structure) and rephrased inputs (different wording, same meaning), as shown in the Table 1 example in the paper.

## Key Notes of This Paper

### Equirationality: The Core Abstraction

For a query template $q(\bullet_1, ..., \bullet_n)$ with instance $q(o)$, there exists a set $\Pi$ such that for all $\pi \in \Pi$, $q(\pi)$ is **equirational** to $q(o)$ — solvable with identical reasoning.

**Assumption A1**: For every solvable $q(o)$, there exists a program $p(\bullet_1, ..., \bullet_n)$ with SLM access such that $p(\pi) = \mathcal{A}(q(\pi))$ for all $\pi \in \Pi$.

This says every solvable problem is solvable with code + SLM, in a way that generalizes to equirational variants. The correctness claim is about the class, not just the instance.

**Assumption A2**: Builder $\mathcal{B}$ can construct $p$ from $q(o)$ and $\mathcal{S}(q(o))$.

Together, these assumptions are the theoretical foundation of ARC. The experimental results on hard benchmarks suggest they hold broadly.

### Semantic Discovery at Runtime

When an ARC entry is invoked on a new parameter vector, the runtime uses vector similarity to match descriptions to functions. No hard-coded routing. This means:

1. A cache built for "count divisors of N" is automatically discoverable as "how many factors does N have?" — same description, different phrasing
2. When a semantic function is updated (say, generalized), callers that matched its old description continue working without updates

The vector similarity is computed by the **target SLM** $\mathcal{T}$ — the small, cheap model that runs the cache. This is a clever design: the same SLM that runs the cache also handles natural language grounding.

## Limitations

- **High construction cost**: 12–84x the source model's cost. Only amortized by high reuse
- **No hard correctness guarantees**: ARCs are probabilistic — if the source model reasoned incorrectly, the error is cached. Async human review is suggested but not automated
- **Coverage bounded by equirationality**: Problems without equirational variants gain no benefit
- **Currently validated on structured domains**: Math and science benchmarks. Generalization to open-ended agentic tasks (which the authors claim) remains empirically unverified beyond the initial exploration

## Future Work

**Authors' direction**: Deploy on real production agents. The choice to test on hard math/science benchmarks is deliberate — if ARCs can match frontier performance there, they should handle most agentic workloads.

**Additional promising directions**:
1. **API-call agents**: Enterprise systems making repetitive API calls with variable payloads are ideal ARC targets — same structure, different data
2. **Domain-specific SLM as target model**: A medical or legal SLM fine-tuned as the ARC target could provide domain-expert-level cached reasoning at dramatically lower cost than frontier models
3. **Continuous learning ARCs**: The paper notes ARCs are continually trainable. A system that grows its cache over time, with periodic human validation of high-frequency entries, could approach near-zero marginal cost for well-covered problem classes
4. **ARC + on-device prefill**: Combine with KV cache techniques to serve the SLM-based ARC with minimal memory footprint on mobile hardware

## Implications for Edge / On-Device Deployment

CPU-ARC is architecturally ideal for edge deployment:

1. **CPU execution**: No GPU required. The ARC runs on the same CPU available in any smartphone or IoT device
2. **SLM-only at inference**: The small target model $\mathcal{T}$ handles all inference-time NLP. Models at the 1–3B scale fit comfortably in smartphone memory
3. **Zero cloud cost on cache hits**: For agentic applications with repetitive patterns (home automation, sensor analysis, educational tutoring), hit rates can be very high — approaching zero cloud API calls
4. **Offline operation**: Once constructed, ARCs run without network connectivity. Edge devices that work offline can use cached frontier reasoning
5. **Tiered deployment**: Build caches from cloud frontier models, deploy on edge with SLMs. The intelligence of the frontier model is encoded in the cache; the edge pays only SLM costs

## Links

[Original Paper](https://arxiv.org/abs/2611.09891)
