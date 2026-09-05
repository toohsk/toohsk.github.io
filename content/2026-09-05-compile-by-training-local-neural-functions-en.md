Title: Compile by Training: Turning Natural-Language Specifications into Local Neural Functions
Date: 2026-09-05
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Compile by Training converts a natural-language description of a text transformation into a small, reusable neural adapter that runs locally on a compact interpreter — achieving 83.6% semantic accuracy on FuzzyBench-Hard without any remote model dependency at inference time.

## Why This Paper Matters

A wide class of software tasks can be described easily in natural language but implemented poorly with rules: "translate this informally-phrased user text into structured JSON," "normalize names across formats," "extract medical entities from free text." For these, two options exist today: hand-craft brittle regex rules, or call a large remote language model at every inference. The first is fragile; the second introduces latency, cost, and dependency on an external provider for every function call.

The fundamental tension is between **expressiveness** (what rules can capture) and **deployability** (whether the solution can run locally at low cost). Prior work on "Program-as-Weights" showed that a fast compiler can synthesize a small network from a natural language spec in seconds, but leaves a large fraction of hard cases unsolved.

This paper asks a different question: what if compilation is allowed to be *slow* (minutes), but the resulting artifact is a proper **neural function** — deterministic, storable, versionable, and composable — that runs on local hardware with no remote calls? This is the "compile by training" paradigm, and it extends the program-as-weights family in the same direction that traditional compilers extended hand-written assembly: investing compute once at compile time to produce an efficient, portable artifact that runs efficiently forever after.

For on-device and edge deployment, this matters enormously. A compiled neural function carries no runtime dependency on a cloud provider, no variable latency, and no per-call cost. Once compiled, it is a file you can store, version, and ship like any other software artifact.

## Core Technical Contribution

**Compile by Training** proceeds in two phases:

### Compile Time (~1 minute)

Given a natural-language specification $s$ describing a desired text function:

1. **Teacher models generate examples.** One or more large teacher LLMs produce input-output pairs $(x_i, y_i)$ that instantiate the specification. These examples are task-specific: the teachers are not retained at inference, only their examples are.

2. **Fine-tune a small adapter on the examples.** A compact adapter is trained on top of a small, pre-trained "interpreter" model to handle the specific task. The adapter is the compiled artifact.

The key design choices:
- The interpreter is **small and fixed** — it is the runtime cost at inference, not the teacher.
- The adapter is **task-specific and lightweight** — it captures the semantics of the specification without encoding a general language model.
- The compilation is **expensive once, free forever** — one minute of GPU time produces an artifact that runs in milliseconds.

### Inference Time (milliseconds)

The compiled function executes: the small interpreter + adapter processes each input, producing the specified output. No teacher, no remote call, no provider dependency. The function is deterministic (for the same input, it always produces the same output) and fast.

### Benchmark: FuzzyBench-Hard

The evaluation uses FuzzyBench-Hard, a subset of test cases where the Program-as-Weights fast compiler (a prior method that compiles in seconds) produces **zero exact matches**. This isolates the class of tasks that require the full "compile by training" approach.

On FuzzyBench-Hard, compile by training achieves **83.6% semantic accuracy** — from 0% with the fast compiler to 83.6% by trading 1 minute of compile time for a trained adapter.

### Deployments

The authors deploy the compiler in a public interactive service and demonstrate three compiled functions:
- A **multi-site website helper** that maps natural-language user intent to site-specific actions
- A **language-controlled 3D avatar** that translates text instructions into avatar behavior
- A **bidirectional English-Claudish translator** (Claudish is a constructed language used as a test bed)

These demonstrate that compiled functions compose — a website helper can invoke a translator, for example — and that the paradigm works across domains beyond standard NLP benchmarks.

## Comparison to Prior Work

| Method | Compile time | Semantic accuracy (FuzzyBench-Hard) | Runtime dependency |
|---|---|---|---|
| Program-as-Weights (fast) | Seconds | 0% (exact match) | None (weights) |
| **Compile by Training (ours)** | ~1 minute | **83.6%** | None (adapter) |
| Zero-shot GPT-4 / Claude | None | High (varies) | Remote API always |

The comparison reveals the core trade-off: fast compilation (seconds) produces shareable weights but fails on hard cases; compile by training trades compile time for dramatically higher accuracy on hard cases while preserving the zero-inference-dependency property.

## Reading the Results

**83.6% on FuzzyBench-Hard** is particularly meaningful because this subset was chosen precisely as the failure mode of the prior approach. For the cases where fast compilation works, it remains the preferred choice (seconds vs. minutes). For cases requiring richer semantics — complex transformations, nuanced conditions, multi-step logic — compile by training fills the gap.

**Compile time of ~1 minute** is within the tolerance of a software development workflow: it is comparable to running a test suite or a Docker build. A developer writing a specification once and waiting one minute for a deployable function is a reasonable human-time cost.

**Runtime behavior:** Because the resulting adapter runs on a local interpreter with no remote call, the per-input latency is bounded by local hardware and the adapter size, not by network round trips. For applications where the same transformation is applied to thousands of inputs (batch processing, real-time classification), this economics improvement compounds.

## Key Notes of This Paper

### The "Compile" Analogy

The paper draws a precise analogy to traditional compilation:
- **Source code** = the natural-language specification
- **Compiler** = the process that runs teacher models + fine-tuning
- **Binary** = the adapter weights stored to disk
- **Runtime** = the small, fixed interpreter model

Just as a C compiler converts human-readable code into machine instructions that run efficiently on hardware, compile by training converts a human-readable specification into neural weights that run efficiently on a compact interpreter. The "binary" is the adapter — not the full LLM, not the teacher — and it can be stored, versioned in git, and deployed to any device running the interpreter.

### What the Adapter Actually Learns

The training signal comes entirely from teacher-generated examples. The adapter is not fine-tuning on a general corpus; it is learning to implement a specific function. Mathematically, the training objective minimizes:
$$\mathcal{L} = -\sum_i \log p_{\text{interpreter+adapter}}(y_i | x_i)$$

where $(x_i, y_i)$ are the teacher-generated input-output pairs for specification $s$. This is standard supervised fine-tuning, but the key is what it is fine-tuning *on*: a synthetic dataset that was generated to instantiate exactly the semantics of $s$. The adapter weights are, in a real sense, a compiled encoding of the specification.

### Why Hard Cases Require Training (Not Just Prompting)

FuzzyBench-Hard contains cases where the specification describes transformations with:
- Complex conditional logic ("if the first token matches X, then Y, else Z")
- Implicit context-dependence that is hard to specify exhaustively
- Fuzzy boundaries that require learned generalization beyond explicit rules

The fast compiler fails here because it produces hard-coded weight patterns that cannot generalize. A trained adapter generalizes because it learns a mapping, not a lookup table. This is the same reason why learned models outperform hand-crafted rules for most real-world NLP: the training signal encodes generalization over the distribution of inputs.

### Composability

Compiled functions can call each other, enabling a **functional composition** model:
```
translate_to_json = compile("convert user description to JSON with fields: name, age, location")
normalize_names = compile("standardize person names to Last, First format")
pipeline = compose(normalize_names, translate_to_json)
```
Each compiled function is a small adapter; composition stacks adapters. This enables complex pipelines where each step was specified in natural language and compiled independently.

## Limitations

- **Compile time vs. fast compiler:** For specifications that the fast compiler handles (the easy subset), the 1-minute compile time is unnecessary overhead.
- **Teacher quality determines ceiling:** If teacher models generate incorrect or noisy examples, the adapter learns wrong behaviors. The specification must be clear enough for teachers to instantiate correctly.
- **Adapter-to-interpreter coupling:** An adapter trained for interpreter model A does not transfer to interpreter model B. Upgrading the interpreter requires recompilation.
- **Evaluation scope:** FuzzyBench-Hard is a specific benchmark; the 83.6% accuracy number covers the paper's test distribution. Specifications that fall outside this distribution (very long outputs, multi-modal tasks, real-time streaming) were not evaluated.
- **No formal correctness guarantee:** Unlike traditional compilation (where correctness is proven for well-typed programs), compile by training offers probabilistic accuracy. The 16.4% failure rate on FuzzyBench-Hard represents cases where the trained adapter does not correctly implement the specification.

## Future Work

**Authors' suggested directions:** The interactive service deployment suggests moving toward a user-facing compilation-as-a-service model — users write specifications, the service compiles adapters, and adapters are served or downloaded. Version control for specifications + adapters (like a package registry) is a natural extension.

**Additional promising directions:**
- **Continuous recompilation:** If the specification is refined based on test failures, recompiling with updated teacher examples and merging adapters (low-rank merge) would allow iterative development of neural functions without full retraining.
- **Interpreter standardization:** A shared, widely-deployed small interpreter (similar to how the JVM standardized Java bytecode) would enable adapter portability across devices. An adapter compiled once could run on any device with the standardized interpreter.
- **Few-shot specification:** Today, the specification must be written in natural language detailed enough for teacher models to generate correct examples. A meta-learning approach could learn to compile functions from fewer specification examples, reducing the human authoring burden.
- **Formal verification of compiled functions:** For safety-critical applications, post-compilation testing (exhaustive on a finite specification domain, or property-based testing) could provide bounded correctness guarantees.

## Implications for Edge / On-Device Deployment

This is one of the most directly edge-relevant papers in the current cycle. The core value proposition: **write the specification on a powerful machine, compile once, deploy the adapter to any device running the interpreter.**

Practical implications:
1. **No cloud dependency at inference.** A deployed adapter has no network calls. This enables air-gapped deployments, privacy-preserving processing, and low-latency edge applications.
2. **Predictable inference cost.** The inference cost is bounded by the adapter + interpreter, not by a remote model's latency or availability.
3. **Versionable and auditable.** Adapters are files. They can be diffed, rolled back, and audited — properties that are impossible with cloud API calls.
4. **Natural fit for function libraries.** An organization can build a library of compiled neural functions (entity extraction, intent classification, text normalization) that run locally in their applications, with no per-call cost.

For mobile and embedded deployment, the critical open question is interpreter size. The paper uses a "compact interpreter" without specifying exact parameters. If future work establishes a 1B-3B parameter standardized interpreter, compiled adapters could enable sophisticated NLP functions on modern smartphones.

## Links

[Original Paper](https://arxiv.org/abs/2609.04199)
