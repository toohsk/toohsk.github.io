Title: DFM Mimir v1: Frontier 1B Performance Using Only Permissible Data
Date: 2026-08-18
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: DFM Mimir v1 is a 1B-parameter Hierarchical Reasoning Model trained exclusively on permissible, openly-licensed data that matches or exceeds 2–3B class models on English and dominates all competitors on Danish, demonstrating that legal data constraints need not mean capability constraints.

## Why This Paper Matters

Most publicly available SLMs are trained on web-crawled data of uncertain provenance—copyrighted books, personal blog posts, scraped social media content. For real-world deployment in healthcare, legal systems, and government, this is a fundamental blocker. You cannot ship an AI assistant to a hospital if you cannot account for the legal basis of every token it learned from.

This constraint is even more acute for low-resource languages. Danish—the target language for the Danish Foundation Models (DFM) project—has far less high-quality public data than English. Training a capable language model from scratch on exclusively permissible Danish data was widely considered impractical.

DFM Mimir v1 directly disproves this assumption. By focusing on post-training data and leveraging the Hierarchical Reasoning Model (HRM) framework, they trained a 1B-parameter model from scratch using only 70.5B tokens of permissible data—and achieved results that beat models twice or three times its size on multiple benchmarks, while *dramatically* outperforming all competitors on Danish tasks, including 8–9B Danish-specialized models.

The core insight: **architectural efficiency can compensate for data scarcity, if you choose the right architecture.**

## Core Technical Contribution

### The HRM-Text Architecture

Mimir v1 is built on the HRM-Text architecture, which differs fundamentally from standard flat Transformer stacks.

Instead of simply running a token through L independent transformer layers once, HRM-Text uses a **hierarchical cycle structure**:

- **H-cycles (High-level cycles)**: Process at a more abstract reasoning level
- **L-cycles (Low-level cycles)**: Process at the token-level detail layer

Mimir v1 uses **2 H-cycles × 3 L-cycles**, meaning any given token's representation passes through the full layer stack 6 times before producing logits. The intuition: first H-cycle handles pattern recognition and fact retrieval; second H-cycle performs logical operations and answer synthesis.

**Model hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Hidden size | 1,536 |
| Attention heads | 12 |
| FFN expansion | 4× |
| H-cycles | 2 |
| L-cycles | 3 |
| Positional encoding | RoPE (θ=10,000) |
| Normalization | Pre-norm LayerNorm (ε=10⁻⁶) |
| Backprop truncation | 5 steps |
| Total parameters | ~1B |

**Truncated backpropagation (5 steps)** prevents gradient explosion across cycles while still allowing the model to optimize the reasoning steps most proximal to the output.

### Data Strategy: "Transplant Datasets"

The most operationally innovative contribution is the **synthetic data transplantation pipeline**:

```
Non-permissible source data
    ↓
Generate synthetic replacement via Gemma4 31B
    ↓
Quality audit (acceptance rates vary from single digits to >90%)
    ↓
Permissible transplant dataset
```

When a high-value dataset (e.g., Flan NIV2, Platypus) doesn't meet DFM's permissibility standards, they don't just drop it—they regenerate it synthetically and audit the quality. The acceptance rate variance (from <10% to >90% depending on the task type) reveals how some task types are genuinely hard to synthesize faithfully.

**Training corpus composition (161 datasets, 70.5B tokens/epoch):**
- Danish instruction & knowledge: 22.1% (15.56B tokens)
- English instruction: 19.3% (13.58B tokens)
- Sapient mixed: 17.0% (11.92B tokens base + 70M transplant)
- Math & reasoning: 14.8% (10.40B tokens)
- Synthetic, agentic, translation, science: 27.0%

### Shift from MCQ to Free-Form Generation

The original Sapient data was dominated by multiple-choice questions. Mimir v1 deliberately **shifted training toward free-form generation**:

- Math (OpenMathInstruct-2, AceReason): produces numerical/symbolic answers scored by exact match
- Danish instruction: open-ended QA and summarization
- English instruction: Dolci, Tulu 3, Nemotron SFT (primarily free-form)

This explains why Mimir v1 scores particularly well on generative benchmarks (GSM8K: 89.9, DROP: 83.1) but has more modest MCQ performance on MMLU (57.5).

### Training Setup

Trained from scratch for 1.65M steps over ~3 weeks on 8× NVIDIA B200 GPUs (180 GB HBMe3 each), using FSDP with bfloat16 computation:

- Optimizer: AdamW, peak LR 3×10⁻⁴, 2,000-step linear warmup, constant thereafter
- Global batch size: 262,144 tokens (4 contexts × 4,096 length × 16 per-device, 2-step grad accumulation)
- Tokenizer: Gemma-4 tokenizer

## Comparison to Prior Work

Mimir v1 is evaluated against 1B, 2-3B, and 4-5B class models on 20 benchmarks across English, Math & Code, and Danish tasks.

**English benchmarks** (BoolQ, Winogrande, Hellaswag, MMLU, ARC-C, DROP, GovRep):

| Model | Avg |
|-------|-----|
| **Mimir 1B** | **69.0** |
| HRM-Text 1B | 66.1 |
| SmolLM3 3B | 63.1 |
| Qwen 3.5 2B | 58.2 |
| Qwen 3.5 0.8B | 50.5 |

Mimir 1B outperforms SmolLM3 3B (3× its parameter count) on English average.

**Math & Code** (GSM8K, MATH, HumanEval):

| Model | Avg |
|-------|-----|
| SmolLM3 3B | 67.9 |
| Gemma 4 E2B (think) | 70.5 |
| **Mimir 1B** | **64.1** |
| HRM-Text 1B | 46.9 |
| Gemma 3 1B | 43.2 |

Mimir 1B shows a 36.7% improvement over HRM-Text 1B on Math & Code—the same base architecture trained with Mimir's data strategy.

**Danish benchmarks** (10 tasks including NER, GEC, PIQA, WikiQA, translation):

| Model | Avg |
|-------|-----|
| **Mimir 1B** | **56.8** |
| Munin-Apertus 8B | 45.6 |
| Munin-Mistral 8B | 45.6 |
| Munin-Qwen 9B | 43.9 |
| Gemma 3 1B | 36.8 |
| HRM-Text 1B | 21.7 |

**Mimir 1B beats all Danish-specialized 8–9B models.** This is the headline result of the paper: at 1/8th the parameter count, it achieves 25%+ better average performance on Danish tasks than models explicitly designed for Danish.

## Reading the Results

**What makes these numbers significant:**

1. **The 1B vs 3B gap is closing.** Mimir 1B at 69.0 English average beats SmolLM3 3B at 63.1. For practitioners considering model size vs. capability tradeoffs, this suggests HRM's cycle structure provides an effective parameter multiplier.

2. **GSM8K 89.9 at 1B is remarkable.** For comparison, GPT-3.5 (175B) scored ~57% on GSM8K when it was released. Getting 89.9% from 1B parameters—using only permissible data—demonstrates how much post-training data quality matters relative to model scale.

3. **The Danish result challenges the "need English data" assumption.** HRM-Text 1B trained on standard English-heavy data gets only 21.7 on Danish. Mimir 1B, trained with 24.7% Danish data and the transplant strategy, gets 56.8. The data composition difference explains most of this 162% improvement.

4. **MATH at 45.8 is the weak spot.** Getting MATH right requires complex multi-step symbolic reasoning. The 4,096 token context limit may be a constraint here—long mathematical proofs need more room.

## Key Notes of this Paper

### Why HRM's Hierarchical Cycles Help Reasoning

The HRM framework can be understood through the lens of **computational depth vs. parameter count**:

Standard Transformer (depth L): each token gets L processing passes through unique weights.

HRM (2H × 3L cycles): each token gets 6 passes through shared weights.

The mathematical consequence: given the same parameter budget, HRM achieves greater **effective depth** by reusing parameters across cycles. The tradeoff is that cycles share weights, so each pass cannot be maximally specialized—but the authors argue (and the results support) that the extra depth compensates.

**Truncated Backpropagation (5-step)**:

```
Forward: x → Cycle1 → Cycle2 → ... → Cycle6 → Loss
Backward: Loss → ∇Cycle6 → ∇Cycle5 → ... → ∇Cycle2 → STOP (at 5 steps back)
```

By stopping gradient flow at 5 cycles back, the optimization prioritizes the processing steps closest to the output—essentially telling the model "focus on getting the last few reasoning steps right." This is computationally efficient and empirically effective.

**Why the MCQ-to-generation shift matters:**

Multiple-choice problems reward **discrimination** (identifying which option is best). Free-form problems reward **generation** (producing a correct answer from scratch). These two objectives train different model behaviors. By shifting to free-form generation, Mimir v1 trains toward more robust knowledge representations—you can't pick the right answer if you don't actually understand the concept.

### The Transplant Dataset Pipeline in Practice

The key insight in the transplant approach: **the distillation target matters more than the distillation source.** By using Gemma4 31B as the generator, the synthetic data can still embody reasoning patterns from a much larger model, compressed into the smaller Mimir architecture. The quality audit filters out cases where the synthetic data fails to preserve the task's informational content.

## Limitations

1. **Math & Code gap**: MATH score of 45.8 vs Gemma 4 E2B's 64.2 indicates meaningful underperformance on complex mathematical reasoning. The 4,096 token context limit may be a contributing factor.

2. **Data concentration risk**: Top 10 datasets account for 66.5% of all tokens, with Sapient (16.9%) and lærebogen (11.8%) dominating. Heavy repetition (4× for lærebogen, up to 20× for some small datasets) risks overfitting to those distributions.

3. **English MCQ performance**: MMLU at 57.5 is modest compared to Qwen 3.5 4B at 75.8. The free-form training shift may have come at the cost of MCQ discriminative ability.

4. **Context window**: 4,096 tokens is limited for modern agentic tasks and long-document reasoning.

5. **Training compute**: 8× B200 GPUs for 3 weeks remains non-trivial for most research groups, despite being cheaper than typical LLM pre-training.

## Future Work

**Authors' suggested directions:**
- Investigating scaling behavior of HRM models
- Full compatibility with Gemma 4 chat template conventions

**Promising follow-on directions:**

1. **HRM + quantization interactions**: Because HRM cycles reuse weights, the precision-performance tradeoff under INT4/INT8 quantization may differ significantly from standard Transformers. Weight-sharing across cycles could mean quantization errors compound or cancel in unexpected ways.

2. **Longer context for Danish**: Danish GEC (grammar error correction) and document understanding tasks would benefit from 8K+ context. The current 4,096 limit leaves capability on the table.

3. **Cross-lingual transplant generalization**: If the transplant strategy works for Danish, can it work for Icelandic, Faroese, or Welsh? The paper suggests yes—the key ingredient is a capable teacher model for synthesis and a well-designed audit protocol.

4. **HRM + RLHF**: The cyclical processing structure may create interesting dynamics in RL-based post-training, where rewards could guide which cycle learns what.

5. **Permissible pre-training at scale**: Mimir v1 focuses on post-training. A follow-up exploring permissible pre-training from scratch (using Common Pile + similar open-licensed text corpora) at the 3-7B scale would be a natural next step.

## Implications for Edge / On-Device Deployment

**Memory footprint:**
- 1B parameters × 2 bytes (bfloat16) = ~2 GB model weight
- INT4 quantization: ~500 MB — well within range of flagship smartphones (8-12 GB RAM)
- iPhone 16 Pro (A18 Pro NPU), Pixel 9 (Tensor G4), Galaxy S25 (Snapdragon 8 Elite) can all run this comfortably

**Permissibility for regulated edge deployments:**
- Healthcare devices: HIPAA-compliant devices need data provenance clarity
- Government on-device AI: EU AI Act compliance is easier with permissible data
- Enterprise productivity: legal procurement teams can clear a permissible-data model faster than a web-scrape-trained one

**Low-resource language edge AI:**
- National-scale deployments (government apps, national news assistants, educational tools in minority languages) can now build on a principled, performant base
- The Danish results suggest this approach generalizes to other languages with similar data characteristics

**HRM architecture on edge hardware:**
- The cyclic weight-reuse means fewer unique parameters to load into SRAM/cache
- On devices with fast memory bandwidth but limited total RAM (Apple M-series, Snapdragon NPUs), this property could allow higher effective context capacity
- Inference throughput will be lower than a single-pass Transformer of the same parameter count due to the 6 passes—this tradeoff needs profiling on specific hardware targets

## Links

[Original Paper](https://arxiv.org/abs/2608.13517)
