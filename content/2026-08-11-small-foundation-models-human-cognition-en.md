Title: Small Foundation Models of Human Cognition and Behaviour
Date: 2026-08-11
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: Training 135M-to-14B parameter models on 10.7 million human behavioral choices shows that in-distribution cognitive modeling saturates at 0.6–1B parameters, but out-of-distribution generalization to novel task structure improves sharply with scale — revealing both the promise and hard limits of small foundation models as cognitive proxies.

## Why This Paper Matters

Large language models are increasingly used as stand-ins for human participants in psychology experiments — a practice sometimes called "LLM-as-subject." The idea is appealing: running a model costs a fraction of recruiting and paying human subjects, and the model can participate in thousands of experiments simultaneously. But two fundamental questions remain open:

1. **Does this actually require a large model?** If a 1B-parameter model can match a 70B model on behavioral prediction, the case for using LLMs as cognitive proxies becomes much more practically accessible.
2. **Do these models process task structure, or exploit statistical shortcuts?** A model that memorizes behavioral patterns without understanding the task structure is a flawed proxy — it will fail on any experiment that differs meaningfully from its training distribution.

This paper answers both questions empirically by training fourteen models from 135M to 14B parameters on Psych-101, a curated dataset of 10.7 million trial-level human choices from 160 controlled psychological experiments. The scale sweep and diagnostic experiments provide unusual clarity about what small models can and cannot do as cognitive proxies.

## Core Technical Contribution

The core contribution is empirical: a systematic scaling study of foundation models on human behavioral data, paired with ablation diagnostics that reveal *what* the models learn rather than just *how well* they perform.

**Dataset: Psych-101**
160 published psychology experiments covering diverse choice paradigms (decision under risk, social preferences, learning tasks, cognitive games). Each trial provides the experimental context, stimuli, feedback from prior trials, and the participant's choice. Total: 10.7 million trial-level observations. The train/test split separates held-out participants (in-distribution) from held-out task structures (out-of-distribution).

**Model sweep:**
Fourteen models from 135M to 14B parameters, across four architecture families. All fine-tuned on the same Psych-101 training set.

**Key findings:**

*In-distribution (held-out participants from seen experiments):*
Models from 135M to 14B fall within a narrow performance band — effectively against a ceiling. The 0.6B-to-1B range matches the 70B baseline on held-out participants. Scale barely matters once you exceed a few hundred million parameters.

*Out-of-distribution (held-out task structures):*
The band opens into a markedly steeper scaling gradient. Larger models generalize better to novel task structures not seen during training. This gap is what distinguishes a genuine cognitive model from a pattern-matcher.

## Comparison to Prior Work

Prior work using LLMs as cognitive proxies has typically used off-the-shelf instruction-tuned models (GPT-4, Claude, Llama) without systematic scale comparison or diagnostic probing. The key advances here are:

1. **Scale sweep with in-distribution/out-of-distribution separation:** Prior work rarely distinguished between these two evaluation regimes, which have very different implications for practical use.
2. **Diagnostic channel ablations:** Rather than treating the model as a black box, the authors systematically remove information channels and measure the effect.
3. **The 70B baseline:** Using a 70B model as the reference reveals that smaller models already match its in-distribution performance, which changes the cost calculus for this application domain.

No explicit numerical comparison to other cognitive modeling baselines (reinforcement learning agents, Bayesian models, etc.) is reported in the available abstract.

## Reading the Results

The most striking result is the **in-distribution ceiling**: the performance gap between a 135M and a 14B model on held-out participants is small. Practitioners who want to use an LLM as a cognitive proxy for a psychology experiment that falls within a known paradigm space can do so with a 0.6-1B model at dramatically lower cost than a frontier model.

The **out-of-distribution result** is the important warning. When the task structure is genuinely novel — outside the distribution of the 160 training experiments — larger models are clearly advantaged. A small fine-tuned model that performs well in-distribution may fail on a novel paradigm, limiting its usefulness for open-ended psychological research.

The **channel ablation** is the most diagnostic result. Stripping stimuli and outcome feedback content (while keeping task instructions and choice history) destroys 75.7% of learned information and pushes models below chance. This is strong evidence that the models are not relying on surface patterns in choice history (which would be a shallow statistical shortcut). The models need the actual content of stimuli and feedback — they are doing something like task-directed processing, not history extrapolation.

The **trial-order permutation** result complements this: permuting trial order has no effect on tasks with independent trials (where order carries no information) but does hurt performance on tasks where trial order determines state (like sequential learning tasks). The models are sensitive to temporal structure when it matters, and correctly indifferent to it when it doesn't.

## Key Notes of This Paper

The **scaling-regime split** is the paper's conceptual core. In-distribution, the task is essentially pattern completion: predict how a human with a given behavioral history responds to a known stimulus type. This saturates at relatively small scale. Out-of-distribution, the task requires generalization to novel task structure — which requires more representational capacity and scales with model size.

This split maps onto a well-known dichotomy in cognitive science between **exploitation** (applying a known pattern to a familiar situation) and **exploration** (generalizing structure to a new situation). Small models are good exploiters but weak explorers.

The **M factor in the KL truncation context** (from the companion paper) has an interesting analogy here: partial-mass supervision — the models receive rich supervision from stimuli and feedback but only partial mass (not all behavioral information is captured by the prompt). Just as in distillation, the content of what you condition on matters more than the quantity of parameters once you're past a capacity threshold.

The **75.7% information destruction** result is quantitatively striking. The four prompt channels (instructions, stimuli, feedback, choice history) are not equally informative: stimuli and feedback together carry the vast majority of behavioral signal, while choice history alone is nearly uninformative. This mirrors findings in human cognitive psychology that stimulus salience dominates over history in many choice paradigms.

## Limitations

The study is bounded by the 160 experiments in Psych-101. These represent a specific slice of behavioral psychology — primarily Western, laboratory-based, English-language paradigms. Generalization claims must be evaluated against this training distribution.

Out-of-distribution performance, while better for larger models, is not reported quantitatively in the available summary. The claimed "markedly steeper scaling gradient" needs the actual numbers to assess whether the improvement is practically significant or only marginal.

The architectural comparison (four families) is mentioned but the specific families and their relative performance are not reported in the abstract. Understanding whether architecture choice or scale drives performance at the lower end of the parameter range is an open question.

The models are fine-tuned LLMs — they carry behavioral biases from their pretraining on text corpora, which may not align with human behavioral tendencies. The extent to which these biases affect the cognitive proxy validity is not addressed.

## Future Work

The authors explicitly position small cognitively fine-tuned models as **noise ceiling estimators** for psychological experiments — a tool for determining the theoretical maximum performance achievable by a computational model on a given experiment before investing in human data collection. Promising extensions:

- **Psych-101 expansion:** 160 experiments is a useful start but small relative to the diversity of behavioral paradigms in the literature. A larger training distribution would improve both in-distribution ceilings and OOD generalization.
- **Architecture-specific findings:** The four-family sweep contains unpublished information about which architectures are more or less suited to behavioral prediction.
- **Cross-cultural validation:** Behavioral experiments are culturally situated; whether models trained on Western laboratory data can proxy for behavior in other cultural contexts is unexplored.
- **Active experimental design:** A sufficiently good behavioral model could be used to *design* experiments — predict which stimuli would maximize information gain about a psychological construct — rather than just simulate participants.
- **Alignment with mechanistic interpretability:** The channel ablation shows what information the models use; mechanistic interpretability tools could reveal *how* they use it, connecting model internals to cognitive theory.

## Implications for Edge / On-Device Deployment

The in-distribution saturation result has a direct practical implication: **you don't need a frontier model to simulate behavioral data** in domains that are well-represented in training. A 0.6-1B model fine-tuned on behavioral data can match a 70B model for held-out participants in known experimental paradigms. For applications like:

- Personalizing user interfaces by predicting behavioral responses
- Simulating survey respondents for pre-deployment testing
- Building cognitive assistants that model user mental states

...the performance is already achievable at SLM scale. The hard constraint is the training distribution: if the target application involves genuinely novel user behaviors or tasks outside the fine-tuning distribution, larger models or continued fine-tuning on in-domain data are necessary.

The channel ablation result also informs SLM system design: behavioral prediction systems need to provide full stimulus and feedback context, not just compressed action histories. Compressing the context to save tokens will degrade prediction quality disproportionately.

## Links

[Original Paper](https://arxiv.org/abs/2608.05224)
