Title: Harness-Aware Distillation: Teaching Small Language Model Agents to Use Their Framework
Date: 2026-10-08
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: HAD (Harness-Aware Distillation) fixes three fundamental failures in on-policy distillation for harness-equipped agents, raising ALFWorld performance from 43% to 57% by explicitly teaching small models to leverage harness information.

## Why This Paper Matters

When deploying language model agents in practice, the model never runs alone. It operates inside a *harness*: the surrounding software that manages context, provides feedback, tracks environment state, and interfaces with tools. Yet when researchers distill a large agent into a smaller one, they typically treat the harness as just another piece of the input — a passive context string.

This paper identifies why that assumption breaks down and proposes a principled fix. The insight is sharp: since the harness stays fixed when we swap the teacher for a smaller student, what the student really needs to learn is not the full teacher output, but specifically the teacher's ability to act on what the harness reports. Standard on-policy distillation (OPD) doesn't teach this distinction.

The problem is genuinely hard because harness-dependent behavior is sparse and contextual. The harness changes the teacher's action only at certain steps — when it reports a collision, an invalid action, or a state update that matters. OPD treats all steps uniformly, so those critical harness-responsive steps get no special emphasis. Meanwhile, the harness can also reveal that the teacher made an error (proposing to use an object the agent doesn't hold), but OPD blindly imitates those errors too.

## Core Technical Contribution

HAD adds two components on top of standard harness-equipped response distillation (OPD with the harness enabled for both teacher and student):

**Harness-Aware Action Preference Learning**

The key idea: query the *same* teacher twice — once with full harness information ($\tilde{x}_t^{\theta}$) and once without ($x_t^{\theta}$, stripped of harness annotations). This gives two teacher actions for the same student state:

$$\tilde{a}_t^{\phi} \sim \pi_{\phi}(\cdot \mid \tilde{x}_t^{\theta}) \quad \text{(with harness)}$$
$$a_t^{\phi} \sim \pi_{\phi}(\cdot \mid x_t^{\theta}) \quad \text{(without harness)}$$

The harness-conditioned action is treated as preferred. Both are scored under the *student's own reasoning trace* $\tilde{z}_t^{\theta}$:

$$\Delta_t(\theta) = \frac{1}{|\tilde{a}_t^{\phi}|}\log\pi_{\theta}(\tilde{a}_t^{\phi} \mid \tilde{x}_t^{\theta}, \tilde{z}_t^{\theta}) - \frac{1}{|a_t^{\phi}|}\log\pi_{\theta}(a_t^{\phi} \mid \tilde{x}_t^{\theta}, \tilde{z}_t^{\theta})$$

A reference-free pairwise logistic loss is applied:
$$\ell_{\mathrm{pref},t}(\theta) = -\log\sigma(\beta\Delta_t(\theta))$$

Critically, scoring happens under student reasoning, not teacher reasoning. This means the student learns "after I reason about the situation, prefer the action the harness justified" — which is what it actually needs to do at deployment.

**Harness-Aware Action Validity Filtering**

A validity checker $V(a \mid \tilde{x}) \in \{0,1\}$ tests whether the preferred teacher action is consistent with the harness-recorded state (e.g., does the action require holding an object the harness says the agent doesn't have?). Invalid pairs are dropped from the preference term:

$$m_t = V(\tilde{a}_t^{\phi} \mid \tilde{x}_t^{\theta})$$

**Final Objective**:

$$\mathcal{L}_{\mathrm{HAD}}(\theta;\lambda) = \mathbb{E}_{\nu}\left[\ell_{\mathrm{dist},t}(\theta) + \lambda m_t \ell_{\mathrm{pref},t}(\theta)\right]$$

$\lambda$ is automatically balanced each update using gradient-norm balancing between the two loss terms.

## Comparison to Prior Work

The most important comparison is against adding the harness to standard OPD:

| Method | Harness Utilization | Performance (ALFWorld) |
|--------|--------------------|-----------------------|
| Zero-shot student | 65.7% | 43.1% |
| OPD + harness | 73.1% | 43.5% |
| **HAD** | **81.0%** | **57.4%** |

The striking finding: simply adding the harness to OPD barely improves performance (43.1% → 43.5%), despite improving harness utilization. The student reads harness information but doesn't act on it correctly. HAD closes this gap by explicitly teaching the behavioral consequence of harness information.

Ablation results show each HAD component matters independently. Removing the validity filter degrades performance (invalid teacher targets add noise). Using full-response preferences instead of action-only preferences hurts. Positive-only supervision (imitating the harness-conditioned action without contrasting it with the harness-free action) is also weaker — the contrastive signal matters.

## Reading the Results

The paper evaluates on ALFWorld (household tasks in a simulated home environment) and multiple other long-horizon agent benchmarks. Key behavioral differences:

- **Fewer unproductive loops**: Baseline OPD students tend to repeat the same failed action sequence ("stall"). HAD students read the harness message "Your last action had no effect" and change strategy, because they were trained to attend to exactly that signal.
- **Higher error recovery rate**: When a HAD student makes a mistake, subsequent harness feedback redirects it more reliably than in OPD baselines.
- **No task rewards required**: HAD uses only the teacher and the harness structure — no success labels, no step-level reward signals. This is practically important since defining task success is itself often expensive.

## Key Notes of This Paper

The mathematical core is the **contrastive preference signal** in equation (5). Let's understand why it contains information that response distillation alone cannot provide.

Standard distillation teaches $\pi_\theta$ to predict $\tilde{y}_t^{\phi}$ given $\tilde{x}_t^{\theta}$. This supervises the full response but conflates what the harness provided with what the teacher decided. The student doesn't learn *which part of the teacher's behavior was caused by the harness*.

The preference term provides exactly this missing signal. By computing $\Delta_t(\theta)$ under the **student's reasoning** (not the teacher's), it measures: "given how the student currently reasons about the situation, does it prefer the harness-influenced teacher action over the baseline teacher action?" If $\Delta_t(\theta) < 0$, the student currently assigns higher probability to the harness-free action — meaning it hasn't learned to leverage the harness feedback. The preference loss pushes $\Delta_t > 0$.

The scoring formula $\frac{1}{|a|}\log\pi_\theta(a \mid \tilde{x}, \tilde{z})$ averages log-likelihood over action tokens only (not reasoning tokens). This is intentional: reasoning quality is supervised by the response distillation term, while the preference term focuses specifically on the final action decision given the same reasoning context.

The validity mask $m_t$ handles a subtle issue: teachers can hallucinate in student-visited states. When the student's trajectory diverges from the teacher's distribution, the teacher may propose actions that contradict observable facts in the harness. $V$ catches these by checking factual consistency (e.g., "use knife" when harness shows agent holds no knife), preventing the student from learning contradictory signals.

## Limitations

- **Validity checker implementation**: The checker $V$ is rule-based and environment-specific. Generalizing to open-domain environments where validity isn't easily checkable remains an open problem.
- **2× teacher query cost**: Each training step requires querying the teacher with and without harness information — roughly doubling inference costs during distillation.
- **Deterministic harness assumption**: HAD assumes harness information is deterministically computed from environment state. Non-deterministic harnesses (e.g., external API responses that vary) need additional treatment.
- **Simulated environment evaluation**: Results are on ALFWorld-style benchmarks. Real-world deployment with API-based tools and noisy environments remains to be validated.

## Future Work

Authors suggest: training agents robust to varying harness configurations, combining HAD with reinforcement learning for hybrid post-training.

Promising follow-on directions the paper opens:
- **Harness-agnostic distillation**: Learning to distill across different harness formats — crucial for SLM deployment where harnesses differ across platforms (iOS vs Android vs embedded)
- **On-device continual learning**: HAD's reward-free nature makes it attractive for continual fine-tuning using harness logs collected in production, without needing humans to label success/failure
- **Diagnostic harness utilization metrics**: The paper's harness utilization rate could become a standard evaluation metric for agentic SLM benchmarks
- **Multi-agent harness coordination**: Extending to scenarios where multiple small agents share a harness and need to coordinate their responses to harness feedback

## Implications for Edge / On-Device Deployment

HAD directly targets the core challenge of on-device agentic AI: you need a small model that can reliably act on feedback from the surrounding system.

Voice assistants on mobile devices run inside substantial software stacks — conversation history managers, calendar APIs, OS permission systems, sensor interfaces. These are all harnesses. A 3B model distilled without HAD may "see" the harness feedback but fail to act on it, creating the frustrating experience of an assistant that ignores error messages.

HAD's practical implication: **when distilling large agents for edge deployment, harness compatibility should be a first-class training objective, not an afterthought**. Benchmark metrics should include harness utilization rate alongside task success rate.

The reward-free property also matters here. On-device models often need continual adaptation as users' personal data and preferences evolve. HAD could power this adaptation using only harness interaction logs — no human labels needed.

## Links

[Original Paper](https://arxiv.org/abs/2610.02858)
