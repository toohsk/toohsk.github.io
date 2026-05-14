Title: PAAC: Privacy-Aware Agentic Device-Cloud LLM Collaboration
Date: 2026-05-14
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: PAAC resolves the cloud-vs-device privacy dilemma for LLM agents by aligning a planner-executor decomposition with the device-cloud trust boundary, using typed placeholder tokens and a deterministic registry so that cloud reasoning never touches raw sensitive data — achieving +15–36% accuracy with 2–6× less leakage than prior device-cloud baselines.

## Why This Paper Matters

LLM agents working on personal tasks — scheduling a meeting from email threads, answering a health question, filling in a tax form — necessarily touch sensitive data. The structural problem is that the two extremes of deployment architecture each sacrifice something fundamental.

**Cloud agents** can draw on the full power of frontier models. A GPT-4 class agent can reason across a complex email chain and produce a flawless calendar event. But to do so, it must receive the email content — names, medical details, financial figures, personal relationships — in plaintext. For many users and many use cases, this is not acceptable.

**On-device agents** preserve privacy by keeping all data local. But current on-device SLMs are significantly less capable than cloud models, especially on complex multi-step reasoning. An agent that can only run a 1B-parameter model on a phone will fail on the tasks where users most need help.

The naive hybrid — classify queries as sensitive or not-sensitive and route accordingly — doesn't work for agentic tasks. Real tasks blend sensitive and non-sensitive content at a fine-grained level: a task might require reasoning about a person's calendar availability (non-sensitive), their health condition (sensitive), and a publicly available restaurant's phone number (non-sensitive), all in one planning step. You can't cleanly separate these at the query level.

Existing sanitization approaches (NER-based redaction, differential privacy mechanisms) have a critical failure mode in agentic contexts: they break **structural fidelity**. When an agent needs to call `create_calendar_event(name="John Smith", time="3pm")`, redacting "John Smith" to `[PERSON]` produces an action that cannot execute. The sanitizer must preserve the semantic role of each value even while hiding its content — and current tools don't do this.

## Core Technical Contribution

PAAC (Privacy-Aware Agentic Collaboration) introduces a planner-executor decomposition where the decomposition axis is the device-cloud **trust boundary**, not the usual compute boundary.

### Architecture

**On-device agent (Sensitive Span Identifier + Key-Findings Distiller):**
- At each agentic step, the on-device SLM identifies which spans in the current context are privacy-sensitive
- It proposes replacements using **typed placeholder tokens** from a structured taxonomy: `<PERSON_0>`, `<LOCATION_1>`, `<HEALTH_CONDITION_0>`, `<FINANCIAL_VALUE_2>`, etc.
- After the cloud agent returns an action, the on-device agent distills the execution outcome into compact **key findings** that capture what was learned without re-exposing raw data to the cloud on subsequent steps

**Deterministic registry (on-device, non-LLM):**
- A lookup table that maps typed placeholders to their original values and back
- Performs all substitution (sensitive value → placeholder) and reversal (placeholder → original value) deterministically
- The LLM never touches original sensitive values; it only proposes which spans to tag and which placeholder type to assign

**Cloud agent (Reasoner + Planner):**
- Receives sanitized context with typed placeholders intact
- Reasons over placeholders as if they were opaque tokens: `<PERSON_0>` is treated as "some person" whose identity is known to the device but not to the cloud
- Returns action sequences using placeholders (e.g., `send_email(to=<PERSON_0>, subject="Follow-up")`)
- The device registry substitutes the real value before executing the action

### Why Typed Placeholders Work

The key insight is that **reasoning roles can be preserved even when content is hidden**. "Meet with `<PERSON_0>` at `<LOCATION_1>` on `<DATE_0>`" conveys enough structure for the cloud agent to reason about scheduling conflicts, generate follow-up reminders, and produce coherent multi-step plans — without ever knowing that `<PERSON_0>` is "Dr. Sarah Chen" or that `<LOCATION_1>` is a cancer treatment center.

The type annotation is essential: knowing that a placeholder is of type `<HEALTH_CONDITION>` allows the cloud model to apply appropriate reasoning (medical context, sensitivity level, follow-up actions) even without the literal value. This is fundamentally different from simple redaction, which destroys type information along with content.

### The Security Model

PAAC's threat model is **honest-but-curious cloud providers**: the cloud model faithfully executes its role but observes everything sent to it. The guarantee is that no raw sensitive values appear in cloud-visible context. The deterministic registry ensures that even if the cloud could observe traffic patterns, it sees only opaque placeholder tokens.

The on-device LLM's role is deliberately limited: it proposes span boundaries and placeholder types, but does not perform substitution itself. This means errors in span identification are bounded — the worst case is a missed sensitive span (leaking one value), not systematic misidentification of non-sensitive content as sensitive (breaking task utility).

## Comparison to Prior Work

**Simple routing baselines**: Route all sensitive queries to device, all non-sensitive to cloud. Fails on mixed-sensitivity agentic tasks and doesn't leverage cloud reasoning for sensitive reasoning steps (only for non-sensitive ones).

**NER-based redaction**: Replace named entities with `[PERSON]`, `[LOCATION]` etc. Breaks structural fidelity for tool calls (you lose the ability to use the value in a downstream API call). Also limited to entity types in a fixed taxonomy, missing sensitive information that doesn't fit standard NER categories (e.g., "the treatment she started last month").

**Differential privacy mechanisms**: Add noise to embeddings or outputs to prevent membership inference. Incompatible with discrete, structured agentic actions where adding noise produces invalid API calls.

**State-of-the-art device-cloud baselines**: PAAC improves average accuracy by **15–36%** and reduces average leakage by **2–6×** on three agentic benchmarks under strict privacy settings. The largest margins appear on privacy targets outside fixed entity taxonomies — exactly where NER-based approaches fail.

The consistent improvements across **17 additional benchmarks spanning 10 domains** (math, science, finance, among others) suggest the framework's benefit extends beyond privacy-specific tasks.

## Reading the Results

The +15–36% accuracy improvement over device-cloud baselines is large and significant. The range reflects task difficulty: on tasks where the cloud agent can usefully reason over typed placeholders (most scheduling, planning, and factual tasks), accuracy gains are at the high end. On tasks where the sensitive span structure is too dense for coherent placeholder-based reasoning, gains are smaller.

The 2–6× leakage reduction is measured against baselines that already apply some form of sanitization. This means PAAC isn't being compared to "send everything to cloud" — it's outperforming dedicated privacy-preserving baselines by a factor of 2–6 in leakage while also being more accurate.

The fact that improvements hold across 17 benchmarks spanning domains beyond the privacy-focused training distribution suggests the planner-executor decomposition itself is a useful structural prior, not just a privacy mechanism.

## Key Notes of This Paper

The central algorithmic mechanism is the **typed placeholder substitution pipeline**:

```
Step 1 (on-device): Identify sensitive spans
  context: "Schedule lunch with Dr. Chen at Mayo Clinic re: her chemo schedule"
  spans: [("Dr. Chen", PERSON), ("Mayo Clinic", LOCATION), ("chemo schedule", HEALTH_CONDITION)]

Step 2 (on-device registry): Substitute
  sanitized: "Schedule lunch with <PERSON_0> at <LOCATION_0> re: her <HEALTH_CONDITION_0>"
  registry: {PERSON_0: "Dr. Chen", LOCATION_0: "Mayo Clinic", HEALTH_CONDITION_0: "chemo schedule"}

Step 3 (cloud): Reason and plan
  input: sanitized context
  output: create_calendar_event(attendee=<PERSON_0>, location=<LOCATION_0>, note="Discuss <HEALTH_CONDITION_0>")

Step 4 (on-device registry): Reverse substitution
  action: create_calendar_event(attendee="Dr. Chen", location="Mayo Clinic", note="Discuss chemo schedule")

Step 5 (on-device): Execute action and distill key findings
  finding: "Lunch scheduled with PERSON_0 at LOCATION_0 for Thursday 1pm"
```

The key findings distillation in Step 5 is important: it summarizes what was learned in a form that can be passed to the cloud on the next step without re-exposing raw values in a fresh context. This prevents sensitive information from "re-entering" the cloud context through the back door of accumulated conversation history.

The type system is what makes this work at scale. The paper proposes a taxonomy of placeholder types that covers the most common sensitive information categories in personal agentic tasks. The on-device SLM is fine-tuned to produce accurate type assignments, which is a simpler and more reliable task than full information extraction.

## Limitations

- The on-device SLM must be capable enough to accurately identify sensitive spans. A very weak SLM may miss sensitive information or over-redact, degrading task utility.
- The typed placeholder taxonomy must be designed in advance and may not cover all domains (e.g., legal or specialized medical sub-fields may require domain-specific placeholder types).
- Privacy targets that span multiple steps in an agentic chain (e.g., inferring a person's identity from a combination of non-sensitive details) are not addressed — PAAC focuses on within-step span privacy.
- The paper evaluates on agentic benchmarks; real deployment involves additional attack surfaces (side-channel timing attacks, model output analysis) not addressed here.

## Future Work

**Authors' directions:**
- Extending the framework to multi-modal agentic tasks where sensitive information appears in images or audio
- Formal privacy guarantees under stronger threat models (not just honest-but-curious cloud)
- Adaptive typed placeholder taxonomy that grows with new domain deployments

**Additional promising directions:**
- **Federated span detection**: Multiple devices collaboratively train the on-device span detection model without sharing sensitive examples, using federated learning.
- **Semantic-preserving placeholder embeddings**: Instead of opaque tokens, use embeddings that capture semantic properties (valence, category, approximate magnitude for numbers) while hiding the exact value — enabling richer cloud reasoning.
- **Auditable leakage tracking**: A formal audit log of what types of information left the device and when, allowing users to review and set policies.
- **Integration with on-device LLM improvement**: As on-device models improve, the split point can shift dynamically — PAAC's architecture naturally accommodates this without redesign.

## Implications for Edge / On-Device Deployment

PAAC makes a compelling case that the right role for a small on-device LLM in a hybrid system is not "the model that runs when the network is down" — it is **the privacy guardian** that mediates all contact between user data and cloud reasoning.

This reframing has practical implications for SLM design. Instead of optimizing an on-device SLM purely for general language generation quality, you should optimize it for:
1. **Sensitive span detection accuracy**: High recall on identifying sensitive information
2. **Type classification accuracy**: Correct assignment of placeholder type categories  
3. **Key-findings distillation quality**: Compact, information-rich summaries of action outcomes

These are narrower, more achievable tasks than general reasoning. A 1–3B parameter model fine-tuned specifically for these roles can function as an effective privacy guardian even for queries that would stump it as a standalone assistant.

For product teams, this means the on-device LLM's role in a hybrid system is worth investing in even if the on-device model never produces the final user-facing response. The privacy and accuracy gains from a well-designed device-side sanitizer justify the engineering investment.

## Links

[Original Paper](https://huggingface.co/papers/2605.08646)
