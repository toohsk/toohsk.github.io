Title: nanoMuse: An Open-Source Personal Agent for Every Device You Own
Date: 2026-10-09
Category: AI/ML
Tags: SLM, AI, paper-review
Summary: nanoMuse presents a GPL-3.0 open-source personal AI agent that runs locally on Android and PC, with a Sentinel permission architecture, file-based transparent memory, cross-device relay, and screen-level hands — an open counterpart to Meta's Muse designed for device sovereignty.

## Why This Paper Matters

In September 2026, Meta released Muse — the first product to assemble a personal AI agent for one person: persistent memory, device access, proactive messaging, and a wallet. Within five weeks, OpenAI, Manus, and Today released comparable systems. The common architecture: one cloud computer per person in the vendor's cloud, thin clients on the person's devices, and a proprietary model.

**This is the wrong shape for the first half of the description.**

An agent that holds a person's life — their accounts, files, calendar, messages — should be inspectable, controllable, and ideally run on hardware the person already owns. The cloud computer architecture can only reach services with APIs or web pages; many apps people use daily (banking, government services, local utilities) have neither, and are only accessible through screen-level interaction that requires running on the device itself.

nanoMuse is the open counterpart: **a complete personal agent on every device you own, under GPL-3.0, with hands on Android and PC screens.**

## Core Technical Contribution

### System Architecture

nanoMuse places the full agent on each device:

- **Android phone and PC**: The complete agent runs here, with screen-level "hands" (accessibility executor on Android, screenshot loop on PC)
- **iPhone and web app**: Full conversation participation, but no screen operation (iOS prohibits apps from operating other apps)
- **Relay server**: An optional message broker for synchronizing conversations across devices — anyone can run it, and it carries the conversation but cannot read credentials or execute actions

**The Sentinel** is the sole permission authority for all actions. The agent can propose actions; only the Sentinel grants execution rights. This separation is architectural, not just a prompt instruction.

### Memory Design

From analyzing Muse's design (Meta's public documents + a copy of its production system prompt), nanoMuse adopts a transparent, file-based memory system:

- **Three agent-maintained files**: Who the agent is, who the person is, and a relationship map of people in the person's life
- **Indexed memory bank**: Past interactions stored as addressable, provenance-tagged files
- **Read/edit by the person**: All memory files are plain text the person can open, read, and modify — not a black-box vector database
- **Provenance tracking**: Every memory records where it came from, enabling the agent (and the person) to verify its basis

The most significant principle learned from Muse's prompt design: *the agent is instructed to distrust its own recollection and to search memory before answering anything about prior work, dates, people, or preferences.* This is a practical solution to LLM hallucination in a long-running personal context.

### Sentinel Permission Architecture

Inspired by Meta's publicly documented Sentinel design, nanoMuse implements:

- **Surrogate tokens**: The agent never holds real credentials (passwords, API keys). Only surrogate tokens, which are swapped for real credentials at the network boundary after Sentinel approval
- **Scoped grants**: One-time, per-session, per-task, time-bounded, or standing permissions — never a blanket "always approved"
- **Prompt injection defense**: The browser sub-agent sees an accessibility-tree snapshot, not the DOM, and pauses on form fills involving credentials
- **Approval interface**: Approval requests go directly to the person's interface, not through the conversation (the conversation could be injected)

### Model Freedom

Unlike Muse (locked to Meta's Muse Spark family), nanoMuse explicitly decouples the agent architecture from the underlying model:

- Cloud APIs (Anthropic, OpenAI, etc.)
- Local models via Ollama or similar
- Future: open "hands models" — small models fine-tuned specifically for screen operation

This design is significant for SLM deployment: the "brain" model need not be the same as the "hands" model. A 7B general model could handle reasoning while a purpose-built 1B screen-operation model handles UI interaction.

## Comparison to Prior Work

The personal agent landscape as of October 2026:

| System | Where it runs | Phone screen | Computer screen | Open |
|--------|--------------|-------------|----------------|------|
| Hermes Agent | User's server | No | No | MIT |
| OpenMinis | Inside Android app | Android | No | GPL |
| Meta Muse | Meta cloud (Linux VM) | No | macOS only | No |
| OpenAI dots | Cloud per agent | No | No | No |
| Manus Cue | Cloud per agent | No | No | No |
| **nanoMuse** | **Phone + computer** | **Android** | **Yes** | **GPL-3.0** |

nanoMuse is the only system combining: runs on your own hardware, has Android screen control, runs on the computer too, and is fully open source.

The paper also catalogs earlier open projects (Hermes Agent, OpenClaw, OpenMinis, OpenMuse from CopilotKit) and explains what each is missing to constitute a "personal agent" under the paper's definition.

## Reading the Results

This is a system design and implementation report, not a benchmark paper. The evaluation framework the authors set up (and acknowledge is incomplete) centers on:

1. **Architectural completeness**: Does the system satisfy all five defining questions of a personal agent? (Who runs it, what does it remember, when does it speak first, how does it act, what does it answer for?)
2. **Hands evaluation suite**: An evaluation benchmark for Android screen operation, described as a roadmap item
3. **Cost transparency**: The paper provides estimated model size and monthly inference cost as specific numbers

The Muse analysis is itself a valuable result: by cross-referencing Meta's three public documents and the production system prompt, the paper reverse-engineers Muse's system design with source citations for every claim (documented / prompt / observed / inferred). This is a rare rigorous analysis of a closed commercial product.

## Key Notes of This Paper

### The five defining questions

The paper proposes five questions that define a personal agent:

1. **Who runs it?** — Where does the agent's execution loop run, and who controls the hardware?
2. **What does it remember?** — What persists across sessions, in what form, and who can read/edit it?
3. **When does it speak first?** — What triggers proactive messages, and who controls that threshold?
4. **How does it act?** — Which tools, what permissions model, and what happens on prompt injection?
5. **What does it answer for?** — What is the audit trail, and how does the person know what the agent did?

Each question has engineering implications. "Who runs it" is not a deployment preference — it determines which apps the agent can reach (device-local apps vs. API-accessible services). "What does it remember" is not a UX choice — it determines whether the person can verify, correct, or delete the agent's beliefs about them.

### The phone screen gap

The most technically significant observation: all cloud-based personal agents (Muse, OpenAI dots, Manus Cue) can only reach services with an API or a web page. A large fraction of a typical person's digital life runs through apps with neither — native banking apps, government services, local business apps, health apps. The only way to act in these apps is through screen-level interaction, and screen-level interaction can only run on the device itself.

This is not a temporary limitation that APIs will fix. Many services deliberately avoid public APIs for security or competitive reasons. The hands-on-device model is not a compromise — it's the only architecture that can reach the full scope of a person's digital life.

### Memory provenance as a technical problem

The paper identifies **memory confidence** as a core unsolved problem in personal agents. A multi-week personal agent accumulates hundreds of memories from various sources — conversations, web fetches, file reads, proactive background jobs. The agent must act on these memories while correctly representing its uncertainty about them.

Muse's solution (requiring the agent to search memory before trusting its own recollection, and to attribute claims to their source on request) is the most rigorous response to this problem seen in a shipped product. nanoMuse inherits this design, making it a baseline for future research on long-term memory management in personal agents.

## Limitations

1. **Hands model gap**: Purpose-built small models for screen operation on Android are not yet available under open licenses. The roadmap depends on community development of open hands models.
2. **iOS lockout**: Apple's restrictions on cross-app operation mean iPhones participate in conversation only, with no ability to operate other apps.
3. **Relay requirement**: Cross-device conversation sync requires a relay server — technically sophisticated for typical users, even though the codebase is open.
4. **Evaluation immaturity**: The Android screen operation evaluation suite is described as future work; current evaluation is qualitative.
5. **Not production-ready**: The authors explicitly list what they "would not yet trust it with" — financial transactions, medical information access, decisions involving children's care.

## Future Work

**Authors' roadmap:**
- Open hands model (a small model purpose-built for Android accessibility interaction)
- Standardized evaluation suite for personal agent "hands" on Android
- Full memory provenance system (complete audit trail of memory sources and replacements)

**Research directions this work opens:**
- **Dedicated screen-operation SLMs**: The "hands" component is a perfect target for a specialized sub-1B model — fixed action space (tap, scroll, type, read), latency-critical, runs locally
- **Prompt injection characterization**: The accessibility-tree snapshot defense is heuristic; formal adversarial evaluation of injection resistance is needed
- **Cross-device context merging**: How to maintain a coherent agent state across multiple devices with different context windows, battery states, and network connectivity
- **Graceful degradation under connectivity loss**: Personal agents need to handle offline or intermittent connectivity, with clear failure modes
- **Privacy-preserving memory**: Techniques to store long-term personal memory in a form that is useful to the local agent but not recoverable if the device is compromised

## Implications for Edge / On-Device Deployment

nanoMuse reframes the central challenge of on-device AI deployment.

The research community has made substantial progress on model efficiency — quantization, distillation, speculative decoding. These advances make it possible to run 7B-class models on a flagship smartphone. But nanoMuse shows that **running a model on device is necessary but not sufficient** for a personal agent.

The harder problems are:
1. **Permission architecture**: How does the agent act on accounts and services without exposing credentials or acting outside authorized scope?
2. **Memory management**: How does the agent maintain a consistent, trustworthy, user-visible world model across weeks?
3. **Multi-device coordination**: How do separate model instances on different devices share context without a trusted intermediary?

For SLM practitioners, nanoMuse suggests a split architecture worth designing for:
- A **larger reasoning model** (4–7B, quantized to INT4) handles planning and memory synthesis
- A **dedicated hands model** (<1B) handles screen interaction with low latency
- A **Sentinel module** (rule-based or tiny classifier) handles permission decisions without calling the LLM

This decomposition keeps latency and battery costs manageable while preserving the full capability of a personal agent.

## Links

[Original Paper](https://huggingface.co/papers/2610.08699)
