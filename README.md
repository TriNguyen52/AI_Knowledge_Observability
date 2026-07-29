# AI-Ready

AI systems are only as reliable as the knowledge they retrieve.

Even the strongest language models can hallucinate for missing context, disconnected concepts, broken relationships, ambiguous headings, or inconsistent organization. These issues silently reduce retrieval quality, increase hallucinations, and make AI agents less reliable.

**AI-Ready** is a deterministic knowledge observability and continuous improvement platform that evaluates knowledge before it reaches an AI system. Rather than measuring language models or retrieval algorithms, AI-Ready measures the quality of the knowledge itself.

The platform transforms documentation, APIs, codebases, runbooks, internal wikis, and other knowledge sources into a unified Knowledge Artifact model. From these artifacts it collects deterministic Knowledge Signals, interprets them into Knowledge Problems, produces versioned Knowledge Assessments, and continuously improves the knowledge base through verified remediation workflows.

---

## How It Works

AI-Ready collects knowledge from one or more repositories and transforming every source into a unified Knowledge Artifact representation. Documentation, API specifications, source code, runbooks, and other supported sources are normalized into the same internal model, allowing every analysis to operate independently of where the knowledge originally came from.

Once the knowledge has been normalized, AI-Ready performs deterministic analysis to collect Knowledge Signals. These signals describe properties such as context independence, topic purity, heading quality, connectivity, retrieval readiness, workflow completeness, and other characteristics that directly influence how effectively AI systems consume knowledge.

Signals are then interpreted into higher-level Knowledge Problems, representing underlying issues that may affect across many artifacts simultaneously. Multiple signals can contribute to the same problem, allowing AI-Ready to reason about causes rather than symptoms.

Each assessment is stored as a versioned snapshot so organizations can compare historical assessments, detect regressions, monitor improvements over time, and understand how changes affect AI readiness.

---

## Continuous Knowledge Improvement

The improvement workflow begins by ranking discovered problems according to their expected impact on AI reasoning. Then. AI-Ready combines analysis with large language models to generate fixing proposals. 

Every proposed modification passes through human approval before execution. After changes are applied, AI-Ready performs another assessment to verify whether the knowledge actually improved. Improvements are accepted only when the measured assessment demonstrates meaningful progress rather than simply producing plausible edits.

Over time the platform records proposals outcomes, learns which strategies consistently improve different categories of Knowledge Problems, and uses those historical outcomes to prioritize future remediation attempts.

---

## Workflow Orchestration

The improvement pipeline is orchestrated using Apache Burr.

This provides reproducible execution, persistent workflow history, resumable approval checkpoints, workflow forking after failed remediation attempts, OpenTelemetry observability, and automatic regression test generation from production workflows.

Apache Burr manages the workflow itself. AI-Ready remains responsible for knowledge collection, deterministic analysis, assessment, verification, and reasoning about Knowledge Problems.

---

## Architecture

```text
Knowledge Sources
        │
        ▼
Knowledge Artifact Model
        │
        ▼
Deterministic Signal Collection
        │
        ▼
Knowledge Problem Discovery
        │
        ▼
Knowledge Assessment
        │
        ▼
Problem Prioritization
        │
        ▼
LLM Proposal Generation
        │
        ▼
Human Approval
        │
        ▼
Knowledge Modification
        │
        ▼
Verification
        │
        ▼
Historical Learning
```

---

## Design Principles

AI-Ready is built on a small number of principles that remain consistent throughout the system.

- Deterministic analysis always comes before AI reasoning. Every assessment should be reproducible regardless of which language model is available.

- Knowledge is treated as infrastructure rather than application data. 

- Evidence is never discarded. Every assessment, proposed modification, verification result, and remediation outcome becomes part of the historical record, allowing organizations to understand not only what changed, but why it changed.

---

## Getting Started

```bash
pip install -e .

# Scan a knowledge base
ai-ready scan ./knowledge

# View the assessment
ai-ready assess

# Compare historical assessments
ai-ready diff

# Start continuous monitoring
ai-ready monitor
```

---

## License

Apache 2.0
