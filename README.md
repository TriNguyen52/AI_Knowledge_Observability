# AI-Ready

AI-Ready is a knowledge observability that ensure the quality of knowledge before it reaches an AI system. Rather than evaluating language models or retrieval algorithms, it analyzes the knowledge repositories that those systems depend on, including documentation, APIs, runbooks, and internal wikis. The engine transforms various sources into a unified artifact model, collects deterministic signals that describe measurable properties of the knowledge, and interprets those signals into assessments that reflect their potential impact on AI reasoning. These assessments are persisted over time, allowing organizations to detect regressions, monitor the evolution of their knowledge base, and understand how changes affect AI readiness. The analysis pipeline is orchestrated as a deterministic workflow using Burr, ensuring every stage of collection, interpretation, and assessment follows an explicit, reproducible execution graph.

## What it does
- **`ai-ready collect`** — Discovers and normalizes knowledge from documentation, APIs, runbooks, and other supported sources into a unified Knowledge Artifact model.
- **`ai-ready assess`** — Collects deterministic Knowledge Signals, applies interpretation policies, and produces a versioned Knowledge Assessment.
- **`ai-ready diff`** — Compares two assessments to identify new issues, resolved issues, and knowledge regressions.
- **`ai-ready history`** — Tracks how knowledge quality evolves across historical assessments.
- **`ai-ready explain`** — Shows the evidence behind every signal and assessment, making every result explainable and reproducible.
- **`ai-ready check`** — Executes the complete analysis pipeline and returns results suitable for CI/CD quality gates.
- **`ai-ready serve`** — Exposes AI-Ready as a service for integration with engineering workflows and AI platforms.
- **`ai-ready connectors`** — Lists available connectors and supported knowledge sources.

## Quick Start

```bash
# Install
pip install -e .

# Scan a knowledge base
ai-ready scan docs/

# JSON output for CI/CD
ai-ready scan docs/ --json

# SARIF output for GitHub code scanning
ai-ready scan docs/ --sarif

# View snapshot history
ai-ready history

# Diff two snapshots for regression detection
ai-ready diff baseline.db current.db

# Continuous monitoring
ai-ready monitor --path docs/ --interval 300
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Scan successful, thresholds passed |
| 1 | Readiness threshold failed |
| 2 | High severity findings present |
| 3 | Internal analyzer error |

## Configuration

Create `.ai-ready.yml` in your docs directory:

```yaml
version: 1

thresholds:
  overall_score: 85

fail_on:
  - CRITICAL

weights:
  retrieval: 0.25
  context: 0.15
  consistency: 0.20
  trust: 0.20
  connectivity: 0.10
  workflow: 0.10

rules:
  topic_purity:
    enabled: true
  context_independence:
    enabled: true
  heading_quality:
    enabled: true
  link_integrity:
    enabled: true
```

## AI Readiness Dimensions

| Dimension | Signals |
|-----------|---------|
| Retrieval Readiness | Topic Purity, Heading Quality |
| Context Readiness | Context Independence |
| Consistency | Terminology, Contradictions |
| Trustworthiness | Canonical Sources, Freshness |
| Connectivity | Knowledge Connectivity, Links |
| Task Completion | Workflow Completeness |

## GitHub Actions Integration

```yaml
- name: AI readiness scan
  run: ai-ready scan docs/ --json > report.json

- name: Fail on regression
  run: ai-ready diff baseline.db current.db
```

## Architecture

```
Connectors (Markdown, Git, GitBook*, Notion*, Confluence*)
    ↓
Knowledge Representation Layer (Canonical Document Model)
    ↓
AI Readiness Rule Engine (collect → measure → evaluate → report)
    ↓
Snapshot Store (SQLite)
    ↓
Diff / Regression Engine
    ↓
CLI / JSON / SARIF / Prometheus* / GitHub Checks
```

*Future connectors and output formats

## Design Principles

- **Infrastructure-first** — runs as CLI, CI step, daemon, or container
- **Deterministic** — no LLM-based scoring, every finding has evidence
- **Machine-readable** — JSON, SARIF, Prometheus metrics
- **CI/CD native** — exit codes for pipeline integration
- **Never modifies the KB** — read-only analysis

## License

MIT
