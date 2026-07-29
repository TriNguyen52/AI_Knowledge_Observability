"""Prompt context builders for LLM actions.

These functions construct the text summaries that get passed to the LLM
inside Burr actions. They are the security boundary: the LLM sees only
what these functions show it, never the knowledge base directly.

Extracted from actions.py to keep action functions focused on
orchestration logic rather than prompt construction.
"""

from __future__ import annotations

import json
from typing import Any


def build_signal_context(
    signal_ids: list[str],
    assessment_store: Any,
) -> str:
    """Build a text summary of signals for the LLM prompt.

    The LLM receives signal descriptions, NOT access to the knowledge base.
    This is the security boundary — the LLM sees what we show it.
    """
    if not assessment_store:
        return "No assessment store available."

    assessment = assessment_store.latest()
    if not assessment:
        return "No assessment available."

    lines: list[str] = []
    for signal in assessment.signals:
        if signal.signal_id in signal_ids:
            lines.append(
                f"- Signal {signal.signal_id}: "
                f"collector={signal.collector_id}, "
                f"type={signal.signal_type}, "
                f"severity={signal.severity.value}, "
                f"artifact={signal.artifact_uri}, "
                f"evidence={json.dumps(signal.evidence, default=str)[:200]}, "
                f"recommendation={signal.recommendation}, "
                f"ai_impact={signal.ai_impact}"
            )
    return "\n".join(lines) if lines else "No matching signals found."


def build_artifact_context(
    artifact_uris: list[str],
    assessment_store: Any,
) -> str:
    """Build a text summary of affected artifacts for the LLM prompt."""
    if not assessment_store:
        return "No assessment store available."

    assessment = assessment_store.latest()
    if not assessment:
        return "No assessment available."

    lines: list[str] = []
    for artifact_uri in artifact_uris:
        # Try to find artifact metadata in the assessment
        for signal in assessment.signals:
            if signal.artifact_uri == artifact_uri:
                lines.append(f"- URI: {artifact_uri} (referenced by signal {signal.signal_id})")
                break
        else:
            lines.append(f"- URI: {artifact_uri} (no signal reference found)")

    return "\n".join(lines) if lines else "No artifacts found."


def build_history_context(
    prior_failures: list[dict[str, Any]],
    history_store: Any = None,
) -> str:
    """Build context from prior improvement outcomes for the LLM prompt."""
    lines: list[str] = []

    # Include prior failures from forking
    for failure in prior_failures:
        lines.append(
            f"- Prior attempt #{failure.get('attempt_number', '?')}: "
            f"strategy={failure.get('strategy', 'unknown')}, "
            f"result={failure.get('result', 'unknown')}, "
            f"reason={failure.get('failure_reason', 'unknown')}"
        )

    # Include historical outcomes from the history store
    if history_store:
        try:
            recent = history_store.get_recent_outcomes(limit=5)
            for outcome in recent:
                lines.append(
                    f"- Historical: issue={outcome.get('issue_type', '?')}, "
                    f"strategy={outcome.get('strategy', '?')}, "
                    f"result={outcome.get('result', '?')}, "
                    f"score_change={outcome.get('score_change', 0)}"
                )
        except Exception:
            pass

    return "\n".join(lines) if lines else "No prior history available."


def cluster_signals(
    signal_ids: list[str],
    assessment_store: Any,
) -> list[dict[str, Any]]:
    """Cluster similar signals and return representative examples.

    Instead of sending every signal to the LLM (which wastes tokens),
    group signals by (collector_id, signal_type, artifact_uri) and
    send one representative per cluster plus a count. This reduces
    noise and token consumption while preserving diagnostic information.

    Returns:
        List of cluster dicts, each with:
          - representative: the first signal in the cluster (full description)
          - count: how many signals are in this cluster
          - cluster_key: the grouping key
          - all_signal_ids: list of all signal IDs in this cluster
    """
    if not assessment_store:
        return []

    assessment = assessment_store.latest()
    if not assessment:
        return []

    # Group signals by (collector_id, signal_type, artifact_uri)
    clusters: dict[str, dict[str, Any]] = {}
    for signal in assessment.signals:
        if signal.signal_id not in signal_ids:
            continue
        cluster_key = f"{signal.collector_id}|{signal.signal_type}|{signal.artifact_uri}"
        if cluster_key not in clusters:
            clusters[cluster_key] = {
                "representative": (
                    f"collector={signal.collector_id}, "
                    f"type={signal.signal_type}, "
                    f"severity={signal.severity.value}, "
                    f"artifact={signal.artifact_uri}, "
                    f"evidence={json.dumps(signal.evidence, default=str)[:200]}, "
                    f"recommendation={signal.recommendation}, "
                    f"ai_impact={signal.ai_impact}"
                ),
                "count": 1,
                "cluster_key": cluster_key,
                "all_signal_ids": [signal.signal_id],
            }
        else:
            clusters[cluster_key]["count"] += 1
            clusters[cluster_key]["all_signal_ids"].append(signal.signal_id)

    return list(clusters.values())


def build_assessment_summary(
    signal_ids: list[str],
    artifact_uris: list[str],
    assessment_store: Any,
    clusters: list[dict[str, Any]] | None = None,
) -> str:
    """Build a compact, reusable assessment summary for LLM prompts.

    This summary is built once in analyze_issue and persisted in Burr
    state as `assessment_summary`. Later actions (generate_proposal)
    reuse it instead of rebuilding context from scratch, saving both
    compute and tokens.

    The summary includes:
      - Signal cluster representatives (not every signal)
      - Artifact references with signal counts
      - Overall severity distribution

    Returns:
        A compact text summary suitable for LLM prompts.
    """
    if not assessment_store:
        return "No assessment store available."

    assessment = assessment_store.latest()
    if not assessment:
        return "No assessment available."

    if clusters is None:
        clusters = cluster_signals(signal_ids, assessment_store)

    lines: list[str] = []
    lines.append(f"Assessment: {assessment.assessment_id}")
    lines.append(f"Total signals: {len(signal_ids)} (in {len(clusters)} cluster(s))")
    lines.append(f"Affected artifacts: {len(artifact_uris)}")
    lines.append("")

    # Severity distribution
    severity_counts: dict[str, int] = {}
    for signal in assessment.signals:
        if signal.signal_id in signal_ids:
            sev = signal.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
    if severity_counts:
        lines.append("Severity distribution: " + ", ".join(
            f"{sev}={count}" for sev, count in sorted(severity_counts.items())
        ))
        lines.append("")

    # Signal cluster representatives
    lines.append("Signal clusters (representative shown, count indicates cluster size):")
    for cluster in clusters:
        rep = cluster["representative"]
        count = cluster["count"]
        if count > 1:
            lines.append(f"  [{count}x] {rep}")
        else:
            lines.append(f"  {rep}")

    return "\n".join(lines)
