"""Output formatters — terminal, JSON, SARIF.

All user-facing text uses the canonical domain model:
  KnowledgeArtifact, KnowledgeSignal, KnowledgeAssessment.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ai_ready.models import AssessmentDiff, KnowledgeAssessment, Severity


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WIDTH = 72
_SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _health_label(score: int) -> str:
    """Map a score to a human-readable health label."""
    if score >= 80:
        return "Healthy"
    if score >= 50:
        return "Needs Attention"
    return "Critical"


def _impact_label(severity: str) -> str:
    """Map a severity enum value to an observation impact label."""
    return {
        "CRITICAL": "Critical",
        "HIGH": "High Impact",
        "MEDIUM": "Moderate",
        "LOW": "Low Impact",
        "INFO": "Informational",
    }.get(severity, severity)


def _collector_label(collector_id: str) -> str:
    """Convert a snake_case collector ID to a human-readable label."""
    return collector_id.replace("_", " ").title()


def _score_bar(score: int, width: int = 15) -> str:
    """Generate a visual bar chart for a score using block characters."""
    filled = int(score / 100 * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _format_artifact_uri(uri: str, max_width: int = 55) -> str:
    """Truncate long artifact URIs intelligently, keeping the last path components."""
    if len(uri) <= max_width:
        return uri
    parts = uri.replace("\\", "/").split("/")
    for n in (3, 2):
        short = "/".join(parts[-n:])
        if len(short) + 3 <= max_width:
            return "..." + "/" + short
    return "..." + uri[-(max_width - 3):]


def _severity_counts(signals: list) -> dict[str, int]:
    """Count signals by severity."""
    counts: dict[str, int] = {}
    for s in signals:
        sev = s.severity.value
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def _separator(width: int | None = None) -> str:
    """Generate a horizontal separator line."""
    w = width or (_WIDTH - 4)
    return "  " + "\u2500" * w


def _assessment_summary(assessment: KnowledgeAssessment) -> list[str]:
    """Generate a natural-language assessment summary.

    Explains the reasoning behind the assessment score in plain language
    rather than listing rule statistics. Identifies the primary and secondary
    factors contributing to score loss and estimates recovery potential.
    """
    lines: list[str] = []
    score = assessment.score

    if not assessment.signals:
        lines.append(f"The knowledge base scored {score}/100. No signals were observed")
        lines.append("that would reduce AI readiness.")
        return lines

    explanation = assessment.explain_score()
    contributors = explanation.dominant_contributors

    # Opening statement
    if score >= 80:
        lines.append(f"The knowledge base scored {score}/100, indicating healthy")
        lines.append("AI readiness.")
    elif score >= 50:
        lines.append(f"The knowledge base scored {score}/100. While some aspects")
        lines.append("are adequate, attention is needed to improve AI readiness.")
    else:
        lines.append(f"The knowledge base scored {score}/100, indicating critical")
        lines.append("gaps in AI readiness that should be addressed.")

    if contributors:
        top = contributors[0]
        lines.append("")
        lines.append(f"The primary factor is {top.cause.lower()}: {top.signal_count}")
        lines.append(f"signal(s) across {top.artifact_count} artifact(s), worth up to")
        lines.append(f"{top.estimated_score_gain:.1f} points of score loss.")

        if len(contributors) > 1:
            second = contributors[1]
            lines.append("")
            lines.append(f"A secondary concern is {second.cause.lower()}:")
            lines.append(f"{second.signal_count} signal(s) across {second.artifact_count}")
            lines.append(f"artifact(s), worth up to {second.estimated_score_gain:.1f} points.")

        # Recovery estimate
        total_recoverable = sum(c.estimated_score_gain for c in contributors[:3])
        potential = min(100, score + int(total_recoverable))
        if potential > score + 5:
            lines.append("")
            lines.append("Addressing these areas would raise the overall score to")
            lines.append(f"approximately {potential}/100.")

    return lines


# ---------------------------------------------------------------------------
# Terminal formatters
# ---------------------------------------------------------------------------

def format_terminal(
    assessment: KnowledgeAssessment,
    *,
    prev_score: int | None = None,
    score_delta: int | None = None,
    is_baseline: bool = True,
) -> str:
    """Format a KnowledgeAssessment as a human-readable terminal report.

    The report reads like an assessment report, not a linter dump.
    Layout (assessment first, evidence last):
      1. Header with tool identity
      2. Knowledge Base metadata
      3. Knowledge Health Score (the assessment result)
      4. Historical context (baseline or trend)
      5. Assessment Dimensions (explain the score)
      6. Assessment Summary (natural language reasoning)
      7. Knowledge Signals (supporting evidence)
      8. Conclusion
    """
    lines: list[str] = []
    w = _WIDTH

    # 1. Header
    lines.append("")
    lines.append("=" * w)
    lines.append("  AI-Ready \u00b7 Knowledge Assessment")
    lines.append("=" * w)
    lines.append("")

    # 2. Knowledge Base metadata
    source = assessment.metadata.get("source", "N/A")
    lines.append(f"  Knowledge Base:    {source}")
    lines.append(f"  Artifacts:         {assessment.metadata.get('artifact_count', 0):,}")
    if assessment.metadata.get("relationship_count"):
        lines.append(f"  Relationships:     {assessment.metadata.get('relationship_count', 0):,}")
    if assessment.metadata.get("git_commit"):
        lines.append(f"  Git Commit:        {assessment.metadata['git_commit']}")
    lines.append("")

    # 3. Knowledge Health Score (the primary assessment result)
    health = _health_label(assessment.score)
    lines.append(f"  Knowledge Health Score    {assessment.score}/100    {health}")
    lines.append("")

    # 4. Historical context
    if is_baseline:
        lines.append("  Baseline assessment \u2014 no previous assessments were found.")
    else:
        if score_delta is not None and prev_score is not None:
            if score_delta > 0:
                lines.append(
                    f"  Previous: {prev_score}/100  \u2192  Improving (+{score_delta})"
                )
            elif score_delta < 0:
                lines.append(
                    f"  Previous: {prev_score}/100  \u2192  Regressing ({score_delta})"
                )
            else:
                lines.append(
                    f"  Previous: {prev_score}/100  \u2192  Stable (no change)"
                )
    lines.append("")

    # 5. Assessment Dimensions
    if assessment.dimensions:
        lines.append(_separator())
        lines.append("")
        lines.append("  Assessment Dimensions")
        lines.append("")
        for name, dim in sorted(assessment.dimensions.items()):
            bar = _score_bar(dim.score)
            lines.append(f"    {name:20s}  {dim.score:>3}/100  {bar}  {dim.signals_count} signals")
        lines.append("")

    # 6. Assessment Summary (natural language reasoning)
    lines.append(_separator())
    lines.append("")
    lines.append("  Assessment Summary")
    lines.append("")
    summary_lines = _assessment_summary(assessment)
    for sl in summary_lines:
        lines.append(f"    {sl}")
    lines.append("")

    # 7. Knowledge Signals (supporting evidence)
    if assessment.signals:
        lines.append(_separator())
        lines.append("")
        lines.append("  Knowledge Signals \u2014 Supporting Evidence")
        lines.append("")

        total = len(assessment.signals)
        artifact_count = assessment.metadata.get("artifact_count", 0)
        lines.append(f"    {total:,} signal(s) were observed across the knowledge base.")
        lines.append("")

        # Summary by impact
        counts = _severity_counts(assessment.signals)
        impact_parts = []
        for sev in _SEVERITY_ORDER:
            if sev in counts:
                label = _impact_label(sev)
                impact_parts.append(f"{label}: {counts[sev]}")
        lines.append(f"    {'   '.join(impact_parts)}")
        lines.append("")

        # Show examples per severity (limited)
        max_per_severity = {
            "CRITICAL": 5,
            "HIGH": 5,
            "MEDIUM": 5,
            "LOW": 3,
            "INFO": 3,
        }

        for sev in _SEVERITY_ORDER:
            sev_signals = [s for s in assessment.signals if s.severity.value == sev]
            if not sev_signals:
                continue

            show = max_per_severity.get(sev, 3)
            label = _impact_label(sev)
            lines.append(f"    {label} ({len(sev_signals)})")
            for s in sev_signals[:show]:
                uri = _format_artifact_uri(s.artifact_uri)
                collector = _collector_label(s.collector_id)
                line_info = f":{s.line}" if s.line else ""
                lines.append(f"      {uri}{line_info}  \u2014  {collector}")
                if s.ai_impact:
                    # Truncate long ai_impact descriptions
                    impact = s.ai_impact
                    if len(impact) > 100:
                        impact = impact[:97] + "..."
                    lines.append(f"        {impact}")
            remaining = len(sev_signals) - show
            if remaining > 0:
                lines.append(f"      ... and {remaining:,} more")
            lines.append("")

        lines.append("    Use 'ai-ready signals' to explore individual signals.")
        lines.append("    Use 'ai-ready diff' to compare with previous assessments.")
    else:
        lines.append(_separator())
        lines.append("")
        lines.append("  No knowledge signals were observed. The knowledge base is AI-ready.")
        lines.append("")

    # 8. Conclusion
    lines.append(_separator())
    lines.append("")
    lines.append(f"  Assessment complete. Knowledge health: {health} ({assessment.score}/100).")
    lines.append("=" * w)

    return "\n".join(lines)


def format_regression_terminal(report: AssessmentDiff) -> str:
    """Format an AssessmentDiff as a human-readable terminal report.

    Presents the comparison as an assessment evolution report rather
    than a regression dump. Layout:
      1. Header
      2. Assessment comparison (previous vs current)
      3. Score change with trajectory
      4. Dimension changes
      5. Why the assessment changed (natural language)
      6. New and resolved observations
      7. Persistent observations count
      8. Conclusion
    """
    lines: list[str] = []
    w = _WIDTH

    # 1. Header
    lines.append("")
    lines.append("=" * w)
    lines.append("  AI-Ready \u00b7 Assessment Comparison")
    lines.append("=" * w)
    lines.append("")

    # 2. Assessment comparison
    lines.append(f"  Previous:  {report.prev_assessment_id}  \u00b7  {report.prev_score}/100  \u00b7  {_health_label(report.prev_score)}")
    lines.append(f"  Current:   {report.curr_assessment_id}  \u00b7  {report.curr_score}/100  \u00b7  {_health_label(report.curr_score)}")
    lines.append("")

    # 3. Score change with trajectory
    delta = report.score_delta
    if delta > 0:
        lines.append(f"  Score Change:  {report.prev_score} \u2192 {report.curr_score}  (+{delta})  \u00b7  Improving")
    elif delta < 0:
        lines.append(f"  Score Change:  {report.prev_score} \u2192 {report.curr_score}  ({delta})  \u00b7  Regressing")
    else:
        lines.append(f"  Score Change:  {report.prev_score} \u2192 {report.curr_score}  \u00b7  Stable")
    lines.append("")

    # 4. Dimension changes
    if report.dimension_deltas:
        lines.append(_separator())
        lines.append("")
        lines.append("  Dimension Changes")
        lines.append("")
        for name, d in sorted(report.dimension_deltas.items()):
            if d > 0:
                indicator = "[+]"
            elif d < 0:
                indicator = "[!]"
            else:
                indicator = "[ ]"
            lines.append(f"    {indicator}  {name:20s}  {d:+d}")
        lines.append("")

    # 5. Why the assessment changed
    if report.score_change_explanation:
        regressions = report.score_change_explanation.get("top_regressions", [])
        improvements = report.score_change_explanation.get("top_improvements", [])
        if regressions or improvements:
            lines.append(_separator())
            lines.append("")
            lines.append("  Why the Assessment Changed")
            lines.append("")
            for change in regressions[:3]:
                lines.append(
                    f"    [!] {change['cause']} "
                    f"({change['prev_signals']} \u2192 {change['curr_signals']} signals, "
                    f"{change['delta_estimated_score_gain']:+.1f} pts)"
                )
            for change in improvements[:3]:
                lines.append(
                    f"    [+] {change['cause']} "
                    f"({change['prev_signals']} \u2192 {change['curr_signals']} signals, "
                    f"{change['delta_estimated_score_gain']:+.1f} pts)"
                )
            lines.append("")

    # 6. New observations
    if report.new_signals:
        lines.append(_separator())
        lines.append("")
        lines.append(f"  New Observations ({len(report.new_signals)})")
        for s in report.new_signals[:15]:
            uri = _format_artifact_uri(s.artifact_uri)
            collector = _collector_label(s.collector_id)
            lines.append(f"    {uri}  \u2014  {collector}")
        remaining = len(report.new_signals) - 15
        if remaining > 0:
            lines.append(f"    ... and {remaining:,} more")
        lines.append("")

    # 7. Resolved observations
    if report.resolved_signals:
        lines.append(f"  Resolved Observations ({len(report.resolved_signals)})")
        for s in report.resolved_signals[:15]:
            uri = _format_artifact_uri(s.artifact_uri)
            collector = _collector_label(s.collector_id)
            lines.append(f"    {uri}  \u2014  {collector}")
        remaining = len(report.resolved_signals) - 15
        if remaining > 0:
            lines.append(f"    ... and {remaining:,} more")
        lines.append("")

    # 8. Persistent observations
    if report.persistent_signals:
        lines.append(f"  Persistent Observations: {len(report.persistent_signals):,}")
        lines.append("")

    # Severity changes (impact changes)
    if report.severity_changes:
        lines.append(f"  Impact Changes ({len(report.severity_changes)})")
        for sc in report.severity_changes[:10]:
            direction = "[!]" if sc["direction"] == "worse" else "[+]"
            uri = _format_artifact_uri(sc["artifact_uri"])
            lines.append(
                f"    {direction} {uri} "
                f"({sc['prev_severity']} \u2192 {sc['curr_severity']})"
            )
        lines.append("")

    # Alerts (integrated, not as separate section)
    if report.new_high_count:
        lines.append(f"  [!] {report.new_high_count} new high-impact observations")
    if report.increased_contradictions:
        lines.append(f"  [!] Contradictions increased by {report.increased_contradictions}")
    if report.increased_topic_entropy:
        lines.append(f"  [!] Topic entropy increased by {report.increased_topic_entropy}")
    if report.new_orphan_artifacts:
        lines.append(f"  [!] {report.new_orphan_artifacts} new orphan artifacts")
    if report.new_duplicate_clusters:
        lines.append(f"  [!] {report.new_duplicate_clusters} new duplicate clusters")

    # Conclusion
    lines.append("")
    if report.recommendation:
        lines.append(f"  {report.recommendation}")
    lines.append("=" * w)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON formatters
# ---------------------------------------------------------------------------

def format_json(assessment: KnowledgeAssessment) -> str:
    """Format a KnowledgeAssessment as JSON."""
    return json.dumps(assessment.to_dict(), indent=2, default=str)


def format_regression_json(report: AssessmentDiff) -> str:
    """Format an AssessmentDiff as JSON."""
    return json.dumps(report.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# SARIF formatter
# ---------------------------------------------------------------------------

def format_sarif(assessment: KnowledgeAssessment) -> str:
    """Format a KnowledgeAssessment as SARIF."""
    sarif: dict[str, Any] = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/SARIF/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "ai-ready",
                        "version": "0.2.0",
                        "informationUri": "https://github.com/jacks/ai-ready",
                        "rules": [
                            {
                                "id": s.collector_id,
                                "name": _collector_label(s.collector_id),
                                "shortDescription": {"text": s.recommendation[:200]},
                            }
                            for s in assessment.signals
                        ],
                    }
                },
                "results": [
                    {
                        "ruleId": s.collector_id,
                        "level": _severity_to_sarif_level(s.severity.value),
                        "message": {"text": s.recommendation},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": s.artifact_uri,
                                    },
                                    "region": {"startLine": s.line} if s.line else {},
                                }
                            }
                        ],
                        "partialFingerprints": {
                            "primaryLocationLineHash": s.signal_id,
                        },
                    }
                    for s in assessment.signals
                ],
            }
        ],
    }
    return json.dumps(sarif, indent=2)


def _severity_to_sarif_level(severity: str) -> str:
    """Map severity to SARIF level."""
    mapping = {
        "CRITICAL": "error",
        "HIGH": "error",
        "MEDIUM": "warning",
        "LOW": "note",
        "INFO": "none",
    }
    return mapping.get(severity, "warning")
