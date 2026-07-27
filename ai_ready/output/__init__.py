"""Output formatters - terminal, JSON, SARIF."""

from __future__ import annotations

import json
import sys
from typing import Any

from ai_ready.models import AssessmentDiff, KnowledgeAssessment


def format_terminal(assessment: KnowledgeAssessment) -> str:
    """Format assessment as human-readable terminal output."""
    lines: list[str] = []
    lines.append("")
    lines.append("AI Readiness Assessment")
    lines.append("=" * 60)
    lines.append(f"  Assessment ID:  {assessment.assessment_id}")
    lines.append(f"  Source:          {assessment.metadata.get('source', 'N/A')}")
    lines.append(f"  Artifacts:       {assessment.metadata.get('artifact_count', 0)}")
    if assessment.metadata.get("relationship_count"):
        lines.append(f"  Relations:       {assessment.metadata.get('relationship_count', 0)}")
    lines.append(f"  Git Commit:      {assessment.metadata.get('git_commit', 'N/A')}")
    lines.append("")
    lines.append(f"  Overall Score: {assessment.score}/100")
    lines.append("")
    lines.append("  Dimensions:")
    for name, dim in sorted(assessment.dimensions.items()):
        lines.append(f"    {name:20s} {dim.score:>3}/100  ({dim.signals_count} signals)")
    lines.append("")

    explanation = assessment.explain_score()
    if assessment.signals:
        lines.append("  Why this score:")
        lines.append(f"    {explanation.summary}")
        for contributor in explanation.dominant_contributors[:5]:
            gain = contributor.estimated_score_gain
            lines.append(
                f"    - {contributor.cause} [{contributor.dimension}/{contributor.collector_id}] "
                f"{contributor.signal_count} signal(s), "
                f"{contributor.artifact_count} artifact(s), up to +{gain:.1f} score"
            )
            if contributor.sample_signals:
                sample = contributor.sample_signals[0]
                lines.append(
                    f"      Example: {sample['artifact_uri']} "
                    f"({sample['signal_id']})"
                )
        lines.append("")

    if assessment.signals:
        lines.append(f"  Signals ({len(assessment.signals)}):")
        lines.append("")

        # Group by severity
        severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
        for sev in severity_order:
            sev_signals = [s for s in assessment.signals if s.severity.value == sev]
            if not sev_signals:
                continue
            lines.append(f"  [{sev}] ({len(sev_signals)})")
            for s in sev_signals[:20]:  # Limit output
                lines.append(f"    {s.artifact_uri}:{s.line} - {s.collector_id}")
                lines.append(f"      {s.recommendation}")
                if s.evidence:
                    for k, v in list(s.evidence.items())[:3]:
                        lines.append(f"      {k}: {v}")
            if len(sev_signals) > 20:
                lines.append(f"    ... and {len(sev_signals) - 20} more")
            lines.append("")
    else:
        lines.append("  No signals. Knowledge base is AI-ready.")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_json(assessment: KnowledgeAssessment) -> str:
    """Format assessment as JSON."""
    return json.dumps(assessment.to_dict(), indent=2, default=str)


def format_sarif(assessment: KnowledgeAssessment) -> str:
    """Format assessment as SARIF (Static Analysis Results Interchange Format)."""
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
                                "name": s.collector_id.replace("_", " ").title(),
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


def format_regression_terminal(report: AssessmentDiff) -> str:
    """Format assessment diff as human-readable terminal output."""
    lines: list[str] = []
    lines.append("")
    lines.append("AI Readiness Diff")
    lines.append("=" * 60)
    lines.append(f"  Previous: {report.prev_assessment_id} (score: {report.prev_score})")
    lines.append(f"  Current:  {report.curr_assessment_id} (score: {report.curr_score})")
    lines.append(f"  Score:    {report.prev_score} -> {report.curr_score} ({report.score_delta:+d})")
    lines.append("")

    if report.explanation:
        lines.append(f"  Summary: {report.explanation}")
        lines.append("")

    if report.dimension_deltas:
        lines.append("  Dimension Changes:")
        for name, delta in sorted(report.dimension_deltas.items()):
            indicator = "[+]" if delta > 0 else "[!]" if delta < 0 else "   "
            lines.append(f"    {indicator} {name:20s} {delta:+d}")
        lines.append("")

    if report.score_change_explanation:
        regressions = report.score_change_explanation.get("top_regressions", [])
        improvements = report.score_change_explanation.get("top_improvements", [])
        if regressions or improvements:
            lines.append("  Why the score changed:")
            for change in regressions[:3]:
                lines.append(
                    f"    [!] {change['cause']} "
                    f"({change['prev_signals']} -> {change['curr_signals']} signals, "
                    f"{change['delta_estimated_score_gain']:+.1f} score pressure)"
                )
            for change in improvements[:3]:
                lines.append(
                    f"    [+] {change['cause']} "
                    f"({change['prev_signals']} -> {change['curr_signals']} signals, "
                    f"{change['delta_estimated_score_gain']:+.1f} score pressure)"
                )
            lines.append("")

    if report.new_signals:
        lines.append(f"  New Issues ({len(report.new_signals)}):")
        for s in report.new_signals[:20]:
            lines.append(f"    * {s.collector_id}: {s.artifact_uri}")
        if len(report.new_signals) > 20:
            lines.append(f"    ... and {len(report.new_signals) - 20} more")
        lines.append("")

    if report.resolved_signals:
        lines.append(f"  Resolved Issues ({len(report.resolved_signals)}):")
        for s in report.resolved_signals[:20]:
            lines.append(f"    [OK] {s.collector_id}: {s.artifact_uri}")
        if len(report.resolved_signals) > 20:
            lines.append(f"    ... and {len(report.resolved_signals) - 20} more")
        lines.append("")

    if report.persistent_signals:
        lines.append(f"  Persistent Issues: {len(report.persistent_signals)}")
        lines.append("")

    if report.severity_changes:
        lines.append(f"  Severity Changes ({len(report.severity_changes)}):")
        for sc in report.severity_changes[:10]:
            direction = "[!]" if sc["direction"] == "worse" else "[+]"
            lines.append(f"    {direction} {sc['collector_id']}: {sc['artifact_uri']} "
                         f"({sc['prev_severity']} -> {sc['curr_severity']})")
        lines.append("")

    if report.new_high_count:
        lines.append(f"  [!] {report.new_high_count} new high-severity signals")

    if report.increased_contradictions:
        lines.append(f"  [!] Contradictions increased by {report.increased_contradictions}")
    if report.increased_topic_entropy:
        lines.append(f"  [!] Topic entropy increased by {report.increased_topic_entropy}")
    if report.new_orphan_artifacts:
        lines.append(f"  [!] {report.new_orphan_artifacts} new orphan artifacts")
    if report.new_duplicate_clusters:
        lines.append(f"  [!] {report.new_duplicate_clusters} new duplicate clusters")

    lines.append("")
    if report.recommendation:
        lines.append(f"  Recommendation: {report.recommendation}")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_regression_json(report: AssessmentDiff) -> str:
    """Format assessment diff as JSON."""
    return json.dumps(report.to_dict(), indent=2, default=str)


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
