"""Collect and Assess operations — extracted from AssessmentPipeline.

CollectOperation runs collectors against an artifact bundle and produces
raw KnowledgeSignals (bare facts without interpretation).

AssessOperation applies interpretation policy to signals, then aggregates
dimensions and computes the overall score to produce a KnowledgeAssessment.

AssessmentPipeline.run() delegates to CollectOperation then AssessOperation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ai_ready.evaluation_policy import InterpretationPolicy
from ai_ready.models import (
    ArtifactBundle,
    AssessmentDiff,
    CollectorResult,
    DimensionScore,
    KnowledgeArtifact,
    KnowledgeAssessment,
    KnowledgeSignal,
    Relationship,
    Severity,
    _severity_rank,
)
from ai_ready.rules import SignalCollector, all_collectors


class CollectOperation:
    """Runs collectors against an artifact bundle and produces raw signals.

    Extracted from AssessmentPipeline.run() — the collection phase.

    Collectors (SignalCollectors) emit KnowledgeSignal objects as the
    primary output. CollectOperation passes an ArtifactBundle to each
    collector and collects the resulting CollectorResult objects.
    """

    def __init__(self, enabled_collectors: list[str] | None = None) -> None:
        self.enabled_collectors = enabled_collectors

    def run(
        self,
        artifacts: list[KnowledgeArtifact],
        relationships: list[Relationship] | None = None,
        source: str = "",
    ) -> tuple[list[CollectorResult], list[KnowledgeSignal], ArtifactBundle]:
        """Run all enabled collectors and return results, signals, and the bundle.

        Returns:
            A tuple of (collector_results, all_signals, artifact_bundle).
        """
        bundle = ArtifactBundle(
            artifacts=artifacts,
            relationships=relationships or [],
            source=source,
        )

        registry = all_collectors()
        if self.enabled_collectors:
            collector_ids = [c for c in self.enabled_collectors if c in registry]
        else:
            collector_ids = list(registry.keys())

        results: list[CollectorResult] = []
        all_signals: list[KnowledgeSignal] = []
        for collector_id in collector_ids:
            collector_cls = registry[collector_id]
            collector = collector_cls()
            result = collector.run(bundle)
            results.append(result)
            all_signals.extend(result.signals)

        return results, all_signals, bundle


class AssessOperation:
    """Applies interpretation policy, aggregates dimensions, computes score.

    Extracted from AssessmentPipeline.run() — the assessment phase.

    Applies InterpretationPolicy to enrich KnowledgeSignals with severity,
    score, recommendation, and ai_impact, then aggregates dimensions and
    computes the overall score to produce a KnowledgeAssessment.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        policy: InterpretationPolicy | None = None,
    ) -> None:
        from ai_ready.pipeline import DEFAULT_WEIGHTS
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS
        self.policy = policy if policy is not None else InterpretationPolicy()

    def run(
        self,
        results: list[CollectorResult],
        signals: list[KnowledgeSignal],
        artifacts: list[KnowledgeArtifact],
        bundle: ArtifactBundle,
        source: str = "",
        git_commit: str = "",
    ) -> KnowledgeAssessment:
        """Apply policy to signals, aggregate, and produce an assessment.

        Args:
            results: Collector results from CollectOperation.
            signals: Bare KnowledgeSignals from CollectOperation.
            artifacts: Original artifact list (for metadata).
            bundle: ArtifactBundle from CollectOperation.
            source: Source path string for metadata.
            git_commit: Git commit hash for metadata.

        Returns:
            A KnowledgeAssessment with assessed signals, dimension scores, and overall score.
        """
        # Apply policy to populate severity, score, ai_impact, recommendation
        self.apply_policy(signals)

        # Aggregate into dimensions
        dimensions = self.aggregate_dimensions(results, signals)

        # Compute overall score
        overall_score = self.compute_overall_score(dimensions)

        # Collect metrics
        metrics: dict[str, Any] = {}
        for r in results:
            metrics.update(r.metrics)

        # Create assessment
        assessment_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata: dict[str, Any] = {"source": source}
        if git_commit:
            metadata["git_commit"] = git_commit
        metadata["artifact_count"] = len(artifacts)
        metadata["relationship_count"] = len(bundle.relationships)

        return KnowledgeAssessment(
            assessment_id=assessment_id,
            score=overall_score,
            dimensions=dimensions,
            signals=signals,
            metrics=metrics,
            metadata=metadata,
        )

    def aggregate_dimensions(
        self, results: list[CollectorResult], signals: list[KnowledgeSignal]
    ) -> dict[str, DimensionScore]:
        """Aggregate collector results into dimension scores.

        Collector scores are recomputed from policy-assessed signals rather than
        the placeholder scores emitted by collectors.
        """
        dim_results: dict[str, list[CollectorResult]] = {}
        for r in results:
            collector_cls = all_collectors().get(r.collector_id)
            if collector_cls:
                dim = collector_cls.dimension
                dim_results.setdefault(dim, []).append(r)

        dimensions: dict[str, DimensionScore] = {}
        for dim_name, dim_results_list in dim_results.items():
            # Recompute collector scores from assessed signals
            collector_scores = []
            for r in dim_results_list:
                collector_signals = [s for s in signals if s.collector_id == r.collector_id]
                if collector_signals:
                    total_penalty = sum(100 - s.score for s in collector_signals)
                    collector_score = max(0, 100 - total_penalty)
                else:
                    collector_score = 100
                collector_scores.append(collector_score)

            avg_score = int(sum(collector_scores) / len(collector_scores)) if collector_scores else 100
            collector_ids = [r.collector_id for r in dim_results_list]
            signals_count = sum(
                len([s for s in signals if s.collector_id == r.collector_id]) for r in dim_results_list
            )

            dimensions[dim_name] = DimensionScore(
                name=dim_name,
                score=avg_score,
                collector_ids=collector_ids,
                signals_count=signals_count,
            )

        return dimensions

    def apply_policy(self, signals: list[KnowledgeSignal]) -> None:
        """Populate severity, score, ai_impact, and recommendation from policy."""
        for signal in signals:
            entry = self.policy.lookup(signal.collector_id, signal.signal_type)
            if entry:
                signal.severity = entry.severity
                signal.score = 100 - entry.score_penalty
                signal.ai_impact = entry.ai_impact
                signal.recommendation = entry.recommendation_template.format(
                    **signal.evidence
                )
            else:
                signal.ai_impact = "No AI impact description available."
                signal.recommendation = f"Issue: {signal.signal_type}"

    def compute_overall_score(self, dimensions: dict[str, DimensionScore]) -> int:
        """Compute weighted average of dimension scores."""
        total_weight = 0.0
        weighted_sum = 0.0
        for dim_name, dim_score in dimensions.items():
            weight = self.weights.get(dim_name, 0)
            weighted_sum += dim_score.score * weight
            total_weight += weight

        if total_weight == 0:
            return 100
        return int(weighted_sum / total_weight)


class DiffOperation:
    """Computes diffs between two assessments.

    Extracted from AssessmentStore._compute_diff — the diff/regression logic
    is pure computation that doesn't need DB access. This makes it testable
    in isolation and reusable by both AssessmentStore and AssessmentStoreFacade.
    """

    @staticmethod
    def compute_diff(prev: KnowledgeAssessment, curr: KnowledgeAssessment) -> AssessmentDiff:
        """Compute diff between two assessments using stable signal IDs."""
        prev_by_id = {s.signal_id: s for s in prev.signals}
        curr_by_id = {s.signal_id: s for s in curr.signals}

        prev_ids = set(prev_by_id.keys())
        curr_ids = set(curr_by_id.keys())

        new_signals = [curr_by_id[sid] for sid in curr_ids - prev_ids]
        resolved_signals = [prev_by_id[sid] for sid in prev_ids - curr_ids]
        persistent_signals = [curr_by_id[sid] for sid in curr_ids & prev_ids]

        # Detect severity changes
        severity_changes: list[dict[str, Any]] = []
        for sid in curr_ids & prev_ids:
            ps = prev_by_id[sid]
            cs = curr_by_id[sid]
            if ps.severity != cs.severity:
                severity_changes.append({
                    "signal_id": sid,
                    "collector_id": cs.collector_id,
                    "artifact_uri": cs.artifact_uri,
                    "prev_severity": ps.severity.value,
                    "curr_severity": cs.severity.value,
                    "direction": "worse" if _severity_rank(cs.severity) > _severity_rank(ps.severity) else "better",
                })

        # Dimension deltas
        dim_deltas: dict[str, int] = {}
        for dim_name in set(list(prev.dimensions.keys()) + list(curr.dimensions.keys())):
            prev_score = prev.dimensions[dim_name].score if dim_name in prev.dimensions else 0
            curr_score = curr.dimensions[dim_name].score if dim_name in curr.dimensions else 0
            dim_deltas[dim_name] = curr_score - prev_score

        new_high = sum(1 for s in new_signals if s.severity in (Severity.HIGH, Severity.CRITICAL))

        # Metric deltas
        prev_contradictions = prev.metrics.get("contradiction_count", 0)
        curr_contradictions = curr.metrics.get("contradiction_count", 0)
        prev_entropy = prev.metrics.get("avg_topic_entropy", 0)
        curr_entropy = curr.metrics.get("avg_topic_entropy", 0)
        prev_orphans = prev.metrics.get("orphan_documents", 0)
        curr_orphans = curr.metrics.get("orphan_documents", 0)
        prev_dupes = prev.metrics.get("duplicate_clusters", 0)
        curr_dupes = curr.metrics.get("duplicate_clusters", 0)

        contributor_changes = DiffOperation._compute_contributor_changes(prev, curr)
        dimension_change_explanations = [
            {
                "dimension": name,
                "prev_score": prev.dimensions[name].score if name in prev.dimensions else None,
                "curr_score": curr.dimensions[name].score if name in curr.dimensions else None,
                "delta": delta,
                "direction": "worse" if delta < 0 else "better" if delta > 0 else "unchanged",
            }
            for name, delta in sorted(
                dim_deltas.items(),
                key=lambda item: (abs(item[1]), item[0]),
                reverse=True,
            )
            if delta != 0
        ]

        # Build explanation
        explanation_parts: list[str] = []
        if new_signals:
            explanation_parts.append(f"{len(new_signals)} new signals")
        if resolved_signals:
            explanation_parts.append(f"{len(resolved_signals)} resolved")
        if severity_changes:
            worse = sum(1 for s in severity_changes if s["direction"] == "worse")
            better = sum(1 for s in severity_changes if s["direction"] == "better")
            if worse:
                explanation_parts.append(f"{worse} signals got worse")
            if better:
                explanation_parts.append(f"{better} signals improved")
        top_regression = next(
            (c for c in contributor_changes if c["delta_estimated_score_gain"] > 0),
            None,
        )
        if top_regression:
            explanation_parts.append(
                f"largest new score pressure: {top_regression['cause']}"
            )
        if not explanation_parts:
            explanation_parts.append("No changes detected")
        explanation = "; ".join(explanation_parts) + "."

        score_change_explanation = {
            "summary": explanation,
            "score_delta": curr.score - prev.score,
            "dimension_changes": dimension_change_explanations,
            "top_regressions": [
                c for c in contributor_changes
                if c["delta_estimated_score_gain"] > 0
            ][:5],
            "top_improvements": [
                c for c in contributor_changes
                if c["delta_estimated_score_gain"] < 0
            ][:5],
        }

        recommendation = ""
        if curr.score < prev.score:
            recommendation = f"Score dropped {prev.score} -> {curr.score}. Review {len(new_signals)} new signals before deployment."
        elif new_high > 0:
            recommendation = f"{new_high} new high-severity signals. Review before deployment."
        elif len(resolved_signals) > 0 and not new_signals:
            recommendation = f"Improving. {len(resolved_signals)} signals resolved, no new issues."
        elif not new_signals:
            recommendation = "No regressions detected."

        return AssessmentDiff(
            prev_assessment_id=prev.assessment_id,
            curr_assessment_id=curr.assessment_id,
            prev_score=prev.score,
            curr_score=curr.score,
            score_delta=curr.score - prev.score,
            new_signals=new_signals,
            resolved_signals=resolved_signals,
            persistent_signals=persistent_signals,
            severity_changes=severity_changes,
            dimension_deltas=dim_deltas,
            new_high_count=new_high,
            increased_contradictions=max(0, curr_contradictions - prev_contradictions),
            increased_topic_entropy=max(0, curr_entropy - prev_entropy),
            new_orphan_artifacts=max(0, curr_orphans - prev_orphans),
            new_duplicate_clusters=max(0, curr_dupes - prev_dupes),
            score_change_explanation=score_change_explanation,
            contributor_changes=contributor_changes[:10],
            recommendation=recommendation,
            explanation=explanation,
        )

    @staticmethod
    def _compute_contributor_changes(
        prev: KnowledgeAssessment,
        curr: KnowledgeAssessment,
    ) -> list[dict[str, Any]]:
        """Compare derived score contributors between two assessments."""
        prev_contributors = {
            c.key: c for c in prev.explain_score(max_contributors=50).dominant_contributors
        }
        curr_contributors = {
            c.key: c for c in curr.explain_score(max_contributors=50).dominant_contributors
        }

        changes: list[dict[str, Any]] = []
        for key in set(prev_contributors) | set(curr_contributors):
            prev_c = prev_contributors.get(key)
            curr_c = curr_contributors.get(key)
            representative = curr_c or prev_c
            if representative is None:
                continue

            prev_gain = prev_c.estimated_score_gain if prev_c else 0.0
            curr_gain = curr_c.estimated_score_gain if curr_c else 0.0
            prev_count = prev_c.signal_count if prev_c else 0
            curr_count = curr_c.signal_count if curr_c else 0
            prev_ids = set(prev_c.signal_ids if prev_c else [])
            curr_ids = set(curr_c.signal_ids if curr_c else [])

            delta = curr_gain - prev_gain
            if delta == 0 and curr_count == prev_count:
                continue

            changes.append({
                "key": key,
                "dimension": representative.dimension,
                "collector_id": representative.collector_id,
                "signal_type": representative.signal_type,
                "cause": representative.cause,
                "prev_estimated_score_gain": round(prev_gain, 2),
                "curr_estimated_score_gain": round(curr_gain, 2),
                "delta_estimated_score_gain": round(delta, 2),
                "prev_signals": prev_count,
                "curr_signals": curr_count,
                "signal_count_delta": curr_count - prev_count,
                "added_signal_ids": sorted(curr_ids - prev_ids)[:20],
                "removed_signal_ids": sorted(prev_ids - curr_ids)[:20],
            })

        return sorted(
            changes,
            key=lambda c: (
                abs(c["delta_estimated_score_gain"]),
                abs(c["signal_count_delta"]),
            ),
            reverse=True,
        )
