"""Assessment pipeline - runs collectors against an artifact bundle and aggregates results into an assessment.

AssessmentPipeline is a facade that delegates to CollectOperation and
AssessOperation. The collect phase runs collectors and produces raw signals.
The assess phase applies interpretation policy, aggregates dimensions, and
computes the overall score.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_ready.evaluation_policy import InterpretationPolicy
from ai_ready.models import (
    ArtifactBundle,
    CollectorResult,
    DimensionScore,
    KnowledgeArtifact,
    KnowledgeAssessment,
    KnowledgeSignal,
    Relationship,
    Severity,
)
from ai_ready.rules import SignalCollector, all_collectors
from ai_ready.operations import CollectOperation, AssessOperation

# Default dimension weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "retrieval": 0.25,
    "context": 0.15,
    "consistency": 0.20,
    "trust": 0.20,
    "connectivity": 0.10,
    "workflow": 0.10,
}

# Default thresholds
DEFAULT_THRESHOLDS = {
    "overall_score": 0,
}

DEFAULT_FAIL_ON: list[str] = ["CRITICAL"]


class AssessmentPipeline:
    """Orchestrates collector execution, dimension aggregation, and assessment creation.

    Delegates to CollectOperation (run collectors -> raw signals) and
    AssessOperation (apply policy -> aggregate dimensions -> compute score).
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        enabled_collectors: list[str] | None = None,
        thresholds: dict[str, Any] | None = None,
        fail_on: list[str] | None = None,
    ) -> None:
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS
        self.thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
        self.fail_on = fail_on if fail_on is not None else DEFAULT_FAIL_ON
        self.enabled_collectors = enabled_collectors
        self.policy = InterpretationPolicy()

        # Create delegate operations
        self._collect_op = CollectOperation(enabled_collectors=enabled_collectors)
        self._assess_op = AssessOperation(weights=self.weights, policy=self.policy)

    def run(
        self,
        artifacts: list[KnowledgeArtifact],
        source: str = "",
        git_commit: str = "",
        relationships: list[Relationship] | None = None,
    ) -> KnowledgeAssessment:
        """Run all enabled collectors and produce an assessment.

        Delegates to CollectOperation then AssessOperation.

        Args:
            artifacts: List of KnowledgeArtifact objects to analyze.
            source: Source path string for metadata.
            git_commit: Git commit hash for metadata.
            relationships: Optional list of Relationship objects for artifact relationships.
        """
        # Collect phase: run collectors -> raw signals
        results, all_signals, bundle = self._collect_op.run(
            artifacts=artifacts,
            relationships=relationships,
            source=source,
        )

        # Assess phase: signals -> apply policy -> aggregate -> score
        return self._assess_op.run(
            results=results,
            signals=all_signals,
            artifacts=artifacts,
            bundle=bundle,
            source=source,
            git_commit=git_commit,
        )

    def run_incremental(
        self,
        prev_assessment: KnowledgeAssessment,
        change_events: list,
        artifacts: list[KnowledgeArtifact],
        relationships: list[Relationship] | None = None,
        source: str = "",
        git_commit: str = "",
    ) -> KnowledgeAssessment:
        """Run an incremental assessment, updating only affected signals.

        Delegates to IncrementalExecutor, reusing this pipeline's helper
        methods for policy application, dimension aggregation, and score
        computation.

        Args:
            prev_assessment: The previous complete assessment.
            change_events: List of ChangeEvent objects since the previous assessment.
            artifacts: ALL current KnowledgeArtifact objects.
            relationships: ALL current Relationship objects.
            source: Source path string for metadata.
            git_commit: Git commit hash for metadata.

        Returns:
            A KnowledgeAssessment identical to what a full scan would produce.
        """
        from ai_ready.incremental import IncrementalExecutor

        executor = IncrementalExecutor(self)
        return executor.run(
            prev_assessment=prev_assessment,
            change_events=change_events,
            all_artifacts=artifacts,
            relationships=relationships or [],
            source=source,
            git_commit=git_commit,
        )

    # --- Delegate methods ---

    def aggregate_dimensions(
        self, results: list[CollectorResult], signals: list[KnowledgeSignal]
    ) -> dict[str, DimensionScore]:
        """Aggregate collector results into dimension scores. Delegates to AssessOperation."""
        return self._assess_op.aggregate_dimensions(results, signals)

    def apply_policy(self, signals: list[KnowledgeSignal]) -> None:
        """Populate severity, score, ai_impact, and recommendation from policy. Delegates to AssessOperation."""
        self._assess_op.apply_policy(signals)

    def compute_overall_score(self, dimensions: dict[str, DimensionScore]) -> int:
        """Compute weighted average of dimension scores. Delegates to AssessOperation."""
        return self._assess_op.compute_overall_score(dimensions)

    def get_exit_code(self, assessment: KnowledgeAssessment) -> int:
        """Determine exit code based on assessment results."""
        fail_severities = set()
        for s in self.fail_on:
            try:
                fail_severities.add(Severity(s))
            except ValueError:
                pass

        has_fail_severity = any(
            sig.severity in fail_severities for sig in assessment.signals
        )
        if has_fail_severity:
            return 2

        threshold = self.thresholds.get("overall_score", 0)
        if assessment.score < threshold:
            return 1

        return 0

