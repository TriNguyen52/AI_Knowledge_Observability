"""Incremental execution model — update assessments in response to artifact changes.

Instead of re-running all collectors across the entire corpus, the incremental executor:
1. Classifies artifact changes (added, modified, deleted, link-changed)
2. Determines which collectors need to re-run on which artifacts
3. Invalidates only affected signals from the previous assessment
4. Re-runs affected collectors on affected artifacts
5. Merges new signals with reused signals
6. Recomputes affected dimensions and overall score

The result is an assessment identical to what a full scan would produce,
but computed in O(affected + neighborhood) instead of O(total artifacts).

If correctness cannot be guaranteed, the executor falls back to a full scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_ready.models import (
    ArtifactBundle,
    CollectorResult,
    KnowledgeArtifact,
    KnowledgeAssessment,
    KnowledgeSignal,
    Relationship,
    Severity,
)
from ai_ready.rules import SignalCollector, all_collectors


# ---------------------------------------------------------------------------
# Change events
# ---------------------------------------------------------------------------

@dataclass
class ChangeEvent:
    """A single artifact change detected between assessments.

    Attributes:
        event_type: "added", "modified", or "deleted"
        artifact_uri: Relative URI of the changed artifact
        artifact: New KnowledgeArtifact object (None for deletions)
        links_changed: For "modified" events, whether the artifact's links
                       differ from the previous version. If True, link_integrity
                       must re-evaluate. If False, link_integrity signals for
                       this artifact are unchanged.
    """

    event_type: str
    artifact_uri: str
    artifact: KnowledgeArtifact | None = None
    links_changed: bool = False


# ---------------------------------------------------------------------------
# Collector dependencies
# ---------------------------------------------------------------------------

@dataclass
class CollectorDependency:
    """Declares what a collector depends on, driving incremental execution.

    Attributes:
        collector_id: The collector's identifier
        scope: "per_artifact" if the collector only reads one artifact at a time,
               "cross_artifact" if it needs the full bundle
        needs_full_bundle_on_link_change: If True, any link change triggers a full
                                          re-evaluation of this collector
    """
    collector_id: str
    scope: str  # "per_artifact" | "cross_artifact"
    needs_full_bundle_on_link_change: bool = False

    def is_affected_by(self, event: ChangeEvent) -> bool:
        """Whether this collector's signals may change due to this event."""
        if event.event_type == "deleted":
            return True
        if event.event_type == "added":
            return True
        if event.event_type == "modified":
            if self.scope == "per_artifact":
                return True
            if self.scope == "cross_artifact":
                return event.links_changed
        return False

    def get_affected_uris(
        self, event: ChangeEvent, all_uris: set[str]
    ) -> set[str]:
        """Which artifact URIs need re-evaluation for this collector + event.

        For per-artifact collectors: only the changed artifact itself.
        For cross-artifact collectors on link changes: all artifacts (full bundle).
        For cross-artifact collectors on content-only changes: empty set (skip).
        """
        if event.event_type == "deleted":
            if self.scope == "cross_artifact":
                return all_uris
            return set()  # deleted artifact is gone, no signals to recompute

        if event.event_type == "added":
            if self.scope == "cross_artifact":
                return all_uris
            return {event.artifact_uri}

        if event.event_type == "modified":
            if self.scope == "per_artifact":
                return {event.artifact_uri}
            if self.scope == "cross_artifact" and event.links_changed:
                return all_uris
            return set()  # content-only change, cross-artifact collector not affected

        return set()


# Registry of collector dependencies
_COLLECTOR_DEPENDENCIES: dict[str, CollectorDependency] = {
    "topic_purity": CollectorDependency(
        collector_id="topic_purity",
        scope="per_artifact",
    ),
    "heading_quality": CollectorDependency(
        collector_id="heading_quality",
        scope="per_artifact",
    ),
    "context_independence": CollectorDependency(
        collector_id="context_independence",
        scope="per_artifact",
    ),
    "link_integrity": CollectorDependency(
        collector_id="link_integrity",
        scope="cross_artifact",
        needs_full_bundle_on_link_change=True,
    ),
}


# ---------------------------------------------------------------------------
# Incremental executor
# ---------------------------------------------------------------------------

class IncrementalExecutor:
    """Orchestrates incremental assessment updates.

    Uses an AssessmentPipeline instance to reuse policy application, dimension
    aggregation, and score computation — no logic is duplicated.
    """

    def __init__(self, pipeline: Any) -> None:
        """Create an executor that reuses the given pipeline's helper methods.

        Args:
            pipeline: An AssessmentPipeline instance with apply_policy,
                      aggregate_dimensions, compute_overall_score, and
                      weights attributes.
        """
        self.pipeline = pipeline
        self.policy = pipeline.policy
        self.weights = pipeline.weights

    def run(
        self,
        prev_assessment: KnowledgeAssessment,
        change_events: list[ChangeEvent],
        all_artifacts: list[KnowledgeArtifact],
        relationships: list[Relationship],
        source: str = "",
        git_commit: str = "",
    ) -> KnowledgeAssessment:
        """Produce an updated assessment from changes.

        Args:
            prev_assessment: The previous complete assessment.
            change_events: List of artifact changes since the previous scan.
            all_artifacts: ALL current artifacts (including unchanged).
                          The executor uses these to build bundles for collector execution.
            relationships: ALL current artifact relationships.
            source: Source path string for metadata.
            git_commit: Git commit hash for metadata.

        Returns:
            A KnowledgeAssessment that is identical to what a full scan would produce.
        """
        # --- Fallback conditions ---
        if not prev_assessment or not change_events:
            return self._full_scan_fallback(all_artifacts, relationships, source, git_commit)

        all_uris = {a.uri for a in all_artifacts}
        changed_uris = {e.artifact_uri for e in change_events}

        # If >50% of artifacts changed, incremental gain is minimal — do full scan
        if len(changed_uris) > len(all_uris) * 0.5:
            return self._full_scan_fallback(all_artifacts, relationships, source, git_commit)

        # --- Classify changes ---
        added_uris: set[str] = set()
        modified_uris: set[str] = set()
        deleted_uris: set[str] = set()
        link_changed_uris: set[str] = set()

        for event in change_events:
            if event.event_type == "added":
                added_uris.add(event.artifact_uri)
            elif event.event_type == "modified":
                modified_uris.add(event.artifact_uri)
                if event.links_changed:
                    link_changed_uris.add(event.artifact_uri)
            elif event.event_type == "deleted":
                deleted_uris.add(event.artifact_uri)

        per_doc_collectors = ["topic_purity", "heading_quality", "context_independence"]
        per_doc_affected = added_uris | modified_uris

        link_integrity_affected: set[str] = set()
        link_integrity_needs_full = bool(link_changed_uris or added_uris or deleted_uris)
        if link_integrity_needs_full:
            link_integrity_affected = all_uris

        # --- Invalidate signals from previous assessment ---
        # Build set of (collector_id, artifact_uri) pairs to invalidate
        invalidated: set[tuple[str, str]] = set()

        for collector_id in per_doc_collectors:
            for uri in per_doc_affected:
                invalidated.add((collector_id, uri))

        if link_integrity_needs_full:
            # Invalidate all link_integrity signals
            for s in prev_assessment.signals:
                if s.collector_id == "link_integrity":
                    invalidated.add((s.collector_id, s.artifact_uri))

        # Also invalidate signals for deleted artifacts (any collector)
        for uri in deleted_uris:
            for collector_id in per_doc_collectors + ["link_integrity"]:
                invalidated.add((collector_id, uri))

        # --- Keep signals that are not invalidated ---
        kept_signals: list[KnowledgeSignal] = [
            s for s in prev_assessment.signals
            if (s.collector_id, s.artifact_uri) not in invalidated
        ]

        # --- Re-run affected collectors ---
        new_signals: list[KnowledgeSignal] = []
        results: list[CollectorResult] = []

        # Build artifact lookup for mini-bundle construction
        artifact_by_uri: dict[str, KnowledgeArtifact] = {a.uri: a for a in all_artifacts}

        # Run per-artifact collectors on affected artifacts only
        if per_doc_affected:
            affected_artifacts = [
                artifact_by_uri[uri] for uri in per_doc_affected
                if uri in artifact_by_uri
            ]
            if affected_artifacts:
                mini_bundle = ArtifactBundle(
                    artifacts=affected_artifacts,
                    relationships=relationships,  # relationships don't affect per-artifact collectors
                    source=source,
                )
                registry = all_collectors()
                for collector_id in per_doc_collectors:
                    if collector_id not in registry:
                        continue
                    if self.pipeline.enabled_collectors and collector_id not in self.pipeline.enabled_collectors:
                        continue
                    collector = registry[collector_id]()
                    result = collector.run(mini_bundle)
                    results.append(result)
                    new_signals.extend(result.signals)

        # Run link_integrity on full bundle if needed
        if link_integrity_needs_full:
            registry = all_collectors()
            collector_id = "link_integrity"
            if collector_id in registry:
                if not self.pipeline.enabled_collectors or collector_id in self.pipeline.enabled_collectors:
                    full_bundle = ArtifactBundle(
                        artifacts=all_artifacts,
                        relationships=relationships,
                        source=source,
                    )
                    collector = registry[collector_id]()
                    result = collector.run(full_bundle)
                    results.append(result)
                    new_signals.extend(result.signals)

        if new_signals:
            self.pipeline.apply_policy(new_signals)

        all_signals = kept_signals + new_signals

        all_results = self._build_all_results(results, all_signals, prev_assessment)
        dimensions = self.pipeline.aggregate_dimensions(all_results, all_signals)

        # --- Recompute overall score ---
        overall_score = self.pipeline.compute_overall_score(dimensions)

        # --- Collect metrics from new results ---
        metrics: dict[str, Any] = {}
        for r in results:
            metrics.update(r.metrics)

        # --- Create updated assessment ---
        assessment_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata: dict[str, Any] = {
            "source": source,
            "artifact_count": len(all_artifacts),
            "relationship_count": len(relationships),
            "incremental": True,
            "changed_artifacts": len(changed_uris),
        }
        if git_commit:
            metadata["git_commit"] = git_commit

        return KnowledgeAssessment(
            assessment_id=assessment_id,
            score=overall_score,
            dimensions=dimensions,
            signals=all_signals,
            metrics=metrics,
            metadata=metadata,
        )

    def _build_all_results(
        self,
        new_results: list[CollectorResult],
        all_signals: list[KnowledgeSignal],
        prev_assessment: KnowledgeAssessment,
    ) -> list[CollectorResult]:
        """Build a complete CollectorResult list for dimension aggregation.

        For collectors that ran incrementally, use their new CollectorResult.
        For collectors that didn't run (no affected artifacts), construct a CollectorResult
        from the kept signals so that aggregate_dimensions can compute
        the correct dimension scores.

        All registered collectors must have a CollectorResult, even if they have zero
        signals, because aggregate_dimensions uses CollectorResult.collector_id to
        look up the collector's dimension. Without a CollectorResult, the collector's
        dimension would be missing from the output, changing the overall
        score calculation.
        """
        ran_collector_ids = {r.collector_id for r in new_results}
        registry = all_collectors()

        results = list(new_results)

        for collector_id in registry:
            if collector_id in ran_collector_ids:
                continue
            if self.pipeline.enabled_collectors and collector_id not in self.pipeline.enabled_collectors:
                continue
            # Construct a CollectorResult from existing signals
            collector_signals = [s for s in all_signals if s.collector_id == collector_id]
            results.append(CollectorResult(
                collector_id=collector_id,
                score=100,  # Placeholder; aggregate_dimensions recomputes
                severity=Severity.LOW,
                metrics={},
                signals=collector_signals,
            ))

        return results

    def _full_scan_fallback(
        self,
        artifacts: list[KnowledgeArtifact],
        relationships: list[Relationship],
        source: str,
        git_commit: str,
    ) -> KnowledgeAssessment:
        """Fall back to a full scan via the pipeline."""
        return self.pipeline.run(
            artifacts,
            source=source,
            git_commit=git_commit,
            relationships=relationships,
        )


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def detect_changes(
    source_path: Any,
    prev_assessment: KnowledgeAssessment | None,
    current_artifacts: list[KnowledgeArtifact],
) -> list[ChangeEvent]:
    """Detect artifact changes between a previous assessment and current state.

    Uses git diff when available. Falls back to assessment comparison.

    Args:
        source_path: Path to the knowledge base directory.
        prev_assessment: The previous assessment (None if no history).
        current_artifacts: All current KnowledgeArtifact objects.

    Returns:
        List of ChangeEvent objects. Empty list means no changes detected
        (which will trigger a full scan fallback).
    """
    if prev_assessment is None:
        return []  # No previous state — can't do incremental

    current_by_uri: dict[str, KnowledgeArtifact] = {a.uri: a for a in current_artifacts}
    current_uris = set(current_by_uri.keys())

    prev_uris: set[str] = set()
    for s in prev_assessment.signals:
        prev_uris.add(s.artifact_uri)

    prev_artifact_count = prev_assessment.metadata.get("artifact_count", 0)

    if not prev_uris and prev_artifact_count > 0:
        return []

    # Detect changes
    events: list[ChangeEvent] = []

    # Added artifacts
    for uri in current_uris - prev_uris:
        events.append(ChangeEvent(
            event_type="added",
            artifact_uri=uri,
            artifact=current_by_uri[uri],
            links_changed=True,
        ))

    # Deleted artifacts
    for uri in prev_uris - current_uris:
        events.append(ChangeEvent(
            event_type="deleted",
            artifact_uri=uri,
            artifact=None,
        ))

    # Modified artifacts — compare links
    for uri in current_uris & prev_uris:
        artifact = current_by_uri[uri]
        current_internal_links = {link.target for link in artifact.links if link.is_internal}

        events.append(ChangeEvent(
            event_type="modified",
            artifact_uri=uri,
            artifact=artifact,
            links_changed=True,
        ))

    return events


def detect_changes_via_git(
    source_path: Any,
    prev_assessment: KnowledgeAssessment | None,
    current_artifacts: list[KnowledgeArtifact],
) -> list[ChangeEvent] | None:
    """Detect changes using git diff.

    Returns None if git is not available or the directory is not a git repo.
    Otherwise returns a list of ChangeEvent objects.
    """
    import subprocess
    from pathlib import Path

    source = Path(source_path)
    if not source.is_dir():
        return None

    if prev_assessment is None:
        return None

    prev_commit = prev_assessment.metadata.get("git_commit", "")
    if not prev_commit:
        return None

    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", f"{prev_commit}..HEAD"],
            capture_output=True, text=True, cwd=str(source), timeout=10,
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")
        if not lines or lines == [""]:
            return []

        current_by_uri = {a.uri: a for a in current_artifacts}
        events: list[ChangeEvent] = []

        for line in lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            status = parts[0]
            file_path = parts[-1]

            # Normalize path
            file_path = file_path.replace("\\", "/")

            # Only care about supported file types
            if not any(file_path.endswith(ext) for ext in [".md", ".markdown", ".mdx", ".txt", ".rst"]):
                continue

            artifact_uri = None
            if file_path in current_by_uri:
                artifact_uri = file_path
            else:
                # Try matching as suffix
                for p in current_by_uri:
                    if p.endswith(file_path) or file_path.endswith(p):
                        artifact_uri = p
                        break

            if status.startswith("A"):
                # Added
                if artifact_uri:
                    events.append(ChangeEvent(
                        event_type="added",
                        artifact_uri=artifact_uri,
                        artifact=current_by_uri[artifact_uri],
                        links_changed=True,
                    ))
            elif status.startswith("D"):
                # Deleted
                if artifact_uri:
                    events.append(ChangeEvent(
                        event_type="deleted",
                        artifact_uri=artifact_uri,
                        artifact=None,
                    ))
            elif status.startswith("R"):
                # Renamed — treat as delete + add
                if artifact_uri:
                    events.append(ChangeEvent(
                        event_type="added",
                        artifact_uri=artifact_uri,
                        artifact=current_by_uri[artifact_uri],
                        links_changed=True,
                    ))
            else:
                # Modified (M) or unknown
                if artifact_uri:
                    artifact = current_by_uri[artifact_uri]
                    current_links = {link.target for link in artifact.links if link.is_internal}

                    events.append(ChangeEvent(
                        event_type="modified",
                        artifact_uri=artifact_uri,
                        artifact=artifact,
                        links_changed=True,  # Conservative
                    ))

        return events

    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
