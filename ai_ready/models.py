"""Canonical domain model for AI-Ready knowledge observability.

Architecture:
  KnowledgeArtifact  — normalized representation of any source context
  KnowledgeSignal    — observable fact + interpretation (severity, score, …)
  KnowledgeAssessment — interpreted view of signals with dimension scores
  SignalCollector    — produces signals from an ArtifactBundle
  InterpretationPolicy — enriches bare signals with severity/score/recommendation
  SignalStore / AssessmentStore — persistence layer
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


_SEVERITY_ORDER = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}


def _severity_rank(s: Severity) -> int:
    """Return numeric rank for severity comparison (higher = worse)."""
    return _SEVERITY_ORDER.get(s, 0)


def make_signal_id(collector_id: str, artifact_uri: str, signal_type: str) -> str:
    """Generate a stable signal ID that survives line shifts and minor edits.

    The ID is derived from (collector_id, artifact_uri, signal_type) where
    signal_type identifies the *what* of the signal, not the *where*.
    Line numbers are deliberately excluded.
    """
    raw = f"{collector_id}:{artifact_uri}:{signal_type}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Structural content types
# ---------------------------------------------------------------------------

@dataclass
class Heading:
    level: int
    text: str
    line: int = 0

    def __post_init__(self) -> None:
        if self.level < 1:
            raise ValueError(f"Heading level must be >= 1, got {self.level}")


@dataclass
class Section:
    heading: Heading | None
    text: str
    line_start: int = 0
    line_end: int = 0


@dataclass
class Paragraph:
    text: str
    section_heading: str = ""
    line: int = 0


@dataclass
class Link:
    text: str
    target: str
    line: int = 0
    is_internal: bool = True


@dataclass
class CodeBlock:
    language: str
    text: str
    line: int = 0


@dataclass
class DocumentContent:
    """Structured content of a knowledge artifact.

    Wraps the heading, section, paragraph, link, and code-block lists
    that a parsed document carries. This gives KnowledgeArtifact a
    single ``content`` attribute instead of a flat dict.
    """

    headings: list[Heading] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Artifact model
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeArtifact:
    """Any external context that contributes to AI reasoning.

    A document is one type of artifact (``artifact_type="document"``).
    Future types may include schemas, API specs, examples, embeddings, etc.
    """

    id: str
    uri: str
    title: str
    artifact_type: str = "document"
    content: DocumentContent | dict[str, Any] = field(default_factory=dict)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def headings(self) -> list[Heading]:
        if isinstance(self.content, DocumentContent):
            return self.content.headings
        if isinstance(self.content, dict):
            return self.content.get("headings", [])
        return []

    @property
    def sections(self) -> list[Section]:
        if isinstance(self.content, DocumentContent):
            return self.content.sections
        if isinstance(self.content, dict):
            return self.content.get("sections", [])
        return []

    @property
    def paragraphs(self) -> list[Paragraph]:
        if isinstance(self.content, DocumentContent):
            return self.content.paragraphs
        if isinstance(self.content, dict):
            return self.content.get("paragraphs", [])
        return []

    @property
    def links(self) -> list[Link]:
        if isinstance(self.content, DocumentContent):
            return self.content.links
        if isinstance(self.content, dict):
            return self.content.get("links", [])
        return []

    @property
    def code_blocks(self) -> list[CodeBlock]:
        if isinstance(self.content, DocumentContent):
            return self.content.code_blocks
        if isinstance(self.content, dict):
            return self.content.get("code_blocks", [])
        return []

    @property
    def word_count(self) -> int:
        total = 0
        for section in self.sections:
            total += len(section.text.split())
        return total

    @property
    def all_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


@dataclass
class Relationship:
    """A relationship between two artifacts in the knowledge ecosystem."""

    source_uri: str
    target_uri: str
    relation_type: str  # "parent_child", "cross_reference", "navigation"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactBundle:
    """A collection of artifacts and their relationships."""

    artifacts: list[KnowledgeArtifact] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_count(self) -> int:
        return len(self.artifacts)

    def get_artifact(self, uri: str) -> KnowledgeArtifact | None:
        for artifact in self.artifacts:
            if artifact.uri == uri:
                return artifact
        return None

    def get_children(self, uri: str) -> list[str]:
        return [
            r.target_uri for r in self.relationships
            if r.source_uri == uri and r.relation_type == "parent_child"
        ]

    def get_parents(self, uri: str) -> list[str]:
        return [
            r.source_uri for r in self.relationships
            if r.target_uri == uri and r.relation_type == "parent_child"
        ]

    def get_cross_references(self, uri: str) -> list[str]:
        """Get artifacts that this artifact cross-references."""
        return [
            r.target_uri for r in self.relationships
            if r.source_uri == uri and r.relation_type == "cross_reference"
        ]

    def is_linked(self, uri: str) -> bool:
        for r in self.relationships:
            if r.target_uri == uri and r.source_uri != uri:
                return True
        return False


# ---------------------------------------------------------------------------
# Signal model
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeSignal:
    """A single observable fact about a knowledge artifact.

    Signals are produced by SignalCollectors. They carry both bare facts
    (signal_id, collector_id, signal_type, artifact_uri, evidence, line) and
    interpretation (severity, score, recommendation, ai_impact). The
    interpretation fields are populated by the assessment layer via
    InterpretationPolicy.
    """

    collector_id: str
    signal_type: str
    artifact_uri: str
    evidence: dict[str, Any]
    signal_id: str = ""
    artifact_id: str = ""
    line: int = 0
    # Interpretation fields — populated by the assessment layer
    severity: Severity = Severity.LOW
    score: int = 100
    recommendation: str = ""
    ai_impact: str = ""

    def __post_init__(self) -> None:
        if not self.signal_id:
            self.signal_id = make_signal_id(
                self.collector_id, self.artifact_uri, self.signal_type
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "collector_id": self.collector_id,
            "signal_type": self.signal_type,
            "severity": self.severity.value,
            "score": self.score,
            "artifact_id": self.artifact_id,
            "artifact_uri": self.artifact_uri,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "ai_impact": self.ai_impact,
            "line": self.line,
        }


class SignalStatus(str, Enum):
    """Lifecycle status for a signal across assessments."""

    NEW = "new"
    PERSISTENT = "persistent"
    RESOLVED = "resolved"
    RECURRING = "recurring"


@dataclass
class SignalLifecycle:
    """Lifecycle tracking for a single signal across assessments."""

    signal_id: str
    collector_id: str
    artifact_uri: str
    first_seen: str = ""
    last_seen: str = ""
    status: SignalStatus = SignalStatus.NEW
    severity_history: list[dict[str, Any]] = field(default_factory=list)
    assessment_ids: list[str] = field(default_factory=list)
    resolved_assessment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "collector_id": self.collector_id,
            "artifact_uri": self.artifact_uri,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "status": self.status.value,
            "severity_history": self.severity_history,
            "assessment_ids": self.assessment_ids,
            "resolved_assessment": self.resolved_assessment,
            "age_assessments": len(self.assessment_ids),
        }


@dataclass
class CollectorResult:
    """Output from a single SignalCollector evaluation."""

    collector_id: str
    score: int
    severity: Severity
    metrics: dict[str, Any] = field(default_factory=dict)
    signals: list[KnowledgeSignal] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "collector_id": self.collector_id,
            "score": self.score,
            "severity": self.severity.value,
            "metrics": self.metrics,
            "signals": [s.to_dict() for s in self.signals],
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Scoring model
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """Score for a single AI readiness dimension."""

    name: str
    score: int
    collector_ids: list[str] = field(default_factory=list)
    signals_count: int = 0


@dataclass
class ScoreContributor:
    """Derived explanation for a repeated score contributor.

    This is intentionally not persisted as its own entity. It is computed from
    signals that were already interpreted by the InterpretationPolicy.
    """

    key: str
    dimension: str
    collector_id: str
    signal_type: str
    cause: str
    signal_count: int
    artifact_count: int
    severity: Severity
    total_penalty: int
    estimated_collector_gain: int
    estimated_dimension_gain: float
    estimated_score_gain: float
    recommendation: str
    ai_impact: str
    signal_ids: list[str] = field(default_factory=list)
    sample_signals: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dimension": self.dimension,
            "collector_id": self.collector_id,
            "signal_type": self.signal_type,
            "cause": self.cause,
            "signal_count": self.signal_count,
            "artifact_count": self.artifact_count,
            "severity": self.severity.value,
            "total_penalty": self.total_penalty,
            "estimated_collector_gain": self.estimated_collector_gain,
            "estimated_dimension_gain": round(self.estimated_dimension_gain, 2),
            "estimated_score_gain": round(self.estimated_score_gain, 2),
            "recommendation": self.recommendation,
            "ai_impact": self.ai_impact,
            "signal_ids": self.signal_ids,
            "sample_signals": self.sample_signals,
        }


@dataclass
class ScoreExplanation:
    """Derived explanation of why an assessment received its score."""

    score: int
    possible_score: int = 100
    lost_score_points: int = 0
    summary: str = ""
    dominant_dimensions: list[dict[str, Any]] = field(default_factory=list)
    dominant_contributors: list[ScoreContributor] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "possible_score": self.possible_score,
            "lost_score_points": self.lost_score_points,
            "summary": self.summary,
            "dominant_dimensions": self.dominant_dimensions,
            "dominant_contributors": [c.to_dict() for c in self.dominant_contributors],
        }


# ---------------------------------------------------------------------------
# Assessment model
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeAssessment:
    """Interpretation of signals into dimensions and scores.

    An assessment is the interpreted view of collected signals, with
    severity, scores, and recommendations applied via InterpretationPolicy.
    """

    assessment_id: str
    score: int
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    signals: list[KnowledgeSignal] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def explain_score(self, max_contributors: int = 5) -> ScoreExplanation:
        """Explain the assessment score using existing dimensions and signals.

        The calculation estimates how much score could recover if a contributor's
        signals were fixed. It does not re-score policy; it only removes the
        penalties already present on signals.
        """
        contributors = self._score_contributors(max_contributors=max_contributors)
        dominant_dimensions = self._dominant_dimensions()

        if not self.signals:
            summary = "No signals contributed to score degradation."
        elif contributors:
            top = contributors[0]
            summary = (
                f"{top.cause} is the largest contributor: {top.signal_count} "
                f"signal(s) across {top.artifact_count} artifact(s), worth up to "
                f"{top.estimated_score_gain:.1f} overall point(s)."
            )
        else:
            summary = "Signals exist, but no score contributor could be estimated."

        return ScoreExplanation(
            score=self.score,
            lost_score_points=max(0, 100 - self.score),
            summary=summary,
            dominant_dimensions=dominant_dimensions,
            dominant_contributors=contributors,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "score": self.score,
            "dimensions": {
                name: {
                    "name": d.name,
                    "score": d.score,
                    "collector_ids": d.collector_ids,
                    "signals_count": d.signals_count,
                }
                for name, d in self.dimensions.items()
            },
            "score_explanation": self.explain_score().to_dict(),
            "signals": [s.to_dict() for s in self.signals],
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def _dominant_dimensions(self) -> list[dict[str, Any]]:
        weights = self._effective_weights()
        total_weight = sum(weights.get(name, 0) for name in self.dimensions)
        if total_weight <= 0:
            total_weight = len(self.dimensions) or 1
            weights = {name: 1 for name in self.dimensions}

        dimensions: list[dict[str, Any]] = []
        for name, dim in self.dimensions.items():
            normalized_weight = weights.get(name, 0) / total_weight
            lost_points = (100 - dim.score) * normalized_weight
            dimensions.append({
                "dimension": name,
                "score": dim.score,
                "signals_count": dim.signals_count,
                "weight": round(normalized_weight, 3),
                "estimated_lost_score_points": round(lost_points, 2),
            })
        return sorted(
            dimensions,
            key=lambda d: (d["estimated_lost_score_points"], d["signals_count"]),
            reverse=True,
        )

    def _score_contributors(self, max_contributors: int = 5) -> list[ScoreContributor]:
        collector_to_dimension = self._collector_to_dimension()
        dimension_collector_counts = {
            name: max(1, len(dim.collector_ids)) for name, dim in self.dimensions.items()
        }
        weights = self._effective_weights()
        total_weight = sum(weights.get(name, 0) for name in self.dimensions)
        if total_weight <= 0:
            total_weight = len(self.dimensions) or 1
            weights = {name: 1 for name in self.dimensions}

        signals_by_collector: dict[str, list[KnowledgeSignal]] = {}
        for signal in self.signals:
            signals_by_collector.setdefault(signal.collector_id, []).append(signal)

        collector_total_penalties = {
            cid: sum(self._signal_penalty(s) for s in sigs)
            for cid, sigs in signals_by_collector.items()
        }

        groups: dict[str, list[KnowledgeSignal]] = {}
        for signal in self.signals:
            dimension = collector_to_dimension.get(signal.collector_id, "unknown")
            signal_type = self._effective_signal_type(signal)
            cause_key, _ = self._signal_cause(signal, signal_type)
            key = f"{dimension}:{signal.collector_id}:{signal_type}:{cause_key}"
            groups.setdefault(key, []).append(signal)

        contributors: list[ScoreContributor] = []
        for key, group_signals in groups.items():
            first = group_signals[0]
            dimension = collector_to_dimension.get(first.collector_id, "unknown")
            signal_type = self._effective_signal_type(first)
            _, cause = self._signal_cause(first, signal_type)

            group_penalty = sum(self._signal_penalty(s) for s in group_signals)
            collector_total_penalty = collector_total_penalties.get(first.collector_id, 0)
            current_collector_score = max(0, 100 - collector_total_penalty)
            fixed_collector_score = max(0, 100 - max(0, collector_total_penalty - group_penalty))
            estimated_collector_gain = fixed_collector_score - current_collector_score

            collector_count = dimension_collector_counts.get(dimension, 1)
            dimension_gain = estimated_collector_gain / collector_count
            normalized_weight = weights.get(dimension, 0) / total_weight
            score_gain = dimension_gain * normalized_weight

            artifacts = sorted({s.artifact_uri for s in group_signals})
            severities = [s.severity for s in group_signals]
            highest_severity = max(severities, key=lambda s: _severity_rank(s)) if severities else Severity.INFO

            contributors.append(ScoreContributor(
                key=key,
                dimension=dimension,
                collector_id=first.collector_id,
                signal_type=signal_type,
                cause=cause,
                signal_count=len(group_signals),
                artifact_count=len(artifacts),
                severity=highest_severity,
                total_penalty=group_penalty,
                estimated_collector_gain=estimated_collector_gain,
                estimated_dimension_gain=dimension_gain,
                estimated_score_gain=score_gain,
                recommendation=first.recommendation,
                ai_impact=first.ai_impact,
                signal_ids=[s.signal_id for s in group_signals],
                sample_signals=[
                    {
                        "signal_id": s.signal_id,
                        "artifact_uri": s.artifact_uri,
                        "line": s.line,
                    }
                    for s in group_signals[:5]
                ],
            ))

        return sorted(
            contributors,
            key=lambda c: (
                c.estimated_score_gain,
                c.signal_count,
                _severity_rank(c.severity),
            ),
            reverse=True,
        )[:max_contributors]

    def _effective_weights(self) -> dict[str, float]:
        raw = self.metadata.get("weights", {})
        if not isinstance(raw, dict):
            return {}
        weights: dict[str, float] = {}
        for key, value in raw.items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return weights

    def _collector_to_dimension(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for dim_name, dim in self.dimensions.items():
            for collector_id in dim.collector_ids:
                mapping[collector_id] = dim_name
        return mapping

    def _effective_signal_type(self, signal: KnowledgeSignal) -> str:
        if signal.signal_type:
            return signal.signal_type
        if isinstance(signal.evidence, dict):
            if signal.evidence.get("issue_type"):
                return str(signal.evidence["issue_type"])
            if signal.evidence.get("orphan"):
                return "orphan"
            if signal.evidence.get("broken_links"):
                return "broken_link"
            if signal.evidence.get("clusters"):
                return "mixed_topics"
            if signal.evidence.get("dangling_reference_count") or signal.evidence.get("issues"):
                return "dangling_reference"
        return "unknown"

    def _signal_cause(self, signal: KnowledgeSignal, signal_type: str) -> tuple[str, str]:
        evidence = signal.evidence if isinstance(signal.evidence, dict) else {}
        if signal.collector_id == "heading_quality":
            heading = str(evidence.get("heading", "")).strip()
            if heading:
                return self._safe_key(heading), f"{signal_type.replace('_', ' ')} heading '{heading}'"
        if signal.collector_id == "context_independence":
            refs: list[str] = []
            for issue in evidence.get("issues", []):
                if isinstance(issue, dict):
                    refs.extend(str(ref) for ref in issue.get("references", []) if ref)
            if refs:
                ref = Counter(refs).most_common(1)[0][0]
                return self._safe_key(ref), f"dangling reference '{ref}'"
        if signal.collector_id == "link_integrity":
            if signal_type == "broken_link":
                targets = [
                    str(link.get("target"))
                    for link in evidence.get("broken_links", [])
                    if isinstance(link, dict) and link.get("target")
                ]
                if targets:
                    target = Counter(targets).most_common(1)[0][0]
                    return self._safe_key(target), f"broken link target '{target}'"
            if signal_type == "orphan":
                return "orphan", "orphan artifacts"
        if signal.collector_id == "topic_purity":
            return "mixed_topics", "mixed-topic artifacts"
        return signal_type or "unknown", (signal_type or signal.collector_id).replace("_", " ")

    def _signal_penalty(self, signal: KnowledgeSignal) -> int:
        return max(0, 100 - signal.score)

    def _safe_key(self, value: str) -> str:
        return value.lower().strip()[:120]


# ---------------------------------------------------------------------------
# Diff and trend models
# ---------------------------------------------------------------------------

@dataclass
class AssessmentDiff:
    """Diff between two assessments."""

    prev_assessment_id: str = ""
    curr_assessment_id: str = ""
    prev_score: int = 0
    curr_score: int = 0
    score_delta: int = 0
    new_signals: list[KnowledgeSignal] = field(default_factory=list)
    resolved_signals: list[KnowledgeSignal] = field(default_factory=list)
    persistent_signals: list[KnowledgeSignal] = field(default_factory=list)
    recurring_signals: list[KnowledgeSignal] = field(default_factory=list)
    severity_changes: list[dict[str, Any]] = field(default_factory=list)
    dimension_deltas: dict[str, int] = field(default_factory=dict)
    new_high_count: int = 0
    increased_contradictions: int = 0
    increased_topic_entropy: int = 0
    new_orphan_artifacts: int = 0
    new_duplicate_clusters: int = 0
    score_change_explanation: dict[str, Any] = field(default_factory=dict)
    contributor_changes: list[dict[str, Any]] = field(default_factory=list)
    recommendation: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prev_assessment_id": self.prev_assessment_id,
            "curr_assessment_id": self.curr_assessment_id,
            "prev_score": self.prev_score,
            "curr_score": self.curr_score,
            "score_delta": self.score_delta,
            "new_signals": [s.to_dict() for s in self.new_signals],
            "resolved_signals": [s.to_dict() for s in self.resolved_signals],
            "persistent_signals": [s.to_dict() for s in self.persistent_signals],
            "recurring_signals": [s.to_dict() for s in self.recurring_signals],
            "severity_changes": self.severity_changes,
            "dimension_deltas": self.dimension_deltas,
            "new_high_count": self.new_high_count,
            "increased_contradictions": self.increased_contradictions,
            "increased_topic_entropy": self.increased_topic_entropy,
            "new_orphan_artifacts": self.new_orphan_artifacts,
            "new_duplicate_clusters": self.new_duplicate_clusters,
            "score_change_explanation": self.score_change_explanation,
            "contributor_changes": self.contributor_changes,
            "recommendation": self.recommendation,
            "explanation": self.explanation,
        }


@dataclass
class EvolutionView:
    """Health evolution across multiple assessments."""

    assessments: list[dict[str, Any]] = field(default_factory=list)
    score_trend: list[dict[str, Any]] = field(default_factory=list)
    signal_trend: list[dict[str, Any]] = field(default_factory=list)
    dimension_trends: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    trajectory: str = ""  # "improving", "worsening", "stable"
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessments": self.assessments,
            "score_trend": self.score_trend,
            "signal_trend": self.signal_trend,
            "dimension_trends": self.dimension_trends,
            "trajectory": self.trajectory,
            "summary": self.summary,
        }
