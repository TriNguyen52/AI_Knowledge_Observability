"""Quantitative salience and eligibility models for knowledge improvement.

All functions are deterministic and require NO LLM calls — they use
pure mathematical computation based on signal properties, assessment
scores, and historical outcomes.

Three core components:

1. **Problem Salience** — Ranks KnowledgeProblems by importance using a
   3-subscore formula:
     salience = (0.40 * encoding + 0.25 * outcome + 0.35 * retrieval) * size_penalty

   - encoding: signal severity distribution (how bad the signals are)
   - outcome: historical success rate for this problem type (learned from history)
   - retrieval: dimension impact (how much room to improve)
   - size_penalty: problems affecting many artifacts are harder to fix

2. **Signal-Delta Eligibility** — Deterministic gate that checks whether
   new signals have appeared since the last analysis. Prevents wasted
   LLM calls on unchanged signals.

3. **Heuristic Problem Discovery** — Deterministic signal clustering into
   KnowledgeProblem candidates using rule-based patterns. Provides a
   baseline analysis that works without any LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_ready.improvement.models import KnowledgeProblem, RootCauseHypothesis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity weights (deterministic, no LLM)
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 1.0,
    "high": 0.7,
    "medium": 0.4,
    "low": 0.1,
}

# Salience sub-score weights
W_ENCODING = 0.40
W_OUTCOME = 0.25
W_RETRIEVAL = 0.35

# Default salience threshold — problems below this are skipped
DEFAULT_SALIENCE_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# 1. Problem Salience
# ---------------------------------------------------------------------------

@dataclass
class ProblemSalience:
    """Salience decomposition for a single KnowledgeProblem.

    Every field is deterministic — no LLM involvement. The decomposition
    allows explainability: "this problem was prioritized because
    encoding=0.9, outcome=0.78, retrieval=0.85".
    """

    problem_id: str
    encoding: float       # signal severity distribution [0, 1]
    outcome: float        # historical success rate for this problem type [0, 1]
    retrieval: float      # dimension impact [0, 1]
    size_penalty: float   # 1 / max(artifact_count, 1) [0, 1]
    total: float          # weighted sum * size_penalty [0, 1]
    explanation: str      # human-readable decomposition

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "encoding": round(self.encoding, 4),
            "outcome": round(self.outcome, 4),
            "retrieval": round(self.retrieval, 4),
            "size_penalty": round(self.size_penalty, 4),
            "total": round(self.total, 4),
            "explanation": self.explanation,
        }


def compute_problem_salience(
    problem: KnowledgeProblem,
    assessment: Any,
    history_store: Any = None,
) -> ProblemSalience:
    """Compute a unified salience score for a KnowledgeProblem.

    Pure deterministic computation — no LLM calls.

    Args:
        problem: The KnowledgeProblem to score.
        assessment: The KnowledgeAssessment containing signals and dimensions.
        history_store: Optional ImprovementHistoryStore for outcome sub-score.

    Returns:
        ProblemSalience with full decomposition.
    """
    # --- Encoding sub-score: signal severity distribution ---
    # Higher severity signals -> higher encoding score.
    signal_severities: list[float] = []
    if assessment:
        for signal in assessment.signals:
            if signal.signal_id in problem.signal_ids:
                weight = SEVERITY_WEIGHTS.get(signal.severity.value, 0.1)
                signal_severities.append(weight)
    if signal_severities:
        encoding = sum(signal_severities) / len(signal_severities)
    else:
        encoding = 0.0

    # --- Outcome sub-score: historical success rate for this problem type ---
    # If we have no history, default to neutral 0.5.
    outcome = 0.5  # prior probability of success
    if history_store:
        try:
            metrics = history_store.get_remediation_metrics()
            strategy_success = metrics.get("strategy_success_rate", {})
            problem_category = problem.category
            for strategy_name, stats in strategy_success.items():
                if stats.get("total", 0) > 0:
                    rate = stats.get("success_rate", 0.5)
                    outcome = max(outcome, rate)
                    break
        except Exception:
            pass  # Keep default 0.5 if history unavailable

    # --- Retrieval sub-score: dimension impact ---
    # Lower dimension scores -> higher retrieval salience (more room to improve).
    retrieval = 0.5  # neutral default
    if assessment and hasattr(assessment, "dimensions"):
        affected_dims: set[str] = set()
        if assessment:
            for signal in assessment.signals:
                if signal.signal_id in problem.signal_ids:
                    affected_dims.add(signal.collector_id)
        if affected_dims:
            dim_scores: list[float] = []
            for dim_name in affected_dims:
                if dim_name in assessment.dimensions:
                    score = assessment.dimensions[dim_name].score
                    dim_scores.append(1.0 - (score / 100.0))
                else:
                    dim_scores.append(0.5)
            retrieval = sum(dim_scores) / len(dim_scores)

    # --- Size penalty: problems affecting many artifacts are harder to fix ---
    artifact_count = len(problem.artifact_uris) if problem.artifact_uris else 1
    size_penalty = 1.0 / max(artifact_count, 1)

    # --- Total: weighted sum * size_penalty ---
    total = (W_ENCODING * encoding + W_OUTCOME * outcome + W_RETRIEVAL * retrieval) * size_penalty

    explanation = (
        f"Prioritized because: encoding={encoding:.2f} "
        f"(severity of {len(signal_severities)} signal(s)), "
        f"outcome={outcome:.2f} (historical success rate for '{problem.category}'), "
        f"retrieval={retrieval:.2f} (dimension impact), "
        f"size_penalty={size_penalty:.2f} ({artifact_count} artifact(s)). "
        f"Total salience={total:.4f}."
    )

    return ProblemSalience(
        problem_id=problem.problem_id,
        encoding=encoding,
        outcome=outcome,
        retrieval=retrieval,
        size_penalty=size_penalty,
        total=total,
        explanation=explanation,
    )


def rank_problems_by_salience(
    problems: list[KnowledgeProblem],
    assessment: Any,
    history_store: Any = None,
    threshold: float = DEFAULT_SALIENCE_THRESHOLD,
) -> tuple[list[KnowledgeProblem], list[ProblemSalience], list[KnowledgeProblem]]:
    """Rank KnowledgeProblems by salience and filter below threshold.

    Pure deterministic computation — no LLM calls.

    Args:
        problems: List of KnowledgeProblems to rank.
        assessment: The KnowledgeAssessment containing signals and dimensions.
        history_store: Optional ImprovementHistoryStore for outcome sub-score.
        threshold: Minimum salience to include (default 0.15).

    Returns:
        Tuple of (ranked_problems, salience_scores, skipped_problems).
        ranked_problems: sorted by salience descending, above threshold.
        salience_scores: corresponding ProblemSalience for each ranked problem.
        skipped_problems: problems below threshold (for skip event recording).
    """
    scored: list[tuple[KnowledgeProblem, ProblemSalience]] = []
    for problem in problems:
        salience = compute_problem_salience(problem, assessment, history_store)
        scored.append((problem, salience))

    # Sort by total salience descending
    scored.sort(key=lambda x: x[1].total, reverse=True)

    ranked = []
    saliences = []
    skipped = []
    for problem, salience in scored:
        if salience.total >= threshold:
            ranked.append(problem)
            saliences.append(salience)
        else:
            skipped.append(problem)

    return ranked, saliences, skipped


# ---------------------------------------------------------------------------
# 2. Signal-Delta Eligibility
# ---------------------------------------------------------------------------

@dataclass
class EligibilityResult:
    """Result of a signal-delta eligibility check.

    Deterministic — no LLM. Only process when there are new signals
    since the last analysis to avoid wasting tokens on unchanged state.
    """

    eligible: bool
    new_signal_ids: list[str] = field(default_factory=list)
    unchanged_signal_ids: list[str] = field(default_factory=list)
    removed_signal_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "new_signal_ids": self.new_signal_ids,
            "unchanged_signal_ids": self.unchanged_signal_ids,
            "removed_signal_ids": self.removed_signal_ids,
            "reason": self.reason,
        }


def check_signal_delta(
    current_signal_ids: list[str],
    last_analysis_signal_ids: list[str] | None,
) -> EligibilityResult:
    """Check whether new signals have appeared since the last analysis.

    Deterministic set comparison — no LLM. An assessment is only
    eligible for analysis if there are new signals since the last
    analysis run.

    Args:
        current_signal_ids: Signal IDs from the current assessment.
        last_analysis_signal_ids: Signal IDs from the last analysis,
            or None if this is the first analysis.

    Returns:
        EligibilityResult with eligibility flag and signal deltas.
    """
    current_set = set(current_signal_ids)

    if last_analysis_signal_ids is None:
        # First analysis — always eligible
        return EligibilityResult(
            eligible=True,
            new_signal_ids=list(current_set),
            unchanged_signal_ids=[],
            removed_signal_ids=[],
            reason="First analysis — all signals are new.",
        )

    last_set = set(last_analysis_signal_ids)
    new_ids = current_set - last_set
    unchanged_ids = current_set & last_set
    removed_ids = last_set - current_set

    if new_ids:
        return EligibilityResult(
            eligible=True,
            new_signal_ids=list(new_ids),
            unchanged_signal_ids=list(unchanged_ids),
            removed_signal_ids=list(removed_ids),
            reason=f"{len(new_ids)} new signal(s) since last analysis.",
        )
    else:
        return EligibilityResult(
            eligible=False,
            new_signal_ids=[],
            unchanged_signal_ids=list(unchanged_ids),
            removed_signal_ids=list(removed_ids),
            reason=(
                f"No new signals since last analysis "
                f"({len(unchanged_ids)} unchanged, {len(removed_ids)} removed). "
                f"Skipping LLM analysis to save tokens."
            ),
        )


# ---------------------------------------------------------------------------
# 3. Heuristic Problem Discovery (no LLM)
# ---------------------------------------------------------------------------

def discover_problems_heuristic(
    signal_ids: list[str],
    assessment: Any,
) -> tuple[list[KnowledgeProblem], list[RootCauseHypothesis]]:
    """Discover KnowledgeProblems from signals using deterministic rules.

    No LLM needed — uses rule-based clustering:

    1. Same (collector, type, artifact) -> single problem (signal cluster)
    2. Same signal_type across multiple artifacts -> systemic problem
    3. Severity escalation (CRITICAL/HIGH signals) -> urgent problem

    This provides a baseline analysis that works without any LLM.
    When an LLM is available, it can enhance these heuristic problems
    with richer root cause hypotheses.

    Args:
        signal_ids: Signal IDs to analyze.
        assessment: The KnowledgeAssessment containing signal details.

    Returns:
        Tuple of (knowledge_problems, hypotheses).
    """
    if not assessment:
        return [], []

    # Collect signals of interest
    target_signals = [s for s in assessment.signals if s.signal_id in set(signal_ids)]
    if not target_signals:
        return [], []

    problems: list[KnowledgeProblem] = []
    hypotheses: list[RootCauseHypothesis] = []

    # --- Rule 1: Cluster by (collector, type, artifact) ---
    clusters: dict[str, list[Any]] = {}
    for signal in target_signals:
        key = f"{signal.collector_id}|{signal.signal_type}|{signal.artifact_uri}"
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(signal)

    for idx, (key, cluster_signals) in enumerate(clusters.items()):
        parts = key.split("|")
        collector, signal_type, artifact_uri = parts[0], parts[1], parts[2]

        # Determine the dominant severity
        severities = [s.severity.value for s in cluster_signals]
        max_severity = max(severities, key=lambda s: SEVERITY_WEIGHTS.get(s, 0.1))

        hypothesis = RootCauseHypothesis(
            hypothesis=(
                f"{collector} reports {signal_type} on {artifact_uri} "
                f"({len(cluster_signals)} signal(s), max severity={max_severity}). "
                f"Likely cause: {signal_type} issue in the knowledge artifact."
            ),
            evidence=[
                f"Signal {s.signal_id}: {s.recommendation}"
                for s in cluster_signals[:3]
            ],
            confidence=min(0.3 + 0.1 * len(cluster_signals), 0.8),
            affected_artifact_uris=[artifact_uri],
            category=signal_type,
        )
        hypotheses.append(hypothesis)

        problem = KnowledgeProblem(
            problem_id=f"kp_heuristic_{idx}",
            category=signal_type,
            description=(
                f"{len(cluster_signals)} {signal_type} signal(s) from {collector} "
                f"on {artifact_uri} (max severity: {max_severity})"
            ),
            root_cause_idx=idx,
            signal_ids=[s.signal_id for s in cluster_signals],
            artifact_uris=[artifact_uri],
            evidence_summary=f"Clustered {len(cluster_signals)} signals by ({collector}, {signal_type}, {artifact_uri})",
        )
        problems.append(problem)

    # --- Rule 2: Systemic problems (same type across multiple artifacts) ---
    type_to_artifacts: dict[str, set[str]] = {}
    type_to_signals: dict[str, list[str]] = {}
    for signal in target_signals:
        if signal.signal_type not in type_to_artifacts:
            type_to_artifacts[signal.signal_type] = set()
            type_to_signals[signal.signal_type] = []
        type_to_artifacts[signal.signal_type].add(signal.artifact_uri)
        type_to_signals[signal.signal_type].append(signal.signal_id)

    for signal_type, artifacts in type_to_artifacts.items():
        if len(artifacts) >= 2:
            idx = len(hypotheses)
            hypothesis = RootCauseHypothesis(
                hypothesis=(
                    f"Systemic issue: {signal_type} appears across {len(artifacts)} "
                    f"artifacts. This suggests a process or pipeline-level problem "
                    f"rather than an individual artifact issue."
                ),
                evidence=[
                    f"Affected artifacts: {', '.join(list(artifacts)[:5])}",
                    f"Total signals: {len(type_to_signals[signal_type])}",
                ],
                confidence=0.6,
                affected_artifact_uris=list(artifacts),
                category=f"systemic_{signal_type}",
            )
            hypotheses.append(hypothesis)

            problem = KnowledgeProblem(
                problem_id=f"kp_systemic_{signal_type}",
                category=f"systemic_{signal_type}",
                description=(
                    f"Systemic {signal_type} issue across {len(artifacts)} artifacts "
                    f"({len(type_to_signals[signal_type])} total signals)"
                ),
                root_cause_idx=idx,
                signal_ids=type_to_signals[signal_type],
                artifact_uris=list(artifacts),
                evidence_summary=f"Same signal type across {len(artifacts)} artifacts indicates a systemic issue.",
            )
            problems.append(problem)

    # --- Rule 3: Urgent problems (CRITICAL/HIGH severity) ---
    urgent_signals = [
        s for s in target_signals
        if SEVERITY_WEIGHTS.get(s.severity.value, 0.1) >= 0.7
    ]
    if urgent_signals and len(urgent_signals) >= 2:
        idx = len(hypotheses)
        hypothesis = RootCauseHypothesis(
            hypothesis=(
                f"Urgent: {len(urgent_signals)} high-severity signal(s) detected. "
                f"These require immediate attention as they significantly impact "
                f"AI readiness."
            ),
            evidence=[
                f"Signal {s.signal_id}: severity={s.severity.value}, {s.recommendation}"
                for s in urgent_signals[:5]
            ],
            confidence=0.7,
            affected_artifact_uris=list({s.artifact_uri for s in urgent_signals}),
            category="urgent_high_severity",
        )
        hypotheses.append(hypothesis)

        problem = KnowledgeProblem(
            problem_id="kp_urgent_high_severity",
            category="urgent_high_severity",
            description=(
                f"{len(urgent_signals)} high-severity signal(s) requiring immediate attention"
            ),
            root_cause_idx=idx,
            signal_ids=[s.signal_id for s in urgent_signals],
            artifact_uris=list({s.artifact_uri for s in urgent_signals}),
            evidence_summary=f"{len(urgent_signals)} signals at HIGH/CRITICAL severity.",
        )
        problems.append(problem)

    return problems, hypotheses


# ---------------------------------------------------------------------------
# 4. Skip Event (for visible no-ops)
# ---------------------------------------------------------------------------

@dataclass
class SkipEvent:
    """Records when the system chose not to act, with reason.

    Transparency mechanism — the system records its decisions NOT to
    act so users can see what was considered and rejected.
    """

    lane: str           # which process was skipped
    reason: str         # why (low_salience, no_new_signals, gated, etc.)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "context": self.context,
        }
