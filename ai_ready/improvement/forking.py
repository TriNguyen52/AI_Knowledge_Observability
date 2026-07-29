"""State forking for retrying failed improvement workflows.

Uses Burr's fork_from_app_id + fork_from_sequence_id to branch from
a prior point in the workflow — specifically from the proposal
generation step — and try an alternative strategy.

This is the PRIMARY reason for integrating Burr. Without forking,
a failed improvement means re-running the entire scan + diagnosis
pipeline. With forking, we reuse the scan results and root cause
analysis, only generating a new proposal.

Fork flow:
  Attempt 1: analyze → propose (strategy A) → approve → execute → verify (FAIL)
  Attempt 2: fork from "generate_proposal" step → propose (strategy B) → approve → execute → verify (SUCCESS)

The forked application preserves:
  - Original assessment_id and signal_ids
  - Root cause analysis from attempt 1
  - Prior failure summary (so the LLM avoids strategy A)
  - Incremented attempt_number

The forked application gets:
  - New app_id (stored in forked_from_app_id for traceability)
  - Fresh proposals (LLM generates alternatives)
  - Clean execution_history and verification_results
"""

from __future__ import annotations

import logging
from typing import Any

from burr.core import ApplicationBuilder

from ai_ready.improvement.application import build_improvement_graph
from ai_ready.improvement.models import (
    RemediationStatus,
    initial_state,
)

logger = logging.getLogger(__name__)


def fork_from_proposal(
    prior_app_id: str,
    prior_state: dict[str, Any],
    prior_sequence_id: int,
    llm_gateway: Any = None,
    assessment_store: Any = None,
    assessment_pipeline: Any = None,
    history_store: Any = None,
    executor: Any = None,
    artifacts: list[Any] = None,
    relationships: list[Any] = None,
    source: str = "",
    git_commit: str = "",
    enable_tracking: bool = True,
    enable_otel: bool = False,
    project_name: str = "ai_ready_improvement",
    partition_key: str | None = None,
) -> Any:
    """Fork a failed improvement workflow from the proposal generation step.

    Creates a new Burr application that starts at generate_proposal,
    carrying forward the root cause analysis and prior failure info,
    but with a fresh attempt at proposal generation.

    Args:
        prior_app_id: The app_id of the failed workflow.
        prior_state: The final state dict of the failed workflow.
        prior_sequence_id: The sequence_id of the generate_proposal step
                          in the prior workflow (to fork from that point).
        llm_gateway: LLMGateway for LLM calls.
        assessment_store: AI-Ready AssessmentStore.
        assessment_pipeline: AI-Ready AssessmentPipeline for verification.
        history_store: ImprovementHistoryStore.
        executor: KnowledgeExecutor.
        artifacts: KnowledgeArtifact list for verification.
        relationships: Relationship list for verification.
        source: Source path string.
        git_commit: Git commit hash.
        enable_tracking: Enable Burr tracking.
        enable_otel: Enable OpenTelemetry.
        project_name: Burr tracking project name.
        partition_key: Optional partition key.

    Returns:
        A new Burr Application ready to run from generate_proposal.
    """
    # Build the new state from the prior state
    attempt_number = prior_state.get("attempt_number", 1) + 1
    prior_failures = list(prior_state.get("prior_failures", []))

    # Add the current failure to prior_failures if not already there
    # (handle_failure action may have already added it)
    proposals = prior_state.get("proposals", [])
    selected_idx = prior_state.get("selected_proposal_idx", 0)
    verification = prior_state.get("verification_results", {})
    root_causes = prior_state.get("root_cause_analysis", [])

    strategy = (proposals[selected_idx].get("strategy", "unknown")
                if selected_idx < len(proposals) else "unknown")
    issue_type = (root_causes[0].get("category", "unknown")
                  if root_causes else "unknown")

    # Check if this failure was already recorded by handle_failure
    already_recorded = any(
        f.get("attempt_number") == prior_state.get("attempt_number", 1)
        and f.get("strategy") == strategy
        for f in prior_failures
    )
    if not already_recorded:
        prior_failures.append({
            "attempt_number": prior_state.get("attempt_number", 1),
            "strategy": strategy,
            "issue_type": issue_type,
            "result": "failure",
            "failure_reason": verification.get("summary", "unknown"),
            "score_change": verification.get("score_difference", 0),
        })

    # Create new initial state for the forked workflow
    new_state = initial_state(
        remediation_id=f"{prior_state.get('remediation_id', 'unknown')}_fork_{attempt_number}",
        assessment_id=prior_state.get("assessment_id", ""),
        signal_ids=prior_state.get("signal_ids", []),
        affected_artifact_ids=prior_state.get("affected_artifact_ids", []),
        affected_artifact_uris=prior_state.get("affected_artifact_uris", []),
        prior_failures=prior_failures,
    )

    # Carry forward root cause analysis — we don't need to re-analyze
    new_state["root_cause_analysis"] = prior_state.get("root_cause_analysis", [])
    # Focus 3: Carry forward knowledge problems, prior diagnosis, strategy,
    # verification, and execution so the forked retry can reason about why
    # the prior attempt failed and generate a genuinely different strategy
    new_state["knowledge_problems"] = prior_state.get("knowledge_problems", [])
    new_state["prior_diagnosis"] = prior_state.get("prior_diagnosis", {})
    new_state["prior_strategy"] = prior_state.get("prior_strategy", {})
    new_state["prior_verification"] = prior_state.get("prior_verification", {})
    new_state["prior_execution"] = prior_state.get("prior_execution", [])
    # Carry forward decision trace so reasoning accumulates across forks
    new_state["decision_trace"] = prior_state.get("decision_trace", {})
    # Obj 2, 9: Carry forward assessment summary and cumulative tokens
    new_state["assessment_summary"] = prior_state.get("assessment_summary", "")
    new_state["cumulative_tokens"] = prior_state.get("cumulative_tokens", 0)
    new_state["attempt_number"] = attempt_number
    new_state["forked_from_app_id"] = prior_app_id
    new_state["forked_from_sequence_id"] = prior_sequence_id
    new_state["current_stage"] = RemediationStatus.ROOT_CAUSE_FOUND.value

    # Build the graph and application
    graph = build_improvement_graph()

    builder = (
        ApplicationBuilder()
        .with_graph(graph)
        .with_state(**new_state)
        .with_entrypoint("generate_proposal")  # Start from proposal generation
        .with_identifiers(partition_key=partition_key)
    )

    if enable_tracking:
        try:
            from burr.tracking import LocalTrackingClient
            tracker = LocalTrackingClient(project=project_name)
            builder = builder.with_tracker(tracker)
        except ImportError:
            pass

    if enable_otel:
        try:
            from burr.integrations.opentelemetry import OpenTelemetryBridge
            builder = builder.with_hooks(OpenTelemetryBridge())
        except ImportError:
            pass

    # Build the app, then register dependencies (same pattern as create_improvement_app)
    app = builder.build()
    from ai_ready.improvement.application import _register_dependencies
    _register_dependencies(app, {
        "llm_gateway": llm_gateway,
        "assessment_store": assessment_store,
        "assessment_pipeline": assessment_pipeline,
        "history_store": history_store,
        "executor": executor,
        "artifacts": artifacts,
        "relationships": relationships,
        "source": source,
        "git_commit": git_commit,
    })

    return app


def _strategy_similarity(strategy_a: str, strategy_b: str) -> float:
    """Compute a simple similarity score between two strategy names.

    Objective 6: Prevent retrying the same strategy under a different name.
    Uses token overlap (Jaccard similarity) on lowercased words.

    Returns:
        Similarity score from 0.0 (completely different) to 1.0 (identical).
    """
    if not strategy_a or not strategy_b:
        return 0.0
    # Normalize: lowercase, replace underscores and hyphens with spaces
    tokens_a = set(strategy_a.lower().replace("_", " ").replace("-", " ").split())
    tokens_b = set(strategy_b.lower().replace("_", " ").replace("-", " ").split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def validate_fork_strategy(
    new_strategy: str,
    prior_failures: list[dict[str, Any]],
    similarity_threshold: float = 0.7,
) -> tuple[bool, str]:
    """Validate that a forked retry strategy is genuinely different.

    Objective 6: Prevents retrying the same strategy under a different name.
    Compares the new strategy against all prior failed strategies using
    token overlap similarity. If any prior strategy is too similar,
    the fork should be rejected.

    Args:
        new_strategy: The strategy name from the new proposal.
        prior_failures: List of prior failure summaries with 'strategy' keys.
        similarity_threshold: Max allowed similarity (0.0 to 1.0).
            Default 0.7 means strategies sharing 70%+ of words are rejected.

    Returns:
        Tuple of (is_valid, explanation). is_valid is True if the strategy
        is sufficiently different from all prior failures.
    """
    if not prior_failures:
        return True, "No prior failures to compare against."

    for failure in prior_failures:
        prior_strategy = failure.get("strategy", "")
        if not prior_strategy:
            continue
        similarity = _strategy_similarity(new_strategy, prior_strategy)
        if similarity >= similarity_threshold:
            return False, (
                f"Strategy '{new_strategy}' is too similar to prior failed "
                f"strategy '{prior_strategy}' (similarity={similarity:.2f}, "
                f"threshold={similarity_threshold}). Never retry the same "
                f"strategy under a different name."
            )

    return True, f"Strategy '{new_strategy}' is sufficiently different from all prior failures."


def should_fork(state: dict[str, Any], max_attempts: int = 3) -> bool:
    """Determine if a failed workflow should be forked.

    Forking is recommended when:
    - The failure was in execution or verification (not analysis)
    - There are alternative proposals to try
    - We haven't exceeded max_attempts

    Args:
        state: The final state of the failed workflow.
        max_attempts: Maximum number of fork attempts.

    Returns:
        True if forking should be attempted.
    """
    current_stage = state.get("current_stage", "")
    attempt_number = state.get("attempt_number", 1)

    # Don't fork analysis failures — the LLM couldn't even analyze
    if current_stage == RemediationStatus.FAILED_ANALYSIS.value:
        return False

    # Don't fork rejections — user explicitly rejected
    if current_stage == RemediationStatus.REJECTED.value:
        return False

    # Check attempt limit
    if attempt_number >= max_attempts:
        return False

    # Fork for execution and verification failures
    if current_stage in (RemediationStatus.FAILED_EXECUTION.value,
                          RemediationStatus.FAILED_VERIFICATION.value):
        return True

    return False


def get_fork_sequence_id(tracker: Any, app_id: str, action_name: str = "generate_proposal") -> int:
    """Get the sequence_id of a specific action from a prior workflow.

    Uses Burr's tracking client to find the sequence_id of the
    generate_proposal step in the prior workflow.

    Args:
        tracker: Burr LocalTrackingClient instance.
        app_id: The prior application ID.
        action_name: The action to find (default: "generate_proposal").

    Returns:
        The sequence_id of the action, or -1 if not found.
    """
    # This requires accessing Burr's tracking data
    # The exact API depends on the Burr version
    try:
        # Burr stores tracking data in ~/.burr/<project>/
        # The load API can retrieve prior state
        import json
        from pathlib import Path

        # Try to find the tracking data
        tracking_dir = Path.home() / ".burr"
        if tracker and hasattr(tracker, 'project'):
            tracking_dir = tracking_dir / tracker.project

        # Search for the app's state log
        # The exact path structure may vary by Burr version
        for state_file in tracking_dir.rglob("*.json"):
            try:
                data = json.loads(state_file.read_text())
                if isinstance(data, list):
                    for entry in data:
                        if (entry.get("app_id") == app_id and
                            entry.get("action") == action_name):
                            return entry.get("sequence_id", -1)
            except (json.JSONDecodeError, KeyError):
                continue
    except Exception as e:
        logger.warning(f"Could not retrieve fork sequence_id: {e}")

    return -1
