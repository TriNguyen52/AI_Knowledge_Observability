"""Burr application graph for the knowledge improvement workflow.

Defines the state machine: actions, transitions, conditions, and
the ApplicationBuilder configuration with tracking, persistence,
and optional OpenTelemetry hooks.

Workflow graph:

  analyze_issue ──→ generate_proposal ──→ review_approval
       │                   │                    │
       ↓ fail              ↓ fail              ↓ rejected
   failed_analysis    failed_analysis       rejected (terminal)

  review_approval ──→ execute_change ──→ verify_improvement
                          │                    │
                          ↓ fail              ↓ fail
                     failed_execution    failed_verification
                                               │
                                          handle_failure
"""

from __future__ import annotations

import logging
from typing import Any

from burr.core import ApplicationBuilder, State, expr, when
from burr.core.graph import GraphBuilder

from ai_ready.improvement.actions import (
    analyze_issue,
    generate_proposal,
    review_approval,
    execute_change,
    verify_improvement,
    handle_failure,
)
from ai_ready.improvement.models import RemediationStatus

logger = logging.getLogger(__name__)


def build_improvement_graph() -> Any:
    """Build the Burr graph for the improvement workflow.

    Returns a Graph object that can be used with ApplicationBuilder.
    """
    graph = (
        GraphBuilder()
        .with_actions(
            analyze_issue,
            generate_proposal,
            review_approval,
            execute_change,
            verify_improvement,
            handle_failure,
        )
        .with_transitions(
            # analyze_issue → generate_proposal (success) or failed_analysis (failure)
            ("analyze_issue", "generate_proposal",
             when(current_stage=RemediationStatus.ROOT_CAUSE_FOUND.value)),
            ("analyze_issue", "handle_failure",
             when(current_stage=RemediationStatus.FAILED_ANALYSIS.value)),

            # generate_proposal → review_approval (success) or handle_failure (failure)
            ("generate_proposal", "review_approval",
             when(current_stage=RemediationStatus.PROPOSAL_GENERATED.value)),
            ("generate_proposal", "handle_failure",
             when(current_stage=RemediationStatus.FAILED_ANALYSIS.value)),

            # review_approval → execute_change (approved) or terminal (rejected)
            ("review_approval", "execute_change",
             when(current_stage=RemediationStatus.WAITING_FOR_APPROVAL.value,
                  approval_status="approved")),
            ("review_approval", "handle_failure",
             when(current_stage=RemediationStatus.WAITING_FOR_APPROVAL.value,
                  approval_status="rejected")),

            # execute_change → verify_improvement (success) or handle_failure (failure)
            ("execute_change", "verify_improvement",
             when(current_stage=RemediationStatus.EXECUTING.value)),
            ("execute_change", "handle_failure",
             when(current_stage=RemediationStatus.FAILED_EXECUTION.value)),

            # verify_improvement → handle_failure (failure)
            # When completed, no transition matches → application stops (terminal)
            ("verify_improvement", "handle_failure",
             when(current_stage=RemediationStatus.FAILED_VERIFICATION.value)),

            # handle_failure is terminal — no outgoing transitions.
            # The application stops when no transition condition matches.
        )
        .build()
    )
    return graph


def create_improvement_app(
    initial_state_dict: dict[str, Any],
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
    app_id: str | None = None,
    partition_key: str | None = None,
) -> Any:
    """Create a Burr Application for the improvement workflow.

    Args:
        initial_state_dict: Initial state from improvement.models.initial_state().
        llm_gateway: LLMGateway instance for LLM calls.
        assessment_store: AI-Ready AssessmentStore for loading assessments.
        assessment_pipeline: AI-Ready AssessmentPipeline for verification.
        history_store: ImprovementHistoryStore for institutional memory.
        executor: KnowledgeExecutor for applying modifications.
        artifacts: List of KnowledgeArtifact objects (for verification re-assessment).
        relationships: List of Relationship objects (for verification).
        source: Source path string.
        git_commit: Git commit hash.
        enable_tracking: Enable Burr local tracking (writes to ~/.burr).
        enable_otel: Enable OpenTelemetry bridge.
        project_name: Burr tracking project name.
        app_id: Optional application ID (auto-generated if None).
        partition_key: Optional partition key.

    Returns:
        A Burr Application ready to run.
    """
    graph = build_improvement_graph()

    builder = (
        ApplicationBuilder()
        .with_graph(graph)
        .with_state(**initial_state_dict)
        .with_entrypoint("analyze_issue")
        .with_identifiers(
            app_id=app_id,
            partition_key=partition_key,
        )
    )

    # Enable tracking
    if enable_tracking:
        try:
            from burr.tracking import LocalTrackingClient
            tracker = LocalTrackingClient(project=project_name)
            builder = builder.with_tracker(tracker)
        except ImportError:
            logger.warning("Burr tracking not available — install apache-burr[start]")

    # Enable OpenTelemetry
    if enable_otel:
        try:
            from burr.integrations.opentelemetry import OpenTelemetryBridge
            builder = builder.with_hooks(OpenTelemetryBridge())
        except ImportError:
            logger.warning("Burr OpenTelemetry integration not available")

    app = builder.build()

    # Register dependencies in Burr's _dependency_factory so they are
    # automatically injected into any action that declares them as
    # optional parameters.  Burr's @action decorator treats **kwargs as
    # a required input, so we removed **kwargs from all action signatures
    # and instead rely on this factory mechanism which works across ALL
    # iterations (unlike iterate(inputs=...) which only applies to the
    # first step).
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


def _register_dependencies(app: Any, deps: dict[str, Any]) -> None:
    """Register dependencies in Burr's _dependency_factory.

    Burr's _process_inputs checks this factory for any missing optional
    inputs that an action declares.  Factory values must be callables
    taking (action, sequence_id) and returning the dependency value.
    """
    for key, value in deps.items():
        if value is not None:
            app._dependency_factory[key] = lambda action, seq, v=value: v


def run_until_approval(app: Any) -> dict[str, Any]:
    """Run the improvement workflow until it reaches the approval checkpoint.

    The workflow halts after review_approval. The caller then needs
    to set approval_status in the state and resume.

    Returns:
        The state dict at the halt point.
    """
    # Run until we hit WAITING_FOR_APPROVAL or a terminal state
    halt_stages = {
        RemediationStatus.WAITING_FOR_APPROVAL.value,
        RemediationStatus.FAILED_ANALYSIS.value,
        RemediationStatus.REJECTED.value,
    }

    # halt_after covers both the approval checkpoint and the terminal
    # failure path (handle_failure), so Burr doesn't warn about missing
    # halt conditions or undefined terminal behaviour.
    for action, result, state in app.iterate(
        halt_after=["review_approval", "handle_failure"]
    ):
        current_stage = state.get("current_stage", "")
        if current_stage in halt_stages:
            break

    return dict(app.state)

def approve_and_run(app: Any, approved: bool, reason: str = "") -> dict[str, Any]:
    """Set approval status and run the rest of the workflow.

    Args:
        app: The halted Burr application.
        approved: Whether the proposal was approved.
        reason: Optional reason for the decision.

    Returns:
        Final state dict.
    """
    if approved:
        app.update_state(app.state.update(approval_status="approved", approval_reason=reason))
    else:
        app.update_state(app.state.update(approval_status="rejected", approval_reason=reason))

    # Run to completion.  halt_after covers every terminal path:
    #   - verify_improvement (success → COMPLETED, no outgoing transition)
    #   - handle_failure (all failure branches converge here, terminal)
    # This prevents Burr's "No halt termination specified" and
    # "trying to return without having computed a single action" warnings.
    for action, result, state in app.iterate(
        halt_after=["verify_improvement", "handle_failure"]
    ):
        pass

    return dict(app.state)

def run_full_workflow(app: Any, auto_approve: bool = False) -> dict[str, Any]:
    """Run the complete improvement workflow.

    If auto_approve is True, the workflow runs without pausing for
    human approval (useful for testing/dry-run mode).

    Args:
        app: The Burr application.
        auto_approve: Skip the approval checkpoint.

    Returns:
        Final state dict.
    """
    if auto_approve:
        # Patch: set approval_status before the approval action runs
        # by running in two phases
        # Phase 1: run until approval
        state = run_until_approval(app)

        if state.get("current_stage") == RemediationStatus.WAITING_FOR_APPROVAL.value:
            # Phase 2: auto-approve and continue
            state = approve_and_run(app, approved=True, reason="auto-approved (dry-run)")
        return state
    else:
        # Just run until approval — caller must resume
        return run_until_approval(app)
