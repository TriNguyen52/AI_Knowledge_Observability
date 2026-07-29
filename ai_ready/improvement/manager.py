"""ImprovementManager — facade tying together the full improvement workflow.

This is the main entry point for AI-Ready to trigger knowledge improvements.
It orchestrates:
  1. Creating a Burr application from an assessment
  2. Running the workflow to the approval checkpoint
  3. Handling approval (human-in-the-loop)
  4. Running execution and verification
  5. Recording outcomes in the history store
  6. Forking failed workflows for retry
  7. Generating regression tests from failures

Usage:
    manager = ImprovementManager(
        llm_gateway=gateway,
        assessment_store=assessment_store,
        assessment_pipeline=pipeline,
        history_store=history_store,
        executor=executor,
    )

    # Start an improvement from an assessment
    app_id = manager.start_improvement(assessment)

    # Wait for approval, then resume
    final_state = manager.approve_and_complete(app_id, approved=True)

    # If it failed, try forking
    if manager.should_retry(app_id):
        forked_app_id = manager.fork_and_retry(app_id)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ai_ready.improvement.models import (
    RemediationStatus,
    RemediationOutcome,
    VerificationResult,
    initial_state,
)
from ai_ready.improvement.application import (
    create_improvement_app,
    run_until_approval,
    approve_and_run,
)
from ai_ready.improvement.forking import fork_from_proposal, should_fork
from ai_ready.improvement.history import ImprovementHistoryStore
from ai_ready.improvement.test_generation import generate_test

logger = logging.getLogger(__name__)


class ImprovementManager:
    """Facade for the knowledge improvement workflow.

    Ties together Burr application lifecycle, LLM gateway, AI-Ready
    stores, executor, history, forking, and test generation.

    The manager is the ONLY component that AI-Ready's pipeline calls
    directly. Everything else is internal to the improvement package.
    """

    def __init__(
        self,
        llm_gateway: Any = None,
        assessment_store: Any = None,
        assessment_pipeline: Any = None,
        history_store: ImprovementHistoryStore | None = None,
        executor: Any = None,
        artifacts: list[Any] = None,
        relationships: list[Any] = None,
        source: str = "",
        git_commit: str = "",
        history_db_path: str | None = None,
        test_output_dir: str = "tests/improvement",
        enable_tracking: bool = True,
        enable_otel: bool = False,
        max_fork_attempts: int = 3,
    ) -> None:
        self.llm_gateway = llm_gateway
        self.assessment_store = assessment_store
        self.assessment_pipeline = assessment_pipeline
        self.executor = executor
        self.artifacts = artifacts
        self.relationships = relationships
        self.source = source
        self.git_commit = git_commit
        self.test_output_dir = test_output_dir
        self.enable_tracking = enable_tracking
        self.enable_otel = enable_otel
        self.max_fork_attempts = max_fork_attempts

        # Initialize history store if not provided
        if history_store:
            self.history_store = history_store
        elif history_db_path:
            self.history_store = ImprovementHistoryStore(history_db_path)
        else:
            self.history_store = None

        # Track active applications
        self._apps: dict[str, Any] = {}

    def start_improvement(
        self,
        assessment: Any,
        signal_ids: list[str] | None = None,
        artifact_uris: list[str] | None = None,
    ) -> str:
        """Start an improvement workflow from an assessment.

        Args:
            assessment: The KnowledgeAssessment that triggered this improvement.
            signal_ids: Optional subset of signal IDs to address (default: all).
            artifact_uris: Optional subset of artifact URIs (default: from signals).

        Returns:
            The app_id of the created Burr application.
        """
        remediation_id = str(uuid.uuid4())
        assessment_id = assessment.assessment_id

        # Determine which signals and artifacts to address
        if signal_ids is None:
            signal_ids = [s.signal_id for s in assessment.signals]

        if artifact_uris is None:
            artifact_uris = list({s.artifact_uri for s in assessment.signals
                                  if s.signal_id in signal_ids})

        artifact_ids = list({s.artifact_id for s in assessment.signals
                             if s.signal_id in signal_ids})

        # Create initial state
        state_dict = initial_state(
            remediation_id=remediation_id,
            assessment_id=assessment_id,
            signal_ids=signal_ids,
            affected_artifact_ids=artifact_ids,
            affected_artifact_uris=artifact_uris,
        )

        # Create the Burr application
        app = create_improvement_app(
            initial_state_dict=state_dict,
            llm_gateway=self.llm_gateway,
            assessment_store=self.assessment_store,
            assessment_pipeline=self.assessment_pipeline,
            history_store=self.history_store,
            executor=self.executor,
            artifacts=self.artifacts,
            relationships=self.relationships,
            source=self.source,
            git_commit=self.git_commit,
            enable_tracking=self.enable_tracking,
            enable_otel=self.enable_otel,
        )

        app_id = app.uid
        self._apps[app_id] = app

        # Run until the approval checkpoint
        run_until_approval(app)

        return app_id

    def get_state(self, app_id: str) -> dict[str, Any]:
        """Get the current state of an improvement workflow."""
        app = self._apps.get(app_id)
        if app is None:
            raise KeyError(f"Unknown app_id: {app_id}")
        return dict(app.state)

    def get_proposals(self, app_id: str) -> list[dict[str, Any]]:
        """Get the generated proposals for an improvement workflow."""
        return self.get_state(app_id).get("proposals", [])

    def approve_and_complete(
        self,
        app_id: str,
        approved: bool,
        reason: str = "",
        selected_proposal_idx: int | None = None,
    ) -> dict[str, Any]:
        """Set approval and run the rest of the workflow.

        Args:
            app_id: The app_id of the workflow waiting for approval.
            approved: Whether the proposal was approved.
            reason: Optional reason for the decision.
            selected_proposal_idx: Optional index to select a different proposal.

        Returns:
            Final state dict.
        """
        app = self._apps.get(app_id)
        if app is None:
            raise KeyError(f"Unknown app_id: {app_id}")

        if selected_proposal_idx is not None:
            app.update_state(app.state.update(selected_proposal_idx=selected_proposal_idx))

        final_state = approve_and_run(app, approved=approved, reason=reason)

        # Record outcome in history
        self._record_outcome(app_id, final_state)

        # Generate regression test if failed
        if final_state.get("current_stage") in {
            RemediationStatus.FAILED_EXECUTION.value,
            RemediationStatus.FAILED_VERIFICATION.value,
        }:
            try:
                generate_test(
                    state=final_state,
                    output_dir=self.test_output_dir,
                    app_id=app_id if self.enable_tracking else None,
                )
            except Exception as e:
                logger.warning(f"Test generation failed: {e}")

        return final_state

    def should_retry(self, app_id: str) -> bool:
        """Check if a failed workflow should be retried via forking."""
        state = self.get_state(app_id)
        return should_fork(state, max_attempts=self.max_fork_attempts)

    def fork_and_retry(self, app_id: str) -> str:
        """Fork a failed workflow and retry with an alternative strategy.

        Args:
            app_id: The app_id of the failed workflow.

        Returns:
            The app_id of the new forked application.
        """
        prior_state = self.get_state(app_id)

        if not should_fork(prior_state, max_attempts=self.max_fork_attempts):
            raise ValueError(f"Workflow {app_id} should not be forked")

        # Fork from the proposal generation step
        app = fork_from_proposal(
            prior_app_id=app_id,
            prior_state=prior_state,
            prior_sequence_id=-1,  # Will be resolved by tracking if available
            llm_gateway=self.llm_gateway,
            assessment_store=self.assessment_store,
            assessment_pipeline=self.assessment_pipeline,
            history_store=self.history_store,
            executor=self.executor,
            artifacts=self.artifacts,
            relationships=self.relationships,
            source=self.source,
            git_commit=self.git_commit,
            enable_tracking=self.enable_tracking,
            enable_otel=self.enable_otel,
        )

        new_app_id = app.uid
        self._apps[new_app_id] = app

        # Run until approval again
        run_until_approval(app)

        return new_app_id

    def _record_outcome(self, app_id: str, state: dict[str, Any]) -> None:
        """Record the workflow outcome in the history store.

        Focus 6: Records the new fields (verification_outcome,
        strategy_description, proposal_reasoning) so that future
        workflows can use richer historical context.
        """
        if self.history_store is None:
            return

        current_stage = state.get("current_stage", "")
        proposals = state.get("proposals", [])
        selected_idx = state.get("selected_proposal_idx", 0)
        verification = state.get("verification_results", {})
        root_causes = state.get("root_cause_analysis", [])
        knowledge_problems = state.get("knowledge_problems", [])
        decision_trace = state.get("decision_trace", {})

        strategy = (proposals[selected_idx].get("strategy", "unknown")
                    if selected_idx < len(proposals) else "unknown")
        issue_type = (root_causes[0].get("category", "unknown")
                      if root_causes else "unknown")

        result = "success" if current_stage == RemediationStatus.COMPLETED.value else "failure"
        score_change = verification.get("score_difference", 0)
        root_cause = root_causes[0].get("hypothesis", "") if root_causes else ""
        artifact_uris = state.get("affected_artifact_uris", [])
        failure_reason = (
            verification.get("failure_explanation", "")
            or verification.get("summary", "")
            if result == "failure" else ""
        )
        attempt_number = state.get("attempt_number", 1)
        forked_from = state.get("forked_from_app_id", "")

        # Focus 6: New fields for richer history
        verification_outcome = verification.get("overall_outcome", "")
        strategy_description = (
            proposals[selected_idx].get("description", "")
            if selected_idx < len(proposals) else ""
        )
        proposal_reasoning = (
            proposals[selected_idx].get("reasoning", "")
            if selected_idx < len(proposals) else ""
        )

        outcome = RemediationOutcome(
            issue_type=issue_type,
            strategy=strategy,
            result=result,
            score_change=score_change,
            root_cause=root_cause,
            artifact_uris=artifact_uris,
            failure_reason=failure_reason,
            attempt_number=attempt_number,
            verification_outcome=verification_outcome,
            strategy_description=strategy_description,
            proposal_reasoning=proposal_reasoning,
        )

        self.history_store.record_outcome(
            outcome=outcome,
            remediation_id=state.get("remediation_id", ""),
            forked_from_app_id=forked_from,
            metadata={
                "app_id": app_id,
                "current_stage": current_stage,
                "llm_metadata": state.get("llm_metadata", {}),
                "knowledge_problems": knowledge_problems,
                "decision_trace": decision_trace,
            },
        )

    def run_auto(
        self,
        assessment: Any,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Run the full improvement workflow with auto-approval (for testing/dry-run).

        Runs the workflow, auto-approving proposals, and automatically
        forks on failure up to max_attempts times.

        Args:
            assessment: The assessment to improve from.
            max_attempts: Maximum number of attempts (including forks).

        Returns:
            Final state dict from the last attempt.
        """
        app_id = self.start_improvement(assessment)
        state = self.approve_and_complete(app_id, approved=True, reason="auto-approved")

        attempts = 1
        while (state.get("current_stage") in {
            RemediationStatus.FAILED_EXECUTION.value,
            RemediationStatus.FAILED_VERIFICATION.value,
        } and attempts < max_attempts):
            if not self.should_retry(app_id):
                break

            logger.info(f"Auto-forking attempt #{attempts + 1}")
            app_id = self.fork_and_retry(app_id)
            state = self.approve_and_complete(app_id, approved=True, reason="auto-approved (forked retry)")
            attempts += 1

        return state

    def get_metrics(self) -> dict[str, Any]:
        """Get aggregate remediation quality metrics (Focus 6).

        Returns metrics showing whether AI-Ready is becoming more
        effective at resolving Knowledge Problems over time:

        - total_workflows, problems_resolved, problems_partially_resolved
        - verification_success_rate, avg_score_improvement
        - strategy_reuse_rate, strategy_success_rate
        - avg_forks_before_resolution

        Returns:
            Dict of remediation quality metrics, or empty dict if
            no history store is configured.
        """
        if self.history_store is None:
            return {}
        return self.history_store.get_remediation_metrics()

    def get_decision_trace(self, app_id: str) -> dict[str, Any]:
        """Get the full decision trace for a workflow (Focus 7).

        Returns the explainability chain showing reasoning at each stage:
        Knowledge Problems → Evidence → Root Cause → Historical Context →
        LLM Proposal → Approval → Execution → Verification → Final Outcome

        Args:
            app_id: The app_id of the workflow.

        Returns:
            DecisionTrace dict, or empty dict if not available.
        """
        return self.get_state(app_id).get("decision_trace", {})
