"""Knowledge modification executor — AI-Ready's execution layer.

This is NOT a Burr action. It is called BY the ExecuteChangeAction
to apply approved modifications to the knowledge base.

The executor belongs to AI-Ready. Burr only orchestrates the call.

Supported modification types:
  - update_document: Modify document content (headings, text, metadata)
  - add_metadata: Add or update metadata fields on an artifact
  - create_relationship: Create a relationship between artifacts
  - remove_relationship: Remove a relationship
  - archive_artifact: Mark an artifact as archived/outdated
  - merge_artifacts: Merge multiple artifacts into one
  - split_artifact: Split one artifact into multiple

The executor operates on artifact URIs (not IDs) and returns
execution results that the Burr action stores in state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ExecutionStep:
    """A single modification step executed by the executor."""

    step_type: str  # "update_document", "add_metadata", etc.
    artifact_uri: str
    description: str
    success: bool = True
    error: str = ""
    timestamp: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_type": self.step_type,
            "artifact_uri": self.artifact_uri,
            "description": self.description,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class KnowledgeExecutor:
    """Executes approved knowledge modifications.

    This class is intentionally pluggable — the actual file/DB operations
    depend on the deployment context. The default implementation logs
    modifications and returns success. Production deployments subclass
    this to implement real file writes, API calls, or DB updates.

    Security: The executor is the ONLY component that modifies knowledge.
    The LLM never calls the executor directly. Burr's ExecuteChangeAction
    calls the executor with the approved proposal's modification steps.
    """

    def __init__(self, artifact_store: Any = None) -> None:
        """Initialize the executor.

        Args:
            artifact_store: Optional reference to the artifact storage layer.
                           Production implementations use this to load and
                           save artifacts. The default implementation doesn't
                           need it (dry-run mode).
        """
        self.artifact_store = artifact_store
        self._dry_run = artifact_store is None

    def execute(
        self,
        modification_steps: list[dict[str, Any]],
        artifact_uris: list[str],
    ) -> list[ExecutionStep]:
        """Execute a list of modification steps.

        Args:
            modification_steps: List of step dicts from the approved proposal.
                               Each dict has: step_type, artifact_uri, and
                               step-specific parameters.
            artifact_uris: All artifact URIs affected by this improvement.

        Returns:
            List of ExecutionStep results (one per input step).
        """
        results: list[ExecutionStep] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        for step in modification_steps:
            step_type = step.get("step_type", "unknown")
            artifact_uri = step.get("artifact_uri", "")
            description = step.get("description", "")

            try:
                handler = self._get_handler(step_type)
                details = handler(step)
                results.append(ExecutionStep(
                    step_type=step_type,
                    artifact_uri=artifact_uri,
                    description=description,
                    success=True,
                    timestamp=timestamp,
                    details=details,
                ))
            except Exception as e:
                results.append(ExecutionStep(
                    step_type=step_type,
                    artifact_uri=artifact_uri,
                    description=description,
                    success=False,
                    error=str(e),
                    timestamp=timestamp,
                ))

        return results

    def _get_handler(self, step_type: str):
        """Get the handler function for a modification type."""
        handlers = {
            "update_document": self._handle_update_document,
            "add_metadata": self._handle_add_metadata,
            "create_relationship": self._handle_create_relationship,
            "remove_relationship": self._handle_remove_relationship,
            "archive_artifact": self._handle_archive_artifact,
            "merge_artifacts": self._handle_merge_artifacts,
            "split_artifact": self._handle_split_artifact,
        }
        handler = handlers.get(step_type)
        if handler is None:
            raise ValueError(f"Unknown modification type: {step_type}")
        return handler

    def _handle_update_document(self, step: dict[str, Any]) -> dict[str, Any]:
        """Handle document content update."""
        if self._dry_run:
            return {"mode": "dry_run", "message": "Document update simulated"}
        # Production: load artifact, apply changes, save
        raise NotImplementedError("Production document update requires artifact_store")

    def _handle_add_metadata(self, step: dict[str, Any]) -> dict[str, Any]:
        """Handle metadata addition/update."""
        if self._dry_run:
            return {"mode": "dry_run", "message": "Metadata update simulated"}
        raise NotImplementedError("Production metadata update requires artifact_store")

    def _handle_create_relationship(self, step: dict[str, Any]) -> dict[str, Any]:
        """Handle relationship creation."""
        if self._dry_run:
            return {"mode": "dry_run", "message": "Relationship creation simulated"}
        raise NotImplementedError("Production relationship creation requires artifact_store")

    def _handle_remove_relationship(self, step: dict[str, Any]) -> dict[str, Any]:
        """Handle relationship removal."""
        if self._dry_run:
            return {"mode": "dry_run", "message": "Relationship removal simulated"}
        raise NotImplementedError("Production relationship removal requires artifact_store")

    def _handle_archive_artifact(self, step: dict[str, Any]) -> dict[str, Any]:
        """Handle artifact archiving."""
        if self._dry_run:
            return {"mode": "dry_run", "message": "Artifact archival simulated"}
        raise NotImplementedError("Production archival requires artifact_store")

    def _handle_merge_artifacts(self, step: dict[str, Any]) -> dict[str, Any]:
        """Handle artifact merging."""
        if self._dry_run:
            return {"mode": "dry_run", "message": "Artifact merge simulated"}
        raise NotImplementedError("Production merge requires artifact_store")

    def _handle_split_artifact(self, step: dict[str, Any]) -> dict[str, Any]:
        """Handle artifact splitting."""
        if self._dry_run:
            return {"mode": "dry_run", "message": "Artifact split simulated"}
        raise NotImplementedError("Production split requires artifact_store")

    def rollback(self, execution_steps: list[ExecutionStep]) -> list[ExecutionStep]:
        """Rollback executed modifications.

        Called when verification fails and we want to undo changes
        before forking to try an alternative strategy.

        Returns:
            List of ExecutionStep results for the rollback operations.
        """
        rollback_results: list[ExecutionStep] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Reverse order — undo last changes first
        for step in reversed(execution_steps):
            if not step.success:
                continue  # Can't rollback a step that didn't succeed

            try:
                # Production: implement actual rollback logic
                rollback_results.append(ExecutionStep(
                    step_type=f"rollback_{step.step_type}",
                    artifact_uri=step.artifact_uri,
                    description=f"Rollback: {step.description}",
                    success=True,
                    timestamp=timestamp,
                    details={"mode": "dry_run", "original_step": step.to_dict()},
                ))
            except Exception as e:
                rollback_results.append(ExecutionStep(
                    step_type=f"rollback_{step.step_type}",
                    artifact_uri=step.artifact_uri,
                    description=f"Rollback: {step.description}",
                    success=False,
                    error=str(e),
                    timestamp=timestamp,
                ))

        return rollback_results
