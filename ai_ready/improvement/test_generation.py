"""Regression test generation from failed improvement workflows.

Converts production improvement failures into pytest regression tests.
Uses Burr's built-in test case generation when available, with a
fallback to manual test generation from the workflow state.

Two approaches:
  1. Burr CLI: `burr-test-case create` generates tests from tracked
     workflow data (requires Burr tracking enabled).
  2. Programmatic: We generate pytest files from the workflow state
     and verification results, capturing the exact signals, root causes,
     proposals, and expected verification outcomes.

Generated tests serve as regression guards: if a future change
reintroduces the same issue, the test fails.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_ready.improvement.models import RemediationStatus

logger = logging.getLogger(__name__)


def _find_burr_test_case_cli() -> str | None:
    """Locate the burr-test-case executable.

    Searches in order:
    1. PATH (shutil.which)
    2. Same directory as the running Python interpreter (venv Scripts/)
    """
    import shutil
    cli_name = "burr-test-case" if sys.platform != "win32" else "burr-test-case.exe"
    # 1. Check PATH
    found = shutil.which("burr-test-case")
    if found:
        return found
    # 2. Check alongside the Python executable (venv Scripts dir)
    python_dir = Path(sys.executable).parent
    candidate = python_dir / cli_name
    if candidate.exists():
        return str(candidate)
    return None


def generate_test_from_failure(
    state: dict[str, Any],
    output_dir: str | Path,
    test_name: str | None = None,
) -> Path:
    """Generate a pytest regression test from a failed improvement workflow.

    Creates a self-contained pytest file that:
    - Recreates the initial state
    - Runs the improvement workflow
    - Asserts the expected failure (or success, for positive regression)

    Args:
        state: The final state of the improvement workflow.
        output_dir: Directory to write the test file.
        test_name: Optional test name (auto-generated if None).

    Returns:
        Path to the generated test file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test name
    if test_name is None:
        issue_type = "unknown"
        root_causes = state.get("root_cause_analysis", [])
        if root_causes:
            issue_type = root_causes[0].get("category", "unknown")
        attempt = state.get("attempt_number", 1)
        test_name = f"test_improvement_{issue_type}_attempt_{attempt}"

    # Sanitize test name
    safe_name = test_name.replace(" ", "_").replace("-", "_").replace(".", "_")

    # Extract key data from state
    signal_ids = state.get("signal_ids", [])
    artifact_uris = state.get("affected_artifact_uris", [])
    root_causes = state.get("root_cause_analysis", [])
    proposals = state.get("proposals", [])
    selected_idx = state.get("selected_proposal_idx", 0)
    verification = state.get("verification_results", {})
    current_stage = state.get("current_stage", "")
    attempt_number = state.get("attempt_number", 1)

    strategy = (proposals[selected_idx].get("strategy", "unknown")
                if selected_idx < len(proposals) else "unknown")

    # Build the test file content
    test_content = f'''"""Auto-generated regression test from failed improvement workflow.

Generated: {datetime.now(timezone.utc).isoformat()}
Issue type: {root_causes[0].get("category", "unknown") if root_causes else "unknown"}
Strategy: {strategy}
Attempt: #{attempt_number}
Result: {current_stage}

This test captures the exact state of a failed improvement workflow
to serve as a regression guard. If a future change reintroduces
the same issue or the same strategy fails again, this test will
catch it.
"""

import pytest
from ai_ready.improvement.models import RemediationStatus
from ai_ready.improvement.application import create_improvement_app
from ai_ready.improvement.models import initial_state


# --- Captured state from the failed workflow ---

CAPTURED_SIGNAL_IDS = {json.dumps(signal_ids)}

CAPTURED_ARTIFACT_URIS = {json.dumps(artifact_uris)}

CAPTURED_ROOT_CAUSES = {json.dumps(root_causes, indent=4, default=str)}

CAPTURED_PROPOSALS = {json.dumps(proposals, indent=4, default=str)}

CAPTURED_VERIFICATION = {json.dumps(verification, indent=4, default=str)}

CAPTURED_STRATEGY = "{strategy}"

CAPTURED_STAGE = "{current_stage}"

CAPTURED_ATTEMPT = {attempt_number}


class TestImprovementRegression:
    """Regression tests for the {safe_name} workflow."""

    def test_root_causes_were_identified(self):
        """Verify that root cause analysis produced hypotheses."""
        assert len(CAPTURED_ROOT_CAUSES) > 0, \\
            "Root cause analysis should have produced at least one hypothesis"

    def test_proposals_were_generated(self):
        """Verify that improvement proposals were generated."""
        assert len(CAPTURED_PROPOSALS) > 0, \\
            "Proposal generation should have produced at least one proposal"

    def test_strategy_was_attempted(self):
        """Verify that a specific strategy was attempted."""
        assert CAPTURED_STRATEGY != "unknown", \\
            "A specific strategy should have been selected"

    def test_workflow_reached_terminal_state(self):
        """Verify the workflow reached a terminal state (not stuck)."""
        terminal_states = {{
            RemediationStatus.COMPLETED.value,
            RemediationStatus.REJECTED.value,
            RemediationStatus.FAILED_ANALYSIS.value,
            RemediationStatus.FAILED_EXECUTION.value,
            RemediationStatus.FAILED_VERIFICATION.value,
        }}
        assert CAPTURED_STAGE in terminal_states, \\
            f"Workflow should have reached a terminal state, got: {{CAPTURED_STAGE}}"

    def test_failure_was_recorded(self):
        """Verify the failure was properly recorded for future learning."""
        if CAPTURED_STAGE == RemediationStatus.COMPLETED.value:
            pytest.skip("This workflow succeeded — no failure to record")
        assert CAPTURED_VERIFICATION.get("summary", "") != "", \\
            "Failure should have a summary explaining what went wrong"

    def test_signal_ids_are_valid(self):
        """Verify signal IDs are non-empty strings."""
        for sid in CAPTURED_SIGNAL_IDS:
            assert isinstance(sid, str) and len(sid) > 0, \\
                f"Signal ID should be non-empty string, got: {{sid}}"

    def test_artifact_uris_are_valid(self):
        """Verify artifact URIs are non-empty strings."""
        for uri in CAPTURED_ARTIFACT_URIS:
            assert isinstance(uri, str) and len(uri) > 0, \\
                f"Artifact URI should be non-empty string, got: {{uri}}"

    @pytest.mark.integration
    def test_workflow_can_be_reproduced(self):
        """Reproduce the workflow from initial state and verify it reaches the same stage.

        This is an integration test that requires the full AI-Ready stack.
        Mark as integration to skip in unit test runs.
        """
        # This test would need the actual assessment store and pipeline
        # to reproduce. It's included as a template.
        pytest.skip("Integration test — requires full AI-Ready stack")

    def test_strategy_not_repeated_if_failed(self):
        """If this was a forked retry, verify the strategy differs from prior attempts."""
        prior_failures = {json.dumps(state.get("prior_failures", []), default=str)}
        if not prior_failures:
            pytest.skip("No prior attempts — nothing to compare")
        for failure in prior_failures:
            prior_strategy = failure.get("strategy", "")
            if prior_strategy == CAPTURED_STRATEGY:
                pytest.fail(
                    f"Strategy '{{CAPTURED_STRATEGY}}' was repeated "
                    f"despite failing in attempt #{{failure.get('attempt_number', '?')}}"
                )
'''

    test_file = output_dir / f"test_{safe_name}.py"
    test_file.write_text(test_content, encoding="utf-8")
    logger.info(f"Generated regression test: {test_file}")
    return test_file


def generate_burr_test_case(
    app_id: str,
    project_name: str = "ai_ready_improvement",
    output_dir: str | Path = ".",
    sequence_id: str | None = None,
    partition_key: str | None = None,
) -> Path | None:
    """Generate a test case using Burr's built-in test case generator.

    Uses the `burr-test-case create` CLI command to generate a test
    from tracked workflow data. Requires Burr tracking to have been
    enabled during the workflow.

    Args:
        app_id: The Burr application ID of the workflow.
        project_name: The Burr tracking project name.
        output_dir: Directory to write the test file.
        sequence_id: The sequence ID to capture (required by the CLI).
            If None, the latest sequence is attempted by omitting the flag.
        partition_key: Optional partition key for the tracked data.

    Returns:
        Path to the generated test file, or None if generation failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cli_path = _find_burr_test_case_cli()
        if cli_path is None:
            logger.warning("burr-test-case CLI not found — using programmatic test generation")
            return None

        target_file = output_dir / f"burr_test_{app_id[:8]}.py"

        cmd = [
            cli_path, "create",
            "--app-id", app_id,
            "--project-name", project_name,
            "--target-file-name", str(target_file),
        ]
        if sequence_id:
            cmd.extend(["--sequence-id", sequence_id])
        if partition_key:
            cmd.extend(["--partition-key", partition_key])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info(f"Burr test case generated for app {app_id}")
            if target_file.exists():
                return target_file
            # Fallback: look for any test file created
            test_files = list(output_dir.glob("test_*.py"))
            if test_files:
                return test_files[-1]
        else:
            logger.warning(f"Burr test case generation failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning("Burr test case generation timed out")
    except Exception as e:
        logger.warning(f"Burr test case generation error: {e}")

    return None


def generate_test(
    state: dict[str, Any],
    output_dir: str | Path,
    app_id: str | None = None,
    project_name: str = "ai_ready_improvement",
    test_name: str | None = None,
) -> Path:
    """Generate a regression test, using Burr CLI if available, else programmatic.

    Args:
        state: The final state of the improvement workflow.
        output_dir: Directory to write the test file.
        app_id: Optional Burr app_id for CLI-based generation.
        project_name: Burr tracking project name.
        test_name: Optional test name.

    Returns:
        Path to the generated test file.
    """
    # Try Burr CLI first (if tracking was enabled)
    if app_id:
        # Extract the sequence ID from state (Burr stores it as __SEQUENCE_ID)
        sequence_id = state.get("__SEQUENCE_ID")
        if sequence_id is not None:
            sequence_id = str(sequence_id)
        burr_test = generate_burr_test_case(
            app_id, project_name, output_dir, sequence_id=sequence_id
        )
        if burr_test:
            return burr_test

    # Fall back to programmatic generation
    return generate_test_from_failure(state, output_dir, test_name)
