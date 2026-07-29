"""End-to-end validation of AI-Ready + Apache Burr integration.

Validates: state machine transitions, LLM integration, approval workflow,
execution, verification, forking, observability, and regression test generation.

Target repo: fastapi (read-only, not modified)
All output files stay inside the ai-ready directory.
"""

from __future__ import annotations

import json
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# ─── Paths ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
FASTAPI_PATH = Path.home() / "Documents" / "khe-validation" / "fastapi"
BASELINE_DB = SCRIPT_DIR / "baseline.db"
HISTORY_DB = SCRIPT_DIR / "improvement_history.db"
TEST_OUTPUT_DIR = SCRIPT_DIR / "generated_tests"
RESULTS_FILE = SCRIPT_DIR / "validation_results.json"

# Load .env
env_path = SCRIPT_DIR / "ai_ready" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

sys.path.insert(0, str(SCRIPT_DIR))

from ai_ready.connectors.markdown import MarkdownConnector
from ai_ready.models import KnowledgeArtifact, Relationship
from ai_ready.stores import AssessmentStore
from ai_ready.pipeline import AssessmentPipeline
from ai_ready.llm.gateway import LLMGateway
from ai_ready.improvement.manager import ImprovementManager
from ai_ready.improvement.models import RemediationStatus
from ai_ready.improvement.history import ImprovementHistoryStore
from ai_ready.improvement.executor import KnowledgeExecutor
from ai_ready.improvement.application import create_improvement_app, run_until_approval, approve_and_run
from ai_ready.improvement.forking import fork_from_proposal, should_fork
from ai_ready.improvement.test_generation import generate_test

# ─── Results collector ──────────────────────────────────────────────────

results: dict = {
    "validation_start": datetime.now(timezone.utc).isoformat(),
    "baseline": {},
    "burr_workflow": {},
    "llm_integration": {},
    "approval_workflow": {},
    "execution": {},
    "post_remediation": {},
    "burr_unique": {},
    "regression_tests": {},
    "errors": [],
}


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def record(section: str, key: str, value) -> None:
    results.setdefault(section, {})[key] = value


# ─── Phase 1: Load baseline assessment ──────────────────────────────────

log("Phase 1: Loading baseline assessment...")
assessment_store = AssessmentStore(BASELINE_DB)
baseline_assessment = assessment_store.latest()

if not baseline_assessment:
    log("ERROR: No baseline assessment found! Run: python -c \"from ai_ready.cli import app; app(['scan', '<fastapi_path>', '--json', '--db', '" + str(BASELINE_DB) + "'])\"")
    sys.exit(1)

log(f"  Assessment ID: {baseline_assessment.assessment_id}")
log(f"  Score: {baseline_assessment.score}/100")
log(f"  Signals: {len(baseline_assessment.signals)}")
log(f"  Dimensions: {list(baseline_assessment.dimensions.keys())}")

signal_categories: dict[str, list] = {}
for sig in baseline_assessment.signals:
    key = f"{sig.collector_id}:{sig.signal_type}"
    signal_categories.setdefault(key, []).append(sig)

log(f"  Signal categories: {len(signal_categories)}")
for cat, sigs in sorted(signal_categories.items(), key=lambda x: -len(x[1])):
    log(f"    {cat}: {len(sigs)} signals")

record("baseline", "assessment_id", baseline_assessment.assessment_id)
record("baseline", "score", baseline_assessment.score)
record("baseline", "signal_count", len(baseline_assessment.signals))
record("baseline", "dimensions", {
    k: {"score": v.score, "signals": v.signals_count}
    for k, v in baseline_assessment.dimensions.items()
})
record("baseline", "signal_categories", {
    cat: len(sigs) for cat, sigs in signal_categories.items()
})

# ─── Phase 2: Select representative signals ─────────────────────────────

selected_signals = []
seen_collectors = set()

for cat, sigs in sorted(signal_categories.items(), key=lambda x: -len(x[1])):
    if cat.split(":")[0] not in seen_collectors:
        selected_signals.append(sigs[0])
        seen_collectors.add(cat.split(":")[0])

for cat, sigs in sorted(signal_categories.items(), key=lambda x: -len(x[1])):
    for sig in sigs[:2]:
        if sig not in selected_signals and len(selected_signals) < 5:
            selected_signals.append(sig)
            seen_collectors.add(sig.collector_id)

log(f"\nPhase 2: Selected {len(selected_signals)} representative signals:")
for sig in selected_signals:
    log(f"  - {sig.signal_id} | {sig.collector_id}:{sig.signal_type} | {sig.severity.value} | {sig.artifact_uri}")

record("burr_workflow", "selected_signals", [
    {
        "signal_id": s.signal_id,
        "collector_id": s.collector_id,
        "signal_type": s.signal_type,
        "severity": s.severity.value,
        "artifact_uri": s.artifact_uri,
    }
    for s in selected_signals
])

# ─── Phase 3: Set up dependencies ────────────────────────────────────────

log("\nPhase 3: Setting up dependencies...")

connector = MarkdownConnector()
connector.connect(str(FASTAPI_PATH))
artifacts = list(connector.iter_artifacts())
relationships = list(connector.iter_relationships())
log(f"  Loaded {len(artifacts)} artifacts, {len(relationships)} relationships")

log("  Initializing LLM Gateway (Groq)...")
try:
    llm_gateway = LLMGateway(provider="groq")
    log(f"    Provider: {llm_gateway.provider_name}")
    log(f"    Model: {llm_gateway.provider.default_model}")
    record("llm_integration", "provider", llm_gateway.provider_name)
    record("llm_integration", "default_model", llm_gateway.provider.default_model)
except Exception as e:
    log(f"    ERROR: LLM Gateway init failed: {e}")
    results["errors"].append(f"LLM Gateway init: {e}")
    llm_gateway = None

assessment_pipeline = AssessmentPipeline()
history_store = ImprovementHistoryStore(HISTORY_DB)
log(f"  History store: {HISTORY_DB}")

executor = KnowledgeExecutor()
log("  Executor: dry-run mode")

if TEST_OUTPUT_DIR.exists():
    shutil.rmtree(TEST_OUTPUT_DIR)
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Phase 4: Run Burr remediation workflow ─────────────────────────────

log("\nPhase 4: Running Burr remediation workflow...")

git_result = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True, cwd=str(FASTAPI_PATH)
)
git_commit = git_result.stdout.strip() if git_result.returncode == 0 else ""
log(f"  Git commit: {git_commit}")

manager = ImprovementManager(
    llm_gateway=llm_gateway,
    assessment_store=assessment_store,
    assessment_pipeline=assessment_pipeline,
    history_store=history_store,
    executor=executor,
    artifacts=artifacts,
    relationships=relationships,
    source=str(FASTAPI_PATH),
    git_commit=git_commit,
    history_db_path=str(HISTORY_DB),
    test_output_dir=str(TEST_OUTPUT_DIR),
    enable_tracking=True,
    enable_otel=False,
    max_fork_attempts=3,
)

signal_ids = [s.signal_id for s in selected_signals]
artifact_uris = list({s.artifact_uri for s in selected_signals})

log(f"  Starting improvement for {len(signal_ids)} signals, {len(artifact_uris)} artifacts...")

workflow_start = time.time()
state_transitions: list[dict] = []
llm_calls: list[dict] = []
final_state = None
app_id = None

try:
    app_id = manager.start_improvement(
        assessment=baseline_assessment,
        signal_ids=signal_ids,
        artifact_uris=artifact_uris,
    )
    log(f"  Burr app created: {app_id}")

    state = manager.get_state(app_id)
    current_stage = state.get("current_stage", "")
    log(f"  State after run_until_approval: {current_stage}")

    state_transitions.append({
        "step": "after_run_until_approval",
        "stage": current_stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    if current_stage == RemediationStatus.WAITING_FOR_APPROVAL.value:
        log("  CHECK: Workflow reached WAITING_FOR_APPROVAL (correct halt point)")

        proposals = state.get("proposals", [])
        log(f"  Proposals generated: {len(proposals)}")
        for i, p in enumerate(proposals):
            log(f"    [{i}] strategy={p.get('strategy', '?')}, impact={p.get('expected_impact', '?')}")
            log(f"        steps={len(p.get('modification_steps', []))}")

        record("burr_workflow", "proposals", proposals)
        record("burr_workflow", "reached_approval", True)

        llm_meta = state.get("llm_metadata", {})
        for action_name, meta in llm_meta.items():
            if "error" in meta:
                log(f"  LLM call {action_name}: ERROR - {meta['error']}")
                llm_calls.append({"action": action_name, "error": meta["error"]})
            else:
                log(f"  LLM call {action_name}: provider={meta.get('provider', '?')}, model={meta.get('model', '?')}, "
                    f"tokens={meta.get('prompt_tokens', 0)}+{meta.get('completion_tokens', 0)}, "
                    f"latency={meta.get('latency_ms', 0):.0f}ms")
                llm_calls.append({
                    "action": action_name,
                    "provider": meta.get("provider"),
                    "model": meta.get("model"),
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "completion_tokens": meta.get("completion_tokens", 0),
                    "latency_ms": meta.get("latency_ms", 0),
                })

        record("llm_integration", "calls", llm_calls)

        root_causes = state.get("root_cause_analysis", [])
        log(f"  Root cause hypotheses: {len(root_causes)}")
        for rc in root_causes:
            log(f"    - {rc.get('category', '?')}: {rc.get('hypothesis', '?')[:80]}...")
            log(f"      confidence={rc.get('confidence', 0)}")

        record("burr_workflow", "root_causes", root_causes)
        record("burr_workflow", "state_transitions", state_transitions)

        # ─── Phase 5: Test approval workflow ────────────────────────────

        log("\nPhase 5: Testing approval workflow...")

        log("  Testing APPROVE path...")
        approve_start = time.time()
        final_state = manager.approve_and_complete(
            app_id=app_id,
            approved=True,
            reason="Validation test - approved",
        )
        approve_duration = time.time() - approve_start

        final_stage = final_state.get("current_stage", "")
        log(f"  Final stage after approval: {final_stage}")

        execution_history = final_state.get("execution_history", [])
        log(f"  Execution steps: {len(execution_history)}")
        for step in execution_history:
            log(f"    - {step.get('step_type', '?')} on {step.get('artifact_uri', '?')}: "
                f"{'SUCCESS' if step.get('success') else 'FAILED'}")

        record("approval_workflow", "approved", True)
        record("approval_workflow", "final_stage", final_stage)
        record("approval_workflow", "approve_duration_s", round(approve_duration, 2))
        record("execution", "steps", execution_history)
        record("execution", "step_count", len(execution_history))
        record("execution", "all_succeeded", all(s.get("success", False) for s in execution_history) if execution_history else False)

        verification = final_state.get("verification_results", {})
        log(f"  Verification: success={verification.get('success', '?')}, "
            f"score {verification.get('before_score', '?')} -> {verification.get('after_score', '?')}")
        record("burr_workflow", "verification", verification)
        record("burr_workflow", "final_stage", final_stage)

        state_transitions.append({
            "step": "after_approval_and_execution",
            "stage": final_stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        workflow_duration = time.time() - workflow_start
        record("burr_workflow", "duration_s", round(workflow_duration, 2))
        log(f"  Workflow duration: {workflow_duration:.1f}s")

    elif current_stage in {
        RemediationStatus.FAILED_ANALYSIS.value,
        RemediationStatus.FAILED_EXECUTION.value,
    }:
        log(f"  WARNING: Workflow failed at stage: {current_stage}")
        record("burr_workflow", "reached_approval", False)
        record("burr_workflow", "failure_stage", current_stage)
        llm_meta = state.get("llm_metadata", {})
        for action_name, meta in llm_meta.items():
            llm_calls.append({"action": action_name, **meta})
        record("llm_integration", "calls", llm_calls)
    else:
        log(f"  WARNING: Unexpected stage: {current_stage}")
        record("burr_workflow", "reached_approval", False)
        record("burr_workflow", "unexpected_stage", current_stage)

except Exception as e:
    log(f"  ERROR during workflow: {e}")
    import traceback
    traceback.print_exc()
    results["errors"].append(f"Workflow execution: {e}")
    record("burr_workflow", "error", str(e))

# ─── Phase 6: Test rejection path ───────────────────────────────────────

log("\nPhase 6: Testing rejection path...")
try:
    app_id_2 = manager.start_improvement(
        assessment=baseline_assessment,
        signal_ids=signal_ids[:2],
        artifact_uris=artifact_uris[:2],
    )
    state_2 = manager.get_state(app_id_2)

    if state_2.get("current_stage") == RemediationStatus.WAITING_FOR_APPROVAL.value:
        reject_state = manager.approve_and_complete(
            app_id=app_id_2,
            approved=False,
            reason="Validation test - rejected",
        )
        reject_stage = reject_state.get("current_stage", "")
        log(f"  Rejection final stage: {reject_stage}")

        record("approval_workflow", "rejected_stage", reject_stage)
        record("approval_workflow", "rejection_works",
              reject_stage == RemediationStatus.REJECTED.value
              or reject_stage == RemediationStatus.FAILED_ANALYSIS.value)
    else:
        log(f"  Second workflow didn't reach approval: {state_2.get('current_stage')}")
        record("approval_workflow", "rejection_works", False)
        record("approval_workflow", "rejection_note", "Second workflow didn't reach approval")

except Exception as e:
    log(f"  ERROR during rejection test: {e}")
    results["errors"].append(f"Rejection test: {e}")
    record("approval_workflow", "rejection_works", False)
    record("approval_workflow", "rejection_error", str(e))

# ─── Phase 7: Test forking ──────────────────────────────────────────────

log("\nPhase 7: Testing state forking...")
try:
    if final_state and final_state.get("current_stage") in {
        RemediationStatus.FAILED_EXECUTION.value,
        RemediationStatus.FAILED_VERIFICATION.value,
    }:
        log("  First workflow failed - testing fork...")

        should_fork_result = should_fork(final_state, max_attempts=3)
        log(f"  should_fork() = {should_fork_result}")

        if should_fork_result and app_id:
            forked_app_id = manager.fork_and_retry(app_id)
            forked_state = manager.get_state(forked_app_id)
            log(f"  Forked app: {forked_app_id}")
            log(f"  Forked state stage: {forked_state.get('current_stage')}")
            log(f"  Attempt number: {forked_state.get('attempt_number')}")
            log(f"  Prior failures: {len(forked_state.get('prior_failures', []))}")
            log(f"  Forked from: {forked_state.get('forked_from_app_id')}")

            record("burr_unique", "forking", {
                "tested": True,
                "forked_app_id": forked_app_id,
                "attempt_number": forked_state.get("attempt_number"),
                "prior_failures_count": len(forked_state.get("prior_failures", [])),
                "forked_from_app_id": forked_state.get("forked_from_app_id"),
                "stage": forked_state.get("current_stage"),
            })

            forked_final = manager.approve_and_complete(
                app_id=forked_app_id,
                approved=True,
                reason="Forked retry - auto-approved",
            )
            log(f"  Forked final stage: {forked_final.get('current_stage')}")
            record("burr_unique", "forking_final_stage", forked_final.get("current_stage"))
        else:
            log("  should_fork() returned False - skipping fork")
            record("burr_unique", "forking", {"tested": False, "reason": "should_fork=False"})
    else:
        log("  First workflow didn't fail - simulating failure for fork test...")

        sim_fail_state = dict(final_state) if final_state else {}
        sim_fail_state["current_stage"] = RemediationStatus.FAILED_VERIFICATION.value
        sim_fail_state["verification_results"] = {
            "success": False,
            "summary": "Simulated failure for fork testing",
            "score_difference": -5,
        }

        should_fork_result = should_fork(sim_fail_state, max_attempts=3)
        log(f"  should_fork() with simulated failure = {should_fork_result}")

        record("burr_unique", "forking", {
            "tested": True,
            "simulated": True,
            "should_fork": should_fork_result,
            "note": "First workflow succeeded; fork logic validated with simulated failure state",
        })

except Exception as e:
    log(f"  ERROR during fork test: {e}")
    import traceback
    traceback.print_exc()
    results["errors"].append(f"Fork test: {e}")
    record("burr_unique", "forking", {"tested": False, "error": str(e)})

# ─── Phase 8: Validate state persistence ─────────────────────────────────

log("\nPhase 8: Validating state persistence...")
try:
    burr_tracking_dir = Path.home() / ".burr"
    tracking_exists = burr_tracking_dir.exists()
    tracking_files = list(burr_tracking_dir.rglob("*.json")) if tracking_exists else []
    log(f"  Burr tracking dir exists: {tracking_exists}")
    log(f"  Tracking files: {len(tracking_files)}")

    if final_state:
        state_keys = list(final_state.keys())
        has_references_only = (
            "assessment_id" in state_keys
            and "signal_ids" in state_keys
            and "affected_artifact_uris" in state_keys
        )
        state_size = len(json.dumps(final_state, default=str))
        log(f"  State keys: {state_keys}")
        log(f"  State size (JSON): {state_size} bytes")
        log(f"  References only (no large content): {has_references_only}")

        record("burr_unique", "state_persistence", {
            "tracking_dir_exists": tracking_exists,
            "tracking_files_count": len(tracking_files),
            "state_keys": state_keys,
            "state_size_bytes": state_size,
            "references_only": has_references_only,
        })
    else:
        record("burr_unique", "state_persistence", {
            "tracking_dir_exists": tracking_exists,
            "tracking_files_count": len(tracking_files),
        })

except Exception as e:
    log(f"  ERROR during persistence validation: {e}")
    results["errors"].append(f"Persistence validation: {e}")
    record("burr_unique", "state_persistence", {"error": str(e)})

# ─── Phase 9: Validate history store ────────────────────────────────────

log("\nPhase 9: Validating history store (institutional memory)...")
try:
    recent_outcomes = history_store.get_recent_outcomes(limit=10)
    log(f"  Recent outcomes in history: {len(recent_outcomes)}")

    for outcome in recent_outcomes:
        log(f"    - issue={outcome.get('issue_type', '?')}, strategy={outcome.get('strategy', '?')}, "
            f"result={outcome.get('result', '?')}, score_change={outcome.get('score_change', 0)}")

    record("burr_unique", "history_store", {
        "outcomes_count": len(recent_outcomes),
        "outcomes": recent_outcomes,
    })

except Exception as e:
    log(f"  ERROR during history validation: {e}")
    results["errors"].append(f"History validation: {e}")
    record("burr_unique", "history_store", {"error": str(e)})

# ─── Phase 10: Validate regression test generation ──────────────────────

log("\nPhase 10: Validating regression test generation...")
try:
    if final_state:
        test_file = generate_test(
            state=final_state,
            output_dir=str(TEST_OUTPUT_DIR),
            app_id=app_id,
        )
        log(f"  Generated test file: {test_file}")
        log(f"  Test file exists: {test_file.exists()}")
        if test_file.exists():
            test_content = test_file.read_text()
            test_lines = test_content.count("\n")
            log(f"  Test file size: {len(test_content)} bytes, {test_lines} lines")

            has_root_cause_test = "test_root_causes_were_identified" in test_content
            has_proposal_test = "test_proposals_were_generated" in test_content
            has_strategy_test = "test_strategy_was_attempted" in test_content
            has_terminal_test = "test_workflow_reached_terminal_state" in test_content
            has_failure_test = "test_failure_was_recorded" in test_content

            log(f"  Test methods found:")
            log(f"    test_root_causes_were_identified: {has_root_cause_test}")
            log(f"    test_proposals_were_generated: {has_proposal_test}")
            log(f"    test_strategy_was_attempted: {has_strategy_test}")
            log(f"    test_workflow_reached_terminal_state: {has_terminal_test}")
            log(f"    test_failure_was_recorded: {has_failure_test}")

            record("regression_tests", "generated", {
                "generated": True,
                "file_path": str(test_file),
                "file_size": len(test_content),
                "line_count": test_lines,
                "has_root_cause_test": has_root_cause_test,
                "has_proposal_test": has_proposal_test,
                "has_strategy_test": has_strategy_test,
                "has_terminal_test": has_terminal_test,
                "has_failure_test": has_failure_test,
            })
        else:
            record("regression_tests", "generated", {"generated": False, "error": "File not created"})
    else:
        log("  No final state available for test generation")
        record("regression_tests", "generated", {"generated": False, "reason": "No final state"})

except Exception as e:
    log(f"  ERROR during test generation: {e}")
    import traceback
    traceback.print_exc()
    results["errors"].append(f"Test generation: {e}")
    record("regression_tests", "generated", {"generated": False, "error": str(e)})

# ─── Phase 11: Validate LLM integration details ─────────────────────────

log("\nPhase 11: Validating LLM integration details...")
try:
    if llm_gateway and llm_gateway.provider:
        provider = llm_gateway.provider
        record("llm_integration", "provider_class", type(provider).__name__)
        record("llm_integration", "provider_name", provider.name)
        record("llm_integration", "default_model", provider.default_model)

        from ai_ready.improvement import actions
        import inspect
        analyze_src = inspect.getsource(actions.analyze_issue)
        propose_src = inspect.getsource(actions.generate_proposal)

        uses_gateway = "llm_gateway" in analyze_src and "llm_gateway" in propose_src
        uses_provider_directly = "GroqProvider" in analyze_src or "GroqProvider" in propose_src

        log(f"  Actions use LLMGateway: {uses_gateway}")
        log(f"  Actions bypass gateway (direct provider): {uses_provider_directly}")

        record("llm_integration", "uses_gateway", uses_gateway)
        record("llm_integration", "bypasses_gateway", uses_provider_directly)
        record("llm_integration", "total_calls", len(llm_calls))
        record("llm_integration", "successful_calls", sum(1 for c in llm_calls if "error" not in c))
        record("llm_integration", "failed_calls", sum(1 for c in llm_calls if "error" in c))

        if llm_calls:
            total_prompt_tokens = sum(c.get("prompt_tokens", 0) for c in llm_calls if "error" not in c)
            total_completion_tokens = sum(c.get("completion_tokens", 0) for c in llm_calls if "error" not in c)
            total_latency = sum(c.get("latency_ms", 0) for c in llm_calls if "error" not in c)
            record("llm_integration", "total_prompt_tokens", total_prompt_tokens)
            record("llm_integration", "total_completion_tokens", total_completion_tokens)
            record("llm_integration", "total_latency_ms", round(total_latency, 1))
            log(f"  Total tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion")
            log(f"  Total latency: {total_latency:.0f}ms")

except Exception as e:
    log(f"  ERROR during LLM validation: {e}")
    results["errors"].append(f"LLM validation: {e}")

# ─── Phase 12: Validate Burr observability ──────────────────────────────

log("\nPhase 12: Validating Burr observability (tracking)...")
try:
    burr_dir = Path.home() / ".burr"
    if burr_dir.exists():
        project_dirs = [d for d in burr_dir.iterdir() if d.is_dir()]
        log(f"  Burr project directories: {len(project_dirs)}")
        for d in project_dirs:
            log(f"    - {d.name}")

        all_tracking_files = list(burr_dir.rglob("*"))
        log(f"  Total tracking entries: {len(all_tracking_files)}")

        record("burr_unique", "observability", {
            "tracking_enabled": True,
            "tracking_dir": str(burr_dir),
            "project_dirs": [d.name for d in project_dirs],
            "total_files": len(all_tracking_files),
        })
    else:
        log("  Burr tracking directory not found")
        record("burr_unique", "observability", {
            "tracking_enabled": False,
            "tracking_dir": str(burr_dir),
        })

except Exception as e:
    log(f"  ERROR during observability validation: {e}")
    results["errors"].append(f"Observability validation: {e}")
    record("burr_unique", "observability", {"error": str(e)})

# ─── Phase 13: Post-remediation assessment ──────────────────────────────

log("\nPhase 13: Running post-remediation assessment...")
try:
    post_pipeline = AssessmentPipeline()
    post_assessment = post_pipeline.run(
        artifacts=artifacts,
        source=str(FASTAPI_PATH),
        git_commit=git_commit,
        relationships=relationships,
    )
    log(f"  Post-remediation score: {post_assessment.score}")
    log(f"  Post-remediation signals: {len(post_assessment.signals)}")

    diff = assessment_store._compute_diff(baseline_assessment, post_assessment)
    log(f"  Diff: score {diff.prev_score} -> {diff.curr_score} (delta={diff.score_delta})")
    log(f"  New signals: {len(diff.new_signals)}")
    log(f"  Resolved signals: {len(diff.resolved_signals)}")
    log(f"  Persistent signals: {len(diff.persistent_signals)}")

    record("post_remediation", "score", post_assessment.score)
    record("post_remediation", "signal_count", len(post_assessment.signals))
    record("post_remediation", "diff", {
        "prev_score": diff.prev_score,
        "curr_score": diff.curr_score,
        "score_delta": diff.score_delta,
        "new_signals": len(diff.new_signals),
        "resolved_signals": len(diff.resolved_signals),
        "persistent_signals": len(diff.persistent_signals),
        "dimension_deltas": diff.dimension_deltas,
        "recommendation": diff.recommendation,
    })
    record("post_remediation", "note", "Dry-run mode: executor simulated changes, repo unchanged")

except Exception as e:
    log(f"  ERROR during post-remediation assessment: {e}")
    import traceback
    traceback.print_exc()
    results["errors"].append(f"Post-remediation: {e}")
    record("post_remediation", "error", str(e))

# ─── Save results ───────────────────────────────────────────────────────

results["validation_end"] = datetime.now(timezone.utc).isoformat()

log("\n" + "=" * 70)
log("VALIDATION COMPLETE")
log("=" * 70)

log(f"\nSummary:")
log(f"  Baseline score: {results.get('baseline', {}).get('score', '?')}/100")
log(f"  Signals analyzed: {results.get('baseline', {}).get('signal_count', '?')}")
log(f"  Workflow reached approval: {results.get('burr_workflow', {}).get('reached_approval', '?')}")
log(f"  LLM calls: {len(llm_calls)} ({sum(1 for c in llm_calls if 'error' not in c)} successful)")
log(f"  Errors: {len(results['errors'])}")
for err in results["errors"]:
    log(f"    - {err}")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nResults saved to: {RESULTS_FILE}")
