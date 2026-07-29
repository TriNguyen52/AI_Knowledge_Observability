"""Burr actions for the knowledge improvement workflow.

Each action declares reads/writes and implements the Burr action contract.
LLM calls happen inside actions via LLMGateway. AI-Ready stores are
accessed for data retrieval and verification.

Actions:
  analyze_issue     — LLM analyzes signals to determine root causes
  generate_proposal — LLM generates improvement strategies
  review_approval    — Human-in-the-loop checkpoint (halt/resume)
  execute_change    — AI-Ready executor applies approved modifications
  verify_improvement — AI-Ready assessment pipeline verifies the result
  handle_failure    — Records failure, prepares state for potential forking
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from burr.core import action, State

from ai_ready.improvement.models import (
    RemediationStatus,
    VerificationOutcome,
    RootCauseHypothesis,
    KnowledgeProblem,
    ProblemVerification,
    DecisionTrace,
    RemediationProposal,
    VerificationResult,
)
from ai_ready.improvement.salience import (
    check_signal_delta,
    discover_problems_heuristic,
    rank_problems_by_salience,
    SkipEvent,
    DEFAULT_SALIENCE_THRESHOLD,
)
from ai_ready.improvement.diagnosis_quality import DiagnosisQualityTracker
from ai_ready.improvement.prompts import (
    build_signal_context,
    build_artifact_context,
    build_history_context,
    cluster_signals,
    build_assessment_summary,
)
from ai_ready.improvement.heuristics import (
    heuristic_analysis,
    heuristic_problems,
    heuristic_proposal,
    reject_generic_proposals,
    parse_hypotheses_and_problems,
    parse_proposals,
)
from ai_ready.llm.base import LLMMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action: Analyze Issue
# ---------------------------------------------------------------------------

@action(reads=["signal_ids", "affected_artifact_uris", "assessment_id", "prior_failures",
               "prior_diagnosis", "prior_strategy", "prior_verification", "cumulative_tokens",
               "last_analysis_signal_ids", "skip_events"],
        writes=["root_cause_analysis", "knowledge_problems", "current_stage",
                "llm_metadata", "decision_trace", "assessment_summary", "cumulative_tokens",
                "last_analysis_signal_ids", "skip_events", "problem_saliences"])
def analyze_issue(state: State, llm_gateway: Any = None, assessment_store: Any = None,
                  history_store: Any = None,
                  diagnosis_quality_tracker: Any = None) -> State:
    """Analyze assessment signals and determine root causes.

    Uses the LLM to reason about why the knowledge quality issues exist.
    The LLM receives signal descriptions and artifact references — it
    never accesses the knowledge base directly.

    Identifies Knowledge Problems (underlying issues that produce
    signals) rather than just listing signals. Multiple signals can
    stem from the same problem.

    Retrieves historical context from prior remediation outcomes for
    similar issue types, making institutional memory actionable.

    If this is a forked retry, includes prior diagnosis, strategy, and
    verification from the failed attempt so the LLM can reason about
    why the prior approach failed.

    Writes:
      root_cause_analysis: list of RootCauseHypothesis dicts
      knowledge_problems: list of KnowledgeProblem dicts
      current_stage: "root_cause_found" or "failed_analysis"
      llm_metadata: updated with call info
      decision_trace: updated with analysis-stage reasoning
    """
    signal_ids = state["signal_ids"]
    artifact_uris = state["affected_artifact_uris"]
    prior_failures = state.get("prior_failures", [])
    prior_diagnosis = state.get("prior_diagnosis", {})
    prior_strategy = state.get("prior_strategy", {})
    prior_verification = state.get("prior_verification", {})
    cumulative_tokens = state.get("cumulative_tokens", 0)
    last_analysis_signal_ids = state.get("last_analysis_signal_ids", [])
    skip_events = list(state.get("skip_events", []))

    # --- Signal-Delta Eligibility Gate (deterministic, no LLM) ---
    # Only invoke LLM analysis when new signals have appeared since the
    # last analysis. This prevents wasted LLM calls on unchanged state.
    eligibility = check_signal_delta(signal_ids, last_analysis_signal_ids if last_analysis_signal_ids else None)

    # --- Heuristic Pre-Analysis (deterministic, no LLM) ---
    # Always run heuristic discovery as a baseline. This provides problem
    # candidates even without an LLM. When an LLM is available, it can
    # enhance these with richer root cause hypotheses.
    assessment = assessment_store.latest() if assessment_store else None
    heuristic_problems_result, heuristic_hypotheses = discover_problems_heuristic(signal_ids, assessment)

    # Cluster signals and build a compact reusable assessment summary
    clusters = cluster_signals(signal_ids, assessment_store)
    assessment_summary = build_assessment_summary(
        signal_ids, artifact_uris, assessment_store, clusters
    )

    # If not eligible (no new signals), use heuristic analysis only and skip LLM
    if not eligibility.eligible and heuristic_problems_result:
        skip_events.append(SkipEvent(
            lane="llm_analysis",
            reason=eligibility.reason,
            context={"signal_count": len(signal_ids), "heuristic_problem_count": len(heuristic_problems_result)},
        ).to_dict())

        # Rank heuristic problems by salience (deterministic)
        ranked_problems, saliences, skipped_problems = rank_problems_by_salience(
            heuristic_problems_result, assessment, history_store,
        )

        # Record skipped low-salience problems
        for sp in skipped_problems:
            skip_events.append(SkipEvent(
                lane="problem_processing",
                reason=f"Low salience problem skipped: {sp.category}",
                context={"problem_id": sp.problem_id},
            ).to_dict())

        decision_trace = DecisionTrace.from_dict(state.get("decision_trace", {}))
        decision_trace.knowledge_problems = [kp.to_dict() for kp in ranked_problems]
        decision_trace.supporting_evidence = [
            {"hypothesis": h.hypothesis, "evidence": h.evidence, "confidence": h.confidence}
            for h in heuristic_hypotheses
        ]
        decision_trace.root_cause_reasoning = (
            f"Heuristic analysis only (no LLM — {eligibility.reason}). "
            f"Identified {len(heuristic_hypotheses)} hypothesis(es) and "
            f"{len(ranked_problems)} knowledge problem(s) from {len(signal_ids)} signal(s). "
            f"Salience ranking: {len(saliences)} above threshold, {len(skipped_problems)} skipped."
        )

        return state.update(
            root_cause_analysis=[h.to_dict() for h in heuristic_hypotheses],
            knowledge_problems=[kp.to_dict() for kp in ranked_problems],
            current_stage=RemediationStatus.ROOT_CAUSE_FOUND.value if ranked_problems
                          else RemediationStatus.FAILED_ANALYSIS.value,
            llm_metadata={**state.get("llm_metadata", {}), "analyze_issue": {
                "provider": "none", "model": "heuristic", "reason": eligibility.reason,
            }},
            decision_trace=decision_trace.to_dict(),
            assessment_summary=assessment_summary,
            cumulative_tokens=cumulative_tokens,
            last_analysis_signal_ids=signal_ids,
            skip_events=skip_events,
            problem_saliences=[s.to_dict() for s in saliences],
        )

    # Use clustered signal context (representative examples) instead of raw signal list
    signal_context = "\n".join(
        f"  [{c['count']}x] {c['representative']}" if c["count"] > 1 else f"  {c['representative']}"
        for c in clusters
    ) if clusters else build_signal_context(signal_ids, assessment_store)

    artifact_context = build_artifact_context(artifact_uris, assessment_store)
    history_context = build_history_context(prior_failures, history_store)

    # Build prior attempt context for forked retries
    prior_attempt_context = ""
    if prior_diagnosis or prior_strategy:
        prior_attempt_context = f"""## Prior Failed Attempt (Forked Retry)

### Prior Diagnosis
{json.dumps(prior_diagnosis, indent=2, default=str) if prior_diagnosis else "None"}

### Prior Strategy That Failed
{json.dumps(prior_strategy, indent=2, default=str) if prior_strategy else "None"}

### Prior Verification Result
{json.dumps(prior_verification, indent=2, default=str) if prior_verification else "None"}

IMPORTANT: The prior strategy above FAILED. You must identify why it failed and propose a DIFFERENT root cause or a different approach to the same root cause."""

    # Retrieve historical remediation context
    remediation_context_str = ""
    if history_store:
        try:
            # Use artifact URIs for similarity matching if available
            remediation_context = history_store.get_remediation_context(
                issue_type="",  # Will be filled after analysis; use recent for now
                artifact_uris=artifact_uris,
            )
            if remediation_context.get("total_prior_attempts", 0) > 0:
                remediation_context_str = f"""## Historical Remediation Outcomes
{json.dumps(remediation_context, indent=2, default=str)}"""
        except Exception:
            pass

    # Enhanced system prompt — distinguish observations from causes,
    # reason about tradeoffs, focus on improving the knowledge system
    system_prompt = """You are a knowledge quality analyst. You analyze signals from a knowledge assessment system and determine root causes for quality issues.

Your job:
1. Distinguish OBSERVATIONS (what the signals show) from CAUSES (why the signals exist). Signals are symptoms; your job is to find the underlying disease.
2. Identify the most likely root cause(s) — the underlying issues in the knowledge system that produce these signals
3. Provide evidence from the signals to support each hypothesis — cite specific signal IDs and evidence
4. Assign a confidence score (0.0 to 1.0) to each hypothesis — be honest about uncertainty
5. Categorize the root cause (e.g., "missing_metadata", "poor_structure", "duplicate_content", "broken_links", "missing_context")
6. For each root cause, identify which specific signals it explains (signal_ids)
7. Group signals into Knowledge Problems — each problem is an underlying issue that produces one or more signals
8. Discuss tradeoffs: if multiple root causes are possible, explain why you prefer one over another

Think about the knowledge system holistically. Do not just list signals — explain WHY they exist. A signal like "document has no title" is an observation; the root cause might be "the ingestion pipeline doesn't enforce metadata requirements."

Respond in JSON format:
{
  "hypotheses": [
    {
      "hypothesis": "Description of the root cause — the underlying system issue",
      "evidence": ["Evidence point 1 from signal X", "Evidence point 2 from signal Y"],
      "confidence": 0.85,
      "affected_artifact_uris": ["uri1", "uri2"],
      "category": "missing_metadata",
      "signal_ids": ["sig_1", "sig_2"]
    }
  ],
  "knowledge_problems": [
    {
      "category": "missing_metadata",
      "description": "Documents lack topic tags, causing retrieval failures",
      "root_cause_idx": 0,
      "signal_ids": ["sig_1", "sig_2"],
      "artifact_uris": ["uri1"],
      "evidence_summary": "Both signals reference the same artifact and indicate missing metadata"
    }
  ]
}

Be specific and evidence-based. Do not speculate beyond what the signals show.
If prior attempts failed, explain WHY the prior strategy failed and how your analysis differs."""

    # Diagnosis quality calibration (deterministic feedback loop)
    calibration_text = ""
    if diagnosis_quality_tracker:
        try:
            calibration_text = diagnosis_quality_tracker.get_calibration_text()
            if calibration_text:
                calibration_text = f"## Diagnosis Quality Calibration\n{calibration_text}\n"
        except Exception:
            pass

    user_prompt = f"""Analyze the following knowledge quality signals and determine root causes.

## Assessment Summary (clustered signals — representative shown with cluster count)
{assessment_summary}

## Affected Artifacts
{artifact_context}

## Prior Improvement History
{history_context}

{prior_attempt_context}

{remediation_context_str}

{calibration_text}
Provide your root cause analysis as JSON. Remember: distinguish observations from causes."""

    try:
        if llm_gateway is None:
            # No LLM available — produce a basic heuristic analysis
            hypotheses = [heuristic_analysis(signal_ids, artifact_uris, assessment_store)]
            knowledge_problems = heuristic_problems(hypotheses, signal_ids, artifact_uris)
            llm_meta = {"provider": "none", "model": "heuristic"}
        else:
            response = llm_gateway.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                temperature=0.3,
                max_tokens=2048,
            )

            hypotheses, knowledge_problems = parse_hypotheses_and_problems(
                response.content, signal_ids, artifact_uris
            )
            llm_meta = {
                "provider": response.provider,
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_ms": response.latency_ms,
            }

        # Track cumulative tokens across all actions
        action_tokens = (llm_meta.get("prompt_tokens", 0) or 0) + (llm_meta.get("completion_tokens", 0) or 0)
        cumulative_tokens += action_tokens

        # Populate DecisionTrace with analysis-stage reasoning
        decision_trace = DecisionTrace.from_dict(state.get("decision_trace", {}))
        decision_trace.knowledge_problems = [kp.to_dict() for kp in knowledge_problems]
        decision_trace.supporting_evidence = [
            {"hypothesis": h.hypothesis, "evidence": h.evidence, "confidence": h.confidence}
            for h in hypotheses
        ]
        decision_trace.root_cause_reasoning = (
            f"Identified {len(hypotheses)} root cause hypothesis(es) and "
            f"{len(knowledge_problems)} knowledge problem(s) from {len(signal_ids)} signal(s) "
            f"in {len(clusters)} cluster(s)."
        )
        if prior_attempt_context:
            decision_trace.root_cause_reasoning += (
                " Prior attempt context was provided for forked retry."
            )

        # --- Salience Ranking (deterministic, no LLM) ---
        # Rank knowledge problems by quantitative salience score and
        # filter out low-salience ones to reduce token waste.
        ranked_problems, saliences, skipped_problems = rank_problems_by_salience(
            knowledge_problems, assessment, history_store,
        )
        for sp in skipped_problems:
            skip_events.append(SkipEvent(
                lane="problem_processing",
                reason=f"Low salience problem skipped: {sp.category}",
                context={"problem_id": sp.problem_id},
            ).to_dict())

        decision_trace.root_cause_reasoning += (
            f" Salience ranking: {len(saliences)} above threshold, "
            f"{len(skipped_problems)} skipped."
        )

        updates = {
            "root_cause_analysis": [h.to_dict() for h in hypotheses],
            "knowledge_problems": [kp.to_dict() for kp in ranked_problems],
            "current_stage": RemediationStatus.ROOT_CAUSE_FOUND.value,
            "llm_metadata": {**state.get("llm_metadata", {}), "analyze_issue": llm_meta},
            "decision_trace": decision_trace.to_dict(),
            "assessment_summary": assessment_summary,
            "cumulative_tokens": cumulative_tokens,
            "last_analysis_signal_ids": signal_ids,
            "skip_events": skip_events,
            "problem_saliences": [s.to_dict() for s in saliences],
        }

        if not hypotheses or not ranked_problems:
            updates["current_stage"] = RemediationStatus.FAILED_ANALYSIS.value

        return state.update(**updates)

    except Exception as e:
        logger.error(f"analyze_issue failed: {e}")
        return state.update(
            current_stage=RemediationStatus.FAILED_ANALYSIS.value,
            llm_metadata={**state.get("llm_metadata", {}), "analyze_issue": {"error": str(e)}},
            last_analysis_signal_ids=signal_ids,
            skip_events=skip_events,
        )


# ---------------------------------------------------------------------------
# Action: Generate Proposal
# ---------------------------------------------------------------------------

@action(reads=["root_cause_analysis", "knowledge_problems", "affected_artifact_uris",
               "prior_failures", "assessment_id", "prior_diagnosis", "prior_strategy",
               "prior_verification", "decision_trace", "assessment_summary", "cumulative_tokens",
               "problem_saliences"],
        writes=["proposals", "selected_proposal_idx", "current_stage", "llm_metadata",
                "decision_trace", "cumulative_tokens"])
def generate_proposal(state: State, llm_gateway: Any = None, history_store: Any = None) -> State:
    """Generate improvement strategies based on root cause analysis.

    Retrieves structured historical context (successful/failed strategies,
    strategy stats) from the history store and passes it to the LLM.

    Provides 3-source context to the LLM:
      1. Root cause analysis + Knowledge Problems (what's wrong)
      2. Historical remediation outcomes (what worked/failed before)
      3. Prior failed attempts from forking (what to avoid)

    Writes:
      proposals: list of RemediationProposal dicts
      selected_proposal_idx: 0 (first proposal by default)
      current_stage: "proposal_generated"
      llm_metadata: updated with call info
      decision_trace: updated with proposal-stage reasoning
    """
    root_causes = state.get("root_cause_analysis", [])
    knowledge_problems = state.get("knowledge_problems", [])
    artifact_uris = state["affected_artifact_uris"]
    prior_failures = state.get("prior_failures", [])
    prior_diagnosis = state.get("prior_diagnosis", {})
    prior_strategy = state.get("prior_strategy", {})
    cumulative_tokens = state.get("cumulative_tokens", 0)

    # Reuse the assessment summary from analyze_issue instead of
    # rebuilding context from scratch. Fall back to building it if missing.
    assessment_summary = state.get("assessment_summary", "")
    if not assessment_summary:
        assessment_summary = build_assessment_summary(
            state.get("signal_ids", []), artifact_uris, None
        )

    # Build context for LLM — keep prompts compact to avoid rate limits
    root_cause_text = json.dumps(root_causes, indent=2, default=str)[:1500]
    problems_text = json.dumps(knowledge_problems, indent=2, default=str)[:1000]
    prior_strategies = [f.get("strategy", "") for f in prior_failures]
    prior_text = "\n".join(
        f"- Attempt #{f.get('attempt_number', '?')}: strategy={f.get('strategy', '?')}, "
        f"result={f.get('result', '?')}, reason={f.get('failure_reason', '?')}"
        for f in prior_failures
    ) or "No prior attempts."

    # Include salience ranking so the LLM knows which problems are most important
    salience_scores = state.get("problem_saliences", [])
    salience_text = ""
    if salience_scores:
        salience_lines = [
            f"- {s['problem_id']}: total={s['total']:.4f} "
            f"(encoding={s['encoding']:.2f}, outcome={s['outcome']:.2f}, "
            f"retrieval={s['retrieval']:.2f})"
            for s in salience_scores
        ]
        salience_text = f"""## Problem Salience Ranking (deterministic)
Problems are ranked by quantitative salience score. Focus on highest-salience problems first.
{chr(10).join(salience_lines)}
"""

    # Prior forked attempt context (compact — just strategy name and why it failed)
    prior_attempt_text = ""
    if prior_strategy:
        prior_strategy_summary = (
            f"Strategy: {prior_strategy.get('strategy', 'unknown')}, "
            f"Description: {prior_strategy.get('description', 'N/A')[:200]}"
        )
        prior_attempt_text = f"""## Prior Failed Strategy (Forked Retry)
The following strategy was tried in a prior attempt and FAILED:
{prior_strategy_summary}

You MUST propose a DIFFERENT strategy. Explain why your proposal differs from the prior failed attempt."""

    # Get structured historical context from history store
    remediation_context_str = "No historical context available."
    ema_context_str = ""
    if history_store and root_causes:
        issue_type = root_causes[0].get("category", "")
        if issue_type:
            try:
                remediation_context = history_store.get_remediation_context(
                    issue_type=issue_type,
                    artifact_uris=artifact_uris,
                )
                if remediation_context.get("total_prior_attempts", 0) > 0:
                    remediation_context_str = json.dumps(remediation_context, indent=2, default=str)[:1500]
                else:
                    remediation_context_str = "No prior outcomes for this issue type."

                # EMA-based strategy scoring (deterministic, no LLM)
                ema_data = remediation_context.get("strategy_ema", {})
                ema_strategies = ema_data.get("strategies", [])
                if ema_strategies:
                    ema_lines = [
                        f"- {s['strategy']}: EMA={s['adjusted_ema']:.2f} "
                        f"(observations={s['observation_count']}, "
                        f"last={s['last_outcome']})"
                        for s in ema_strategies
                    ]
                    ema_context_str = f"""## Strategy EMA Scores (deterministic — learned from history)
Strategies ranked by predicted success probability (EMA). Higher = more likely to succeed.
Recent outcomes weigh more than old ones. Diversity floor prevents rare strategies from being suppressed.
{chr(10).join(ema_lines)}

PREFER strategies with EMA > 0.5. AVOID strategies with EMA < 0.3 unless you have a specific reason.
"""
            except Exception:
                remediation_context_str = "History store unavailable."

    # Enhanced system prompt — focus on improving the knowledge system,
    # not just fixing signals. Ask for tradeoffs, minimal modifications,
    # confidence, assumptions, rollback considerations.
    system_prompt = """You are a knowledge improvement strategist. Given root cause analysis of knowledge quality issues, propose specific improvement strategies.

Your goal: improve the OVERALL KNOWLEDGE SYSTEM, not just fix individual signals. Signals are symptoms — your strategies should address the underlying system issues that produce them.

Your job:
1. Propose 1-3 concrete strategies for improving the knowledge quality system
2. For each strategy, describe specific modification steps — recommend MINIMAL modifications that maximize long-term knowledge quality
3. Estimate the expected impact on quality scores — be specific and realistic
4. List potential risks — including the risk of making things worse
5. AVOID strategies that have failed in prior attempts — never retry the same strategy under a different name
6. PREFER strategies that have succeeded historically for similar issues
7. Explain your reasoning for each strategy — why this strategy, why not alternatives, what tradeoffs you considered
8. For each strategy, state your confidence (0.0 to 1.0) that it will work
9. List assumptions your strategy depends on — what must be true for this to work
10. Describe how to roll back the change if it makes things worse

Each strategy should include modification_steps as a list of dicts with:
  - step_type: one of "update_document", "add_metadata", "create_relationship", "remove_relationship", "archive_artifact", "merge_artifacts", "split_artifact"
  - artifact_uri: which artifact to modify
  - description: what to do
  - parameters: step-specific parameters

Respond in JSON format:
{
  "proposals": [
    {
      "strategy": "strategy_name",
      "description": "What this strategy does and why it improves the knowledge system",
      "expected_impact": "e.g., +15 retrieval score, 3 signals resolved",
      "risks": ["risk 1", "risk 2"],
      "affected_artifact_uris": ["uri1"],
      "modification_steps": [
        {"step_type": "add_metadata", "artifact_uri": "uri1", "description": "Add topic tags", "parameters": {}}
      ],
      "root_cause_idx": 0,
      "reasoning": "Why this strategy was chosen over alternatives, what tradeoffs were considered",
      "affected_dimensions": ["completeness", "accuracy"],
      "confidence": 0.8,
      "assumptions": ["The documents are still actively maintained", "Metadata schema supports topic tags"],
      "rollback_considerations": "Remove the added metadata tags to revert to prior state"
    }
  ]
}"""

    user_prompt = f"""Generate improvement proposals for the following root causes.

## Assessment Summary (reused from analysis)
{assessment_summary}

## Root Cause Analysis (Source 1: What's Wrong)
{root_cause_text}

## Knowledge Problems
{problems_text}

{salience_text}

## Affected Artifacts
{json.dumps(artifact_uris, indent=2)}

## Prior Failed Attempts — AVOID These (Source 3: What to Avoid)
{prior_text}

{prior_attempt_text}

## Historical Remediation Outcomes (Source 2: What Worked Before)
{remediation_context_str}

{ema_context_str}
Propose 1-3 improvement strategies as JSON. Focus on improving the knowledge system, not just fixing signals. Include reasoning, confidence, assumptions, and rollback considerations."""

    try:
        if llm_gateway is None:
            proposals = [heuristic_proposal(root_causes, artifact_uris, prior_strategies)]
            llm_meta = {"provider": "none", "model": "heuristic"}
        else:
            response = llm_gateway.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                temperature=0.4,
                max_tokens=4096,
            )

            proposals = parse_proposals(response.content)
            llm_meta = {
                "provider": response.provider,
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_ms": response.latency_ms,
            }

        # Reject generic proposals — strategies that are too vague
        # to be actionable are filtered out
        proposals = reject_generic_proposals(proposals, prior_strategies)

        # Track cumulative tokens across all actions
        action_tokens = (llm_meta.get("prompt_tokens", 0) or 0) + (llm_meta.get("completion_tokens", 0) or 0)
        cumulative_tokens += action_tokens

        # Update DecisionTrace with proposal-stage reasoning
        decision_trace = DecisionTrace.from_dict(state.get("decision_trace", {}))
        decision_trace.historical_context_used = remediation_context_str[:500]
        if proposals:
            decision_trace.proposal_reasoning = (
                f"Generated {len(proposals)} proposal(s). "
                f"Top strategy: {proposals[0].strategy}. "
                f"Reasoning: {getattr(proposals[0], 'reasoning', 'N/A')}"
            )

        updates = {
            "proposals": [p.to_dict() for p in proposals],
            "selected_proposal_idx": 0,
            "current_stage": RemediationStatus.PROPOSAL_GENERATED.value,
            "llm_metadata": {**state.get("llm_metadata", {}), "generate_proposal": llm_meta},
            "decision_trace": decision_trace.to_dict(),
            "cumulative_tokens": cumulative_tokens,
        }

        if not proposals:
            updates["current_stage"] = RemediationStatus.FAILED_ANALYSIS.value

        return state.update(**updates)

    except Exception as e:
        logger.error(f"generate_proposal failed: {e}")
        return state.update(
            current_stage=RemediationStatus.FAILED_ANALYSIS.value,
            llm_metadata={**state.get("llm_metadata", {}), "generate_proposal": {"error": str(e)}},
        )


# ---------------------------------------------------------------------------
# Action: Review Approval
# ---------------------------------------------------------------------------

@action(reads=["proposals", "selected_proposal_idx", "root_cause_analysis",
               "assessment_id", "decision_trace"],
        writes=["current_stage", "approval_status", "decision_trace"])
def review_approval(state: State) -> State:
    """Human-in-the-loop approval checkpoint.

    This action prepares the approval request and sets the stage to
    WAITING_FOR_APPROVAL. The Burr application halts after this action.
    When resumed, the approval_status will have been set externally
    (by the approval callback) to "approved" or "rejected".

    Writes:
      current_stage: "waiting_for_approval"
      decision_trace: updated with approval-stage info
    """
    proposals = state.get("proposals", [])
    selected_idx = state.get("selected_proposal_idx", 0)

    if not proposals:
        return state.update(current_stage=RemediationStatus.FAILED_ANALYSIS.value)

    # Record what's being submitted for approval
    decision_trace = DecisionTrace.from_dict(state.get("decision_trace", {}))
    selected_proposal = proposals[selected_idx] if selected_idx < len(proposals) else {}
    decision_trace.approval_decision = (
        f"Pending approval for strategy '{selected_proposal.get('strategy', 'unknown')}': "
        f"{selected_proposal.get('description', 'N/A')}"
    )

    return state.update(
        current_stage=RemediationStatus.WAITING_FOR_APPROVAL.value,
        approval_status="pending",
        decision_trace=decision_trace.to_dict(),
    )


# ---------------------------------------------------------------------------
# Action: Execute Change
# ---------------------------------------------------------------------------

@action(reads=["proposals", "selected_proposal_idx", "approval_status", "affected_artifact_uris"],
        writes=["execution_history", "current_stage"])
def execute_change(state: State, executor: Any = None) -> State:
    """Execute approved knowledge modifications.

    Calls AI-Ready's KnowledgeExecutor to apply the approved proposal's
    modification steps. The executor belongs to AI-Ready — Burr only
    orchestrates the call.

    Writes:
      execution_history: list of ExecutionStep dicts
      current_stage: "executing" or "failed_execution"
    """
    approval_status = state.get("approval_status", "pending")
    if approval_status != "approved":
        return state.update(current_stage=RemediationStatus.REJECTED.value)

    proposals = state.get("proposals", [])
    selected_idx = state.get("selected_proposal_idx", 0)

    if selected_idx >= len(proposals):
        return state.update(current_stage=RemediationStatus.FAILED_EXECUTION.value)

    proposal = proposals[selected_idx]
    modification_steps = proposal.get("modification_steps", [])
    artifact_uris = state.get("affected_artifact_uris", [])

    if executor is None:
        # Safety-net fallback: when no executor is injected (e.g., direct
        # action calls in tests or validation scripts), create a dry-run
        # executor. The manager always injects a real executor via
        # _register_dependencies, so this path only triggers outside
        # the normal workflow.
        from ai_ready.improvement.executor import KnowledgeExecutor
        executor = KnowledgeExecutor()  # Dry-run mode

    try:
        execution_steps = executor.execute(modification_steps, artifact_uris)
        all_succeeded = all(step.success for step in execution_steps)

        updates = {
            "execution_history": [step.to_dict() for step in execution_steps],
            "current_stage": RemediationStatus.EXECUTING.value if all_succeeded
                             else RemediationStatus.FAILED_EXECUTION.value,
        }
        return state.update(**updates)

    except Exception as e:
        logger.error(f"execute_change failed: {e}")
        return state.update(
            current_stage=RemediationStatus.FAILED_EXECUTION.value,
            execution_history=[{"error": str(e), "success": False}],
        )


# ---------------------------------------------------------------------------
# Action: Verify Improvement
# ---------------------------------------------------------------------------

@action(reads=["execution_history", "assessment_id", "signal_ids", "affected_artifact_uris",
               "proposals", "selected_proposal_idx", "knowledge_problems", "decision_trace",
               "root_cause_analysis"],
        writes=["verification_results", "current_stage", "decision_trace"])
def verify_improvement(state: State, assessment_store: Any = None,
                       assessment_pipeline: Any = None,
                       artifacts: list[Any] = None,
                       relationships: list[Any] = None,
                       source: str = "",
                       git_commit: str = "",
                       diagnosis_quality_tracker: Any = None) -> State:
    """Verify whether the improvement worked by re-running the assessment.

    Verification operates at the Knowledge Problem level, not individual
    signal level. For each Knowledge Problem identified during analysis,
    we check whether its associated signals are resolved, partially
    resolved, unchanged, regressed, or misdiagnosed.

    Writes:
      verification_results: VerificationResult dict with problem_verifications
      current_stage: "completed" or "failed_verification"
      decision_trace: updated with verification-stage reasoning
    """
    execution_history = state.get("execution_history", [])
    if not execution_history or not all(step.get("success", False) for step in execution_history):
        return state.update(
            current_stage=RemediationStatus.FAILED_VERIFICATION.value,
            verification_results=VerificationResult(
                success=False, summary="Execution failed — cannot verify",
                failure_explanation="One or more execution steps failed. "
                    "The proposed modifications were not fully applied to the artifacts.",
            ).to_dict(),
        )

    # Get the before-assessment
    before_assessment = None
    if assessment_store:
        before_assessment = assessment_store.load(state["assessment_id"])
    if not before_assessment and assessment_store:
        before_assessment = assessment_store.latest()

    if not before_assessment:
        return state.update(
            current_stage=RemediationStatus.FAILED_VERIFICATION.value,
            verification_results=VerificationResult(
                success=False, summary="Could not load before-assessment",
                failure_explanation="The assessment that triggered this workflow "
                    "could not be loaded. Cannot compare before/after states.",
            ).to_dict(),
        )

    before_score = before_assessment.score
    before_signal_ids = {s.signal_id for s in before_assessment.signals}
    target_signal_ids = set(state.get("signal_ids", []))
    knowledge_problems = state.get("knowledge_problems", [])

    # Run a new assessment to get the after-state
    if assessment_pipeline is None or artifacts is None:
        # Cannot re-assess — mark as completed with caveat
        return state.update(
            current_stage=RemediationStatus.COMPLETED.value,
            verification_results=VerificationResult(
                success=True,
                before_score=before_score,
                summary="Verification skipped — no assessment pipeline or artifacts provided",
                failure_explanation="No assessment pipeline or artifacts were provided, "
                    "so verification could not be performed. Marked as completed by default.",
            ).to_dict(),
        )

    try:
        after_assessment = assessment_pipeline.run(
            artifacts=artifacts,
            source=source,
            git_commit=git_commit,
            relationships=relationships or [],
        )

        after_score = after_assessment.score
        after_signal_ids = {s.signal_id for s in after_assessment.signals}

        # Compute resolved/new/remaining signals (signal-level, for backward compat)
        resolved_ids = list(before_signal_ids - after_signal_ids)
        new_ids = list(after_signal_ids - before_signal_ids)
        remaining_target_ids = list(target_signal_ids & after_signal_ids)

        # Dimension deltas
        dim_deltas: dict[str, int] = {}
        for dim_name in set(list(before_assessment.dimensions.keys()) +
                           list(after_assessment.dimensions.keys())):
            prev_s = before_assessment.dimensions[dim_name].score if dim_name in before_assessment.dimensions else 0
            curr_s = after_assessment.dimensions[dim_name].score if dim_name in after_assessment.dimensions else 0
            dim_deltas[dim_name] = curr_s - prev_s

        score_diff = after_score - before_score

        # Problem-level verification
        problem_verifications = []
        for kp_dict in knowledge_problems:
            kp = KnowledgeProblem.from_dict(kp_dict)
            kp_signal_ids = set(kp.signal_ids)
            before_kp_signals = kp_signal_ids & before_signal_ids
            after_kp_signals = kp_signal_ids & after_signal_ids
            resolved_kp_signals = before_kp_signals - after_kp_signals
            remaining_kp_signals = before_kp_signals & after_kp_signals
            new_kp_signals = after_kp_signals - before_kp_signals

            # Determine outcome
            if not before_kp_signals:
                outcome = VerificationOutcome.MISDIAGNOSED.value
                explanation = (
                    f"Problem '{kp.problem_id}' had no signals in the before-assessment. "
                    f"The root cause diagnosis may have been incorrect."
                )
            elif len(remaining_kp_signals) == 0 and len(new_kp_signals) == 0:
                outcome = VerificationOutcome.RESOLVED.value
                explanation = (
                    f"Problem '{kp.problem_id}' ({kp.category}): all {len(before_kp_signals)} "
                    f"signal(s) resolved. Signals gone: {resolved_kp_signals}."
                )
            elif len(remaining_kp_signals) < len(before_kp_signals):
                outcome = VerificationOutcome.PARTIALLY_RESOLVED.value
                explanation = (
                    f"Problem '{kp.problem_id}' ({kp.category}): {len(resolved_kp_signals)} "
                    f"of {len(before_kp_signals)} signal(s) resolved. "
                    f"Remaining: {remaining_kp_signals}."
                )
            elif len(new_kp_signals) > 0 and len(remaining_kp_signals) == len(before_kp_signals):
                outcome = VerificationOutcome.REGRESSED.value
                explanation = (
                    f"Problem '{kp.problem_id}' ({kp.category}): no signals resolved, "
                    f"and {len(new_kp_signals)} new signal(s) appeared. "
                    f"Remediation may have made things worse."
                )
            else:
                outcome = VerificationOutcome.UNCHANGED.value
                explanation = (
                    f"Problem '{kp.problem_id}' ({kp.category}): no change. "
                    f"All {len(remaining_kp_signals)} signal(s) remain."
                )

            pv = ProblemVerification(
                problem_id=kp.problem_id,
                outcome=outcome,
                before_signal_count=len(before_kp_signals),
                after_signal_count=len(after_kp_signals),
                resolved_signal_ids=list(resolved_kp_signals),
                remaining_signal_ids=list(remaining_kp_signals),
                new_signal_ids=list(new_kp_signals),
                explanation=explanation,
            )
            problem_verifications.append(pv.to_dict())

        # Determine overall outcome
        if not problem_verifications:
            # No knowledge problems identified — fall back to signal-level
            overall_outcome = (
                VerificationOutcome.RESOLVED.value
                if len(remaining_target_ids) < len(target_signal_ids)
                else VerificationOutcome.UNCHANGED.value
            )
        else:
            outcomes = [pv["outcome"] for pv in problem_verifications]
            if all(o == VerificationOutcome.RESOLVED.value for o in outcomes):
                overall_outcome = VerificationOutcome.RESOLVED.value
            elif any(o == VerificationOutcome.REGRESSED.value for o in outcomes):
                overall_outcome = VerificationOutcome.REGRESSED.value
            elif any(o == VerificationOutcome.MISDIAGNOSED.value for o in outcomes):
                overall_outcome = VerificationOutcome.MISDIAGNOSED.value
            elif all(o in (VerificationOutcome.RESOLVED.value,
                          VerificationOutcome.PARTIALLY_RESOLVED.value) for o in outcomes):
                overall_outcome = VerificationOutcome.PARTIALLY_RESOLVED.value
            else:
                overall_outcome = VerificationOutcome.UNCHANGED.value

        # Success = resolved or partially resolved
        success = overall_outcome in (
            VerificationOutcome.RESOLVED.value,
            VerificationOutcome.PARTIALLY_RESOLVED.value,
        ) or (score_diff > 0)

        # Build failure explanation if not successful
        failure_explanation = ""
        if not success:
            failed_problems = [
                pv for pv in problem_verifications
                if pv["outcome"] not in (
                    VerificationOutcome.RESOLVED.value,
                    VerificationOutcome.PARTIALLY_RESOLVED.value,
                )
            ]
            failure_explanation = (
                f"Verification outcome: {overall_outcome}. "
                f"{len(failed_problems)} problem(s) not resolved. "
                f"Details: " + "; ".join(
                    f"[{pv['problem_id']}] {pv['outcome']}: {pv['explanation']}"
                    for pv in failed_problems
                )
            )

        # Determine the proposal that was executed
        proposals = state.get("proposals", [])
        selected_idx = state.get("selected_proposal_idx", 0)
        strategy = proposals[selected_idx].get("strategy", "unknown") if selected_idx < len(proposals) else "unknown"

        result = VerificationResult(
            success=success,
            before_score=before_score,
            after_score=after_score,
            score_difference=score_diff,
            resolved_signal_ids=resolved_ids,
            new_signal_ids=new_ids,
            remaining_signal_ids=remaining_target_ids,
            dimension_deltas=dim_deltas,
            summary=f"Score: {before_score} -> {after_score} ({'+' if score_diff >= 0 else ''}{score_diff}). "
                    f"Overall: {overall_outcome}. "
                    f"Resolved: {len(resolved_ids)}, New: {len(new_ids)}, "
                    f"Remaining targets: {len(remaining_target_ids)}/{len(target_signal_ids)}. "
                    f"Strategy: {strategy}.",
            problem_verifications=problem_verifications,
            overall_outcome=overall_outcome,
            failure_explanation=failure_explanation,
        )

        # Save the after-assessment if we have a store
        if assessment_store:
            assessment_store.save(after_assessment)

        # Update DecisionTrace with verification reasoning
        decision_trace = DecisionTrace.from_dict(state.get("decision_trace", {}))
        decision_trace.execution_summary = (
            f"Executed {len(execution_history)} step(s) with strategy '{strategy}'."
        )
        decision_trace.verification_reasoning = (
            f"Overall outcome: {overall_outcome}. "
            f"Score: {before_score} -> {after_score} ({'+' if score_diff >= 0 else ''}{score_diff}). "
            f"{len(problem_verifications)} problem(s) verified."
        )
        decision_trace.final_outcome = (
            f"{'SUCCESS' if success else 'FAILURE'}: {result.summary}"
        )

        # Diagnosis quality feedback: record (confidence, outcome) observation
        # This feeds the rolling correlation that detects diagnosis degradation.
        if diagnosis_quality_tracker:
            try:
                root_causes = state.get("root_cause_analysis", [])
                mean_confidence = 0.5
                if root_causes:
                    confidences = [rc.get("confidence", 0.5) for rc in root_causes]
                    mean_confidence = sum(confidences) / len(confidences)
                outcome_value = 1.0 if success else 0.0
                if overall_outcome == VerificationOutcome.PARTIALLY_RESOLVED.value:
                    outcome_value = 0.5
                diagnosis_quality_tracker.record(
                    confidence=mean_confidence,
                    outcome=outcome_value,
                    remediation_id=state.get("remediation_id", ""),
                    metadata={"strategy": strategy, "overall_outcome": overall_outcome},
                )
            except Exception as e:
                logger.warning(f"Diagnosis quality recording failed: {e}")

        return state.update(
            verification_results=result.to_dict(),
            current_stage=RemediationStatus.COMPLETED.value if success
                          else RemediationStatus.FAILED_VERIFICATION.value,
            decision_trace=decision_trace.to_dict(),
        )

    except Exception as e:
        logger.error(f"verify_improvement failed: {e}")
        return state.update(
            current_stage=RemediationStatus.FAILED_VERIFICATION.value,
            verification_results=VerificationResult(
                success=False, summary=f"Verification error: {e}",
                failure_explanation=f"Exception during verification: {e}",
            ).to_dict(),
        )


# ---------------------------------------------------------------------------
# Action: Handle Failure
# ---------------------------------------------------------------------------

@action(reads=["current_stage", "proposals", "selected_proposal_idx", "verification_results",
               "root_cause_analysis", "knowledge_problems", "attempt_number", "execution_history",
               "approval_status", "decision_trace"],
        writes=["prior_failures", "prior_diagnosis", "prior_strategy", "prior_verification",
                "current_stage", "decision_trace"])
def handle_failure(state: State) -> State:
    """Record failure and prepare state for potential forking.

    Preserves the full prior diagnosis (root causes + knowledge
    problems), the prior strategy (proposal that was selected and
    executed), and the prior verification result (with problem-level
    outcomes and failure explanation). This allows a forked retry
    to reason about WHY the prior approach failed and generate a
    genuinely different strategy.

    Writes:
      prior_failures: updated with this attempt's failure summary
      prior_diagnosis: root causes + knowledge problems from this attempt
      prior_strategy: the selected proposal that was executed
      prior_verification: verification result with failure explanation
      current_stage: set to REJECTED if rejection, preserved otherwise
      decision_trace: updated with final outcome
    """
    proposals = state.get("proposals", [])
    selected_idx = state.get("selected_proposal_idx", 0)
    verification = state.get("verification_results", {})
    attempt_number = state.get("attempt_number", 1)
    root_causes = state.get("root_cause_analysis", [])
    knowledge_problems = state.get("knowledge_problems", [])
    execution_history = state.get("execution_history", [])
    approval_status = state.get("approval_status", "")
    current_stage = state.get("current_stage", "")

    strategy = proposals[selected_idx].get("strategy", "unknown") if selected_idx < len(proposals) else "unknown"
    issue_type = root_causes[0].get("category", "unknown") if root_causes else "unknown"

    # If this is a rejection, set stage to REJECTED (terminal)
    if approval_status == "rejected":
        new_stage = RemediationStatus.REJECTED.value
    else:
        new_stage = current_stage  # Preserve the failure stage for fork logic

    failure_summary = {
        "attempt_number": attempt_number,
        "strategy": strategy,
        "issue_type": issue_type,
        "result": "failure",
        "failure_reason": verification.get("failure_explanation", "")
                         or verification.get("summary", "unknown"),
        "score_change": verification.get("score_difference", 0),
        "verification_outcome": verification.get("overall_outcome", ""),
    }

    prior_failures = list(state.get("prior_failures", []))
    prior_failures.append(failure_summary)

    # Preserve full context for forked retry
    prior_diagnosis = {
        "root_causes": root_causes,
        "knowledge_problems": knowledge_problems,
    }
    prior_strategy = proposals[selected_idx] if selected_idx < len(proposals) else {}
    prior_verification = verification

    # Update DecisionTrace with final outcome
    decision_trace = DecisionTrace.from_dict(state.get("decision_trace", {}))
    if approval_status == "rejected":
        decision_trace.final_outcome = (
            f"REJECTED (attempt #{attempt_number}): proposal with strategy "
            f"'{strategy}' was rejected by the reviewer."
        )
    else:
        decision_trace.final_outcome = (
            f"FAILURE (attempt #{attempt_number}): strategy='{strategy}' did not resolve "
            f"the knowledge problem(s). Verification outcome: "
            f"{verification.get('overall_outcome', 'unknown')}. "
            f"Failure explanation: {verification.get('failure_explanation', 'N/A')}"
        )

    return state.update(
        prior_failures=prior_failures,
        prior_diagnosis=prior_diagnosis,
        prior_strategy=prior_strategy,
        prior_verification=prior_verification,
        current_stage=new_stage,
        decision_trace=decision_trace.to_dict(),
    )
