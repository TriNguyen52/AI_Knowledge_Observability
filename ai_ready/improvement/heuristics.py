"""Heuristic fallbacks, proposal filtering, and LLM response parsing.

These functions provide:
  1. Non-LLM fallbacks for root cause analysis and proposal generation
     (used when llm_gateway is None or LLM calls fail)
  2. Proposal filtering to reject generic or duplicate strategies
  3. Domain-specific parsing of LLM JSON responses into typed dataclasses

Extracted from actions.py to keep action functions focused on
orchestration logic rather than parsing and fallback handling.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ai_ready.improvement.models import (
    RootCauseHypothesis,
    KnowledgeProblem,
    RemediationProposal,
)
from ai_ready.improvement.parsing import parse_llm_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic fallbacks (no LLM required)
# ---------------------------------------------------------------------------

def heuristic_analysis(
    signal_ids: list[str],
    artifact_uris: list[str],
    assessment_store: Any,
) -> RootCauseHypothesis:
    """Produce a basic heuristic root cause when no LLM is available."""
    return RootCauseHypothesis(
        hypothesis=f"Quality issues detected in {len(artifact_uris)} artifact(s) with {len(signal_ids)} signal(s). Manual analysis required.",
        evidence=[f"{len(signal_ids)} signals detected", f"{len(artifact_uris)} artifacts affected"],
        confidence=0.3,
        affected_artifact_uris=artifact_uris,
        category="unknown",
    )


def heuristic_problems(
    hypotheses: list[RootCauseHypothesis],
    signal_ids: list[str],
    artifact_uris: list[str],
) -> list[KnowledgeProblem]:
    """Synthesize Knowledge Problems from hypotheses when LLM doesn't provide them.

    Each hypothesis becomes one Knowledge Problem, with all signals and
    artifacts assigned to it.
    """
    problems = []
    for idx, h in enumerate(hypotheses):
        problem_id = f"kp_{h.category or 'unknown'}_{idx}"
        kp = KnowledgeProblem(
            problem_id=problem_id,
            category=h.category or "unknown",
            description=h.hypothesis,
            root_cause_idx=idx,
            signal_ids=signal_ids,
            artifact_uris=h.affected_artifact_uris or artifact_uris,
            evidence_summary="; ".join(h.evidence),
        )
        problems.append(kp)
    return problems


def heuristic_proposal(
    root_causes: list[dict],
    artifact_uris: list[str],
    prior_strategies: list[str],
) -> RemediationProposal:
    """Produce a basic proposal when no LLM is available."""
    strategy = "manual_review"
    if root_causes:
        category = root_causes[0].get("category", "")
        if category and category not in prior_strategies:
            strategy = f"address_{category}"

    return RemediationProposal(
        strategy=strategy,
        description=f"Heuristic proposal: address {len(root_causes)} root cause(s) in {len(artifact_uris)} artifact(s).",
        expected_impact="unknown",
        risks=["No LLM available for detailed analysis"],
        affected_artifact_uris=artifact_uris,
        modification_steps=[
            {"step_type": "update_document", "artifact_uri": uri, "description": "Manual review and update", "parameters": {}}
            for uri in artifact_uris
        ],
    )


# ---------------------------------------------------------------------------
# Proposal filtering
# ---------------------------------------------------------------------------

def reject_generic_proposals(
    proposals: list[RemediationProposal],
    prior_strategies: list[str],
) -> list[RemediationProposal]:
    """Filter out generic or duplicate proposals.

    A proposal is rejected if:
      1. Strategy name is empty or "unknown"
      2. Description is fewer than 20 characters (too vague)
      3. No modification steps provided
      4. Strategy name matches a prior failed strategy (exact match)
    """
    GENERIC_STRATEGIES = {"unknown", "", "manual_review", "fix", "improve"}
    filtered = []
    for p in proposals:
        # Reject empty/unknown strategies
        if p.strategy.lower() in GENERIC_STRATEGIES:
            logger.info(f"Rejecting generic proposal: strategy='{p.strategy}'")
            continue
        # Reject too-short descriptions
        if len(p.description.strip()) < 20:
            logger.info(f"Rejecting vague proposal: description too short ({len(p.description)} chars)")
            continue
        # Reject proposals with no modification steps
        if not p.modification_steps:
            logger.info(f"Rejecting proposal with no steps: strategy='{p.strategy}'")
            continue
        # Reject proposals that match prior failed strategies (case-insensitive)
        if p.strategy.lower() in [s.lower() for s in prior_strategies]:
            logger.info(f"Rejecting duplicate strategy: '{p.strategy}' matches prior failure")
            continue
        filtered.append(p)
    return filtered


# ---------------------------------------------------------------------------
# LLM response parsing (domain-specific)
# ---------------------------------------------------------------------------

def parse_hypotheses_and_problems(
    content: str,
    signal_ids: list[str],
    artifact_uris: list[str],
) -> tuple[list[RootCauseHypothesis], list[KnowledgeProblem]]:
    """Parse LLM response into RootCauseHypothesis and KnowledgeProblem objects.

    The LLM is asked to return both hypotheses and knowledge_problems in a
    single JSON response. If knowledge_problems is missing, we synthesize
    them from the hypotheses using heuristic_problems().
    """
    try:
        data = parse_llm_json(content)

        hypotheses_data = data.get("hypotheses", data if isinstance(data, list) else [])
        hypotheses = [RootCauseHypothesis.from_dict(h) for h in hypotheses_data]

        problems_data = data.get("knowledge_problems", [])
        if problems_data:
            knowledge_problems = [KnowledgeProblem.from_dict(p) for p in problems_data]
        else:
            # Synthesize one problem per hypothesis if LLM didn't provide them
            knowledge_problems = heuristic_problems(hypotheses, signal_ids, artifact_uris)

        return hypotheses, knowledge_problems
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse LLM hypotheses/problems: {e}")
        # Fallback: create a single hypothesis from the raw response
        hypotheses = [RootCauseHypothesis(
            hypothesis=content[:500],
            evidence=["LLM response could not be parsed as JSON"],
            confidence=0.3,
            category="unknown",
        )]
        knowledge_problems = heuristic_problems(hypotheses, signal_ids, artifact_uris)
        return hypotheses, knowledge_problems


def parse_proposals(content: str) -> list[RemediationProposal]:
    """Parse LLM response into RemediationProposal objects."""
    try:
        data = parse_llm_json(content)

        proposals_data = data.get("proposals", data if isinstance(data, list) else [])
        return [RemediationProposal.from_dict(p) for p in proposals_data]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse LLM proposals: {e}")
        return [RemediationProposal(
            strategy="manual_review",
            description=f"LLM response could not be parsed. Raw: {content[:200]}",
            expected_impact="unknown",
            risks=["LLM parsing failure"],
        )]
