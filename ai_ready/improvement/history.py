"""Improvement history store — institutional memory for knowledge improvements.

Stores outcomes of improvement workflows so future workflows can
learn from past successes and failures. This is separate from
Burr's tracking (which stores workflow execution state) — this
stores the *semantic* outcome: what issue, what strategy, what result.

Burr stores workflow history (how the state machine executed).
AI-Ready stores improvement knowledge (what worked and what didn't).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ai_ready.improvement.models import RemediationOutcome


class ImprovementHistoryStore:
    """SQLite-backed store for improvement outcomes.

    Stores outcomes indexed by issue_type and strategy, enabling
    future workflows to query "what strategies have worked for this
    type of issue before?"
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS improvement_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    remediation_id TEXT NOT NULL,
                    issue_type TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    result TEXT NOT NULL,
                    score_change INTEGER DEFAULT 0,
                    root_cause TEXT DEFAULT '',
                    artifact_uris TEXT DEFAULT '[]',
                    failure_reason TEXT DEFAULT '',
                    attempt_number INTEGER DEFAULT 1,
                    forked_from_app_id TEXT DEFAULT '',
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_issue_type
                ON improvement_outcomes(issue_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_strategy
                ON improvement_outcomes(strategy)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_result
                ON improvement_outcomes(result)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_issue_strategy
                ON improvement_outcomes(issue_type, strategy, result)
            """)
            # Add columns for richer outcome tracking (Focus 2 & 6)
            # Use ALTER TABLE for existing databases
            try:
                conn.execute("ALTER TABLE improvement_outcomes ADD COLUMN verification_outcome TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE improvement_outcomes ADD COLUMN strategy_description TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE improvement_outcomes ADD COLUMN proposal_reasoning TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass

    def record_outcome(self, outcome: RemediationOutcome, remediation_id: str = "",
                       forked_from_app_id: str = "", metadata: dict[str, Any] | None = None) -> int:
        """Record an improvement outcome.

        Args:
            outcome: The RemediationOutcome to store.
            remediation_id: The workflow's remediation ID.
            forked_from_app_id: If this was a forked retry, the prior app_id.
            metadata: Additional metadata.

        Returns:
            The inserted row ID.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO improvement_outcomes
                   (remediation_id, issue_type, strategy, result, score_change,
                    root_cause, artifact_uris, failure_reason, attempt_number,
                    forked_from_app_id, timestamp, metadata,
                    verification_outcome, strategy_description, proposal_reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    remediation_id,
                    outcome.issue_type,
                    outcome.strategy,
                    outcome.result,
                    outcome.score_change,
                    outcome.root_cause,
                    json.dumps(outcome.artifact_uris),
                    outcome.failure_reason,
                    outcome.attempt_number,
                    forked_from_app_id,
                    timestamp,
                    json.dumps(metadata or {}),
                    outcome.verification_outcome,
                    outcome.strategy_description,
                    outcome.proposal_reasoning,
                ),
            )
            return cursor.lastrowid

    def get_outcomes_for_issue(self, issue_type: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get prior outcomes for a specific issue type.

        Used by GenerateProposalAction to learn from history.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM improvement_outcomes
                   WHERE issue_type = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (issue_type, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_successful_strategies(self, issue_type: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get strategies that succeeded for this issue type."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM improvement_outcomes
                   WHERE issue_type = ? AND result = 'success'
                   ORDER BY score_change DESC
                   LIMIT ?""",
                (issue_type, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_failed_strategies(self, issue_type: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get strategies that failed for this issue type."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM improvement_outcomes
                   WHERE issue_type = ? AND result = 'failure'
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (issue_type, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_recent_outcomes(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent outcomes regardless of issue type."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM improvement_outcomes
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_strategy_stats(self, issue_type: str) -> dict[str, Any]:
        """Get success/failure statistics for all strategies tried for an issue type.

        Returns:
            Dict mapping strategy name to stats: {count, success_count, failure_count, avg_score_change, success_rate}
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT strategy, result, score_change FROM improvement_outcomes
                   WHERE issue_type = ?""",
                (issue_type,),
            ).fetchall()

        stats: dict[str, dict[str, Any]] = {}
        for row in rows:
            strategy = row["strategy"]
            if strategy not in stats:
                stats[strategy] = {
                    "count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "total_score_change": 0,
                }
            s = stats[strategy]
            s["count"] += 1
            if row["result"] == "success":
                s["success_count"] += 1
            else:
                s["failure_count"] += 1
            s["total_score_change"] += row["score_change"]

        # Compute averages
        for s in stats.values():
            s["avg_score_change"] = s["total_score_change"] / s["count"] if s["count"] else 0
            s["success_rate"] = s["success_count"] / s["count"] if s["count"] else 0

        return stats

    def get_similar_problems(
        self,
        issue_type: str,
        artifact_uris: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve similar Knowledge Problems from history (Focus 2).

        Finds prior outcomes for the same issue type, optionally filtered
        by overlapping artifact URIs. Used to provide historical context
        to the LLM during proposal generation.

        Args:
            issue_type: The root cause category to match.
            artifact_uris: Optional list of artifact URIs to match (any overlap).
            limit: Maximum number of results.

        Returns:
            List of outcome dicts with strategy, result, score_change,
            failure_reason, verification_outcome, and strategy_description.
        """
        with self._conn() as conn:
            if artifact_uris:
                # Match any outcome that shares at least one artifact URI
                rows = conn.execute(
                    """SELECT * FROM improvement_outcomes
                       WHERE issue_type = ?
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (issue_type, limit * 2),  # over-fetch, filter in Python
                ).fetchall()
                results = [self._row_to_dict(r) for r in rows]
                # Filter by artifact overlap
                uri_set = set(artifact_uris)
                filtered = [
                    r for r in results
                    if uri_set & set(r.get("artifact_uris", []))
                ]
                return filtered[:limit] if filtered else results[:limit]
            else:
                rows = conn.execute(
                    """SELECT * FROM improvement_outcomes
                       WHERE issue_type = ?
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (issue_type, limit),
                ).fetchall()
                return [self._row_to_dict(r) for r in rows]

    def get_remediation_context(
        self,
        issue_type: str,
        artifact_uris: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build structured historical context for proposal generation (Focus 2 & 5).

        Retrieves and organizes prior remediation attempts into a structured
        context that the LLM can use during proposal generation:

        - similar_problems: prior outcomes for this issue type
        - successful_strategies: what worked
        - failed_strategies: what didn't work and why
        - strategy_stats: success/failure rates per strategy

        Args:
            issue_type: The root cause category.
            artifact_uris: Optional artifact URIs for similarity matching.

        Returns:
            Structured context dict for LLM prompt construction.
        """
        similar = self.get_similar_problems(issue_type, artifact_uris, limit=10)
        successful = [o for o in similar if o.get("result") == "success"]
        failed = [o for o in similar if o.get("result") == "failure"]
        stats = self.get_strategy_stats(issue_type)

        return {
            "similar_problems": similar,
            "successful_strategies": successful,
            "failed_strategies": failed,
            "strategy_stats": stats,
            "total_prior_attempts": len(similar),
        }

    def get_remediation_metrics(self) -> dict[str, Any]:
        """Compute aggregate remediation quality metrics (Focus 6).

        Measures whether AI-Ready is becoming more effective over time:

        - total_workflows: total number of recorded outcomes
        - problems_resolved: count of successful outcomes
        - problems_partially_resolved: count with verification_outcome=partially_resolved
        - verification_success_rate: success / total
        - avg_score_improvement: mean score_change for all workflows
        - strategy_reuse_rate: how often strategies are repeated
        - strategy_success_rate: per-strategy success rates
        - avg_forks_before_resolution: mean attempt_number for successful outcomes

        Returns:
            Dict of remediation quality metrics.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM improvement_outcomes ORDER BY timestamp ASC"
            ).fetchall()
            all_outcomes = [self._row_to_dict(r) for r in rows]

        total = len(all_outcomes)
        if total == 0:
            return {
                "total_workflows": 0,
                "problems_resolved": 0,
                "problems_partially_resolved": 0,
                "verification_success_rate": 0.0,
                "avg_score_improvement": 0.0,
                "strategy_reuse_rate": 0.0,
                "strategy_success_rate": {},
                "avg_forks_before_resolution": 0.0,
            }

        resolved = sum(1 for o in all_outcomes if o.get("result") == "success")
        partially = sum(
            1 for o in all_outcomes
            if o.get("verification_outcome") == "partially_resolved"
        )
        success_rate = resolved / total if total else 0.0
        avg_score = sum(o.get("score_change", 0) for o in all_outcomes) / total

        # Strategy reuse rate: how many strategies appear more than once
        strategy_counts: dict[str, int] = {}
        for o in all_outcomes:
            s = o.get("strategy", "")
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        repeated = sum(1 for c in strategy_counts.values() if c > 1)
        reuse_rate = repeated / len(strategy_counts) if strategy_counts else 0.0

        # Per-strategy success rate
        strategy_results: dict[str, dict[str, Any]] = {}
        for o in all_outcomes:
            s = o.get("strategy", "")
            if s not in strategy_results:
                strategy_results[s] = {"total": 0, "success": 0}
            strategy_results[s]["total"] += 1
            if o.get("result") == "success":
                strategy_results[s]["success"] += 1
        for s in strategy_results:
            strategy_results[s]["success_rate"] = (
                strategy_results[s]["success"] / strategy_results[s]["total"]
                if strategy_results[s]["total"] else 0.0
            )

        # Average forks before resolution (for successful outcomes)
        successful_attempts = [
            o.get("attempt_number", 1) for o in all_outcomes
            if o.get("result") == "success"
        ]
        avg_forks = (
            sum(successful_attempts) / len(successful_attempts)
            if successful_attempts else 0.0
        )

        return {
            "total_workflows": total,
            "problems_resolved": resolved,
            "problems_partially_resolved": partially,
            "verification_success_rate": round(success_rate, 4),
            "avg_score_improvement": round(avg_score, 2),
            "strategy_reuse_rate": round(reuse_rate, 4),
            "strategy_success_rate": strategy_results,
            "avg_forks_before_resolution": round(avg_forks, 2),
        }

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "remediation_id": row["remediation_id"],
            "issue_type": row["issue_type"],
            "strategy": row["strategy"],
            "result": row["result"],
            "score_change": row["score_change"],
            "root_cause": row["root_cause"],
            "artifact_uris": json.loads(row["artifact_uris"]),
            "failure_reason": row["failure_reason"],
            "attempt_number": row["attempt_number"],
            "forked_from_app_id": row["forked_from_app_id"],
            "timestamp": row["timestamp"],
            "metadata": json.loads(row["metadata"]),
            "verification_outcome": row["verification_outcome"] if "verification_outcome" in row.keys() else "",
            "strategy_description": row["strategy_description"] if "strategy_description" in row.keys() else "",
            "proposal_reasoning": row["proposal_reasoning"] if "proposal_reasoning" in row.keys() else "",
        }
