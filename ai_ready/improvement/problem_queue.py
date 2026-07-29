"""Problem Queue — continuous discovery and ranking of knowledge problems.

Runs deterministic heuristic problem discovery and salience ranking on
assessments to maintain a ranked queue of knowledge problems across all
assessments. No LLM calls — pure mathematical computation.

The queue makes AI-Ready proactive rather than reactive: instead of
requiring someone to manually identify which signals to investigate,
the system continuously surfaces the most salient problems across
all assessments and presents them in priority order.

Usage:
    queue = ProblemQueue(history_store=store)
    queue.update(assessment=assessment, signal_ids=signal_ids)
    ranked = queue.get_ranked_problems(limit=10)
    # ranked[0] -> {"problem_id": ..., "salience": ..., "assessment_id": ...}
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_ready.improvement.salience import (
    discover_problems_heuristic,
    rank_problems_by_salience,
    ProblemSalience,
    DEFAULT_SALIENCE_THRESHOLD,
)
from ai_ready.improvement.models import KnowledgeProblem


# ---------------------------------------------------------------------------
# Problem Queue Entry
# ---------------------------------------------------------------------------

@dataclass
class ProblemQueueEntry:
    """A single entry in the problem queue.

    Combines a KnowledgeProblem with its salience score and the
    assessment it was discovered from. All fields are deterministic.
    """

    problem: KnowledgeProblem
    salience: ProblemSalience
    assessment_id: str
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "open"  # "open", "in_progress", "resolved", "wont_fix"

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "salience": self.salience.to_dict(),
            "assessment_id": self.assessment_id,
            "discovered_at": self.discovered_at,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Problem Queue
# ---------------------------------------------------------------------------

class ProblemQueue:
    """Ranked queue of knowledge problems discovered across assessments.

    Continuously discovers problems using heuristic signal clustering
    (no LLM) and ranks them by salience (no LLM). The queue persists
    across assessments and can be queried for the most salient open
    problems.

    All computation is deterministic — no LLM calls.
    """

    def __init__(
        self,
        history_store: Any = None,
        db_path: str | Path | None = None,
        salience_threshold: float = DEFAULT_SALIENCE_THRESHOLD,
    ) -> None:
        """Initialize the problem queue.

        Args:
            history_store: Optional ImprovementHistoryStore for outcome
                sub-scores in salience computation.
            db_path: Optional SQLite path for persistence. If None,
                uses in-memory database (non-persistent).
            salience_threshold: Minimum salience to include in the queue.
        """
        self.history_store = history_store
        self.salience_threshold = salience_threshold
        self.db_path = str(db_path) if db_path else ":memory:"
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite table for problem queue entries."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS problem_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL,
                problem_json TEXT NOT NULL,
                salience_json TEXT NOT NULL,
                assessment_id TEXT NOT NULL,
                discovered_at TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                category TEXT DEFAULT '',
                artifact_uris TEXT DEFAULT '[]',
                salience_total REAL DEFAULT 0.0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queue_status
            ON problem_queue(status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queue_salience
            ON problem_queue(salience_total DESC)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queue_problem_id
            ON problem_queue(problem_id)
        """)
        self._conn.commit()

    def update(
        self,
        assessment: Any,
        signal_ids: list[str] | None = None,
    ) -> list[ProblemQueueEntry]:
        """Discover and queue problems from an assessment.

        Runs heuristic problem discovery (no LLM) and salience ranking
        (no LLM) on the assessment's signals. New problems are added
        to the queue; existing problems are updated.

        Args:
            assessment: The KnowledgeAssessment to analyze.
            signal_ids: Optional list of signal IDs to focus on.
                If None, all signals in the assessment are used.

        Returns:
            List of ProblemQueueEntry objects that were added or updated.
        """
        if not assessment:
            return []

        # Use all signals if none specified
        if signal_ids is None:
            signal_ids = [s.signal_id for s in assessment.signals]

        if not signal_ids:
            return []

        # Heuristic problem discovery (no LLM)
        problems, _hypotheses = discover_problems_heuristic(signal_ids, assessment)

        if not problems:
            return []

        # Salience ranking (no LLM)
        ranked_problems, saliences, _skipped = rank_problems_by_salience(
            problems,
            assessment,
            self.history_store,
            threshold=0.0,  # queue all problems, filter on retrieval
        )

        # Persist to queue
        entries = []
        assessment_id = getattr(assessment, "assessment_id", "")

        for problem, salience in zip(ranked_problems, saliences):
            entry = ProblemQueueEntry(
                problem=problem,
                salience=salience,
                assessment_id=assessment_id,
            )
            self._upsert_entry(entry)
            entries.append(entry)

        self._conn.commit()
        return entries

    def _upsert_entry(self, entry: ProblemQueueEntry) -> None:
        """Insert or update a problem queue entry.

        If a problem with the same problem_id already exists and is open,
        update its salience and assessment_id. If it's closed, skip.
        """
        problem_id = entry.problem.problem_id

        # Check if problem already exists
        existing = self._conn.execute(
            "SELECT id, status FROM problem_queue WHERE problem_id = ? ORDER BY id DESC LIMIT 1",
            (problem_id,),
        ).fetchone()

        if existing and existing[1] not in ("open",):
            return  # Don't update closed problems

        if existing:
            # Update existing open problem
            self._conn.execute(
                """UPDATE problem_queue
                   SET problem_json = ?, salience_json = ?, assessment_id = ?,
                       discovered_at = ?, salience_total = ?, category = ?,
                       artifact_uris = ?
                   WHERE id = ?""",
                (
                    json.dumps(entry.problem.to_dict()),
                    json.dumps(entry.salience.to_dict()),
                    entry.assessment_id,
                    entry.discovered_at,
                    entry.salience.total,
                    entry.problem.category,
                    json.dumps(entry.problem.artifact_uris),
                    existing[0],
                ),
            )
        else:
            # Insert new problem
            self._conn.execute(
                """INSERT INTO problem_queue
                   (problem_id, problem_json, salience_json, assessment_id,
                    discovered_at, status, category, artifact_uris, salience_total)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    problem_id,
                    json.dumps(entry.problem.to_dict()),
                    json.dumps(entry.salience.to_dict()),
                    entry.assessment_id,
                    entry.discovered_at,
                    entry.status,
                    entry.problem.category,
                    json.dumps(entry.problem.artifact_uris),
                    entry.salience.total,
                ),
            )

    def get_ranked_problems(
        self,
        limit: int = 20,
        status: str = "open",
        category: str | None = None,
        min_salience: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get the ranked problem queue.

        Returns problems sorted by salience (descending). All
        deterministic — no LLM.

        Args:
            limit: Maximum number of problems to return.
            status: Filter by status ("open", "in_progress", "resolved",
                "wont_fix", or "all" for any status).
            category: Optional category filter.
            min_salience: Optional minimum salience threshold.

        Returns:
            List of problem queue entry dicts, sorted by salience descending.
        """
        query = "SELECT * FROM problem_queue"
        conditions = []
        params: list[Any] = []

        if status != "all":
            conditions.append("status = ?")
            params.append(status)

        if category is not None:
            conditions.append("category = ?")
            params.append(category)

        if min_salience is not None:
            conditions.append("salience_total >= ?")
            params.append(min_salience)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY salience_total DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_problem(self, problem_id: str) -> dict[str, Any] | None:
        """Get a single problem from the queue by ID.

        Args:
            problem_id: The problem ID to look up.

        Returns:
            Problem queue entry dict, or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM problem_queue WHERE problem_id = ? ORDER BY id DESC LIMIT 1",
            (problem_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def set_status(self, problem_id: str, status: str) -> bool:
        """Update the status of a problem in the queue.

        Args:
            problem_id: The problem ID to update.
            status: New status ("open", "in_progress", "resolved", "wont_fix").

        Returns:
            True if the problem was found and updated, False otherwise.
        """
        cursor = self._conn.execute(
            "UPDATE problem_queue SET status = ? WHERE problem_id = ?",
            (status, problem_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_queue_summary(self) -> dict[str, Any]:
        """Get a summary of the problem queue.

        Returns counts by status and category, plus the top 5 most
        salient open problems. All deterministic.

        Returns:
            Summary dict with counts and top problems.
        """
        # Count by status
        status_rows = self._conn.execute(
            "SELECT status, COUNT(*) as count FROM problem_queue GROUP BY status"
        ).fetchall()
        by_status = {row[0]: row[1] for row in status_rows}

        # Count by category
        category_rows = self._conn.execute(
            "SELECT category, COUNT(*) as count FROM problem_queue WHERE status = 'open' GROUP BY category ORDER BY count DESC"
        ).fetchall()
        by_category = {row[0]: row[1] for row in category_rows}

        # Top 5 open problems
        top_problems = self.get_ranked_problems(limit=5, status="open")

        # Total salience of open problems
        total_salience_row = self._conn.execute(
            "SELECT COALESCE(SUM(salience_total), 0) FROM problem_queue WHERE status = 'open'"
        ).fetchone()
        total_salience = total_salience_row[0] if total_salience_row else 0.0

        return {
            "total_open": by_status.get("open", 0),
            "total_in_progress": by_status.get("in_progress", 0),
            "total_resolved": by_status.get("resolved", 0),
            "total_wont_fix": by_status.get("wont_fix", 0),
            "by_category": by_category,
            "total_salience_open": round(total_salience, 4),
            "top_problems": top_problems,
        }

    def _row_to_dict(self, row: sqlite3.Row | tuple) -> dict[str, Any]:
        """Convert a database row to a dict."""
        if isinstance(row, sqlite3.Row):
            row = tuple(row)
        return {
            "id": row[0],
            "problem_id": row[1],
            "problem": json.loads(row[2]),
            "salience": json.loads(row[3]),
            "assessment_id": row[4],
            "discovered_at": row[5],
            "status": row[6],
            "category": row[7],
            "artifact_uris": json.loads(row[8]),
            "salience_total": row[9],
        }
