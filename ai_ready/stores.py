"""Signal and Assessment stores — persistence layer.

SignalStore handles signal persistence + lifecycle tracking.
AssessmentStore handles assessment persistence + diff + trend analysis.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ai_ready.models import (
    AssessmentDiff,
    DimensionScore,
    EvolutionView,
    KnowledgeAssessment,
    KnowledgeSignal,
    Severity,
    SignalLifecycle,
    SignalStatus,
    _severity_rank,
)


class SignalStore:
    """SQLite-backed signal storage with lifecycle tracking.

    Mirrors the signal_lifecycle table in AssessmentStore but
    uses signal-centric table names (signals, signal_lifecycle).
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
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assessment_id TEXT NOT NULL,
                    signal_id TEXT NOT NULL,
                    collector_id TEXT NOT NULL,
                    signal_type TEXT NOT NULL DEFAULT '',
                    severity TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    artifact_id TEXT NOT NULL,
                    artifact_uri TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    ai_impact TEXT NOT NULL DEFAULT '',
                    line INTEGER DEFAULT 0,
                    FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_assessment ON signals(assessment_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_signal_id ON signals(signal_id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_lifecycle (
                    signal_id TEXT PRIMARY KEY,
                    collector_id TEXT NOT NULL,
                    artifact_uri TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'new',
                    severity_history TEXT NOT NULL DEFAULT '[]',
                    assessment_ids TEXT NOT NULL DEFAULT '[]',
                    resolved_assessment TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_lifecycle_status ON signal_lifecycle(status)
            """)

    def save_signals(
        self, assessment_id: str, signals: list[KnowledgeSignal], timestamp: str | None = None
    ) -> None:
        """Persist signals for an assessment and update lifecycle tracking."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            for s in signals:
                conn.execute(
                    """INSERT INTO signals
                       (assessment_id, signal_id, collector_id, signal_type, severity, score,
                        artifact_id, artifact_uri, evidence, recommendation, ai_impact, line)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        assessment_id,
                        s.signal_id,
                        s.collector_id,
                        s.signal_type,
                        s.severity.value,
                        s.score,
                        s.artifact_id,
                        s.artifact_uri,
                        json.dumps(s.evidence),
                        s.recommendation,
                        s.ai_impact,
                        s.line,
                    ),
                )

            self._update_lifecycle(conn, assessment_id, signals, timestamp)

    def _update_lifecycle(
        self, conn: sqlite3.Connection, assessment_id: str, signals: list[KnowledgeSignal], timestamp: str
    ) -> None:
        """Update signal lifecycle table based on current assessment signals."""
        prev_assessment = self._get_previous_assessment_id(conn, assessment_id)
        prev_signal_ids: set[str] = set()
        if prev_assessment:
            rows = conn.execute(
                "SELECT signal_id FROM signals WHERE assessment_id = ?", (prev_assessment,)
            ).fetchall()
            prev_signal_ids = {r["signal_id"] for r in rows}

        curr_signal_ids = {s.signal_id for s in signals}

        # Mark resolved signals
        resolved_ids = prev_signal_ids - curr_signal_ids
        for sid in resolved_ids:
            row = conn.execute(
                "SELECT * FROM signal_lifecycle WHERE signal_id = ?", (sid,)
            ).fetchone()
            if row and row["status"] != "resolved":
                conn.execute(
                    "UPDATE signal_lifecycle SET status = ?, resolved_assessment = ? WHERE signal_id = ?",
                    (SignalStatus.RESOLVED.value, assessment_id, sid),
                )

        # Upsert current signals
        for s in signals:
            existing = conn.execute(
                "SELECT * FROM signal_lifecycle WHERE signal_id = ?", (s.signal_id,)
            ).fetchone()

            severity_entry = {
                "assessment_id": assessment_id,
                "severity": s.severity.value,
                "score": s.score,
            }

            if existing:
                sev_history = json.loads(existing["severity_history"])
                sev_history.append(severity_entry)

                assessment_ids = json.loads(existing["assessment_ids"])
                if assessment_id not in assessment_ids:
                    assessment_ids.append(assessment_id)

                if existing["status"] == SignalStatus.RESOLVED.value:
                    status = SignalStatus.RECURRING.value
                elif existing["status"] == SignalStatus.NEW.value:
                    status = SignalStatus.PERSISTENT.value
                else:
                    status = existing["status"]

                conn.execute(
                    """UPDATE signal_lifecycle
                       SET last_seen = ?, status = ?, severity_history = ?, assessment_ids = ?,
                           resolved_assessment = ''
                       WHERE signal_id = ?""",
                    (timestamp, status, json.dumps(sev_history), json.dumps(assessment_ids), s.signal_id),
                )
            else:
                conn.execute(
                    """INSERT INTO signal_lifecycle
                       (signal_id, collector_id, artifact_uri, first_seen, last_seen, status,
                        severity_history, assessment_ids, resolved_assessment)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        s.signal_id,
                        s.collector_id,
                        s.artifact_uri,
                        timestamp,
                        timestamp,
                        SignalStatus.NEW.value,
                        json.dumps([severity_entry]),
                        json.dumps([assessment_id]),
                        "",
                    ),
                )

    def _get_previous_assessment_id(self, conn: sqlite3.Connection, assessment_id: str) -> str | None:
        """Get the assessment ID immediately before the given one."""
        row = conn.execute(
            "SELECT timestamp FROM assessments WHERE assessment_id = ?", (assessment_id,)
        ).fetchone()
        if not row:
            return None
        prev = conn.execute(
            "SELECT assessment_id FROM assessments WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 1",
            (row["timestamp"],),
        ).fetchone()
        return prev["assessment_id"] if prev else None

    def get_lifecycle(self, signal_id: str) -> SignalLifecycle | None:
        """Get lifecycle info for a specific signal."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM signal_lifecycle WHERE signal_id = ?", (signal_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_lifecycle(row)

    def get_open_signals(self, limit: int = 100) -> list[SignalLifecycle]:
        """Get all currently open signals (new or persistent or recurring)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM signal_lifecycle
                   WHERE status IN (?, ?, ?)
                   ORDER BY first_seen ASC
                   LIMIT ?""",
                (SignalStatus.NEW.value, SignalStatus.PERSISTENT.value, SignalStatus.RECURRING.value, limit),
            ).fetchall()
            return [self._row_to_lifecycle(r) for r in rows]

    def get_recurring_signals(self, limit: int = 50) -> list[SignalLifecycle]:
        """Get signals that resolved and then came back."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM signal_lifecycle
                   WHERE status = ?
                   ORDER BY last_seen DESC
                   LIMIT ?""",
                (SignalStatus.RECURRING.value, limit),
            ).fetchall()
            return [self._row_to_lifecycle(r) for r in rows]

    def get_signal_stats(self) -> dict[str, Any]:
        """Get summary stats about signal lifecycle."""
        with self._conn() as conn:
            stats = {}
            for status in SignalStatus:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM signal_lifecycle WHERE status = ?",
                    (status.value,),
                ).fetchone()
                stats[status.value] = row["count"]

            row = conn.execute(
                """SELECT AVG(julianday('now') - julianday(first_seen)) as avg_age
                   FROM signal_lifecycle
                   WHERE status IN (?, ?, ?)""",
                (SignalStatus.NEW.value, SignalStatus.PERSISTENT.value, SignalStatus.RECURRING.value),
            ).fetchone()
            stats["avg_age_days"] = round(row["avg_age"], 1) if row["avg_age"] else 0

            return stats

    def _row_to_lifecycle(self, row: sqlite3.Row) -> SignalLifecycle:
        return SignalLifecycle(
            signal_id=row["signal_id"],
            collector_id=row["collector_id"],
            artifact_uri=row["artifact_uri"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            status=SignalStatus(row["status"]),
            severity_history=json.loads(row["severity_history"]),
            assessment_ids=json.loads(row["assessment_ids"]),
            resolved_assessment=row["resolved_assessment"],
        )


class AssessmentStore:
    """SQLite-backed assessment storage with diff and trend analysis.

    Uses assessment-centric
    table names (assessments). Works with SignalStore for signal persistence.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.signal_store = SignalStore(db_path)
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
                CREATE TABLE IF NOT EXISTS assessments (
                    assessment_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    dimensions TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)

    def save(self, assessment: KnowledgeAssessment) -> None:
        """Persist an assessment and its signals."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO assessments
                   (assessment_id, timestamp, score, dimensions, metrics, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    assessment.assessment_id,
                    timestamp,
                    assessment.score,
                    json.dumps({
                        k: {"name": v.name, "score": v.score, "collector_ids": v.collector_ids,
                             "signals_count": v.signals_count}
                        for k, v in assessment.dimensions.items()
                    }),
                    json.dumps(assessment.metrics),
                    json.dumps(assessment.metadata),
                ),
            )

        # Save signals via signal store
        self.signal_store.save_signals(assessment.assessment_id, assessment.signals, timestamp)

    def load(self, assessment_id: str) -> KnowledgeAssessment | None:
        """Load an assessment by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_assessment(conn, row)

    def latest(self) -> KnowledgeAssessment | None:
        """Get the most recent assessment."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM assessments ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            return self._row_to_assessment(conn, row)

    def history(self, limit: int = 10) -> list[KnowledgeAssessment]:
        """Get recent assessments ordered by time descending."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM assessments ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_assessment(conn, row) for row in rows]

    def previous(self, assessment_id: str) -> KnowledgeAssessment | None:
        """Get the assessment immediately before the given one."""
        with self._conn() as conn:
            target = conn.execute(
                "SELECT timestamp FROM assessments WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
            if not target:
                return None
            row = conn.execute(
                """SELECT * FROM assessments
                   WHERE timestamp < ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (target["timestamp"],),
            ).fetchone()
            if not row:
                return None
            return self._row_to_assessment(conn, row)

    # --- Diff ---

    def diff(self, prev_id: str, curr_id: str) -> AssessmentDiff | None:
        """Compute diff between two assessments by ID."""
        prev = self.load(prev_id)
        curr = self.load(curr_id)
        if not prev or not curr:
            return None
        return self._compute_diff(prev, curr)

    def diff_latest(self) -> AssessmentDiff | None:
        """Diff the two most recent assessments."""
        history = self.history(limit=2)
        if len(history) < 2:
            return None
        return self._compute_diff(history[1], history[0])

    def _compute_diff(self, prev: KnowledgeAssessment, curr: KnowledgeAssessment) -> AssessmentDiff:
        """Compute diff between two assessments using stable signal IDs."""
        prev_by_id = {s.signal_id: s for s in prev.signals}
        curr_by_id = {s.signal_id: s for s in curr.signals}

        prev_ids = set(prev_by_id.keys())
        curr_ids = set(curr_by_id.keys())

        new_signals = [curr_by_id[sid] for sid in curr_ids - prev_ids]
        resolved_signals = [prev_by_id[sid] for sid in prev_ids - curr_ids]
        persistent_signals = [curr_by_id[sid] for sid in curr_ids & prev_ids]

        # Detect severity changes
        severity_changes: list[dict[str, Any]] = []
        for sid in curr_ids & prev_ids:
            ps = prev_by_id[sid]
            cs = curr_by_id[sid]
            if ps.severity != cs.severity:
                severity_changes.append({
                    "signal_id": sid,
                    "collector_id": cs.collector_id,
                    "artifact_uri": cs.artifact_uri,
                    "prev_severity": ps.severity.value,
                    "curr_severity": cs.severity.value,
                    "direction": "worse" if _severity_rank(cs.severity) > _severity_rank(ps.severity) else "better",
                })

        # Dimension deltas
        dim_deltas: dict[str, int] = {}
        for dim_name in set(list(prev.dimensions.keys()) + list(curr.dimensions.keys())):
            prev_score = prev.dimensions[dim_name].score if dim_name in prev.dimensions else 0
            curr_score = curr.dimensions[dim_name].score if dim_name in curr.dimensions else 0
            dim_deltas[dim_name] = curr_score - prev_score

        new_high = sum(1 for s in new_signals if s.severity in (Severity.HIGH, Severity.CRITICAL))

        # Build explanation
        explanation_parts: list[str] = []
        if new_signals:
            explanation_parts.append(f"{len(new_signals)} new signals")
        if resolved_signals:
            explanation_parts.append(f"{len(resolved_signals)} resolved")
        if severity_changes:
            worse = sum(1 for s in severity_changes if s["direction"] == "worse")
            better = sum(1 for s in severity_changes if s["direction"] == "better")
            if worse:
                explanation_parts.append(f"{worse} signals got worse")
            if better:
                explanation_parts.append(f"{better} signals improved")
        if not explanation_parts:
            explanation_parts.append("No changes detected")
        explanation = "; ".join(explanation_parts) + "."

        recommendation = ""
        if curr.score < prev.score:
            recommendation = f"Score dropped {prev.score} -> {curr.score}. Review {len(new_signals)} new signals before deployment."
        elif new_high > 0:
            recommendation = f"{new_high} new high-severity signals. Review before deployment."
        elif len(resolved_signals) > 0 and not new_signals:
            recommendation = f"Improving. {len(resolved_signals)} signals resolved, no new issues."
        elif not new_signals:
            recommendation = "No regressions detected."

        return AssessmentDiff(
            prev_assessment_id=prev.assessment_id,
            curr_assessment_id=curr.assessment_id,
            prev_score=prev.score,
            curr_score=curr.score,
            score_delta=curr.score - prev.score,
            new_signals=new_signals,
            resolved_signals=resolved_signals,
            persistent_signals=persistent_signals,
            severity_changes=severity_changes,
            dimension_deltas=dim_deltas,
            new_high_count=new_high,
            score_change_explanation={"summary": explanation, "score_delta": curr.score - prev.score},
            recommendation=recommendation,
            explanation=explanation,
        )

    # --- Trend ---

    def get_trend(self, limit: int = 30) -> EvolutionView:
        """Get health evolution across recent assessments."""
        assessments = self.history(limit=limit)
        if len(assessments) < 2:
            return EvolutionView(
                trajectory="unknown",
                summary="Not enough assessments for trend analysis (need at least 2).",
            )

        # Reverse to chronological order
        assessments.reverse()

        score_trend = []
        signal_trend = []
        dim_trends: dict[str, list[dict[str, Any]]] = {}

        for a in assessments:
            score_trend.append({
                "assessment_id": a.assessment_id,
                "score": a.score,
            })
            signal_trend.append({
                "assessment_id": a.assessment_id,
                "total_signals": len(a.signals),
                "high_signals": sum(1 for s in a.signals if s.severity in (Severity.HIGH, Severity.CRITICAL)),
            })
            for dim_name, dim in a.dimensions.items():
                dim_trends.setdefault(dim_name, []).append({
                    "assessment_id": a.assessment_id,
                    "score": dim.score,
                })

        # Determine trajectory
        if len(score_trend) >= 2:
            recent_scores = [s["score"] for s in score_trend[-3:]] if len(score_trend) >= 3 else [s["score"] for s in score_trend]
            first_score = recent_scores[0]
            last_score = recent_scores[-1]
            delta = last_score - first_score
            if delta > 3:
                trajectory = "improving"
            elif delta < -3:
                trajectory = "worsening"
            else:
                trajectory = "stable"
        else:
            trajectory = "stable"

        summary_parts = [
            f"Score: {score_trend[0]['score']} -> {score_trend[-1]['score']} over {len(assessments)} assessments",
            f"Trajectory: {trajectory}",
            f"Signals: {signal_trend[0]['total_signals']} -> {signal_trend[-1]['total_signals']}",
        ]

        return EvolutionView(
            assessments=[{"assessment_id": a.assessment_id, "score": a.score} for a in assessments],
            score_trend=score_trend,
            signal_trend=signal_trend,
            dimension_trends=dim_trends,
            trajectory=trajectory,
            summary=". ".join(summary_parts) + ".",
        )

    # --- Row converter ---

    def _row_to_assessment(self, conn: sqlite3.Connection, row: sqlite3.Row) -> KnowledgeAssessment:
        """Convert a DB row to a KnowledgeAssessment object."""
        dims_data = json.loads(row["dimensions"])
        dimensions: dict[str, DimensionScore] = {}
        for k, v in dims_data.items():
            dimensions[k] = DimensionScore(
                name=v["name"], score=v["score"], collector_ids=v["collector_ids"],
                signals_count=v["signals_count"],
            )

        signals: list[KnowledgeSignal] = []
        srows = conn.execute(
            "SELECT * FROM signals WHERE assessment_id = ?", (row["assessment_id"],)
        ).fetchall()
        for sr in srows:
            signals.append(KnowledgeSignal(
                signal_id=sr["signal_id"],
                collector_id=sr["collector_id"],
                signal_type=sr["signal_type"],
                severity=Severity(sr["severity"]),
                score=sr["score"],
                artifact_id=sr["artifact_id"],
                artifact_uri=sr["artifact_uri"],
                evidence=json.loads(sr["evidence"]),
                recommendation=sr["recommendation"],
                ai_impact=sr["ai_impact"],
                line=sr["line"],
            ))

        return KnowledgeAssessment(
            assessment_id=row["assessment_id"],
            score=row["score"],
            dimensions=dimensions,
            signals=signals,
            metrics=json.loads(row["metrics"]),
            metadata=json.loads(row["metadata"]),
        )
