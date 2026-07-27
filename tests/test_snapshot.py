"""Tests for snapshot store, regression engine, and lifecycle tracking."""

import os
from pathlib import Path

from ai_ready.models import DimensionScore, Finding, FindingStatus, Severity, Snapshot
from ai_ready.snapshot import SnapshotStore

TEST_DB = Path(__file__).parent / "test_snapshots.db"


def _cleanup_db():
    """Remove test database."""
    try:
        if TEST_DB.exists():
            TEST_DB.unlink()
    except PermissionError:
        pass


def _make_finding(
    rule_id: str = "topic_purity",
    severity: Severity = Severity.HIGH,
    doc_path: str = "doc1.md",
    doc_id: str = "doc1",
    evidence: dict | None = None,
    fid: str = "",
    issue_type: str = "mixed_topics",
    score: int = 85,
) -> Finding:
    return Finding(
        finding_id=fid,
        rule_id=rule_id,
        issue_type=issue_type,
        severity=severity,
        score=score,
        document_id=doc_id,
        document_path=doc_path,
        evidence=evidence or {"cluster_count": 3},
        recommendation="Fix",
        ai_impact="Impact",
        line=5,
    )


def _make_snapshot(score: int = 90, findings: list | None = None, sid: str | None = None) -> Snapshot:
    return Snapshot(
        snapshot_id=sid or f"test-{score}",
        score=score,
        dimensions={
            "retrieval": DimensionScore(name="retrieval", score=score, rule_ids=["topic_purity", "heading_quality"]),
            "context": DimensionScore(name="context", score=score, rule_ids=["context_independence"]),
        },
        findings=findings or [],
        metrics={"topic_entropy": 1.5, "orphan_documents": 3},
        metadata={"source": "test/", "document_count": 10},
    )


# --- Basic snapshot CRUD ---

def test_snapshot_save_and_load():
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)
        snap = _make_snapshot(score=85)
        store.save(snap)

        loaded = store.load("test-85")
        assert loaded is not None
        assert loaded.score == 85
        assert loaded.snapshot_id == "test-85"
    finally:
        _cleanup_db()


def test_snapshot_latest():
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)
        store.save(_make_snapshot(score=80))
        store.save(_make_snapshot(score=90, sid="test-90"))

        latest = store.latest()
        assert latest is not None
        assert latest.score == 90
    finally:
        _cleanup_db()


def test_snapshot_history():
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)
        store.save(_make_snapshot(score=70))
        store.save(_make_snapshot(score=80, sid="test-80"))
        store.save(_make_snapshot(score=90, sid="test-90"))

        history = store.history(limit=10)
        assert len(history) == 3
        # Most recent first
        assert history[0].score == 90
    finally:
        _cleanup_db()


def test_snapshot_with_findings():
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)
        findings = [
            _make_finding(),
        ]
        snap = _make_snapshot(score=60, findings=findings)
        store.save(snap)

        loaded = store.load("test-60")
        assert loaded is not None
        assert len(loaded.findings) == 1
        assert loaded.findings[0].rule_id == "topic_purity"
        assert loaded.findings[0].issue_type == "mixed_topics"
        assert loaded.findings[0].ai_impact == "Impact"
        assert loaded.findings[0].severity == Severity.HIGH
        assert loaded.findings[0].finding_id  # Should have a finding_id
    finally:
        _cleanup_db()


def test_snapshot_score_explanation_groups_repeated_causes():
    findings = [
        _make_finding(
            rule_id="heading_quality",
            issue_type="generic",
            doc_path="a.md",
            doc_id="a",
            evidence={"heading": "Overview", "issue_type": "generic"},
            fid="h1",
            score=90,
        ),
        _make_finding(
            rule_id="heading_quality",
            issue_type="generic",
            doc_path="b.md",
            doc_id="b",
            evidence={"heading": "Overview", "issue_type": "generic"},
            fid="h2",
            score=90,
        ),
    ]
    snap = _make_snapshot(score=95, findings=findings, sid="snap-explain")
    snap.metadata["weights"] = {"retrieval": 1.0, "context": 0.0}

    explanation = snap.explain_score()

    assert explanation.dominant_contributors
    contributor = explanation.dominant_contributors[0]
    assert contributor.cause == "generic heading 'Overview'"
    assert contributor.finding_count == 2
    assert contributor.finding_ids == ["h1", "h2"]
    assert contributor.estimated_score_gain > 0


# --- Regression / Diff ---

def test_regression_diff():
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        prev = _make_snapshot(score=91, sid="snap-1")
        prev.metrics = {"topic_entropy": 1.0, "orphan_documents": 2, "contradiction_count": 1}
        store.save(prev)

        curr = _make_snapshot(score=84, sid="snap-2")
        curr.metrics = {"topic_entropy": 2.0, "orphan_documents": 5, "contradiction_count": 3}
        curr.findings = [
            _make_finding(fid="finding-abc"),
        ]
        store.save(curr)

        report = store.diff("snap-1", "snap-2")
        assert report is not None
        assert report.prev_score == 91
        assert report.curr_score == 84
        assert report.score_delta == -7
        assert len(report.new_findings) == 1
        assert report.increased_contradictions == 2
        assert report.new_orphan_documents == 3
        assert report.prev_snapshot_id == "snap-1"
        assert report.curr_snapshot_id == "snap-2"
        assert report.explanation
    finally:
        _cleanup_db()


def test_regression_explains_contributor_changes():
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        prev = _make_snapshot(score=95, sid="snap-1")
        prev.metadata["weights"] = {"retrieval": 1.0, "context": 0.0}
        prev.findings = [
            _make_finding(
                rule_id="heading_quality",
                issue_type="generic",
                doc_path="a.md",
                doc_id="a",
                evidence={"heading": "Overview", "issue_type": "generic"},
                fid="h1",
                score=90,
            )
        ]
        store.save(prev)

        curr = _make_snapshot(score=85, sid="snap-2")
        curr.metadata["weights"] = {"retrieval": 1.0, "context": 0.0}
        curr.findings = [
            _make_finding(
                rule_id="heading_quality",
                issue_type="generic",
                doc_path="a.md",
                doc_id="a",
                evidence={"heading": "Overview", "issue_type": "generic"},
                fid="h1",
                score=90,
            ),
            _make_finding(
                rule_id="heading_quality",
                issue_type="generic",
                doc_path="b.md",
                doc_id="b",
                evidence={"heading": "Overview", "issue_type": "generic"},
                fid="h2",
                score=90,
            ),
        ]
        store.save(curr)

        report = store.diff("snap-1", "snap-2")

        assert report is not None
        assert report.score_change_explanation["top_regressions"]
        top = report.score_change_explanation["top_regressions"][0]
        assert top["cause"] == "generic heading 'Overview'"
        assert top["prev_findings"] == 1
        assert top["curr_findings"] == 2
        assert top["added_finding_ids"] == ["h2"]
    finally:
        _cleanup_db()


def test_regression_no_change():
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)
        store.save(_make_snapshot(score=90, sid="snap-a"))
        store.save(_make_snapshot(score=90, sid="snap-b"))

        report = store.diff("snap-a", "snap-b")
        assert report is not None
        assert report.score_delta == 0
        assert len(report.new_findings) == 0
    finally:
        _cleanup_db()


def test_regression_resolved_findings():
    """Findings in prev but not in curr should be resolved."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        prev = _make_snapshot(score=60, sid="snap-1")
        prev.findings = [_make_finding(fid="f1"), _make_finding(fid="f2", doc_path="doc2.md", doc_id="doc2")]
        store.save(prev)

        curr = _make_snapshot(score=90, sid="snap-2")
        curr.findings = [_make_finding(fid="f1")]  # Only f1 remains
        store.save(curr)

        report = store.diff("snap-1", "snap-2")
        assert len(report.resolved_findings) == 1
        assert report.resolved_findings[0].finding_id == "f2"
        assert len(report.persistent_findings) == 1
        assert report.persistent_findings[0].finding_id == "f1"
    finally:
        _cleanup_db()


def test_regression_severity_change():
    """Detect when a finding's severity changes between scans."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        prev = _make_snapshot(score=70, sid="snap-1")
        prev.findings = [_make_finding(fid="f1", severity=Severity.LOW)]
        store.save(prev)

        curr = _make_snapshot(score=50, sid="snap-2")
        curr.findings = [_make_finding(fid="f1", severity=Severity.HIGH)]
        store.save(curr)

        report = store.diff("snap-1", "snap-2")
        assert len(report.severity_changes) == 1
        assert report.severity_changes[0]["direction"] == "worse"
        assert report.severity_changes[0]["prev_severity"] == "LOW"
        assert report.severity_changes[0]["curr_severity"] == "HIGH"
    finally:
        _cleanup_db()


# --- Finding lifecycle ---

def test_lifecycle_new_finding():
    """A finding in the first scan should be marked as new."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)
        snap = _make_snapshot(sid="snap-1")
        snap.findings = [_make_finding(fid="f1")]
        store.save(snap)

        lifecycle = store.get_lifecycle("f1")
        assert lifecycle is not None
        assert lifecycle.status == FindingStatus.NEW
        assert lifecycle.first_seen
        assert lifecycle.last_seen
        assert len(lifecycle.snapshot_ids) == 1
    finally:
        _cleanup_db()


def test_lifecycle_persistent_finding():
    """A finding in consecutive scans should become persistent."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        snap1 = _make_snapshot(sid="snap-1")
        snap1.findings = [_make_finding(fid="f1")]
        store.save(snap1)

        snap2 = _make_snapshot(score=80, sid="snap-2")
        snap2.findings = [_make_finding(fid="f1")]
        store.save(snap2)

        lifecycle = store.get_lifecycle("f1")
        assert lifecycle is not None
        assert lifecycle.status == FindingStatus.PERSISTENT
        assert len(lifecycle.snapshot_ids) == 2
    finally:
        _cleanup_db()


def test_lifecycle_resolved_finding():
    """A finding that disappears should be marked as resolved."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        snap1 = _make_snapshot(sid="snap-1")
        snap1.findings = [_make_finding(fid="f1")]
        store.save(snap1)

        snap2 = _make_snapshot(score=90, sid="snap-2")
        snap2.findings = []  # f1 is gone
        store.save(snap2)

        lifecycle = store.get_lifecycle("f1")
        assert lifecycle is not None
        assert lifecycle.status == FindingStatus.RESOLVED
        assert lifecycle.resolved_snapshot == "snap-2"
    finally:
        _cleanup_db()


def test_lifecycle_recurring_finding():
    """A finding that resolves and comes back should be marked as recurring."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        # Scan 1: finding exists
        snap1 = _make_snapshot(sid="snap-1")
        snap1.findings = [_make_finding(fid="f1")]
        store.save(snap1)

        # Scan 2: finding resolved
        snap2 = _make_snapshot(score=90, sid="snap-2")
        snap2.findings = []
        store.save(snap2)

        # Scan 3: finding comes back
        snap3 = _make_snapshot(score=60, sid="snap-3")
        snap3.findings = [_make_finding(fid="f1")]
        store.save(snap3)

        lifecycle = store.get_lifecycle("f1")
        assert lifecycle is not None
        assert lifecycle.status == FindingStatus.RECURRING
        assert len(lifecycle.snapshot_ids) == 2  # snap-1 and snap-3
    finally:
        _cleanup_db()


def test_get_oldest_findings():
    """get_oldest_findings should return findings sorted by first_seen."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        snap1 = _make_snapshot(sid="snap-1")
        snap1.findings = [_make_finding(fid="old-finding", doc_path="old.md")]
        store.save(snap1)

        snap2 = _make_snapshot(score=80, sid="snap-2")
        snap2.findings = [_make_finding(fid="old-finding", doc_path="old.md"), _make_finding(fid="new-finding", doc_path="new.md")]
        store.save(snap2)

        oldest = store.get_oldest_findings(limit=10)
        assert len(oldest) == 2
        # Old finding should be first (earlier first_seen)
        assert oldest[0].finding_id == "old-finding"
    finally:
        _cleanup_db()


def test_get_recently_resolved():
    """get_recently_resolved should return findings resolved in latest scan."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        snap1 = _make_snapshot(sid="snap-1")
        snap1.findings = [_make_finding(fid="f1"), _make_finding(fid="f2", doc_path="doc2.md")]
        store.save(snap1)

        snap2 = _make_snapshot(score=90, sid="snap-2")
        snap2.findings = [_make_finding(fid="f1")]  # f2 resolved
        store.save(snap2)

        resolved = store.get_recently_resolved(limit=10)
        assert len(resolved) == 1
        assert resolved[0].finding_id == "f2"
        assert resolved[0].resolved_snapshot == "snap-2"
    finally:
        _cleanup_db()


def test_finding_stats():
    """get_finding_stats should return correct counts."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        snap1 = _make_snapshot(sid="snap-1")
        snap1.findings = [_make_finding(fid="f1"), _make_finding(fid="f2", doc_path="doc2.md")]
        store.save(snap1)

        snap2 = _make_snapshot(score=80, sid="snap-2")
        snap2.findings = [_make_finding(fid="f1")]  # f2 resolved, f1 persistent
        store.save(snap2)

        stats = store.get_finding_stats()
        assert stats["persistent"] == 1  # f1
        assert stats["resolved"] == 1  # f2
    finally:
        _cleanup_db()


# --- Trend analysis ---

def test_trend_report():
    """get_trend should return score and finding trends."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        for i, score in enumerate([60, 70, 80, 85]):
            snap = _make_snapshot(score=score, sid=f"snap-{i}")
            snap.findings = [_make_finding(fid="f1")] if i < 3 else []
            store.save(snap)

        trend = store.get_trend(limit=10)
        assert len(trend.score_trend) == 4
        assert trend.score_trend[0]["score"] == 60
        assert trend.score_trend[-1]["score"] == 85
        assert trend.trajectory == "improving"
    finally:
        _cleanup_db()


def test_trend_worsening():
    """Trend should detect worsening scores."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        for i, score in enumerate([85, 80, 70, 60]):
            snap = _make_snapshot(score=score, sid=f"snap-{i}")
            store.save(snap)

        trend = store.get_trend(limit=10)
        assert trend.trajectory == "worsening"
    finally:
        _cleanup_db()


def test_trend_stable():
    """Trend should detect stable scores."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)

        for i, score in enumerate([80, 81, 80, 81]):
            snap = _make_snapshot(score=score, sid=f"snap-{i}")
            store.save(snap)

        trend = store.get_trend(limit=10)
        assert trend.trajectory == "stable"
    finally:
        _cleanup_db()


# --- Finding ID stability ---

def test_finding_id_persisted():
    """Finding IDs should be persisted and loaded correctly."""
    _cleanup_db()
    try:
        store = SnapshotStore(TEST_DB)
        snap = _make_snapshot(sid="snap-1")
        snap.findings = [_make_finding(fid="stable-id-123")]
        store.save(snap)

        loaded = store.load("snap-1")
        assert loaded.findings[0].finding_id == "stable-id-123"
    finally:
        _cleanup_db()
