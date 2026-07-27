"""Tests for the canonical document model."""

from ai_ready.models import (
    AssessedSignal,
    CodeBlock,
    Document,
    Finding,
    FindingStatus,
    Heading,
    KnowledgeSignal,
    Link,
    Paragraph,
    RuleResult,
    Section,
    Severity,
    Snapshot,
)


def test_heading_creation():
    h = Heading(level=1, text="Introduction", line=1)
    assert h.level == 1
    assert h.text == "Introduction"


def test_heading_invalid_level():
    try:
        Heading(level=0, text="Bad")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_document_word_count():
    doc = Document(
        id="test",
        path="test.md",
        title="Test",
        sections=[Section(heading=None, text="hello world foo bar")],
    )
    assert doc.word_count == 4


def test_document_all_text():
    doc = Document(
        id="test",
        path="test.md",
        title="Test",
        sections=[
            Section(heading=None, text="First section"),
            Section(heading=None, text="Second section"),
        ],
    )
    assert "First section" in doc.all_text
    assert "Second section" in doc.all_text


def test_severity_enum():
    assert Severity.CRITICAL.value == "CRITICAL"
    assert Severity.HIGH.value == "HIGH"
    assert Severity.LOW.value == "LOW"


def test_finding_to_dict():
    f = Finding(
        rule_id="test_rule",
        severity=Severity.HIGH,
        score=50,
        document_id="doc1",
        document_path="path/to/doc.md",
        evidence={"key": "value"},
        recommendation="Fix it",
    )
    d = f.to_dict()
    assert d["rule"] == "test_rule"
    assert d["severity"] == "HIGH"
    assert d["score"] == 50


def test_rule_result_to_dict():
    r = RuleResult(
        rule_id="test",
        score=80,
        severity=Severity.LOW,
    )
    d = r.to_dict()
    assert d["rule"] == "test"
    assert d["score"] == 80
    assert d["findings"] == []


def test_finding_to_signal_roundtrip():
    """Finding.to_signal() extracts bare fact; from_signal() rebuilds with placeholders."""
    f = Finding(
        rule_id="test_rule",
        severity=Severity.HIGH,
        score=50,
        document_id="doc1",
        document_path="path/to/doc.md",
        evidence={"key": "value"},
        recommendation="Fix it",
        ai_impact="Bad impact",
    )
    sig = f.to_signal()
    assert sig.collector_id == "test_rule"
    assert sig.signal_type == f.issue_type
    assert sig.artifact_uri == "path/to/doc.md"
    assert sig.evidence == {"key": "value"}
    assert sig.signal_id == f.finding_id
    assert sig.artifact_id == f.document_id
    assert sig.line == f.line

    # Round-trip: from_signal rebuilds a Finding with placeholder interpretation
    rebuilt = Finding.from_signal(sig)
    assert rebuilt.rule_id == f.rule_id
    assert rebuilt.document_path == f.document_path
    assert rebuilt.document_id == f.document_id
    assert rebuilt.evidence == f.evidence
    assert rebuilt.issue_type == f.issue_type
    assert rebuilt.finding_id == f.finding_id
    # Interpretation fields are placeholders
    assert rebuilt.severity == Severity.LOW
    assert rebuilt.score == 100
    assert rebuilt.recommendation == ""
    assert rebuilt.ai_impact == ""


def test_assessed_signal_is_finding():
    """AssessedSignal is a type alias for Finding, not a new class."""
    assert AssessedSignal is Finding
    f = AssessedSignal(
        rule_id="r",
        severity=Severity.LOW,
        score=100,
        document_id="d",
        document_path="p",
        evidence={},
        recommendation="",
    )
    assert isinstance(f, Finding)
    assert f.to_assessed_signal() is f


def test_snapshot_to_dict():
    s = Snapshot(
        snapshot_id="2026-07-22T10:00:00Z",
        score=91,
    )
    d = s.to_dict()
    assert d["snapshot_id"] == "2026-07-22T10:00:00Z"
    assert d["score"] == 91
    assert d["findings"] == []


def test_snapshot_assessment_roundtrip():
    """Snapshot.to_assessment / from_assessment should be lossless."""
    from ai_ready.models import DimensionScore, KnowledgeAssessment

    findings = [
        Finding(
            finding_id="f1",
            rule_id="topic_purity",
            issue_type="mixed_topics",
            severity=Severity.HIGH,
            score=85,
            document_id="doc1",
            document_path="doc1.md",
            evidence={"cluster_count": 3},
            recommendation="Fix",
            ai_impact="Impact",
        ),
    ]
    snap = Snapshot(
        snapshot_id="snap-1",
        score=75,
        dimensions={"retrieval": DimensionScore(name="retrieval", score=70, rule_ids=["topic_purity"])},
        findings=findings,
        metrics={"topic_entropy": 1.5},
        metadata={"source": "test/"},
    )

    assessment = snap.to_assessment()
    assert assessment.assessment_id == "snap-1"
    assert assessment.score == 75
    assert assessment.dimensions["retrieval"].score == 70
    assert len(assessment.signals) == 1
    assert assessment.metrics == {"topic_entropy": 1.5}
    assert assessment.metadata == {"source": "test/"}

    # Round-trip back
    restored = Snapshot.from_assessment(assessment)
    assert restored.snapshot_id == "snap-1"
    assert restored.score == 75
    assert len(restored.findings) == 1
    assert restored.findings[0].finding_id == "f1"
    assert restored.metrics == {"topic_entropy": 1.5}


def test_assessment_snapshot_roundtrip():
    """KnowledgeAssessment.to_snapshot / from_snapshot should be lossless."""
    from ai_ready.models import DimensionScore, KnowledgeAssessment

    signals = [
        Finding(
            finding_id="s1",
            rule_id="heading_quality",
            issue_type="generic",
            severity=Severity.MEDIUM,
            score=90,
            document_id="doc2",
            document_path="doc2.md",
            evidence={"heading": "Overview"},
            recommendation="Fix",
        ),
    ]
    assessment = KnowledgeAssessment(
        assessment_id="assess-1",
        score=80,
        dimensions={"retrieval": DimensionScore(name="retrieval", score=85, rule_ids=["heading_quality"])},
        signals=signals,
        metrics={"avg_topic_entropy": 0.8},
        metadata={"source": "test/"},
    )

    snap = assessment.to_snapshot()
    assert snap.snapshot_id == "assess-1"
    assert snap.score == 80
    assert len(snap.findings) == 1
    assert snap.findings[0].finding_id == "s1"
    assert snap.metrics == {"avg_topic_entropy": 0.8}

    # Round-trip back
    restored = KnowledgeAssessment.from_snapshot(snap)
    assert restored.assessment_id == "assess-1"
    assert len(restored.signals) == 1
    assert restored.signals[0].finding_id == "s1"


def test_assessment_diff_to_regression_report():
    """AssessmentDiff.to_regression_report should map all fields."""
    from ai_ready.models import AssessmentDiff, Finding

    signals = [Finding(
        finding_id="f1", rule_id="r", severity=Severity.HIGH, score=80,
        document_id="d", document_path="p", evidence={}, recommendation="",
    )]
    diff = AssessmentDiff(
        prev_assessment_id="a1",
        curr_assessment_id="a2",
        prev_score=90,
        curr_score=80,
        score_delta=-10,
        new_signals=signals,
        resolved_signals=[],
        persistent_signals=signals,
        severity_changes=[{"direction": "worse"}],
        dimension_deltas={"retrieval": -10},
        new_high_count=1,
        recommendation="Review",
        explanation="Score dropped",
    )

    report = diff.to_regression_report()
    assert report.prev_snapshot_id == "a1"
    assert report.curr_snapshot_id == "a2"
    assert report.score_delta == -10
    assert len(report.new_findings) == 1
    assert report.new_findings[0].finding_id == "f1"
    assert report.new_high_count == 1
    assert report.recommendation == "Review"


def test_evolution_view_to_trend_report():
    """EvolutionView.to_trend_report should map fields correctly."""
    from ai_ready.models import EvolutionView

    evolution = EvolutionView(
        assessments=[{"assessment_id": "a1", "score": 80}, {"assessment_id": "a2", "score": 85}],
        score_trend=[
            {"assessment_id": "a1", "score": 80},
            {"assessment_id": "a2", "score": 85},
        ],
        signal_trend=[
            {"assessment_id": "a1", "total_signals": 5, "high_signals": 2},
            {"assessment_id": "a2", "total_signals": 3, "high_signals": 1},
        ],
        dimension_trends={"retrieval": [{"assessment_id": "a1", "score": 70}]},
        trajectory="improving",
        summary="Score: 80 -> 85",
    )

    trend = evolution.to_trend_report()
    assert len(trend.snapshots) == 2
    assert trend.snapshots[0]["snapshot_id"] == "a1"
    assert len(trend.score_trend) == 2
    assert trend.score_trend[0]["snapshot_id"] == "a1"
    assert len(trend.finding_trend) == 2
    assert trend.finding_trend[0]["total_findings"] == 5
    assert trend.finding_trend[0]["high_findings"] == 2
    assert trend.trajectory == "improving"
    assert trend.summary == "Score: 80 -> 85"


def test_signal_lifecycle_to_finding_lifecycle():
    """SignalLifecycle.to_finding_lifecycle should map all fields."""
    from ai_ready.models import SignalLifecycle, SignalStatus

    sl = SignalLifecycle(
        signal_id="sig-1",
        collector_id="topic_purity",
        artifact_uri="doc.md",
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-02T00:00:00Z",
        status=SignalStatus.PERSISTENT,
        severity_history=[{"severity": "HIGH", "score": 80}],
        assessment_ids=["a1", "a2"],
        resolved_assessment="",
    )

    fl = sl.to_finding_lifecycle()
    assert fl.finding_id == "sig-1"
    assert fl.rule_id == "topic_purity"
    assert fl.document_path == "doc.md"
    assert fl.status == FindingStatus.PERSISTENT
    assert fl.snapshot_ids == ["a1", "a2"]
    assert fl.resolved_snapshot == ""

