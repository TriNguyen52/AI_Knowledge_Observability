"""Tests for the scan pipeline and output formats."""

import json
import tempfile
from pathlib import Path

from ai_ready.connectors.markdown import MarkdownConnector
from ai_ready.models import Severity
from ai_ready.output import format_json, format_sarif, format_terminal
from ai_ready.pipeline import ScanPipeline

FIXTURES = Path(__file__).parent / "fixtures"


def _load_kb(source=FIXTURES):
    """Load documents and relations from source."""
    conn = MarkdownConnector()
    conn.connect(source)
    docs = list(conn.iter_documents())
    relations = list(conn.iter_relations())
    return docs, relations


def test_pipeline_runs_full_scan():
    docs, relations = _load_kb()
    pipeline = ScanPipeline()
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)

    assert snapshot.score >= 0
    assert snapshot.score <= 100
    assert len(snapshot.dimensions) > 0
    assert snapshot.metadata["document_count"] >= 4


def test_pipeline_produces_findings():
    docs, relations = _load_kb()
    pipeline = ScanPipeline()
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)

    # We expect heading quality issues, context independence issues, and broken links
    assert len(snapshot.findings) > 0


def test_pipeline_exit_code_pass():
    docs, relations = _load_kb()
    pipeline = ScanPipeline(thresholds={"overall_score": 0}, fail_on=[])
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)
    assert pipeline.get_exit_code(snapshot) == 0


def test_pipeline_exit_code_threshold_fail():
    docs, relations = _load_kb()
    pipeline = ScanPipeline(thresholds={"overall_score": 200}, fail_on=[])
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)
    assert pipeline.get_exit_code(snapshot) == 1


def test_pipeline_exit_code_high_findings():
    docs, relations = _load_kb()
    pipeline = ScanPipeline(thresholds={"overall_score": 0}, fail_on=["HIGH"])
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)
    # We should have HIGH findings from topic_purity or heading_quality
    exit_code = pipeline.get_exit_code(snapshot)
    assert exit_code in (0, 2)


def test_pipeline_relations_metadata():
    docs, relations = _load_kb()
    pipeline = ScanPipeline()
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)
    assert "relation_count" in snapshot.metadata


def test_format_terminal():
    docs, relations = _load_kb()
    pipeline = ScanPipeline()
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)

    output = format_terminal(snapshot)
    assert "AI Readiness Scan" in output
    assert "Overall Score" in output


def test_format_json():
    docs, relations = _load_kb()
    pipeline = ScanPipeline()
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)

    output = format_json(snapshot)
    data = json.loads(output)
    assert "snapshot_id" in data
    assert "score" in data
    assert "dimensions" in data


def test_format_sarif():
    docs, relations = _load_kb()
    pipeline = ScanPipeline()
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)

    output = format_sarif(snapshot)
    data = json.loads(output)
    assert data["version"] == "2.1.0"
    assert len(data["runs"]) == 1
    assert data["runs"][0]["tool"]["driver"]["name"] == "ai-ready"


def test_config_loading():
    from ai_ready.config import Config

    config = Config.default()
    assert config.weights["retrieval"] == 0.25
    assert "CRITICAL" in config.fail_on


def test_pipeline_signal_conversion_roundtrip():
    """Phase 8: CollectOperation emits KnowledgeSignals, AssessOperation
    converts them to AssessedSignals (Findings) with policy-enriched interpretation.

    Verifies that the signal conversion roundtrip preserves finding IDs,
    document paths, and that policy is correctly applied after conversion.
    """
    from ai_ready.operations import CollectOperation, AssessOperation

    docs, relations = _load_kb()
    collect_op = CollectOperation()
    results, signals, kb = collect_op.run(
        documents=docs, relations=relations, source=str(FIXTURES)
    )

    # CollectOperation should produce KnowledgeSignal objects
    from ai_ready.models import KnowledgeSignal
    assert len(signals) > 0
    assert all(isinstance(s, KnowledgeSignal) for s in signals)

    # AssessOperation should convert signals to AssessedSignals (Findings)
    assess_op = AssessOperation()
    snapshot = assess_op.run(
        results=results, signals=signals, documents=docs, kb=kb,
        source=str(FIXTURES),
    )

    # Findings should have policy-enriched interpretation (not placeholders)
    from ai_ready.models import Finding
    assert len(snapshot.findings) > 0
    assert all(isinstance(f, Finding) for f in snapshot.findings)

    # At least some findings should have non-default severity or score
    # (meaning policy was applied, not just placeholder values)
    has_enriched = any(
        f.severity != Severity.LOW or f.score != 100 or f.recommendation != ""
        for f in snapshot.findings
    )
    assert has_enriched, "Policy should enrich at least some findings"

    # Finding IDs should match signal IDs (roundtrip preserves IDs)
    signal_ids = {s.signal_id for s in signals}
    finding_ids = {f.finding_id for f in snapshot.findings}
    assert finding_ids == signal_ids


def test_pipeline_incremental_still_works():
    """Phase 8: Incremental execution should still work unchanged.
    It uses pipeline helper methods directly with Finding objects,
    bypassing the signal conversion path.
    """
    from ai_ready.incremental import ChangeEvent, IncrementalExecutor

    docs, relations = _load_kb()
    pipeline = ScanPipeline()
    snapshot = pipeline.run(docs, source=str(FIXTURES), relations=relations)

    # Simulate a modification to one document
    from ai_ready.models import Document, Section
    modified_doc = Document(
        id=docs[0].id,
        path=docs[0].path,
        title=docs[0].title,
        sections=docs[0].sections,
    )

    change_events = [ChangeEvent(
        event_type="modified",
        document_path=docs[0].path,
        document=modified_doc,
        links_changed=False,
    )]

    new_snapshot = pipeline.run_incremental(
        prev_snapshot=snapshot,
        change_events=change_events,
        documents=docs,
        relations=relations,
        source=str(FIXTURES),
    )

    assert new_snapshot.score >= 0
    assert new_snapshot.score <= 100


def test_config_from_file():
    from ai_ready.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write("""
version: 1
thresholds:
  overall_score: 85
fail_on:
  - HIGH
weights:
  retrieval: 0.30
  consistency: 0.15
  trust: 0.20
  connectivity: 0.15
  workflow: 0.10
  context: 0.10
rules:
  topic_purity:
    enabled: true
  context_independence:
    enabled: false
""")
        f.flush()

        config = Config.from_file(f.name)
        assert config.thresholds["overall_score"] == 85
        assert config.weights["retrieval"] == 0.30
        assert "topic_purity" in config.enabled_rule_ids
        assert "context_independence" not in config.enabled_rule_ids


def test_config_collectors_key():
    """Config should accept 'collectors' key alongside 'rules'."""
    from ai_ready.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write("""
version: 1
collectors:
  topic_purity:
    enabled: true
  heading_quality:
    enabled: false
""")
        f.flush()

        config = Config.from_file(f.name)
        assert "topic_purity" in config.enabled_collector_ids
        assert "heading_quality" not in config.enabled_collector_ids
        # enabled_rule_ids and enabled_collector_ids should be the same
        assert config.enabled_rule_ids == config.enabled_collector_ids


def test_config_collectors_takes_precedence():
    """If both 'rules' and 'collectors' are present, collectors takes precedence."""
    from ai_ready.config import Config

    data = {
        "rules": {"topic_purity": {"enabled": True}},
        "collectors": {"heading_quality": {"enabled": True}},
    }
    config = Config.from_dict(data)
    # collectors should take precedence
    assert "heading_quality" in config.enabled_rule_ids
    assert "topic_purity" not in config.enabled_rule_ids
