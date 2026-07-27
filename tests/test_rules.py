"""Tests for Phase 1 rules."""

from pathlib import Path

from ai_ready.connectors.markdown import MarkdownConnector
from ai_ready.models import Document, Heading, KnowledgeBase, Paragraph, Section, Severity
from ai_ready.rules.context_independence import ContextIndependenceRule
from ai_ready.rules.heading_quality import HeadingQualityRule
from ai_ready.rules.link_integrity import LinkIntegrityRule
from ai_ready.rules.topic_purity import TopicPurityRule

FIXTURES = Path(__file__).parent / "fixtures"


def _load_kb(docs=None) -> KnowledgeBase:
    """Load documents from fixtures into a KnowledgeBase."""
    if docs is None:
        conn = MarkdownConnector()
        conn.connect(FIXTURES)
        docs = list(conn.iter_documents())
    return KnowledgeBase(documents=docs)


def _load_docs():
    """Load documents from fixtures (for backward compat)."""
    conn = MarkdownConnector()
    conn.connect(FIXTURES)
    return list(conn.iter_documents())


# --- Topic Purity ---

def test_topic_purity_runs():
    kb = _load_kb()
    rule = TopicPurityRule()
    result = rule.run(kb)
    assert result.rule_id == "topic_purity"
    assert 0 <= result.score <= 100


def test_topic_purity_detects_mixed_topics():
    doc = Document(
        id="mixed",
        path="mixed.md",
        title="Mixed Doc",
        headings=[
            Heading(level=1, text="Password Reset", line=1),
            Heading(level=2, text="Reset Steps", line=3),
            Heading(level=1, text="Billing Invoices", line=10),
            Heading(level=2, text="Payment Methods", line=15),
        ],
        sections=[
            Section(heading=Heading(level=1, text="Password Reset"), text="reset content"),
            Section(heading=Heading(level=1, text="Billing Invoices"), text="billing content"),
        ],
    )
    kb = _load_kb([doc])
    rule = TopicPurityRule()
    result = rule.run(kb)
    assert len(result.findings) > 0
    assert result.findings[0].issue_type == "mixed_topics"
    # NOTE: severity is now assigned by policy, not by the rule
    assert result.findings[0].severity == Severity.LOW  # Placeholder before policy


def test_topic_purity_passes_focused_doc():
    doc = Document(
        id="focused",
        path="focused.md",
        title="OAuth Guide",
        headings=[
            Heading(level=1, text="Configure OAuth", line=1),
            Heading(level=2, text="Step 1: Create App", line=3),
            Heading(level=2, text="Step 2: Set Redirect", line=10),
        ],
        sections=[
            Section(heading=Heading(level=1, text="Configure OAuth"), text="oauth content"),
        ],
    )
    kb = _load_kb([doc])
    rule = TopicPurityRule()
    result = rule.run(kb)
    assert len(result.findings) == 0


# --- Context Independence ---

def test_context_independence_runs():
    kb = _load_kb()
    rule = ContextIndependenceRule()
    result = rule.run(kb)
    assert result.rule_id == "context_independence"
    assert 0 <= result.score <= 100


def test_context_independence_detects_dangling_refs():
    doc = Document(
        id="dangling",
        path="dangling.md",
        title="Test",
        paragraphs=[
            Paragraph(text="As mentioned above, you need to configure this."),
            Paragraph(text="See below for more information."),
            Paragraph(text="This is a self-contained paragraph with no references."),
        ],
    )
    kb = _load_kb([doc])
    rule = ContextIndependenceRule()
    result = rule.run(kb)
    assert len(result.findings) > 0
    assert result.findings[0].evidence["dangling_reference_count"] >= 2


def test_context_independence_passes_clean_doc():
    doc = Document(
        id="clean",
        path="clean.md",
        title="Test",
        paragraphs=[
            Paragraph(text="OAuth requires a client ID and secret."),
            Paragraph(text="The redirect URI must match exactly."),
        ],
    )
    kb = _load_kb([doc])
    rule = ContextIndependenceRule()
    result = rule.run(kb)
    assert len(result.findings) == 0


# --- Heading Quality ---

def test_heading_quality_runs():
    kb = _load_kb()
    rule = HeadingQualityRule()
    result = rule.run(kb)
    assert result.rule_id == "heading_quality"
    assert 0 <= result.score <= 100


def test_heading_quality_detects_generic():
    doc = Document(
        id="generic",
        path="generic.md",
        title="Test",
        headings=[
            Heading(level=1, text="Overview", line=1),
            Heading(level=2, text="Configuration", line=3),
            Heading(level=2, text="Configure OAuth Authentication", line=10),
        ],
    )
    kb = _load_kb([doc])
    rule = HeadingQualityRule()
    result = rule.run(kb)
    assert len(result.findings) >= 2  # "Overview" and "Configuration" are generic


def test_heading_quality_detects_placeholder():
    doc = Document(
        id="placeholder",
        path="placeholder.md",
        title="Test",
        headings=[
            Heading(level=1, text="TODO", line=1),
            Heading(level=2, text="Coming Soon", line=3),
        ],
    )
    kb = _load_kb([doc])
    rule = HeadingQualityRule()
    result = rule.run(kb)
    assert len(result.findings) >= 2
    assert any(f.issue_type == "placeholder" for f in result.findings)
    # NOTE: severity is now assigned by policy, not by the rule


def test_heading_quality_detects_numbered_only():
    doc = Document(
        id="numbered",
        path="numbered.md",
        title="Test",
        headings=[
            Heading(level=2, text="Step 1", line=1),
            Heading(level=2, text="Step 2", line=5),
        ],
    )
    kb = _load_kb([doc])
    rule = HeadingQualityRule()
    result = rule.run(kb)
    assert len(result.findings) >= 2
    assert all(f.evidence["issue_type"] == "numbered_only" for f in result.findings)


# --- Link Integrity ---

def test_link_integrity_runs():
    kb = _load_kb()
    rule = LinkIntegrityRule()
    result = rule.run(kb)
    assert result.rule_id == "link_integrity"
    assert 0 <= result.score <= 100


def test_link_integrity_detects_broken_links():
    kb = _load_kb()
    rule = LinkIntegrityRule()
    result = rule.run(kb)
    # billing.md links to missing.md which doesn't exist
    broken_findings = [f for f in result.findings if f.evidence.get("broken_link_count", 0) > 0]
    assert len(broken_findings) > 0
    # Check that the broken link to missing.md was found
    found_broken = False
    for f in broken_findings:
        for link in f.evidence.get("broken_links", []):
            if "missing" in link.get("target", ""):
                found_broken = True
                break
    assert found_broken


def test_link_integrity_detects_orphan_documents():
    kb = _load_kb()
    rule = LinkIntegrityRule()
    result = rule.run(kb)
    orphan_findings = [f for f in result.findings if f.evidence.get("orphan")]
    # orphan_page.md should be detected as orphan
    assert len(orphan_findings) > 0


# --- Finding ID stability ---

def test_finding_id_stable_across_line_shifts():
    """Finding IDs should be stable when content shifts by lines."""
    doc1 = Document(
        id="doc1",
        path="auth.md",
        title="Auth",
        headings=[
            Heading(level=1, text="Password Reset", line=1),
            Heading(level=1, text="Billing Invoices", line=10),
        ],
        sections=[
            Section(heading=Heading(level=1, text="Password Reset"), text="reset"),
            Section(heading=Heading(level=1, text="Billing Invoices"), text="billing"),
        ],
    )
    doc2 = Document(
        id="doc1",
        path="auth.md",
        title="Auth",
        headings=[
            Heading(level=1, text="Password Reset", line=5),  # Different line
            Heading(level=1, text="Billing Invoices", line=20),  # Different line
        ],
        sections=[
            Section(heading=Heading(level=1, text="Password Reset"), text="reset"),
            Section(heading=Heading(level=1, text="Billing Invoices"), text="billing"),
        ],
    )

    kb1 = _load_kb([doc1])
    kb2 = _load_kb([doc2])
    rule = TopicPurityRule()
    r1 = rule.run(kb1)
    r2 = rule.run(kb2)

    # Same finding IDs despite different line numbers
    ids1 = {f.finding_id for f in r1.findings}
    ids2 = {f.finding_id for f in r2.findings}
    assert ids1 == ids2, f"Finding IDs should match: {ids1} vs {ids2}"


def test_finding_id_different_for_different_docs():
    """Finding IDs should differ for the same rule on different documents."""
    doc1 = Document(
        id="doc1", path="auth.md", title="Auth",
        headings=[Heading(level=1, text="Password Reset", line=1), Heading(level=1, text="Billing", line=10)],
        sections=[Section(heading=Heading(level=1, text="Password Reset"), text="x"), Section(heading=Heading(level=1, text="Billing"), text="y")],
    )
    doc2 = Document(
        id="doc2", path="config.md", title="Config",
        headings=[Heading(level=1, text="Password Reset", line=1), Heading(level=1, text="Billing", line=10)],
        sections=[Section(heading=Heading(level=1, text="Password Reset"), text="x"), Section(heading=Heading(level=1, text="Billing"), text="y")],
    )

    kb1 = _load_kb([doc1])
    kb2 = _load_kb([doc2])
    rule = TopicPurityRule()
    r1 = rule.run(kb1)
    r2 = rule.run(kb2)

    ids1 = {f.finding_id for f in r1.findings}
    ids2 = {f.finding_id for f in r2.findings}
    assert ids1 != ids2, "Finding IDs should differ for different documents"


# --- Evaluation Policy ---

def test_policy_enriches_findings():
    """Verify that policy is applied and populates severity, score, ai_impact."""
    from ai_ready.evaluation_policy import EvaluationPolicy
    from ai_ready.pipeline import ScanPipeline

    doc = Document(
        id="doc1", path="test.md", title="Test",
        headings=[Heading(level=1, text="TODO", line=1)],
    )
    kb = _load_kb([doc])
    rule = HeadingQualityRule()
    result = rule.run(kb)

    # Before policy: severity is placeholder
    assert result.findings[0].severity == Severity.LOW
    assert result.findings[0].score == 0
    assert result.findings[0].recommendation == ""
    assert result.findings[0].ai_impact == ""

    # Apply policy
    pipeline = ScanPipeline()
    pipeline.apply_policy(result.findings)

    # After policy: severity from policy registry
    assert result.findings[0].severity == Severity.CRITICAL
    assert result.findings[0].score == 80  # 100 - 20 penalty
    assert result.findings[0].ai_impact != ""
    assert result.findings[0].recommendation != ""


def test_policy_lookup_all_issue_types():
    """Verify all issue types have policy entries."""
    from ai_ready.evaluation_policy import EvaluationPolicy

    policy = EvaluationPolicy()
    expected_entries = [
        ("topic_purity", "mixed_topics"),
        ("heading_quality", "generic"),
        ("heading_quality", "placeholder"),
        ("heading_quality", "numbered_only"),
        ("heading_quality", "too_short"),
        ("context_independence", "dangling_reference"),
        ("context_independence", "undefined_entity"),
        ("link_integrity", "broken_link"),
        ("link_integrity", "orphan"),
    ]

    for rule_id, issue_type in expected_entries:
        entry = policy.lookup(rule_id, issue_type)
        assert entry is not None, f"Missing policy entry for ({rule_id}, {issue_type})"
        assert entry.severity is not None
        assert entry.score_penalty > 0
        assert entry.ai_impact != ""
        assert entry.recommendation_template != ""


def test_policy_validation():
    """Verify policy validation passes with default entries."""
    from ai_ready.evaluation_policy import EvaluationPolicy

    policy = EvaluationPolicy()
    errors = policy.validate()
    assert errors == [], f"Policy validation errors: {errors}"


def test_pipeline_enriches_findings_end_to_end():
    """Verify the full pipeline enriches findings with policy data."""
    from ai_ready.pipeline import ScanPipeline

    docs = _load_docs()
    pipeline = ScanPipeline()
    snapshot = pipeline.run(docs, source=str(FIXTURES))

    # All findings should have been enriched by policy
    for f in snapshot.findings:
        assert f.issue_type != "", f"Finding {f.finding_id} has no issue_type"
        assert f.ai_impact != "", f"Finding {f.finding_id} has no ai_impact"
        assert f.recommendation != "", f"Finding {f.finding_id} has no recommendation"
        assert f.severity != Severity.LOW or f.issue_type == "orphan", (
            f"Finding {f.finding_id} unexpectedly LOW severity"
        )


def test_collector_aliases_are_rule_classes():
    """Collector aliases should be the same class objects as their Rule counterparts."""
    from ai_ready.rules.topic_purity import TopicPurityCollector, TopicPurityRule
    from ai_ready.rules.heading_quality import HeadingQualityCollector, HeadingQualityRule
    from ai_ready.rules.context_independence import ContextIndependenceCollector, ContextIndependenceRule
    from ai_ready.rules.link_integrity import LinkIntegrityCollector, LinkIntegrityRule

    assert TopicPurityCollector is TopicPurityRule
    assert HeadingQualityCollector is HeadingQualityRule
    assert ContextIndependenceCollector is ContextIndependenceRule
    assert LinkIntegrityCollector is LinkIntegrityRule


def test_collector_aliases_accessible_from_rules_init():
    """Collector aliases should be importable from ai_ready.rules."""
    from ai_ready.rules import (
        TopicPurityCollector,
        HeadingQualityCollector,
        ContextIndependenceCollector,
        LinkIntegrityCollector,
    )

    # Verify they're usable (can be instantiated)
    assert TopicPurityCollector.id == "topic_purity"
    assert HeadingQualityCollector.id == "heading_quality"
    assert ContextIndependenceCollector.id == "context_independence"
    assert LinkIntegrityCollector.id == "link_integrity"
