"""Context Independence collector — detect chunks that cannot stand alone when retrieved."""

from __future__ import annotations

import re
from typing import Any

from ai_ready.models import ArtifactBundle, CollectorResult, KnowledgeSignal, Severity
from ai_ready.rules import SignalCollector, register_collector

# Patterns that indicate dangling references
DANGLING_PATTERNS = [
    (r"\bas mentioned above\b", "References content above without local context"),
    (r"\bas mentioned below\b", "References content below without local context"),
    (r"\bas discussed above\b", "References discussion above without local context"),
    (r"\bas discussed earlier\b", "References earlier discussion without local context"),
    (r"\bsee above\b", "References content above without local context"),
    (r"\bsee below\b", "References content below without local context"),
    (r"\bprevious example\b", "References a previous example without local context"),
    (r"\bprevious section\b", "References a previous section without local context"),
    (r"\bfollowing section\b", "References a following section without local context"),
    (r"\bthe above\b", "References content above without local context"),
    (r"\bthe below\b", "References content below without local context"),
    (r"\bthe previous\b", "References something previous without local context"),
    (r"\bthe following\b", "References something following without local context"),
    (r"\bin the previous step\b", "References a previous step without local context"),
    (r"\bin the next step\b", "References a next step without local context"),
    (r"\bthe aforementioned\b", "References aforementioned content without local context"),
    (r"\bthe latter\b", "References 'the latter' without local context"),
    (r"\bthe former\b", "References 'the former' without local context"),
]

# Undefined entity patterns — pronouns that may lack antecedents
UNDEFINED_ENTITY_PATTERNS = [
    (r"^\s*(this|that|these|those)\s+(?:feature|function|method|option|setting|field|parameter|value|property)\b",
     "Starts with a demonstrative reference that may lack context"),
    (r"^\s*it\s+(?:is|was|will|should|can|must|does|has)\b",
     "Starts with 'it' that may lack an antecedent"),
]


@register_collector
class ContextIndependenceCollector(SignalCollector):
    """Detect paragraphs that reference content outside themselves."""

    id = "context_independence"
    severity_default = Severity.MEDIUM
    dimension = "context"

    def collect(self, bundle: ArtifactBundle) -> dict[str, Any]:
        """Scan all paragraphs for dangling references."""
        results: list[dict[str, Any]] = []
        for artifact in bundle.artifacts:
            for i, para in enumerate(artifact.paragraphs):
                text_lower = para.text.lower()
                dangling: list[dict[str, str]] = []

                for pattern, reason in DANGLING_PATTERNS:
                    matches = list(re.finditer(pattern, text_lower))
                    for m in matches:
                        dangling.append({
                            "pattern": m.group(0),
                            "reason": reason,
                            "offset": m.start(),
                        })

                for pattern, reason in UNDEFINED_ENTITY_PATTERNS:
                    matches = list(re.finditer(pattern, para.text, re.MULTILINE))
                    for m in matches:
                        dangling.append({
                            "pattern": m.group(0).strip(),
                            "reason": reason,
                            "offset": m.start(),
                        })

                if dangling:
                    results.append({
                        "doc_id": artifact.id,
                        "doc_path": artifact.uri,
                        "paragraph_index": i,
                        "paragraph_text": para.text[:200],
                        "section_heading": para.section_heading,
                        "line": para.line,
                        "dangling_references": dangling,
                    })

        return {"paragraph_issues": results}

    def measure(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Count dangling references per artifact."""
        doc_counts: dict[str, dict[str, Any]] = {}
        for issue in signals["paragraph_issues"]:
            doc_id = issue["doc_id"]
            if doc_id not in doc_counts:
                doc_counts[doc_id] = {
                    "doc_id": doc_id,
                    "doc_path": issue["doc_path"],
                    "dangling_count": 0,
                    "undefined_entity_count": 0,
                    "issues": [],
                }
            doc_counts[doc_id]["dangling_count"] += len(issue["dangling_references"])
            doc_counts[doc_id]["issues"].append(issue)

        return {"doc_counts": list(doc_counts.values())}

    def evaluate(self, metrics: dict[str, Any]) -> list[KnowledgeSignal]:
        """Flag artifacts with dangling references.

        Emits bare signals with signal_type only. Severity, score, and
        recommendation are assigned by the InterpretationPolicy in the pipeline.
        """
        signals: list[KnowledgeSignal] = []
        for dc in metrics["doc_counts"]:
            if dc["dangling_count"] > 0:
                signals.append(KnowledgeSignal(
                    collector_id=self.id,
                    signal_type="dangling_reference",
                    severity=Severity.LOW,  # Placeholder; overridden by policy
                    score=0,  # Placeholder; overridden by policy
                    artifact_id=dc["doc_id"],
                    artifact_uri=dc["doc_path"],
                    evidence={
                        "dangling_reference_count": dc["dangling_count"],
                        "issues": [
                            {
                                "paragraph": i["paragraph_text"],
                                "section": i["section_heading"],
                                "line": i["line"],
                                "references": [d["pattern"] for d in i["dangling_references"]],
                            }
                            for i in dc["issues"]
                        ],
                    },
                    recommendation="",  # Filled by policy
                ))
        return signals

    def report(self) -> CollectorResult:
        signals = self._findings
        total_dangling = sum(s.evidence.get("dangling_reference_count", 0) for s in signals)

        score = max(0, 100 - len(signals) * 10)
        severity = Severity.MEDIUM if signals else Severity.LOW

        return CollectorResult(
            collector_id=self.id,
            score=score,
            severity=severity,
            metrics={
                "total_dangling_references": total_dangling,
                "flagged_documents": len(signals),
            },
            signals=signals,
            recommendation=f"{total_dangling} dangling references across {len(signals)} artifacts" if signals else "All paragraphs are self-contained",
        )
