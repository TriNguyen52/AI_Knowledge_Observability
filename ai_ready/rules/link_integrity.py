"""Link Integrity collector - detect broken, redirect, and orphan links.

Uses artifact relationships from the ArtifactBundle to avoid false-positive
orphan detection. An artifact is only an orphan if it is not linked from any
other artifact AND not present in the navigation hierarchy.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ai_ready.models import ArtifactBundle, CollectorResult, KnowledgeSignal, Severity
from ai_ready.rules import SignalCollector, register_collector


@register_collector
class LinkIntegrityCollector(SignalCollector):
    """Check internal links for broken references and detect orphan artifacts."""

    id = "link_integrity"
    severity_default = Severity.MEDIUM
    dimension = "connectivity"

    def __init__(self) -> None:
        self._source_path: str | None = None

    def collect(self, bundle: ArtifactBundle) -> dict[str, Any]:
        """Collect all links and check their targets. Use relationships for orphan detection."""
        all_links: list[dict[str, Any]] = []
        doc_paths = {artifact.uri for artifact in bundle.artifacts}
        doc_dirs = {os.path.dirname(artifact.uri) for artifact in bundle.artifacts}

        for artifact in bundle.artifacts:
            for link in artifact.links:
                if not link.is_internal:
                    continue

                target = link.target
                # Strip anchors
                anchor = ""
                if "#" in target:
                    target, _, anchor = target.partition("#")

                # Strip query params
                if "?" in target:
                    target = target.split("?")[0]

                if not target:
                    # Pure anchor link - skip
                    continue

                # Check if target exists relative to artifact's directory and source root
                doc_dir = os.path.dirname(artifact.uri)
                resolved = self._resolve_link(target, doc_dir, doc_paths)

                all_links.append({
                    "doc_id": artifact.id,
                    "doc_path": artifact.uri,
                    "link_text": link.text,
                    "link_target": link.target,
                    "clean_target": target,
                    "anchor": anchor,
                    "line": link.line,
                    "exists": resolved is not None,
                    "resolved_path": resolved,
                    "is_orphan": False,
                })

        # Check if the link target matches an artifact URI
        for al in all_links:
            if al["clean_target"] in doc_paths:
                al["resolved_path"] = al["clean_target"]

        # Compute orphan artifacts using BOTH inline links AND relationships
        # An artifact is orphan if no other artifact links to it AND it has no
        # parent in the navigation hierarchy
        linked_via_links = {al["resolved_path"] for al in all_links if al["resolved_path"]}
        linked_via_relations = {r.target_uri for r in bundle.relationships if r.relation_type in ("parent_child", "cross_reference", "navigation")}

        all_linked = linked_via_links | linked_via_relations

        orphan_docs = []
        for artifact in bundle.artifacts:
            is_linked = artifact.uri in all_linked
            # Also check if any relationship has this artifact as a source (it's a parent, so it's reachable)
            is_parent = any(r.source_uri == artifact.uri for r in bundle.relationships if r.relation_type in ("parent_child", "navigation"))
            if not is_linked and not is_parent and len(bundle.artifacts) > 1:
                orphan_docs.append({"doc_id": artifact.id, "doc_path": artifact.uri})

        return {
            "all_links": all_links,
            "orphan_docs": orphan_docs,
            "total_docs": len(bundle.artifacts),
            "total_relations": len(bundle.relationships),
        }

    def measure(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Count broken links and orphan artifacts."""
        broken = [l for l in signals["all_links"] if not l["exists"]]
        orphans = signals["orphan_docs"]

        return {
            "broken_links": broken,
            "orphan_documents": orphans,
            "total_links": len(signals["all_links"]),
            "total_docs": signals["total_docs"],
            "total_relations": signals["total_relations"],
            "broken_count": len(broken),
            "orphan_count": len(orphans),
        }

    def evaluate(self, metrics: dict[str, Any]) -> list[KnowledgeSignal]:
        """Create signals for broken links and orphan artifacts.

        Emits bare signals with signal_type only. Severity, score, and
        recommendation are assigned by the InterpretationPolicy in the pipeline.
        """
        signals: list[KnowledgeSignal] = []

        # Group broken links by artifact
        broken_by_doc: dict[str, list[dict[str, Any]]] = {}
        for link in metrics["broken_links"]:
            broken_by_doc.setdefault(link["doc_id"], []).append(link)

        for doc_id, links in broken_by_doc.items():
            doc_path = links[0]["doc_path"]
            signals.append(KnowledgeSignal(
                collector_id=self.id,
                signal_type="broken_link",
                severity=Severity.LOW,  # Placeholder; overridden by policy
                score=0,  # Placeholder; overridden by policy
                artifact_id=doc_id,
                artifact_uri=doc_path,
                evidence={
                    "broken_links": [
                        {"target": l["link_target"], "line": l["line"], "text": l["link_text"]}
                        for l in links
                    ],
                    "broken_link_count": len(links),
                },
                recommendation="",  # Filled by policy
            ))

        # Orphan artifacts
        for orphan in metrics["orphan_documents"]:
            signals.append(KnowledgeSignal(
                collector_id=self.id,
                signal_type="orphan",
                severity=Severity.LOW,  # Placeholder; overridden by policy
                score=0,  # Placeholder; overridden by policy
                artifact_id=orphan["doc_id"],
                artifact_uri=orphan["doc_path"],
                evidence={"orphan": True},
                recommendation="",  # Filled by policy
            ))

        return signals

    def report(self) -> CollectorResult:
        signals = self._findings
        broken_count = self._metrics.get("broken_count", 0)
        orphan_count = self._metrics.get("orphan_count", 0)
        total_links = self._metrics.get("total_links", 0)
        total_relations = self._metrics.get("total_relations", 0)

        score = max(0, 100 - broken_count * 5 - orphan_count * 2)
        severity = Severity.HIGH if broken_count > 5 else (Severity.MEDIUM if signals else Severity.LOW)

        return CollectorResult(
            collector_id=self.id,
            score=score,
            severity=severity,
            metrics={
                "broken_links": broken_count,
                "orphan_documents": orphan_count,
                "total_links": total_links,
                "total_relations": total_relations,
            },
            signals=signals,
            recommendation=f"{broken_count} broken links, {orphan_count} orphan artifacts" if signals else "All links resolve correctly",
        )

    def _resolve_link(self, target: str, doc_dir: str, known_paths: set[str]) -> str | None:
        """Try to resolve a relative link target."""
        # Normalize path separators
        target = target.replace("\\", "/")
        doc_dir = doc_dir.replace("\\", "/")

        # Try relative to artifact directory
        candidate = os.path.normpath(os.path.join(doc_dir, target)).replace("\\", "/")
        if candidate in known_paths:
            return candidate

        # Try as-is
        if target in known_paths:
            return target

        # Try without leading ./
        if target.startswith("./"):
            clean = target[2:]
            if clean in known_paths:
                return clean

        # Try with .md extension
        if not target.endswith((".md", ".mdx", ".txt", ".rst")):
            for ext in [".md", ".mdx", ".txt", ".rst"]:
                candidate = target + ext
                if candidate in known_paths:
                    return candidate
                candidate = os.path.normpath(os.path.join(doc_dir, target + ext)).replace("\\", "/")
                if candidate in known_paths:
                    return candidate

        return None
