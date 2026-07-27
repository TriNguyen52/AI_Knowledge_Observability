"""Capability-based navigation relation extraction for SDKs."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from itertools import islice
from pathlib import Path

from ai_ready.models import KnowledgeArtifact, Relationship


def _add_relation(
    relations: list[Relationship],
    source: str,
    target: str,
    relation_type: str,
    metadata: dict | None = None,
) -> None:
    for relation in relations:
        if (
            relation.source_uri == source
            and relation.target_uri == target
            and relation.relation_type == relation_type
        ):
            return

    relations.append(
        Relationship(
            source_uri=source,
            target_uri=target,
            relation_type=relation_type,
            metadata=metadata or {},
        )
    )


def _relative_path(path: Path, source_root: Path | None) -> str:
    if source_root and source_root.is_dir():
        try:
            return str(path.relative_to(source_root)).replace("\\", "/")
        except ValueError:
            pass
    return str(path).replace("\\", "/")


def build_document_lookup(artifacts: list[KnowledgeArtifact]) -> dict[str, str]:
    """Build a generic lookup from artifact ids, stems, and relative paths to a uri."""
    lookup: dict[str, str] = {}
    for artifact in artifacts:
        lookup[artifact.uri] = artifact.uri
        stem = Path(artifact.uri).stem
        lookup.setdefault(stem, artifact.uri)
        lookup.setdefault(stem.replace("/", "_"), artifact.uri)
    return lookup


def resolve_document_reference(
    target: str,
    base_dir: str,
    known_paths: set[str],
) -> str | None:
    """Resolve a relative reference against a document set."""
    target = target.replace("\\", "/")
    base_dir = base_dir.replace("\\", "/")

    candidate = os.path.normpath(os.path.join(base_dir, target)).replace("\\", "/")
    if candidate in known_paths:
        return candidate

    if target in known_paths:
        return target

    if target.startswith("./"):
        clean = target[2:]
        if clean in known_paths:
            return clean

    if not target.endswith((".md", ".mdx", ".txt", ".rst")):
        for ext in [".md", ".mdx", ".txt", ".rst"]:
            candidate = target + ext
            if candidate in known_paths:
                return candidate
            candidate = os.path.normpath(os.path.join(base_dir, target + ext)).replace("\\", "/")
            if candidate in known_paths:
                return candidate

    return None


class NavigationManifestParser(ABC):
    """Parse a navigation manifest into document relations."""

    @abstractmethod
    def matches(self, path: Path, content: str) -> bool:
        """Return True when this parser can read the manifest."""

    @abstractmethod
    def extract(
        self,
        path: Path,
        content: str,
        artifacts: list[KnowledgeArtifact],
        source_root: Path,
    ) -> list[Relationship]:
        """Extract normalized relations from the manifest."""


class MkDocsNavigationParser(NavigationManifestParser):
    """Parse MkDocs nav structures into parent/child relationships."""

    def matches(self, path: Path, content: str) -> bool:
        return path.name == "mkdocs.yml"

    def extract(
        self,
        path: Path,
        content: str,
        artifacts: list[KnowledgeArtifact],
        source_root: Path,
    ) -> list[Relationship]:
        docs_dir_match = re.search(r"^docs_dir:\s*(.+)$", content, re.MULTILINE)
        docs_dir = docs_dir_match.group(1).strip().strip('"').strip("'") if docs_dir_match else "docs"
        docs_path = path.parent / docs_dir
        if not docs_path.exists() and source_root.is_dir():
            if source_root.name in {"docs", "content"}:
                docs_path = source_root

        nav_match = re.search(r"^nav:\s*$", content, re.MULTILINE)
        if not nav_match:
            return []

        lookup = build_document_lookup(artifacts)
        known_paths = set(lookup.values())
        relations: list[Relationship] = []
        nav_lines = content[nav_match.end():].split("\n")
        self._parse_nav_lines(nav_lines, docs_path, source_root, known_paths, relations, parent_path="")
        return relations

    def _parse_nav_lines(
        self,
        lines: list[str],
        docs_path: Path,
        source_root: Path,
        known_paths: set[str],
        relations: list[Relationship],
        parent_path: str,
        parent_index_path: str | None = None,
        indent: int = 0,
    ) -> None:
        section_index_path: str | None = None
        section_files: list[str] = []

        for index, line in enumerate(lines):
            if not line.strip():
                continue

            stripped = line.strip()
            if not stripped.startswith("-"):
                break

            line_indent = len(line) - len(line.lstrip())
            if line_indent < indent:
                break
            if line_indent > indent:
                continue

            item = stripped[1:].strip()
            if item.startswith("- "):
                item = item[2:].strip()

            if ":" in item:
                colon_idx = item.index(":")
                section_name = item[:colon_idx].strip().strip('"').strip("'")
                remainder = item[colon_idx + 1 :].strip()

                if remainder and remainder.endswith(".md"):
                    file_path = self._resolve_doc_path(remainder, docs_path, source_root, known_paths)
                    if file_path:
                        section_files.append(file_path)
                        current_section = section_name
                        section_index_path = file_path
                        _add_relation(relations, "__nav__", file_path, "navigation")
                else:
                    section_index_path = None

                child_lines = lines[index + 1 :]
                child_indent = line_indent + 2
                next_parent_index = section_index_path or parent_index_path
                self._parse_nav_lines(
                    child_lines,
                    docs_path,
                    source_root,
                    known_paths,
                    relations,
                    section_name,
                    next_parent_index,
                    child_indent,
                )
                continue

            if item.endswith(".md"):
                file_path = self._resolve_doc_path(item, docs_path, source_root, known_paths)
                if file_path:
                    section_files.append(file_path)
                    _add_relation(relations, "__nav__", file_path, "navigation")
                    parent_idx = section_index_path or parent_index_path
                    if parent_idx and parent_idx != file_path:
                        _add_relation(relations, parent_idx, file_path, "parent_child")

        if section_files and section_index_path:
            for file_path in section_files:
                if file_path != section_index_path:
                    _add_relation(relations, section_index_path, file_path, "parent_child")

    def _resolve_doc_path(
        self,
        ref: str,
        docs_path: Path,
        source_root: Path,
        known_paths: set[str],
    ) -> str | None:
        candidate = docs_path / ref
        if candidate.exists():
            relative = _relative_path(candidate, source_root)
            if relative in known_paths:
                return relative

        candidate = source_root / ref
        if candidate.exists():
            relative = _relative_path(candidate, source_root)
            if relative in known_paths:
                return relative

        return resolve_document_reference(ref, str(docs_path), known_paths)


class DocusaurusSidebarParser(NavigationManifestParser):
    """Parse Docusaurus sidebar files into normalized relations."""

    def matches(self, path: Path, content: str) -> bool:
        return path.name.startswith("sidebars.")

    def extract(
        self,
        path: Path,
        content: str,
        artifacts: list[KnowledgeArtifact],
        source_root: Path,
    ) -> list[Relationship]:
        relations: list[Relationship] = []
        lookup = build_document_lookup(artifacts)
        known_paths = set(lookup.values())

        for match in re.finditer(r"items:\s*\[([^\]]+)\]", content, re.DOTALL):
            items_content = match.group(1)
            doc_ids = re.findall(r"['\"]([^'\"]+)['\"]", items_content)
            if len(doc_ids) >= 2:
                parent = doc_ids[0]
                parent_path = self._find_doc_path(parent, lookup, source_root, known_paths)
                if parent_path:
                    _add_relation(relations, "__nav__", parent_path, "navigation")
                for child in doc_ids[1:]:
                    child_path = self._find_doc_path(child, lookup, source_root, known_paths)
                    if parent_path and child_path:
                        _add_relation(relations, parent_path, child_path, "parent_child")
                    if child_path:
                        _add_relation(relations, "__nav__", child_path, "navigation")
            elif len(doc_ids) == 1:
                doc_path = self._find_doc_path(doc_ids[0], lookup, source_root, known_paths)
                if doc_path:
                    _add_relation(relations, "__nav__", doc_path, "navigation")

        return relations

    def _find_doc_path(
        self,
        doc_id: str,
        lookup: dict[str, str],
        source_root: Path,
        known_paths: set[str],
    ) -> str | None:
        if doc_id in lookup:
            return lookup[doc_id]
        for candidate in known_paths:
            if doc_id in candidate:
                return candidate
        return resolve_document_reference(doc_id, str(source_root), known_paths)


class AstroNavigationParser(NavigationManifestParser):
    """Parse Astro/Starlight navigation configuration into normalized relations."""

    def matches(self, path: Path, content: str) -> bool:
        return path.name.startswith("astro.config.")

    def extract(
        self,
        path: Path,
        content: str,
        artifacts: list[KnowledgeArtifact],
        source_root: Path,
    ) -> list[Relationship]:
        relations: list[Relationship] = []
        lookup = build_document_lookup(artifacts)
        known_paths = set(lookup.values())

        for match in re.finditer(r"items:\s*\[([^\]]+)\]", content, re.DOTALL):
            items_content = match.group(1)
            doc_ids = re.findall(r"['\"]([^'\"]+)['\"]", items_content)
            if len(doc_ids) >= 2:
                parent = doc_ids[0]
                parent_path = self._find_doc_path(parent, lookup, source_root, known_paths)
                if parent_path:
                    _add_relation(relations, "__nav__", parent_path, "navigation")
                for child in doc_ids[1:]:
                    child_path = self._find_doc_path(child, lookup, source_root, known_paths)
                    if parent_path and child_path:
                        _add_relation(relations, parent_path, child_path, "parent_child")
                    if child_path:
                        _add_relation(relations, "__nav__", child_path, "navigation")
            elif len(doc_ids) == 1:
                doc_path = self._find_doc_path(doc_ids[0], lookup, source_root, known_paths)
                if doc_path:
                    _add_relation(relations, "__nav__", doc_path, "navigation")

        return relations

    def _find_doc_path(
        self,
        doc_id: str,
        lookup: dict[str, str],
        source_root: Path,
        known_paths: set[str],
    ) -> str | None:
        if doc_id in lookup:
            return lookup[doc_id]
        for candidate in known_paths:
            if doc_id in candidate:
                return candidate
        return resolve_document_reference(doc_id, str(source_root), known_paths)


class NavigationRelationExtractor:
    """Extract navigation relationships from known manifest files when available."""

    def __init__(self, parsers: list[NavigationManifestParser] | None = None) -> None:
        self.parsers = parsers or [
            MkDocsNavigationParser(),
            DocusaurusSidebarParser(),
            AstroNavigationParser(),
        ]

    def extract(
        self,
        source_root: Path,
        artifacts: list[KnowledgeArtifact],
        discovered_files: list[Path],
    ) -> list[Relationship]:
        if not source_root.is_dir():
            return []

        candidate_paths = self._candidate_manifest_paths(source_root, discovered_files)
        relations: list[Relationship] = []
        for manifest_path in candidate_paths:
            try:
                content = manifest_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for parser in self.parsers:
                if parser.matches(manifest_path, content):
                    for relation in parser.extract(manifest_path, content, artifacts, source_root):
                        _add_relation(
                            relations,
                            relation.source_uri,
                            relation.target_uri,
                            relation.relation_type,
                            relation.metadata,
                        )
                    break
        return relations

    def _candidate_manifest_paths(self, source_root: Path, discovered_files: list[Path]) -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()

        def add(path: Path) -> None:
            if path.exists() and path not in seen:
                seen.add(path)
                candidates.append(path)

        for path in [source_root / "mkdocs.yml", source_root.parent / "mkdocs.yml"]:
            add(path)

        for path in source_root.rglob("mkdocs.yml"):
            add(path)

        for path in list(source_root.rglob("sidebars.*")):
            add(path)
        for path in islice(source_root.parent.rglob("sidebars.*"), 3):
            add(path)

        for path in list(source_root.rglob("astro.config.*")):
            add(path)
        for path in islice(source_root.parent.rglob("astro.config.*"), 3):
            add(path)

        for path in discovered_files:
            if path.name in {"mkdocs.yml"} or path.name.startswith(("sidebars.", "astro.config.")):
                add(path)

        return candidates


class InlineLinkRelationExtractor:
    """Infer cross-reference relations from internal markdown links."""

    def extract(self, artifacts: list[KnowledgeArtifact]) -> list[Relationship]:
        relations: list[Relationship] = []
        known_paths = {artifact.uri for artifact in artifacts}

        for artifact in artifacts:
            doc_dir = artifact.uri.rsplit("/", 1)[0] if "/" in artifact.uri else ""
            for link in artifact.links:
                if not link.is_internal:
                    continue

                target = link.target.split("#")[0].split("?")[0]
                if not target:
                    continue

                resolved = resolve_document_reference(target, doc_dir, known_paths)
                if resolved:
                    _add_relation(relations, artifact.uri, resolved, "cross_reference")

        return relations