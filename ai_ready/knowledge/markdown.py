"""Markdown Knowledge SDK built from separate discovery, parsing, and relation components."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ai_ready.knowledge.base import KnowledgeCapability, KnowledgeSDK
from ai_ready.knowledge.discovery import LocalFileDiscovery
from ai_ready.knowledge.markdown_parser import MarkdownDocumentParser
from ai_ready.knowledge.navigation import InlineLinkRelationExtractor, NavigationRelationExtractor
from ai_ready.knowledge.registry import register_knowledge_sdk
from ai_ready.models import KnowledgeArtifact, Relationship


@register_knowledge_sdk
class MarkdownKnowledgeSDK(KnowledgeSDK):
    """Local file-backed knowledge SDK for Markdown, RST, and plain text content."""

    name = "markdown"
    supported_capabilities = frozenset({
        KnowledgeCapability.ARTIFACTS,
        KnowledgeCapability.RELATIONS,
        KnowledgeCapability.NAVIGATION,
        KnowledgeCapability.METADATA,
    })

    def __init__(self) -> None:
        super().__init__()
        self._discovery = LocalFileDiscovery()
        self._parser = MarkdownDocumentParser()
        self._inline_relation_extractor = InlineLinkRelationExtractor()
        self._navigation_relation_extractor = NavigationRelationExtractor()
        self._discovered_files: list[Path] = []
        self._artifact_cache: dict[str, KnowledgeArtifact] = {}
        self._relationship_cache: list[Relationship] | None = None

    @classmethod
    def supports(cls, source: str | Path) -> bool:
        path = Path(source)
        return path.exists() and (
            path.is_dir()
            or path.suffix.lower() in {".md", ".markdown", ".mdx", ".txt", ".rst"}
        )

    def connect(self, source: str | Path) -> None:
        self._source = Path(source)
        self._discovered_files = self._discovery.discover(self._source)
        self._artifact_cache = {}
        self._relationship_cache = None

    def iter_artifacts(self) -> Iterator[KnowledgeArtifact]:
        self._ensure_artifacts()
        for file_path in self._discovered_files:
            relative_path = self._parser._relative_path(file_path, self._source)
            artifact = self._artifact_cache.get(relative_path)
            if artifact is not None:
                yield artifact

    def iter_relationships(self) -> Iterator[Relationship]:
        self._ensure_relationships()
        yield from self._relationship_cache or []

    def source_metadata(self) -> dict[str, object]:
        return {
            "sdk": self.name,
            "file_count": len(self._discovered_files),
        }

    def _ensure_artifacts(self) -> None:
        if self._artifact_cache:
            return

        for file_path in self._discovered_files:
            artifact = self._parser.parse(file_path, self._source)
            self._artifact_cache[artifact.uri] = artifact

    def _ensure_relationships(self) -> None:
        if self._relationship_cache is not None:
            return

        self._ensure_artifacts()
        artifacts = list(self._artifact_cache.values())
        relationships = list(self._inline_relation_extractor.extract(artifacts))
        relationships.extend(
            self._navigation_relation_extractor.extract(
                self._source or Path("."),
                artifacts,
                self._discovered_files,
            )
        )

        deduped: list[Relationship] = []
        for rel in relationships:
            if not any(
                existing.source_uri == rel.source_uri
                and existing.target_uri == rel.target_uri
                and existing.relation_type == rel.relation_type
                for existing in deduped
            ):
                deduped.append(rel)

        self._relationship_cache = deduped