"""Knowledge SDK base types and normalized ingestion results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from ai_ready.models import (
    ArtifactBundle,
    KnowledgeArtifact,
    Relationship,
)


class KnowledgeCapability(str, Enum):
    """Capabilities an SDK may expose."""

    DOCUMENTS = "documents"
    RELATIONS = "relations"
    NAVIGATION = "navigation"
    METADATA = "metadata"
    EMBEDDINGS = "embeddings"
    ARTIFACTS = "artifacts"


@dataclass
class KnowledgeSource:
    """Materialized normalized ingestion output from an SDK."""

    artifacts: list[KnowledgeArtifact] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    capabilities: frozenset[KnowledgeCapability] = frozenset()
    source: str = ""

    def to_artifact_bundle(self) -> ArtifactBundle:
        """Wrap this source's artifacts and relationships into an ArtifactBundle."""
        return ArtifactBundle(
            artifacts=list(self.artifacts),
            relationships=list(self.relationships),
            source=self.source,
            metadata=dict(self.metadata),
        )


class KnowledgeSDK(ABC):
    """Source-specific ingestion layer that normalizes knowledge into domain models."""

    name: str = "base"
    supported_capabilities: frozenset[KnowledgeCapability] = frozenset({
        KnowledgeCapability.ARTIFACTS,
    })

    def __init__(self) -> None:
        self._source: Path | None = None

    @property
    def capabilities(self) -> frozenset[KnowledgeCapability]:
        return self.supported_capabilities

    @classmethod
    def supports(cls, source: str | Path) -> bool:
        """Return True when this SDK can ingest the provided source."""
        return False

    @abstractmethod
    def connect(self, source: str | Path) -> None:
        """Open or prepare the source for ingestion."""

    @abstractmethod
    def iter_artifacts(self) -> Iterator[KnowledgeArtifact]:
        """Yield normalized knowledge artifacts."""

    def iter_relationships(self) -> Iterator[Relationship]:
        """Yield normalized relationships between artifacts.

        Default implementation returns empty. SDKs that can extract
        relationships should override this.
        """
        return iter(())

    def source_metadata(self) -> dict[str, Any]:
        """Return source-level metadata to carry into the knowledge base."""
        return {}

    def load(self) -> KnowledgeSource:
        """Materialize the normalized source into an in-memory knowledge bundle."""
        return KnowledgeSource(
            artifacts=list(self.iter_artifacts()),
            relationships=list(self.iter_relationships()),
            metadata=self.source_metadata(),
            capabilities=frozenset(self.capabilities),
            source=str(self._source) if self._source is not None else "",
        )

    def load_artifacts(self) -> ArtifactBundle:
        """Materialize the source as an ArtifactBundle.

        Default implementation calls ``iter_artifacts()`` and
        ``iter_relationships()``. SDKs may override for custom behavior.
        """
        return ArtifactBundle(
            artifacts=list(self.iter_artifacts()),
            relationships=list(self.iter_relationships()),
            source=str(self._source) if self._source is not None else "",
            metadata=self.source_metadata(),
        )