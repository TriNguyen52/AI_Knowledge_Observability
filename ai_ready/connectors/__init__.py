"""Source integrations for the Knowledge SDK."""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Iterator

from ai_ready.knowledge.base import KnowledgeSDK
from ai_ready.models import KnowledgeArtifact


class Connector(KnowledgeSDK, ABC):
    """Base class for source-specific knowledge connectors.

    Subclasses implement iter_artifacts() to produce KnowledgeArtifact
    objects from a specific source (filesystem, API, database, etc.).
    """

    name: str = "base"

    def artifact_count(self) -> int:
        """Count artifacts by iterating the normalized stream."""
        return sum(1 for _ in self.iter_artifacts())
