"""Convenience helpers for loading normalized knowledge sources."""

from __future__ import annotations

from pathlib import Path

from ai_ready.knowledge.base import KnowledgeSource
from ai_ready.knowledge.registry import load_knowledge_source as _load_knowledge_source


def load_knowledge_base(source: str | Path) -> KnowledgeSource:
    """Load any registered source into normalized AI-Ready domain models."""
    return _load_knowledge_source(source)