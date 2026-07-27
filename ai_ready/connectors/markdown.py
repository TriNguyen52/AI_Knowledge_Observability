"""Backward-compatible Markdown connector that delegates to the Markdown SDK."""

from __future__ import annotations

from ai_ready.knowledge.markdown import MarkdownKnowledgeSDK

from ai_ready.connectors import Connector


class MarkdownConnector(MarkdownKnowledgeSDK, Connector):
    """Backward-compatible name for the Markdown Knowledge SDK."""

    pass
