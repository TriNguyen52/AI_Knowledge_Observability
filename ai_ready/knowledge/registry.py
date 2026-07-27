"""SDK registry and loader helpers."""

from __future__ import annotations

from pathlib import Path

from ai_ready.knowledge.base import KnowledgeSDK, KnowledgeSource

_REGISTERED_SDKS: list[type[KnowledgeSDK]] = []
_BUILTINS_LOADED = False


def register_knowledge_sdk(cls: type[KnowledgeSDK]) -> type[KnowledgeSDK]:
    """Register a Knowledge SDK for source discovery."""
    if cls not in _REGISTERED_SDKS:
        _REGISTERED_SDKS.append(cls)
    return cls


def registered_knowledge_sdks() -> list[type[KnowledgeSDK]]:
    """Return the currently registered SDK classes."""
    return list(_REGISTERED_SDKS)


def _ensure_builtin_sdks() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    # Importing the markdown SDK registers the default local-file handler.
    from ai_ready.knowledge import markdown  # noqa: F401

    _BUILTINS_LOADED = True


def get_knowledge_sdk(source: str | Path) -> KnowledgeSDK:
    """Find, instantiate, and connect the first SDK that supports a source."""
    _ensure_builtin_sdks()
    for sdk_cls in _REGISTERED_SDKS:
        if sdk_cls.supports(source):
            sdk = sdk_cls()
            sdk.connect(source)
            return sdk
    raise ValueError(f"No Knowledge SDK is registered for source: {source}")


def load_knowledge_source(source: str | Path) -> KnowledgeSource:
    """Convenience wrapper that returns a normalized in-memory knowledge bundle."""
    return get_knowledge_sdk(source).load()