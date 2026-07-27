"""Discovery helpers for local file-backed knowledge sources."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".mdx", ".txt", ".rst"}


class LocalFileDiscovery:
    """Discover local documents using supported file extensions."""

    def __init__(self, supported_extensions: set[str] | None = None) -> None:
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS

    def discover(self, source: Path) -> list[Path]:
        if source.is_file():
            return [source] if source.suffix.lower() in self.supported_extensions else []

        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in self.supported_extensions
        )