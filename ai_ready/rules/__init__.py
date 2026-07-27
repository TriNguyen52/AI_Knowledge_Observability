"""Signal collector base class and registry.

A SignalCollector implements the collect -> measure -> evaluate -> report
pipeline and emits bare signals (facts without interpretation). The
assessment layer enriches signals with severity, score, and recommendation
via InterpretationPolicy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ai_ready.models import (
    ArtifactBundle,
    CollectorResult,
    KnowledgeArtifact,
    KnowledgeSignal,
    Severity,
)


class SignalCollector(ABC):
    """Base collector contract — every collector implements collect/measure/evaluate/report.

    Collectors receive an ArtifactBundle (artifacts + relationships) instead of a raw
    list of documents. This allows collectors to access artifact relationships
    (navigation hierarchy, cross-references) for richer analysis.

    The pipeline produces signals (bare facts) that are later interpreted by
    the assessment layer via InterpretationPolicy.
    """

    id: str = "base"
    severity_default: Severity = Severity.LOW
    dimension: str = "retrieval"

    @abstractmethod
    def collect(self, bundle: ArtifactBundle) -> dict[str, Any]:
        """Collect raw signals from the artifact bundle."""
        ...

    @abstractmethod
    def measure(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Compute metrics from collected signals."""
        ...

    @abstractmethod
    def evaluate(self, metrics: dict[str, Any]) -> list[KnowledgeSignal]:
        """Evaluate metrics and produce signals."""
        ...

    @abstractmethod
    def report(self) -> CollectorResult:
        """Produce the final collector result."""
        ...

    def run(self, bundle: ArtifactBundle) -> CollectorResult:
        """Execute the full collect -> measure -> evaluate -> report pipeline."""
        self._bundle = bundle
        self._signals = self.collect(bundle)
        self._metrics = self.measure(self._signals)
        self._findings = self.evaluate(self._metrics)
        return self.report()

    @property
    def artifacts(self) -> list[KnowledgeArtifact]:
        """Convenience accessor for the artifacts in the bundle."""
        return self._bundle.artifacts if hasattr(self, "_bundle") else []


# Registry of all available collectors
_COLLECTOR_REGISTRY: dict[str, type[SignalCollector]] = {}


def register_collector(collector_class: type[SignalCollector]) -> type[SignalCollector]:
    """Decorator to register a collector class."""
    _COLLECTOR_REGISTRY[collector_class.id] = collector_class
    return collector_class


def get_collector(collector_id: str) -> type[SignalCollector] | None:
    """Look up a collector class by ID."""
    return _COLLECTOR_REGISTRY.get(collector_id)


def all_collectors() -> dict[str, type[SignalCollector]]:
    """Return all registered collectors."""
    return dict(_COLLECTOR_REGISTRY)


# Dimension -> collector mapping
DIMENSION_COLLECTORS: dict[str, list[str]] = {
    "retrieval": ["topic_purity", "heading_quality"],
    "context": ["context_independence"],
    "consistency": ["terminology_consistency", "contradiction_detection"],
    "trust": ["canonical_source", "freshness"],
    "connectivity": ["knowledge_connectivity", "link_integrity"],
    "workflow": ["workflow_completeness"],
}


# Auto-import all collector modules to trigger @register_collector decorators
def _import_collectors() -> None:
    """Import all collector modules to register them."""
    import importlib
    import pkgutil

    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_") or module_info.name == "__init__":
            continue
        importlib.import_module(f"{__name__}.{module_info.name}")

_import_collectors()


# Re-export collector classes for convenient access.
from ai_ready.rules.topic_purity import TopicPurityCollector  # noqa: E402, F401
from ai_ready.rules.heading_quality import HeadingQualityCollector  # noqa: E402, F401
from ai_ready.rules.context_independence import ContextIndependenceCollector  # noqa: E402, F401
from ai_ready.rules.link_integrity import LinkIntegrityCollector  # noqa: E402, F401
