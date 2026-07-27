"""Configuration parser for .ai-ready.yml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai_ready.pipeline import DEFAULT_WEIGHTS

class Config:
    """Parsed configuration from .ai-ready.yml."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        thresholds: dict[str, Any] | None = None,
        fail_on: list[str] | None = None,
        enabled_collectors: dict[str, bool] | None = None,
    ) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.thresholds = thresholds or {"overall_score": 0}
        self.fail_on = fail_on or ["CRITICAL"]
        self.enabled_collectors = enabled_collectors or {}

    @property
    def enabled_collector_ids(self) -> list[str]:
        """Return list of collector IDs that are enabled (or all if none specified)."""
        if not self.enabled_collectors:
            return []
        result = []
        for collector_id, val in self.enabled_collectors.items():
            if isinstance(val, dict):
                if val.get("enabled", True):
                    result.append(collector_id)
            elif val:
                result.append(collector_id)
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load config from a YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from a parsed dict."""
        enabled_collectors = data.get("collectors", {})
        return cls(
            weights=data.get("weights", dict(DEFAULT_WEIGHTS)),
            thresholds=data.get("thresholds", {"overall_score": 0}),
            fail_on=data.get("fail_on", ["CRITICAL"]),
            enabled_collectors=enabled_collectors,
        )

    @classmethod
    def default(cls) -> "Config":
        """Create default config."""
        return cls()
