"""Diagnosis quality feedback loop — monitors whether diagnosis
confidence correlates with verification outcomes.

All computation is deterministic and quantitative — no LLM calls.

The tracker maintains a rolling window of (diagnosis_confidence,
verification_outcome) pairs and computes Pearson correlation to
detect when the system's diagnoses are no longer predictive of
success:

  - **Healthy** (corr > 0.3): Diagnoses are useful — high confidence
    predictions tend to succeed, low confidence ones tend to fail.
  - **Noise** (|corr| < 0.1 at n >= 20): Diagnoses are not
    predictive — confidence scores are arbitrary. The system should
    stop trusting its own confidence scores.
  - **Inverted** (corr < -0.3): Diagnoses are anti-predictive — high
    confidence predictions tend to FAIL. Something is systematically
    wrong with the diagnosis logic.

When degradation is detected, a calibration adjustment is produced
that can be fed back into the LLM prompt to warn it about the
unreliability of prior diagnoses.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds (deterministic, no tuning needed)
# ---------------------------------------------------------------------------

MIN_SAMPLE_SIZE = 20         # need at least 20 observations before judging
HEALTHY_CORR_THRESHOLD = 0.3  # corr > 0.3 = healthy
NOISE_CORR_THRESHOLD = 0.1   # |corr| < 0.1 = noise
INVERTED_CORR_THRESHOLD = -0.3  # corr < -0.3 = inverted
DEFAULT_WINDOW_SIZE = 100     # rolling window


# ---------------------------------------------------------------------------
# Diagnosis quality status
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisQualityReport:
    """Quantitative assessment of diagnosis quality.

    Every field is deterministic — computed from historical
    (confidence, outcome) pairs using Pearson correlation.
    """

    correlation: float       # Pearson r between confidence and outcome
    sample_size: int         # number of observations used
    status: str              # "healthy", "noise", "inverted", "insufficient_data"
    calibration_adjustment: str  # text to inject into LLM prompt (if degraded)
    mean_confidence: float   # average diagnosis confidence
    mean_outcome: float      # average verification outcome (1.0 = success, 0.0 = failure)
    confidence_interval: tuple[float, float]  # 95% CI for the correlation

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation": round(self.correlation, 4),
            "sample_size": self.sample_size,
            "status": self.status,
            "calibration_adjustment": self.calibration_adjustment,
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_outcome": round(self.mean_outcome, 4),
            "confidence_interval": [
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4),
            ],
        }


# ---------------------------------------------------------------------------
# DiagnosisQualityTracker
# ---------------------------------------------------------------------------

class DiagnosisQualityTracker:
    """Tracks diagnosis quality over time using rolling correlation.

    Stores (diagnosis_confidence, verification_outcome) pairs and
    computes Pearson correlation to detect degradation. All
    computation is deterministic — no LLM calls.

    Usage:
        tracker = DiagnosisQualityTracker(db_path="diagnosis_quality.db")
        tracker.record(confidence=0.85, outcome=1.0)
        tracker.record(confidence=0.3, outcome=0.0)
        report = tracker.get_report()
        if report.status == "noise":
            # Stop trusting diagnosis confidence scores
            pass
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> None:
        self.db_path = str(db_path) if db_path else ":memory:"
        self.window_size = window_size
        # Keep a persistent connection so :memory: databases work correctly
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite table for diagnosis quality observations."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnosis_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                remediation_id TEXT,
                confidence REAL NOT NULL,
                outcome REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        self._conn.commit()

    def record(
        self,
        confidence: float,
        outcome: float,
        remediation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a (confidence, outcome) observation.

        Args:
            confidence: The diagnosis confidence score (0.0 to 1.0).
            outcome: The verification outcome (1.0 = success, 0.0 = failure,
                     0.5 = partial).
            remediation_id: Optional remediation ID for traceability.
            metadata: Optional metadata dict.
        """
        self._conn.execute(
            "INSERT INTO diagnosis_observations "
            "(remediation_id, confidence, outcome, timestamp, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                remediation_id,
                float(confidence),
                float(outcome),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

    def _fetch_observations(self) -> list[tuple[float, float]]:
        """Fetch the most recent observations within the rolling window."""
        cursor = self._conn.execute(
            "SELECT confidence, outcome FROM diagnosis_observations "
            "ORDER BY id DESC LIMIT ?",
            (self.window_size,),
        )
        rows = cursor.fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_report(self) -> DiagnosisQualityReport:
        """Compute a diagnosis quality report from historical observations.

        Returns:
            DiagnosisQualityReport with correlation, status, and
            calibration adjustment.
        """
        observations = self._fetch_observations()
        n = len(observations)

        if n < MIN_SAMPLE_SIZE:
            return DiagnosisQualityReport(
                correlation=0.0,
                sample_size=n,
                status="insufficient_data",
                calibration_adjustment="",
                mean_confidence=0.0,
                mean_outcome=0.0,
                confidence_interval=(0.0, 0.0),
            )

        confidences = [o[0] for o in observations]
        outcomes = [o[1] for o in observations]

        corr = _pearson_correlation(confidences, outcomes)
        mean_conf = sum(confidences) / n
        mean_outcome = sum(outcomes) / n
        ci = _correlation_confidence_interval(corr, n)

        # Determine status
        if corr > HEALTHY_CORR_THRESHOLD:
            status = "healthy"
            calibration = ""
        elif corr < INVERTED_CORR_THRESHOLD:
            status = "inverted"
            calibration = (
                f"WARNING: Diagnosis confidence is ANTI-PREDICTIVE (r={corr:.3f}). "
                f"High-confidence diagnoses tend to FAIL. The diagnosis logic "
                f"has a systematic error. Consider: (1) inverting confidence "
                f"interpretation, (2) reviewing diagnosis criteria, "
                f"(3) checking for confounding variables. "
                f"Mean confidence={mean_conf:.2f}, mean outcome={mean_outcome:.2f}."
            )
        elif abs(corr) < NOISE_CORR_THRESHOLD:
            status = "noise"
            calibration = (
                f"NOTE: Diagnosis confidence is NOT predictive of success "
                f"(r={corr:.3f}, n={n}). Confidence scores appear arbitrary. "
                f"Treat all diagnoses with skepticism regardless of their "
                f"stated confidence. Consider revising diagnosis criteria."
            )
        else:
            status = "weak"
            calibration = (
                f"NOTE: Diagnosis confidence is weakly predictive (r={corr:.3f}). "
                f"Use confidence scores as a weak signal only."
            )

        return DiagnosisQualityReport(
            correlation=corr,
            sample_size=n,
            status=status,
            calibration_adjustment=calibration,
            mean_confidence=mean_conf,
            mean_outcome=mean_outcome,
            confidence_interval=ci,
        )

    def get_calibration_text(self) -> str:
        """Get calibration text for injection into LLM prompts.

        Returns empty string if diagnosis quality is healthy or
        insufficient data. Returns a warning message if degraded.
        """
        report = self.get_report()
        return report.calibration_adjustment


# ---------------------------------------------------------------------------
# Statistical helpers (deterministic, no external dependencies)
# ---------------------------------------------------------------------------

def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient.

    Pure deterministic computation — no external libraries.
    Returns 0.0 if either list has zero variance.
    """
    n = len(x)
    if n < 2:
        return 0.0

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator_sq = (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)

    if denominator_sq <= 0:
        return 0.0

    return numerator / math.sqrt(denominator_sq)


def _correlation_confidence_interval(
    r: float, n: int, z_crit: float = 1.96,
) -> tuple[float, float]:
    """Compute 95% confidence interval for Pearson r using Fisher's z.

    Returns (lower, upper) bounds. Pure deterministic computation.
    """
    if n < 3:
        return (-1.0, 1.0)

    # Fisher's z-transformation
    if abs(r) >= 0.999:
        r = 0.999 * (1 if r > 0 else -1)

    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)

    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    # Inverse Fisher transform
    r_lower = (math.exp(2 * z_lower) - 1) / (math.exp(2 * z_lower) + 1)
    r_upper = (math.exp(2 * z_upper) - 1) / (math.exp(2 * z_upper) + 1)

    return (r_lower, r_upper)
