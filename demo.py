#!/usr/bin/env python
"""End-to-end demonstration of AI-Ready knowledge observability.

This demo proves the full production pipeline works against a real
documentation repository (FastAPI).  It run every layer:

  1. Ingestion   — MarkdownKnowledgeSDK discovers and parses 1,600+ docs
  2. Collection  — SignalCollectors emit KnowledgeSignals (bare facts)
  3. Assessment  — InterpretationPolicy enriches signals → dimension scores
  4. Persistence — AssessmentStore + SignalStore save to SQLite
  5. Regression  — DiffOperation compares two snapshots
  6. Trend       — EvolutionView tracks health over time

Workflow:
  Phase 1  Baseline scan of the real FastAPI docs
  Phase 2  Introduce an intentionally bad doc → re-scan (regression)
  Phase 3  Diff the two assessments
  Phase 4  Remove the bad doc → re-scan (improvement)
  Phase 5  Diff again to show resolved signals
  Phase 6  Trend analysis across all three assessments

Usage:
    python demo.py [--source PATH] [--db PATH] [--keep-temp]

    --source   Path to the knowledge base (default: FastAPI docs)
    --db       Assessment database path (default: .ai-ready/demo.db)
    --keep-temp  Keep the temp copy for inspection
"""

from __future__ import annotations

import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.stdout.reconfigure(line_buffering=True)

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SOURCE = Path("C:/Users/jacks/Documents/khe-validation/fastapi")
DEFAULT_DB = Path(".ai-ready/demo.db")

# A deliberately bad documentation file used to trigger regression signals.
# Contains: placeholder headings, dangling references, and a broken link.
BAD_DOC = """\
# TODO

## TBD

This section needs to be written. See above for context.

As mentioned below, the API will be documented eventually.

For more details, see [nonexistent page](./does-not-exist.md).

"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str, char: str = "=", width: int = 72) -> None:
    """Print a section banner."""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def run_cli(
    args: list[str],
    *,
    check: bool = False,
    timeout: int = 300,
    max_lines: int = 80,
) -> subprocess.CompletedProcess:
    """Run an ai-ready CLI command and print its output to the terminal.

    Output longer than *max_lines* is truncated to show the first and last
    portions so the demo stays readable while still showing real CLI output.

    Returns the CompletedProcess so the caller can inspect the return code.
    """
    cmd = [sys.executable, "-m", "ai_ready.cli", *args]
    print(f"  $ ai-ready {' '.join(args)}")
    print()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    output = result.stdout
    lines = output.splitlines()
    if len(lines) > max_lines:
        half = max_lines // 2
        truncated = lines[:half] + [
            f"  ... ({len(lines) - max_lines} lines omitted) ...",
        ] + lines[-half:]
        output = "\n".join(truncated)
    print(output)
    if result.stderr:
        print(result.stderr, end="")
    print()
    if check and result.returncode not in (0, 1, 2):
        raise RuntimeError(f"CLI command failed with code {result.returncode}: {' '.join(args)}")
    return result


def run_python(code: str, timeout: int = 300) -> None:
    """Run a short Python snippet and stream its output."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=False,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Python snippet failed with code {result.returncode}")


# ---------------------------------------------------------------------------
# Demo phases
# ---------------------------------------------------------------------------

def phase_1_baseline_scan(source: Path, db: Path) -> None:
    """Phase 1: Scan the real FastAPI docs and persist the baseline assessment."""
    banner("PHASE 1: Baseline Scan (Real FastAPI Documentation)")
    print(f"  Source:    {source}")
    print(f"  Database:  {db}")
    print()
    print("  This phase scans the real FastAPI documentation repository")
    print("  (1,600+ markdown files) using the production pipeline:")
    print("    MarkdownKnowledgeSDK → SignalCollectors → InterpretationPolicy")
    print("    → DimensionScores → SQLite persistence")
    print()
    run_cli(["scan", str(source), "--db", str(db)], check=True)


def phase_2_introduce_regression(source: Path, temp_dir: Path) -> Path:
    """Phase 2: Copy the repo, add a bad doc, and re-scan."""
    banner("PHASE 2: Introduce Intentional Documentation Change (Regression)")
    print(f"  Copying {source}")
    print(f"  → {temp_dir}")
    shutil.copytree(source, temp_dir, dirs_exist_ok=True)

    bad_doc_path = temp_dir / "docs" / "demo-regression-test.md"
    bad_doc_path.parent.mkdir(parents=True, exist_ok=True)
    bad_doc_path.write_text(BAD_DOC, encoding="utf-8")
    print()
    print(f"  Created: {bad_doc_path.relative_to(temp_dir)}")
    print("  Contents: placeholder headings (TODO, TBD, Placeholder),")
    print("            dangling references (see above, as mentioned below),")
    print("            broken link (./does-not-exist.md)")
    print()
    print("  Re-scanning the modified knowledge base...")
    print()
    return bad_doc_path


def phase_3_regression_diff(db: Path) -> None:
    """Phase 3: Diff the two most recent assessments."""
    banner("PHASE 3: Regression Detection (Assessment Diff)")
    print("  The diff command compares the two most recent assessments")
    print("  and reports new, resolved, persistent, and recurring signals.")
    print()
    run_cli(["diff", "--db", str(db)], check=False)


def phase_4_fix_regression(temp_dir: Path, bad_doc_path: Path, db: Path) -> None:
    """Phase 4: Remove the bad doc and re-scan to show improvement."""
    banner("PHASE 4: Fix Regression and Re-Scan")
    bad_doc_path.unlink()
    print(f"  Removed: {bad_doc_path.relative_to(temp_dir)}")
    print()
    print("  Re-scanning the fixed knowledge base...")
    print()
    run_cli(["scan", str(temp_dir), "--db", str(db)], check=True)


def phase_5_improvement_diff(db: Path) -> None:
    """Phase 5: Diff after the fix to show resolved signals."""
    banner("PHASE 5: Improvement Detection (Diff After Fix)")
    print("  This diff compares the regression assessment (Phase 2)")
    print("  with the fixed assessment (Phase 4). The signals introduced")
    print("  by the bad doc should now appear as resolved.")
    print()
    run_cli(["diff", "--db", str(db)], check=False)


def phase_6_status_and_history(db: Path) -> None:
    """Phase 6: Show status, history, and trend across all assessments."""
    banner("PHASE 6: Status, History, and Trend Analysis")
    print("  The status command shows the latest assessment, signal")
    print("  lifecycle (new/persistent/recurring/resolved), and recent")
    print("  changes.")
    print()
    run_cli(["status", "--db", str(db)], check=False)

    banner("Assessment History", char="-")
    print()
    run_cli(["history", "--db", str(db), "--limit", "10"], check=False)

    banner("Trend Analysis", char="-")
    print()
    run_cli(["trend", "--db", str(db), "--limit", "10"], check=False)


def phase_7_signal_lifecycle(db: Path) -> None:
    """Phase 7: Show signal lifecycle details."""
    banner("PHASE 7: Signal Lifecycle Details")
    print("  The signals command lists individual signals with their")
    print("  lifecycle status (new, persistent, recurring, resolved).")
    print()
    run_cli(["signals", "--db", str(db), "--limit", "20"], check=False)

    print()
    print("  Resolved signals (those fixed in Phase 4):")
    print()
    run_cli(["signals", "--db", str(db), "--status", "resolved", "--limit", "20"], check=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end AI-Ready demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help=f"Path to the knowledge base (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help=f"Assessment database path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep the temporary copy for inspection",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    db = args.db

    if not source.exists():
        print(f"ERROR: Source path does not exist: {source}")
        sys.exit(1)

    # Clean up previous demo database
    if db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)

    # Create a temp directory for the modified copy
    temp_dir = Path(tempfile.mkdtemp(prefix="ai-ready-demo-"))

    print()
    print("=" * 72)
    print("  AI-Ready End-to-End Demonstration")
    print("  Knowledge Observability for AI-Ready Documentation")
    print("=" * 72)
    print()
    print("  This demo exercises the full production pipeline against")
    print("  a real documentation repository (FastAPI, 1,600+ files).")
    print("  It demonstrates: ingestion, signal collection, assessment,")
    print("  persistence, regression detection, and trend analysis.")
    print()
    print(f"  Source:    {source}")
    print(f"  Database:  {db}")
    print(f"  Temp dir:  {temp_dir}")

    try:
        # Phase 1: Baseline scan
        phase_1_baseline_scan(source, db)

        # Phase 2: Introduce regression
        bad_doc_path = phase_2_introduce_regression(source, temp_dir)
        run_cli(["scan", str(temp_dir), "--db", str(db)], check=True)

        # Phase 3: Regression diff
        phase_3_regression_diff(db)

        # Phase 4: Fix regression
        phase_4_fix_regression(temp_dir, bad_doc_path, db)

        # Phase 5: Improvement diff
        phase_5_improvement_diff(db)

        # Phase 6: Status, history, and trend
        phase_6_status_and_history(db)

        # Phase 7: Signal lifecycle
        phase_7_signal_lifecycle(db)

    finally:
        # Cleanup
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print()
            print(f"  Temp directory cleaned up: {temp_dir}")
            print("  (Use --keep-temp to preserve it for inspection)")
        else:
            print()
            print(f"  Temp directory preserved: {temp_dir}")

    banner("Demo Complete")
    print("  This demo exercised the full AI-Ready pipeline:")
    print()
    print("    1. Ingestion      MarkdownKnowledgeSDK → KnowledgeArtifact")
    print("    2. Collection     SignalCollector → KnowledgeSignal")
    print("    3. Assessment     InterpretationPolicy → DimensionScore")
    print("    4. Persistence    AssessmentStore + SignalStore → SQLite")
    print("    5. Regression      DiffOperation → AssessmentDiff")
    print("    6. Trend           AssessmentStore.get_trend → EvolutionView")
    print()
    print("  CLI commands demonstrated:")
    print("    ai-ready scan    — assess a knowledge base")
    print("    ai-ready diff    — compare two assessments")
    print("    ai-ready status  — current health and signal lifecycle")
    print("    ai-ready history — assessment history")
    print("    ai-ready trend   — health evolution over time")
    print("    ai-ready signals — list signals with lifecycle info")
    print()
    print(f"  Database saved at: {db}")
    print("=" * 72)


if __name__ == "__main__":
    main()
