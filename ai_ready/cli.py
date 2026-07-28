"""CLI entry point — ai-ready command.

AI-Ready · Knowledge Observability Platform

Commands:
  scan      Assess a knowledge base and produce a health report
  diff      Compare two assessments and report what changed
  history   Show assessment history
  status    Show current health, recent changes, and signal lifecycle
  trend     Show health trend over time
  signals   List knowledge signals with lifecycle information
  monitor   Continuously monitor a knowledge base
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.table import Table
import rich.markup

from ai_ready.config import Config
from ai_ready.output import (
    format_json,
    format_regression_json,
    format_regression_terminal,
    format_sarif,
    format_terminal,
)
from ai_ready.knowledge.loader import load_knowledge_base
from ai_ready.pipeline import AssessmentPipeline
from ai_ready.snapshot import AssessmentStoreFacade

app = typer.Typer(
    name="ai-ready",
    help="AI-Ready \u2014 Knowledge observability platform. Assess whether a knowledge base is AI-ready.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console(width=120)

# Default assessment DB location
DEFAULT_DB = Path.cwd() / ".ai-ready" / "assessments.db"

# Standard width for separators
_WIDTH = 72


# ---------------------------------------------------------------------------
# Internal helpers (business logic — preserved from original)
# ---------------------------------------------------------------------------

def _load_config(source: str | Path) -> Config:
    """Load config from .ai-ready.yml in the source directory."""
    source_path = Path(source)
    if source_path.is_dir():
        config_path = source_path / ".ai-ready.yml"
    else:
        config_path = Path.cwd() / ".ai-ready.yml"
    return Config.from_file(config_path)


def _get_store(db_path: str | None) -> AssessmentStoreFacade:
    """Get assessment store."""
    path = Path(db_path) if db_path else DEFAULT_DB
    return AssessmentStoreFacade(path)


def _detect_git_commit(source: str | Path) -> str:
    """Try to detect the current git commit."""
    import subprocess
    try:
        source_path = Path(source)
        cwd = source_path if source_path.is_dir() else source_path.parent
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(cwd), timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _load_kb(source: Path, config: Config) -> tuple[list, list]:
    """Load artifacts and relations from source."""
    knowledge_source = load_knowledge_base(source)
    return knowledge_source.artifacts, knowledge_source.relationships


def _run_full_assessment(
    pipeline: AssessmentPipeline,
    artifacts: list,
    relations: list,
    source: str,
    git_commit: str,
) -> Any:
    """Run a full assessment with progress messages for each stage.

    Splits the pipeline into collect and assess phases to show
    domain-appropriate progress messages.
    """
    console.print("[dim]Collecting knowledge signals...[/dim]")
    results, all_signals, bundle = pipeline._collect_op.run(
        artifacts=artifacts, relationships=relations, source=source
    )
    console.print(f"[dim]  {len(all_signals):,} signal(s) collected from {len(results)} collector(s)[/dim]")

    console.print("[dim]Building knowledge assessment...[/dim]")
    return pipeline._assess_op.run(
        results=results, signals=all_signals, artifacts=artifacts,
        bundle=bundle, source=source, git_commit=git_commit,
    )


def _run_incremental_assessment(
    pipeline: AssessmentPipeline,
    source: Path,
    artifacts: list,
    relations: list,
    git_commit: str,
    db: str | None,
) -> Any:
    """Run an incremental assessment, falling back to full assessment if needed.

    Loads the previous assessment, detects changes via git diff, and
    delegates to pipeline.run_incremental(). Falls back to a full assessment
    if no previous assessment exists or changes can't be determined.
    """
    from ai_ready.incremental import detect_changes_via_git, detect_changes

    store = _get_store(db)
    prev_assessment = store.latest()

    if prev_assessment is None:
        console.print("[dim]No previous assessment found.[/dim]")
        return _run_full_assessment(pipeline, artifacts, relations, str(source), git_commit)

    # Try git-based change detection first
    change_events = detect_changes_via_git(source, prev_assessment, artifacts)

    if change_events is None:
        # Git not available — fall back to assessment comparison
        change_events = detect_changes(source, prev_assessment, artifacts)

    if not change_events:
        console.print("[dim]No changes detected since last assessment.[/dim]")
        return _run_full_assessment(pipeline, artifacts, relations, str(source), git_commit)

    changed_count = len({e.artifact_uri for e in change_events})
    console.print(f"[dim]  {changed_count} artifact(s) changed since last assessment[/dim]")
    console.print("[dim]Collecting updated signals and building assessment...[/dim]")

    return pipeline.run_incremental(
        prev_assessment=prev_assessment,
        change_events=change_events,
        artifacts=artifacts,
        relationships=relations,
        source=str(source),
        git_commit=git_commit,
    )


def _score_bar(score: int, width: int = 15) -> str:
    """Generate a visual bar chart for a score using block characters."""
    filled = int(score / 100 * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _health_label(score: int) -> str:
    """Map a score to a human-readable health label."""
    if score >= 80:
        return "Healthy"
    if score >= 50:
        return "Needs Attention"
    return "Critical"


def _collector_label(collector_id: str) -> str:
    """Convert a snake_case collector ID to a human-readable label."""
    return collector_id.replace("_", " ").title()


def _format_artifact_uri(uri: str, max_width: int = 55) -> str:
    """Truncate long artifact URIs intelligently, keeping the last path components."""
    if len(uri) <= max_width:
        return uri
    parts = uri.replace("\\", "/").split("/")
    for n in (3, 2):
        short = "/".join(parts[-n:])
        if len(short) + 3 <= max_width:
            return "..." + "/" + short
    return "..." + uri[-(max_width - 3):]


def format_json_trend(report) -> str:
    """Format a trend report as JSON."""
    import json
    return json.dumps(report.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def scan(
    path: str = typer.Argument(..., help="Path to the knowledge base directory"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    sarif: bool = typer.Option(False, "--sarif", help="Output as SARIF"),
    db: str = typer.Option(None, "--db", help="Assessment database path"),
    config_path: str = typer.Option(None, "--config", help="Config file path"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save assessment to store"),
    incremental: bool = typer.Option(False, "--incremental", help="Run incremental assessment (only re-evaluate changed artifacts)"),
) -> None:
    """Assess a knowledge base and produce a knowledge health report.

    Evaluates all knowledge artifacts in the target directory, calculates
    a Knowledge Health Score, and reports Knowledge Signals organized by
    impact level. The assessment is saved to the store by default.
    """
    source = Path(path).resolve()
    if not source.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {source}")
        raise typer.Exit(3)

    # Load config
    if config_path:
        config = Config.from_file(config_path)
    else:
        config = _load_config(source)

    # Stage 1: Load knowledge base
    console.print(f"[dim]Loading knowledge base from {source}...[/dim]")
    artifacts, relations = _load_kb(source, config)

    if not artifacts:
        console.print(f"[yellow]Warning:[/yellow] No knowledge artifacts found in {source}")
        raise typer.Exit(0)

    console.print(f"[dim]Loaded {len(artifacts):,} knowledge artifact(s), {len(relations):,} relationship(s).[/dim]")

    # Detect git commit
    git_commit = _detect_git_commit(source)

    # Stage 2-3: Run pipeline (collect signals, build assessment)
    pipeline = AssessmentPipeline(
        weights=config.weights,
        enabled_collectors=config.enabled_collector_ids,
        thresholds=config.thresholds,
        fail_on=config.fail_on,
    )

    if incremental:
        assessment = _run_incremental_assessment(
            pipeline, source, artifacts, relations, git_commit, db
        )
    else:
        assessment = _run_full_assessment(
            pipeline, artifacts, relations, str(source), git_commit
        )

    # Get historical context for the report
    store = _get_store(db)
    prev = store.latest()
    prev_score = None
    score_delta = None
    is_baseline = True

    if prev and prev.assessment_id != assessment.assessment_id:
        prev_score = prev.score
        score_delta = assessment.score - prev.score
        is_baseline = False
    elif prev and prev.assessment_id == assessment.assessment_id:
        # The current assessment was already saved — look for the one before it
        prev_prev = store.previous(assessment.assessment_id)
        if prev_prev:
            prev_score = prev_prev.score
            score_delta = assessment.score - prev_prev.score
            is_baseline = False

    # Save assessment
    if save:
        store.save(assessment)
        console.print(f"[dim]Assessment saved: {assessment.assessment_id}[/dim]")

    # Stage 4: Present the assessment
    if json_output:
        print(format_json(assessment))
    elif sarif:
        print(format_sarif(assessment))
    else:
        print(format_terminal(
            assessment,
            prev_score=prev_score,
            score_delta=score_delta,
            is_baseline=is_baseline,
        ))

    # Exit code
    exit_code = pipeline.get_exit_code(assessment)
    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command()
def diff(
    prev_id: str = typer.Option(None, "--prev", help="Previous assessment ID (default: auto-detect)"),
    curr_id: str = typer.Option(None, "--curr", help="Current assessment ID (default: latest)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    db: str = typer.Option(None, "--db", help="Assessment database path"),
) -> None:
    """Compare two assessments and report what changed.

    Shows score delta, dimension changes, new/resolved/persistent Knowledge
    Signals, and impact changes. By default, compares the two most recent
    assessments.
    """
    store = _get_store(db)

    # Determine which assessments to compare
    if prev_id and curr_id:
        report = store.diff(prev_id, curr_id)
    elif not prev_id and not curr_id:
        report = store.diff_latest()
    else:
        # curr specified, auto-detect prev
        curr = store.load(curr_id) if curr_id else store.latest()
        if not curr:
            console.print("[red]Error:[/red] No current assessment found")
            raise typer.Exit(3)
        prev = store.previous(curr.assessment_id)
        if not prev:
            console.print("[red]Error:[/red] No previous assessment found")
            raise typer.Exit(3)
        report = store.diff(prev.assessment_id, curr.assessment_id)

    if not report:
        console.print("[yellow]Not enough assessments to compare (need at least 2).[/yellow]")
        raise typer.Exit(0)

    if json_output:
        print(format_regression_json(report))
    else:
        print(format_regression_terminal(report))

    if report.score_delta < 0 or report.new_high_count > 0:
        raise typer.Exit(2)


@app.command()
def history(
    limit: int = typer.Option(10, "--limit", help="Number of assessments to show"),
    db: str = typer.Option(None, "--db", help="Assessment database path"),
) -> None:
    """Show assessment history.

    Lists recent assessments with their Knowledge Health Score, signal count,
    and source. Useful for tracking knowledge base evolution over time.
    """
    store = _get_store(db)
    assessments = store.history(limit=limit)

    if not assessments:
        console.print("[yellow]No assessments found. Run 'ai-ready scan' first.[/yellow]")
        return

    console.print("")
    console.print("[bold]AI-Ready \u00b7 Assessment History[/bold]")
    console.print("=" * _WIDTH)

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Assessment ID", style="cyan", width=22)
    table.add_column("Score", justify="right", width=12)
    table.add_column("Trend", width=8)
    table.add_column("Signals", justify="right", width=10)
    table.add_column("Source", width=30)

    prev_score = None
    for a in assessments:
        health = _health_label(a.score)
        score_str = f"{a.score}/100  {health}"

        # Trend indicator
        if prev_score is not None:
            delta = a.score - prev_score
            if delta > 0:
                trend = f"\u2191 +{delta}"
            elif delta < 0:
                trend = f"\u2193 {delta}"
            else:
                trend = "\u2192  0"
        else:
            trend = "baseline"

        table.add_row(
            a.assessment_id,
            score_str,
            trend,
            str(len(a.signals)),
            a.metadata.get("source", "N/A"),
        )
        prev_score = a.score

    console.print(table)
    console.print("=" * _WIDTH)


@app.command()
def status(
    db: str = typer.Option(None, "--db", help="Assessment database path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show current knowledge health, recent changes, and signal lifecycle.

    Displays the latest assessment summary, dimension scores, changes since
    the previous assessment, signal lifecycle statistics, oldest unresolved
    observations, and recently resolved observations.
    """
    store = _get_store(db)
    latest = store.latest()

    if not latest:
        console.print("[yellow]No assessments found. Run 'ai-ready scan' first.[/yellow]")
        raise typer.Exit(0)

    if json_output:
        import json
        result = {
            "latest_assessment": latest.to_dict(),
            "diff": store.diff_latest().to_dict() if store.diff_latest() else None,
            "signal_stats": store.get_signal_stats(),
            "oldest_signals": [s.to_dict() for s in store.get_oldest_signals(limit=10)],
            "recently_resolved": [s.to_dict() for s in store.get_recently_resolved(limit=10)],
        }
        print(json.dumps(result, indent=2, default=str))
        return

    # Terminal output
    console.print("")
    console.print("[bold]AI-Ready \u00b7 Knowledge Health Status[/bold]")
    console.print("=" * _WIDTH)

    # Latest assessment summary
    health = _health_label(latest.score)
    console.print(f"\n  Assessment ID:    {latest.assessment_id}")
    console.print(f"  Health Score:    {latest.score}/100  \u00b7  {health}")
    console.print(f"  Artifacts:       {latest.metadata.get('artifact_count', '?')}")
    console.print(f"  Signals:         {len(latest.signals)}")
    console.print(f"  Source:          {latest.metadata.get('source', 'N/A')}")
    console.print(f"  Dimensions:      {', '.join(sorted(latest.dimensions.keys()))}")
    if latest.metadata.get("git_commit"):
        console.print(f"  Git Commit:      {latest.metadata['git_commit']}")

    # Dimensions
    console.print(f"\n  Dimension Scores:")
    for dim_name, dim in sorted(latest.dimensions.items()):
        bar = _score_bar(dim.score)
        console.print(f"    {dim_name:20s} {dim.score:>3}/100  {rich.markup.escape(bar)}  ({dim.signals_count} signals)")

    # Recent changes
    diff_report = store.diff_latest()
    if diff_report:
        console.print(f"\n  Changes Since Last Assessment:")
        console.print(f"    Score:           {diff_report.prev_score} \u2192 {diff_report.curr_score} ({diff_report.score_delta:+d})")
        console.print(f"    New observations: {len(diff_report.new_signals)}")
        console.print(f"    Resolved:         {len(diff_report.resolved_signals)}")
        console.print(f"    Persistent:       {len(diff_report.persistent_signals)}")
        if diff_report.severity_changes:
            console.print(f"    Impact changes:   {len(diff_report.severity_changes)}")
        if diff_report.explanation:
            console.print(f"    Summary:          {diff_report.explanation}")

    # Signal stats
    stats = store.get_signal_stats()
    console.print(f"\n  Signal Lifecycle:")
    console.print(f"    New:            {stats.get('new', 0)}")
    console.print(f"    Persistent:     {stats.get('persistent', 0)}")
    console.print(f"    Recurring:      {stats.get('recurring', 0)}")
    console.print(f"    Resolved:       {stats.get('resolved', 0)}")
    if stats.get("avg_age_days"):
        console.print(f"    Avg age:        {stats['avg_age_days']} days")

    # Oldest unresolved observations
    oldest = store.get_oldest_signals(limit=5)
    if oldest:
        console.print(f"\n  Long-standing Observations:")
        for s in oldest:
            uri = _format_artifact_uri(s.artifact_uri)
            console.print(f"    {_collector_label(s.collector_id):25s} {uri}  (since {s.first_seen[:10]})")

    # Recently resolved
    resolved = store.get_recently_resolved(limit=5)
    if resolved:
        console.print(f"\n  Recently Resolved:")
        for s in resolved:
            uri = _format_artifact_uri(s.artifact_uri)
            console.print(f"    {_collector_label(s.collector_id):25s} {uri}  (resolved in {s.resolved_assessment})")

    console.print("\n" + "=" * _WIDTH)


@app.command()
def trend(
    limit: int = typer.Option(30, "--limit", help="Number of assessments to analyze"),
    db: str = typer.Option(None, "--db", help="Assessment database path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show knowledge health trend over time.

    Displays score progression, signal counts, and dimension trends across
    recent assessments. Useful for identifying whether knowledge health is
    improving, stable, or degrading.
    """
    store = _get_store(db)
    report = store.get_trend(limit=limit)

    if json_output:
        print(format_json_trend(report))
        return

    console.print("")
    console.print("[bold]AI-Ready \u00b7 Knowledge Health Trend[/bold]")
    console.print("=" * _WIDTH)

    if not report.score_trend:
        console.print(f"\n  {report.summary}")
        return

    # Trajectory
    if report.trajectory:
        console.print(f"\n  Trajectory: {report.trajectory}")

    console.print(f"\n  {report.summary}")

    console.print(f"\n  Score Progression:")
    for entry in report.score_trend:
        bar = _score_bar(entry["score"])
        console.print(f"    {entry['assessment_id']}  {rich.markup.escape(bar)} {entry['score']}/100")

    console.print(f"\n  Signal Progression:")
    for entry in report.signal_trend:
        console.print(f"    {entry['assessment_id']}  Total: {entry['total_signals']:>4}  High: {entry['high_signals']:>4}")

    if report.dimension_trends:
        console.print(f"\n  Dimension Trends:")
        for dim_name, entries in sorted(report.dimension_trends.items()):
            scores = [e["score"] for e in entries]
            if len(scores) >= 2:
                delta = scores[-1] - scores[0]
                trend_str = f"{scores[0]} \u2192 {scores[-1]} ({delta:+d})"
            else:
                trend_str = f"{scores[0]}"
            console.print(f"    {dim_name:20s} {trend_str}")

    console.print("\n" + "=" * _WIDTH)


@app.command()
def signals(
    status_filter: str = typer.Option(None, "--status", help="Filter by status (new, persistent, recurring, resolved)"),
    collector: str = typer.Option(None, "--collector", help="Filter by collector ID"),
    limit: int = typer.Option(50, "--limit", help="Max signals to show"),
    db: str = typer.Option(None, "--db", help="Assessment database path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List knowledge signals with lifecycle information.

    Shows all open Knowledge Signals by default, with their collector,
    artifact, status, age (in assessments), and first-seen date. Filter
    by status or collector to narrow results.
    """
    store = _get_store(db)

    if status_filter:
        from ai_ready.models import SignalStatus
        try:
            status_enum = SignalStatus(status_filter)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid status: {status_filter}. Use: new, persistent, recurring, resolved")
            raise typer.Exit(3)

        if status_enum == SignalStatus.RESOLVED:
            all_signals = store.get_recently_resolved(limit=limit)
        elif status_enum == SignalStatus.RECURRING:
            all_signals = store.get_recurring_signals(limit=limit)
        else:
            all_signals = [s for s in store.get_open_signals(limit=limit) if s.status == status_enum]
    else:
        all_signals = store.get_open_signals(limit=limit)

    # Filter by collector
    if collector:
        all_signals = [s for s in all_signals if s.collector_id == collector]

    if json_output:
        import json
        print(json.dumps([s.to_dict() for s in all_signals], indent=2, default=str))
        return

    if not all_signals:
        console.print("[yellow]No knowledge signals found.[/yellow]")
        return

    console.print("")
    console.print(f"[bold]AI-Ready \u00b7 Knowledge Signals ({len(all_signals)} shown)[/bold]")
    console.print("=" * _WIDTH)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Observation Type", style="cyan", width=22)
    table.add_column("Artifact", style="white", width=35)
    table.add_column("Status", style="yellow", width=12)
    table.add_column("Age", justify="right", width=6)
    table.add_column("First Seen", width=12)

    for s in all_signals:
        table.add_row(
            _collector_label(s.collector_id),
            _format_artifact_uri(s.artifact_uri),
            s.status.value,
            str(len(s.assessment_ids)),
            s.first_seen[:10] if s.first_seen else "",
        )

    console.print(table)
    console.print("=" * _WIDTH)


@app.command()
def monitor(
    path: str = typer.Argument(..., help="Path to the knowledge base directory"),
    interval: int = typer.Option(300, "--interval", help="Assessment interval in seconds"),
    db: str = typer.Option(None, "--db", help="Assessment database path"),
) -> None:
    """Continuously monitor a knowledge base.

    Runs assessments at regular intervals, reporting the Knowledge Health
    Score and detecting changes. Press Ctrl+C to stop.
    """
    console.print("[bold]AI-Ready \u00b7 Knowledge Base Monitor[/bold]")
    console.print("=" * _WIDTH)
    console.print(f"  Path:       {path}")
    console.print(f"  Interval:   {interval}s")
    console.print(f"  Press Ctrl+C to stop.")
    console.print("")

    try:
        while True:
            console.print(f"[dim]{time.strftime('%Y-%m-%d %H:%M:%S')} \u2014 Assessing knowledge health...[/dim]")

            source = Path(path).resolve()
            config = _load_config(source)
            artifacts, relations = _load_kb(source, config)

            if not artifacts:
                console.print("  [yellow]No knowledge artifacts found.[/yellow]")
            else:
                git_commit = _detect_git_commit(source)
                pipeline = AssessmentPipeline(
                    weights=config.weights,
                    enabled_collectors=config.enabled_collector_ids,
                    thresholds=config.thresholds,
                    fail_on=config.fail_on,
                )
                assessment = _run_full_assessment(
                    pipeline, artifacts, relations, str(source), git_commit
                )

                store = _get_store(db)
                store.save(assessment)

                health = _health_label(assessment.score)
                console.print(f"  Knowledge Health Score: {assessment.score}/100 \u00b7 {health}")
                console.print(f"  {len(assessment.signals):,} signals \u00b7 {len(relations):,} relationships")

                # Check for changes
                report = store.diff_latest()
                if report and report.score_delta < 0:
                    console.print(f"  [red]\u2193 Regressing ({report.score_delta:+d}) from previous assessment[/red]")
                elif report and report.score_delta > 0:
                    console.print(f"  [green]\u2191 Improving ({report.score_delta:+d}) from previous assessment[/green]")
                else:
                    console.print(f"  [dim]\u2192 Stable (no change from previous assessment)[/dim]")

            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[bold]Monitor stopped.[/bold]")


if __name__ == "__main__":
    app()
