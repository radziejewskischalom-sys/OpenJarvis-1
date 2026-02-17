"""``jarvis telemetry`` — query and manage telemetry data."""

from __future__ import annotations

import csv as csv_mod
import io
import json as json_mod

import click
from rich.console import Console
from rich.table import Table

from openjarvis.core.config import load_config
from openjarvis.telemetry.aggregator import TelemetryAggregator


def _get_aggregator() -> TelemetryAggregator:
    """Build a TelemetryAggregator from user config."""
    config = load_config()
    return TelemetryAggregator(config.telemetry.db_path)


@click.group()
def telemetry() -> None:
    """Query and manage inference telemetry data."""


@telemetry.command()
@click.option(
    "-n", "--top", "top_n", default=10, type=int,
    help="Number of top models to show.",
)
def stats(top_n: int) -> None:
    """Show aggregated telemetry statistics."""
    console = Console()
    agg = _get_aggregator()
    try:
        summary = agg.summary()

        # Overview
        overview = Table(title="Telemetry Overview")
        overview.add_column("Metric", style="cyan")
        overview.add_column("Value", style="green")
        overview.add_row("Total Calls", str(summary.total_calls))
        overview.add_row("Total Tokens", str(summary.total_tokens))
        overview.add_row("Total Cost (USD)", f"${summary.total_cost:.6f}")
        overview.add_row("Total Latency (s)", f"{summary.total_latency:.2f}")
        console.print(overview)

        # Per-model table
        if summary.per_model:
            model_table = Table(title=f"Top {top_n} Models")
            model_table.add_column("Model", style="cyan")
            model_table.add_column("Calls", justify="right")
            model_table.add_column("Tokens", justify="right")
            model_table.add_column("Avg Latency", justify="right")
            model_table.add_column("Cost", justify="right")
            for ms in summary.per_model[:top_n]:
                model_table.add_row(
                    ms.model_id,
                    str(ms.call_count),
                    str(ms.total_tokens),
                    f"{ms.avg_latency:.3f}s",
                    f"${ms.total_cost:.6f}",
                )
            console.print(model_table)

        # Per-engine table
        if summary.per_engine:
            engine_table = Table(title="Engines")
            engine_table.add_column("Engine", style="cyan")
            engine_table.add_column("Calls", justify="right")
            engine_table.add_column("Tokens", justify="right")
            engine_table.add_column("Avg Latency", justify="right")
            engine_table.add_column("Cost", justify="right")
            for es in summary.per_engine:
                engine_table.add_row(
                    es.engine,
                    str(es.call_count),
                    str(es.total_tokens),
                    f"{es.avg_latency:.3f}s",
                    f"${es.total_cost:.6f}",
                )
            console.print(engine_table)

        if summary.total_calls == 0:
            console.print("[dim]No telemetry data recorded yet.[/dim]")
    finally:
        agg.close()


@telemetry.command()
@click.option(
    "-f", "--format", "fmt", default="json", type=click.Choice(["json", "csv"]),
    help="Output format.",
)
@click.option(
    "-o", "--output", "output_path", default=None, type=click.Path(),
    help="Output file path (default: stdout).",
)
def export(fmt: str, output_path: str | None) -> None:
    """Export telemetry records."""
    agg = _get_aggregator()
    try:
        records = agg.export_records()

        if fmt == "json":
            text = json_mod.dumps(records, indent=2)
        else:
            # CSV
            buf = io.StringIO()
            if records:
                writer = csv_mod.DictWriter(buf, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            text = buf.getvalue()

        if output_path:
            with open(output_path, "w") as fh:
                fh.write(text)
            click.echo(f"Exported {len(records)} records to {output_path}")
        else:
            click.echo(text)
    finally:
        agg.close()


@telemetry.command()
@click.option(
    "-y", "--yes", "confirmed", is_flag=True,
    help="Skip confirmation prompt.",
)
def clear(confirmed: bool) -> None:
    """Delete all telemetry records."""
    if not confirmed:
        if not click.confirm("Delete all telemetry records?"):
            click.echo("Aborted.")
            return

    agg = _get_aggregator()
    try:
        count = agg.clear()
        click.echo(f"Deleted {count} telemetry records.")
    finally:
        agg.close()
