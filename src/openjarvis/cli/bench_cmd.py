"""``jarvis bench`` — run inference benchmarks."""

from __future__ import annotations

import json as json_mod
import sys

import click
from rich.console import Console

from openjarvis.core.config import load_config
from openjarvis.engine import get_engine


@click.group()
def bench() -> None:
    """Run inference benchmarks."""


@bench.command()
@click.option("-m", "--model", "model_name", default=None, help="Model to benchmark.")
@click.option("-e", "--engine", "engine_key", default=None, help="Engine backend.")
@click.option(
    "-n", "--samples", "num_samples", default=10, type=int,
    help="Number of samples per benchmark.",
)
@click.option(
    "-b", "--benchmark", "benchmark_name", default=None,
    help="Specific benchmark to run (default: all).",
)
@click.option(
    "-o", "--output", "output_path", default=None, type=click.Path(),
    help="Write JSONL results to file.",
)
@click.option(
    "--json", "output_json", is_flag=True,
    help="Output JSON summary to stdout.",
)
def run(
    model_name: str | None,
    engine_key: str | None,
    num_samples: int,
    benchmark_name: str | None,
    output_path: str | None,
    output_json: bool,
) -> None:
    """Run benchmarks against an inference engine."""
    console = Console(stderr=True)
    config = load_config()

    # Import and register benchmarks
    from openjarvis.bench import ensure_registered
    from openjarvis.bench._stubs import BenchmarkSuite
    from openjarvis.core.registry import BenchmarkRegistry

    ensure_registered()

    # Get engine
    resolved = get_engine(config, engine_key)
    if resolved is None:
        console.print("[red bold]No inference engine available.[/red bold]")
        sys.exit(1)

    engine_name, engine = resolved

    # Resolve model
    if model_name is None:
        models = engine.list_models()
        if models:
            model_name = models[0]
        else:
            console.print("[red]No model available on engine.[/red]")
            sys.exit(1)

    # Select benchmarks
    if benchmark_name:
        if not BenchmarkRegistry.contains(benchmark_name):
            console.print(
                f"[red]Unknown benchmark: {benchmark_name}. "
                f"Available: {', '.join(BenchmarkRegistry.keys())}[/red]"
            )
            sys.exit(1)
        bench_cls = BenchmarkRegistry.get(benchmark_name)
        benchmarks = [bench_cls()]
    else:
        benchmarks = [cls() for _, cls in BenchmarkRegistry.items()]

    if not benchmarks:
        console.print("[yellow]No benchmarks registered.[/yellow]")
        return

    suite = BenchmarkSuite(benchmarks)
    console.print(
        f"[cyan]Running {len(benchmarks)} benchmark(s) "
        f"on {engine_name}/{model_name} ({num_samples} samples)...[/cyan]"
    )

    results = suite.run_all(engine, model_name, num_samples=num_samples)

    # Output results
    if output_path:
        jsonl = suite.to_jsonl(results)
        with open(output_path, "w") as fh:
            fh.write(jsonl + "\n")
        console.print(f"[green]Results written to {output_path}[/green]")

    if output_json:
        summary = suite.summary(results)
        click.echo(json_mod.dumps(summary, indent=2))
    elif not output_path:
        # Pretty-print to console
        for r in results:
            console.print(
                f"\n[bold]{r.benchmark_name}[/bold] "
                f"({r.samples} samples, {r.errors} errors)"
            )
            for k, v in r.metrics.items():
                console.print(f"  {k}: {v:.4f}")
