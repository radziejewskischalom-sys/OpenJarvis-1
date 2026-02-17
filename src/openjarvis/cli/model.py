"""``jarvis model`` — model management subcommands."""

from __future__ import annotations

import sys

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openjarvis.core.config import load_config
from openjarvis.core.registry import ModelRegistry
from openjarvis.engine import discover_engines, discover_models
from openjarvis.intelligence import merge_discovered_models, register_builtin_models


@click.group()
def model() -> None:
    """Manage language models."""


@model.command("list")
def list_models() -> None:
    """List available models from running engines."""
    console = Console()
    config = load_config()
    register_builtin_models()

    engines = discover_engines(config)
    if not engines:
        console.print(
            "[yellow]No inference engines detected.[/yellow]\n"
            "Start an engine (e.g. [cyan]ollama serve[/cyan]) and try again."
        )
        return

    all_models = discover_models(engines)
    for ek, model_ids in all_models.items():
        merge_discovered_models(ek, model_ids)

    table = Table(title="Available Models")
    table.add_column("Engine", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Parameters", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("VRAM", justify="right")

    for engine_key, model_ids in all_models.items():
        for mid in model_ids:
            try:
                spec = ModelRegistry.get(mid)
                params = f"{spec.parameter_count_b}B" if spec.parameter_count_b else "-"
                ctx = f"{spec.context_length:,}" if spec.context_length else "-"
                vram = f"{spec.min_vram_gb}GB" if spec.min_vram_gb else "-"
            except KeyError:
                params = "-"
                ctx = "-"
                vram = "-"
            table.add_row(engine_key, mid, params, ctx, vram)

    console.print(table)


@model.command()
@click.argument("model_name")
def info(model_name: str) -> None:
    """Show details for a model."""
    console = Console()
    register_builtin_models()

    # Also try discovering from running engines
    config = load_config()
    engines = discover_engines(config)
    all_models = discover_models(engines)
    for ek, model_ids in all_models.items():
        merge_discovered_models(ek, model_ids)

    if not ModelRegistry.contains(model_name):
        console.print(f"[red]Model not found:[/red] {model_name}")
        sys.exit(1)

    spec = ModelRegistry.get(model_name)
    params = f"{spec.parameter_count_b}B" if spec.parameter_count_b else "unknown"
    ctx_len = f"{spec.context_length:,}" if spec.context_length else "unknown"
    vram = f"{spec.min_vram_gb}GB" if spec.min_vram_gb else "-"
    engines = ", ".join(spec.supported_engines) if spec.supported_engines else "-"
    provider = spec.provider or "-"
    api_key = "required" if spec.requires_api_key else "not required"
    lines = [
        f"[bold]Model ID:[/bold]     {spec.model_id}",
        f"[bold]Name:[/bold]         {spec.name}",
        f"[bold]Parameters:[/bold]   {params}",
        f"[bold]Context:[/bold]      {ctx_len}",
        f"[bold]Quantization:[/bold] {spec.quantization.value}",
        f"[bold]Min VRAM:[/bold]     {vram}",
        f"[bold]Engines:[/bold]      {engines}",
        f"[bold]Provider:[/bold]     {provider}",
        f"[bold]API Key:[/bold]      {api_key}",
    ]
    console.print(Panel("\n".join(lines), title=spec.name, border_style="blue"))


@model.command()
@click.argument("model_name")
def pull(model_name: str) -> None:
    """Download a model (Ollama only)."""
    console = Console()
    config = load_config()
    host = config.engine.ollama_host.rstrip("/")

    console.print(f"Pulling [cyan]{model_name}[/cyan] via Ollama...")
    try:
        with httpx.stream(
            "POST",
            f"{host}/api/pull",
            json={"name": model_name, "stream": True},
            timeout=600.0,
        ) as resp:
            resp.raise_for_status()
            import json

            for line in resp.iter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                status = data.get("status", "")
                if "total" in data and "completed" in data:
                    total = data["total"]
                    done = data["completed"]
                    pct = int(done / total * 100) if total else 0
                    console.print(f"  {status}: {pct}%", end="\r")
                elif status:
                    console.print(f"  {status}")
        console.print(f"\n[green]Successfully pulled {model_name}[/green]")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to Ollama.[/red] Is it running?")
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]Ollama error:[/red] {exc.response.status_code}")
        sys.exit(1)
