"""``jarvis init`` — detect hardware, generate config, write to disk."""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel

from openjarvis.core.config import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    detect_hardware,
    generate_default_toml,
)


@click.command()
@click.option(
    "--force", is_flag=True, help="Overwrite existing config without prompting."
)
def init(force: bool) -> None:
    """Detect hardware and generate ~/.openjarvis/config.toml."""
    console = Console()

    if DEFAULT_CONFIG_PATH.exists() and not force:
        console.print(
            f"[yellow]Config already exists at {DEFAULT_CONFIG_PATH}[/yellow]"
        )
        console.print("Use [bold]--force[/bold] to overwrite.")
        raise SystemExit(1)

    console.print("[bold]Detecting hardware...[/bold]")
    hw = detect_hardware()

    console.print(f"  Platform : {hw.platform}")
    console.print(f"  CPU      : {hw.cpu_brand} ({hw.cpu_count} cores)")
    console.print(f"  RAM      : {hw.ram_gb} GB")
    if hw.gpu:
        console.print(
            f"  GPU      : {hw.gpu.name} ({hw.gpu.vram_gb} GB VRAM, x{hw.gpu.count})"
        )
    else:
        console.print("  GPU      : none detected")

    toml_content = generate_default_toml(hw)

    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(toml_content)

    console.print()
    console.print(
        Panel(toml_content, title=str(DEFAULT_CONFIG_PATH), border_style="green")
    )
    console.print("[green]Config written successfully.[/green]")
