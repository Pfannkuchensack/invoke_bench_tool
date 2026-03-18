"""CLI entry point for the InvokeAI benchmark tool."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from invokeai_bench import __version__
from invokeai_bench.client import InvokeAIClient
from invokeai_bench.config import DefaultsConfig, load_config
from invokeai_bench.results import compare_results, export_json, format_summary, load_result
from invokeai_bench.runner import run_all_models, run_benchmark


def _make_client(host: str, config_email: Optional[str] = None, config_password: Optional[str] = None) -> InvokeAIClient:
    client = InvokeAIClient(base_url=host)
    if config_email and config_password:
        click.echo(f"Authenticating as {config_email}...")
        client.login(config_email, config_password)
    return client


@click.group()
@click.version_option(version=__version__, prog_name="invokeai-bench")
def main() -> None:
    """InvokeAI Benchmark Tool — measure generation performance across all model types."""


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path), help="Path to benchmark TOML config")
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None, help="Output JSON path (default: results_<timestamp>.json)")
@click.option("--host", default=None, help="Override InvokeAI host (e.g. http://127.0.0.1:9090)")
@click.option("--dry-run", is_flag=True, help="Print graph JSON without submitting")
@click.option("--verbose", is_flag=True, help="Show per-iteration timing")
def run(config_path: Path, output_path: Optional[Path], host: Optional[str], dry_run: bool, verbose: bool) -> None:
    """Run benchmark scenarios from a config file."""
    config = load_config(config_path)

    if host:
        config.connection.host = host

    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"benchmark_results_{ts}.json")

    client = _make_client(config.connection.host, config.connection.email, config.connection.password)
    try:
        click.echo(f"Connecting to {config.connection.host}...")
        version = client.get_version()
        click.echo(f"InvokeAI version: {version}")
        click.echo(f"Scenarios: {len(config.scenario)}")

        result = run_benchmark(client, config, verbose=verbose, dry_run=dry_run)

        if not dry_run:
            export_json(result, output_path)
            click.echo(f"\n{format_summary(result)}")
            click.echo(f"\nResults saved to: {output_path}")
    finally:
        client.close()


@main.command("run-all")
@click.option("--host", default="http://127.0.0.1:9090", help="InvokeAI host")
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None, help="Output JSON path")
@click.option("--iterations", default=3, help="Number of benchmark iterations per model")
@click.option("--warmup", default=1, help="Number of warmup runs per model")
@click.option("--seed", default=42, help="Base seed")
@click.option("--dry-run", is_flag=True, help="Print graph JSON without submitting")
@click.option("--verbose", is_flag=True, help="Show per-iteration timing")
def run_all(
    host: str,
    output_path: Optional[Path],
    iterations: int,
    warmup: int,
    seed: int,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Auto-discover all installed models and benchmark them."""
    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"benchmark_all_{ts}.json")

    defaults = DefaultsConfig(warmup_runs=warmup, iterations=iterations, seed=seed)

    client = _make_client(host)
    try:
        click.echo(f"Connecting to {host}...")
        version = client.get_version()
        click.echo(f"InvokeAI version: {version}")

        result = run_all_models(client, defaults, verbose=verbose, dry_run=dry_run)

        if not dry_run:
            export_json(result, output_path)
            click.echo(f"\n{format_summary(result)}")
            click.echo(f"\nResults saved to: {output_path}")
    finally:
        client.close()


@main.command("list-models")
@click.option("--host", default="http://127.0.0.1:9090", help="InvokeAI host")
@click.option("--base", default=None, help="Filter by base model type (e.g. sdxl, flux)")
def list_models(host: str, base: Optional[str]) -> None:
    """List available models on the InvokeAI instance."""
    client = _make_client(host)
    try:
        models = client.list_models(base=base, model_type="main")
        if not models:
            click.echo("No models found.")
            return

        click.echo(f"{'Name':<40} {'Base':<10} {'Format':<12} {'Key'}")
        click.echo("-" * 90)
        for m in sorted(models, key=lambda x: (x.get("base", ""), x.get("name", ""))):
            click.echo(
                f"{m.get('name', '?'):<40} "
                f"{m.get('base', '?'):<10} "
                f"{m.get('format', '?'):<12} "
                f"{m.get('key', '?')}"
            )
    finally:
        client.close()


@main.command()
@click.argument("baseline", type=click.Path(exists=True, path_type=Path))
@click.argument("current", type=click.Path(exists=True, path_type=Path))
def compare(baseline: Path, current: Path) -> None:
    """Compare two benchmark result JSON files.

    BASELINE is the reference run (A), CURRENT is the new run (B).
    Positive change % = slower (regression), negative = faster (improvement).
    """
    a = load_result(baseline)
    b = load_result(current)
    click.echo(compare_results(a, b))


if __name__ == "__main__":
    main()
