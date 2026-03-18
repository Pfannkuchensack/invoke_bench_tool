"""Benchmark execution loop — warmup, iterations, polling, timing."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from invokeai_bench.client import InvokeAIClient
from invokeai_bench.config import (
    BASE_DEFAULTS,
    BenchmarkConfig,
    DefaultsConfig,
    ScenarioConfig,
    make_auto_scenario,
)
from invokeai_bench.graphs import SUPPORTED_BASES, build_graph
from invokeai_bench.graphs.common import vary_seed
from invokeai_bench.results import (
    BenchmarkResult,
    ScenarioResult,
    TimingEntry,
    finalize_scenario,
)
from invokeai_bench.submodels import resolve_submodels
from invokeai_bench.sysinfo import collect_sysinfo


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 datetime string."""
    if not s:
        return None
    # Handle Z suffix
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _check_queue_empty(client: InvokeAIClient) -> bool:
    """Warn if the queue has pending/in-progress items."""
    status = client.get_queue_status()
    q = status.get("queue", {})
    busy = q.get("pending", 0) + q.get("in_progress", 0)
    if busy > 0:
        click.secho(
            f"  WARNING: Queue has {busy} pending/in-progress items. "
            "Results may be affected by other workloads.",
            fg="yellow",
        )
        return False
    return True


def run_scenario(
    client: InvokeAIClient,
    scenario: ScenarioConfig,
    defaults: DefaultsConfig,
    verbose: bool = False,
    dry_run: bool = False,
) -> ScenarioResult:
    """Run a single benchmark scenario and return its result."""
    result = ScenarioResult(
        name=scenario.name,
        model_name=scenario.model_name,
        model_base=scenario.base,
        scenario_config=scenario.model_dump(exclude_none=True),
    )

    try:
        # Resolve model
        if scenario.model_key:
            model_info = client.find_model_by_key(scenario.model_key)
        else:
            model_info = client.find_model(scenario.model_name, scenario.base)

        # Resolve auxiliary submodels (VAE, T5 Encoder, etc.)
        submodels = resolve_submodels(client, scenario.base, model_info)
        if submodels and verbose:
            click.echo(f"  Resolved submodels: {list(submodels.keys())}")

        # Upload image if img2img
        image_name: Optional[str] = None
        if scenario.type == "img2img":
            if not scenario.input_image:
                raise ValueError("img2img scenario requires 'input_image' path")
            click.echo(f"  Uploading image: {scenario.input_image}")
            img_dto = client.upload_image(Path(scenario.input_image))
            image_name = img_dto["image_name"]

        # Build graph
        graph = build_graph(model_info, scenario, image_name=image_name, submodels=submodels)

        if dry_run:
            import json
            click.echo(json.dumps(graph, indent=2))
            return result

        warmup_runs = scenario.effective_warmup(defaults)
        iterations = scenario.effective_iterations(defaults)

        # Warmup
        if warmup_runs > 0:
            click.echo(f"  Warmup: {warmup_runs} run(s)...")
            for _ in range(warmup_runs):
                graph_w = vary_seed(graph, 0)
                batch = client.enqueue_batch(graph_w)
                client.wait_for_batch(
                    batch["batch"]["batch_id"],
                    batch.get("item_ids", []),
                )

        # Capture cache stats before
        result.cache_stats_before = client.get_cache_stats()

        # Benchmark iterations
        click.echo(f"  Running {iterations} iteration(s)...")
        for i in range(iterations):
            seed = defaults.seed + i
            graph_i = vary_seed(graph, seed)

            t_start = time.monotonic()
            batch = client.enqueue_batch(graph_i)
            batch_id = batch["batch"]["batch_id"]
            item_ids = batch.get("item_ids", [])
            items = client.wait_for_batch(batch_id, item_ids)
            t_end = time.monotonic()

            client_time = t_end - t_start

            for item in items:
                started = _parse_dt(item.get("started_at"))
                completed = _parse_dt(item.get("completed_at"))
                server_time = (completed - started).total_seconds() if started and completed else client_time

                entry = TimingEntry(
                    iteration=i,
                    server_wall_time_s=round(server_time, 4),
                    client_round_trip_s=round(client_time, 4),
                    status=item.get("status", "unknown"),
                )
                result.timings.append(entry)

                if verbose:
                    status_str = click.style(entry.status, fg="green" if entry.status == "completed" else "red")
                    click.echo(f"    [{i+1}/{iterations}] {status_str} — server: {entry.server_wall_time_s:.2f}s, client: {entry.client_round_trip_s:.2f}s")

        # Capture cache stats after
        result.cache_stats_after = client.get_cache_stats()

        # Compute statistics
        finalize_scenario(result)

    except Exception as e:
        result.error = str(e)
        click.secho(f"  ERROR: {e}", fg="red")

    return result


def run_benchmark(
    client: InvokeAIClient,
    config: BenchmarkConfig,
    verbose: bool = False,
    dry_run: bool = False,
) -> BenchmarkResult:
    """Run all scenarios from a config and return the full benchmark result."""
    result = BenchmarkResult(system_info=collect_sysinfo())

    try:
        result.invokeai_version = client.get_version()
    except Exception:
        pass

    _check_queue_empty(client)

    for scenario in config.scenario:
        click.echo(f"\nScenario: {scenario.name} ({scenario.base} / {scenario.model_name})")
        sr = run_scenario(client, scenario, config.defaults, verbose=verbose, dry_run=dry_run)
        result.scenarios.append(sr)

        if sr.server_stats and not dry_run:
            click.echo(f"  Result: {sr.server_stats.mean_s:.2f}s avg ({sr.server_stats.std_s:.2f}s std)")

    return result


def run_all_models(
    client: InvokeAIClient,
    defaults: DefaultsConfig,
    verbose: bool = False,
    dry_run: bool = False,
) -> BenchmarkResult:
    """Auto-discover all installed main models and benchmark them."""
    result = BenchmarkResult(system_info=collect_sysinfo())

    try:
        result.invokeai_version = client.get_version()
    except Exception:
        pass

    _check_queue_empty(client)

    models = client.list_models(model_type="main")
    click.echo(f"Found {len(models)} main model(s)")

    for model_info in models:
        base = model_info.get("base", "unknown")
        name = model_info.get("name", "unknown")

        if base not in SUPPORTED_BASES:
            click.secho(f"\nSkipping {name} (unsupported base: {base})", fg="yellow")
            continue

        scenario = make_auto_scenario(model_info, defaults)
        click.echo(f"\nScenario: {scenario.name} ({base} / {name})")
        sr = run_scenario(client, scenario, defaults, verbose=verbose, dry_run=dry_run)
        result.scenarios.append(sr)

        if sr.server_stats and not dry_run:
            click.echo(f"  Result: {sr.server_stats.mean_s:.2f}s avg ({sr.server_stats.std_s:.2f}s std)")

    return result
