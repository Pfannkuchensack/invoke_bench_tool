"""Statistics computation and JSON export for benchmark results."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from invokeai_bench import __version__


class TimingEntry(BaseModel):
    iteration: int
    server_wall_time_s: float
    client_round_trip_s: float
    status: str


class TimingStatistics(BaseModel):
    mean_s: float
    median_s: float
    std_s: float
    min_s: float
    max_s: float


class ScenarioResult(BaseModel):
    name: str
    model_name: str
    model_base: str
    scenario_config: dict
    timings: list[TimingEntry] = Field(default_factory=list)
    server_stats: Optional[TimingStatistics] = None
    client_stats: Optional[TimingStatistics] = None
    cache_stats_before: Optional[dict] = None
    cache_stats_after: Optional[dict] = None
    error: Optional[str] = None


class BenchmarkResult(BaseModel):
    tool_version: str = __version__
    invokeai_version: str = "unknown"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    system_info: dict = Field(default_factory=dict)
    scenarios: list[ScenarioResult] = Field(default_factory=list)


def compute_stats(values: list[float]) -> TimingStatistics:
    """Compute basic statistics for a list of values."""
    n = len(values)
    if n == 0:
        return TimingStatistics(mean_s=0, median_s=0, std_s=0, min_s=0, max_s=0)

    sorted_v = sorted(values)
    mean = sum(sorted_v) / n
    median = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2

    if n > 1:
        variance = sum((x - mean) ** 2 for x in sorted_v) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return TimingStatistics(
        mean_s=round(mean, 4),
        median_s=round(median, 4),
        std_s=round(std, 4),
        min_s=round(sorted_v[0], 4),
        max_s=round(sorted_v[-1], 4),
    )


def finalize_scenario(result: ScenarioResult) -> None:
    """Compute statistics from the raw timings."""
    successful = [t for t in result.timings if t.status == "completed"]
    if successful:
        result.server_stats = compute_stats([t.server_wall_time_s for t in successful])
        result.client_stats = compute_stats([t.client_round_trip_s for t in successful])


def export_json(result: BenchmarkResult, path: Path) -> None:
    """Write the benchmark result to a JSON file."""
    data = result.model_dump(mode="json", exclude_none=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_summary(result: BenchmarkResult) -> str:
    """Format a human-readable summary of benchmark results."""
    lines = [
        f"InvokeAI Benchmark Results",
        f"  Version: {result.invokeai_version}",
        f"  Date: {result.timestamp}",
        f"  GPU: {result.system_info.get('gpu', 'N/A')}",
        "",
    ]
    for s in result.scenarios:
        if s.error:
            lines.append(f"  {s.name}: FAILED - {s.error}")
            continue
        if s.server_stats:
            lines.append(
                f"  {s.name}: "
                f"{s.server_stats.mean_s:.2f}s avg "
                f"(median {s.server_stats.median_s:.2f}s, "
                f"std {s.server_stats.std_s:.2f}s, "
                f"range {s.server_stats.min_s:.2f}-{s.server_stats.max_s:.2f}s) "
                f"[{len(s.timings)} runs]"
            )
        else:
            lines.append(f"  {s.name}: No successful runs")
    return "\n".join(lines)


def load_result(path: Path) -> BenchmarkResult:
    """Load a benchmark result from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BenchmarkResult.model_validate(data)


def _pct(old: float, new: float) -> str:
    """Format percentage change with color hint."""
    if old == 0:
        return "N/A"
    pct = ((new - old) / old) * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _delta(old: float, new: float) -> str:
    """Format absolute delta."""
    d = new - old
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.2f}s"


def compare_results(a: BenchmarkResult, b: BenchmarkResult) -> str:
    """Compare two benchmark results side-by-side.

    Returns a formatted comparison table. 'a' is baseline, 'b' is the new run.
    Positive % = slower (regression), negative % = faster (improvement).
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 90)
    lines.append("InvokeAI Benchmark Comparison")
    lines.append("=" * 90)
    lines.append(f"  Baseline (A): v{a.invokeai_version}  {a.timestamp}")
    lines.append(f"  Current  (B): v{b.invokeai_version}  {b.timestamp}")
    gpu_a = a.system_info.get("gpu", "N/A")
    gpu_b = b.system_info.get("gpu", "N/A")
    if gpu_a == gpu_b:
        lines.append(f"  GPU: {gpu_a}")
    else:
        lines.append(f"  GPU (A): {gpu_a}")
        lines.append(f"  GPU (B): {gpu_b}")
    lines.append("")

    # Build lookup for B scenarios by name
    b_by_name: dict[str, ScenarioResult] = {s.name: s for s in b.scenarios}

    # Table header
    header = f"  {'Scenario':<35} {'A (avg)':>10} {'B (avg)':>10} {'Delta':>10} {'Change':>10}"
    lines.append(header)
    lines.append("  " + "-" * 77)

    matched = 0
    for sa in a.scenarios:
        sb = b_by_name.pop(sa.name, None)

        if sa.error:
            lines.append(f"  {sa.name:<35} {'FAILED':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue

        if sb is None:
            # Scenario only in A
            avg_a = f"{sa.server_stats.mean_s:.2f}s" if sa.server_stats else "N/A"
            lines.append(f"  {sa.name:<35} {avg_a:>10} {'---':>10} {'---':>10} {'only A':>10}")
            continue

        if sb.error:
            avg_a = f"{sa.server_stats.mean_s:.2f}s" if sa.server_stats else "N/A"
            lines.append(f"  {sa.name:<35} {avg_a:>10} {'FAILED':>10} {'-':>10} {'-':>10}")
            continue

        if sa.server_stats and sb.server_stats:
            avg_a = sa.server_stats.mean_s
            avg_b = sb.server_stats.mean_s
            delta = _delta(avg_a, avg_b)
            pct = _pct(avg_a, avg_b)
            lines.append(
                f"  {sa.name:<35} {avg_a:>9.2f}s {avg_b:>9.2f}s {delta:>10} {pct:>10}"
            )
            matched += 1
        else:
            lines.append(f"  {sa.name:<35} {'N/A':>10} {'N/A':>10} {'-':>10} {'-':>10}")

    # Scenarios only in B
    for name, sb in b_by_name.items():
        avg_b = f"{sb.server_stats.mean_s:.2f}s" if sb.server_stats else "N/A"
        lines.append(f"  {name:<35} {'---':>10} {avg_b:>10} {'---':>10} {'only B':>10}")

    lines.append("  " + "-" * 77)

    # Summary
    if matched > 0:
        lines.append("")
        lines.append("  Positive delta = slower (regression), negative = faster (improvement)")

    lines.append("")
    return "\n".join(lines)
