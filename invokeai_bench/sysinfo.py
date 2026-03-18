"""Collect system information for benchmark reports."""

from __future__ import annotations

import platform
import subprocess


def collect_sysinfo() -> dict:
    """Gather platform, CPU, and GPU info."""
    info: dict = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "hostname": platform.node(),
    }

    # Try to get GPU info via nvidia-smi
    gpu = _get_nvidia_gpu()
    if gpu:
        info["gpu"] = gpu

    return info


def _get_nvidia_gpu() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
