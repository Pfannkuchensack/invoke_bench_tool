"""TOML config loading and Pydantic models for benchmark scenarios."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import,no-redef]

BaseModelType = Literal[
    "sd-1", "sd-2", "sdxl", "flux", "flux2", "sd-3", "z-image", "cogview4"
]

ScenarioType = Literal["txt2img", "img2img"]

# Sensible defaults per base model type
BASE_DEFAULTS: dict[str, dict] = {
    "sd-1": {"width": 512, "height": 512, "steps": 20, "cfg_scale": 7.5},
    "sd-2": {"width": 768, "height": 768, "steps": 20, "cfg_scale": 7.5},
    "sdxl": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 7.0},
    "flux": {"width": 1024, "height": 1024, "steps": 20, "cfg_scale": 1.0, "guidance": 4.0},
    "flux2": {"width": 1024, "height": 1024, "steps": 20, "cfg_scale": 1.0},
    "sd-3": {"width": 1024, "height": 1024, "steps": 28, "cfg_scale": 3.5},
    "z-image": {"width": 1024, "height": 1024, "steps": 8, "cfg_scale": 1.0},
    "cogview4": {"width": 1024, "height": 1024, "steps": 25, "cfg_scale": 3.5},
}

DEFAULT_PROMPT = "a photo of a mountain landscape at golden hour, highly detailed"
DEFAULT_NEGATIVE = "blurry, bad quality, worst quality"


class ConnectionConfig(BaseModel):
    host: str = "http://127.0.0.1:9090"
    email: Optional[str] = None
    password: Optional[str] = None


class DefaultsConfig(BaseModel):
    warmup_runs: int = 1
    iterations: int = 3
    seed: int = 42
    positive_prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE


class ScenarioConfig(BaseModel):
    name: str
    type: ScenarioType = "txt2img"
    base: BaseModelType
    model_name: str
    model_key: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg_scale: float = 7.0
    guidance: Optional[float] = None  # FLUX-specific guidance parameter
    scheduler: str = "euler"
    positive_prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE
    denoising_start: float = 0.0
    input_image: Optional[str] = None
    warmup_runs: Optional[int] = None
    iterations: Optional[int] = None

    def effective_warmup(self, defaults: DefaultsConfig) -> int:
        return self.warmup_runs if self.warmup_runs is not None else defaults.warmup_runs

    def effective_iterations(self, defaults: DefaultsConfig) -> int:
        return self.iterations if self.iterations is not None else defaults.iterations


class BenchmarkConfig(BaseModel):
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    scenario: list[ScenarioConfig] = Field(default_factory=list)


def load_config(path: Path) -> BenchmarkConfig:
    """Load benchmark config from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return BenchmarkConfig.model_validate(data)


def make_auto_scenario(model_info: dict, defaults: DefaultsConfig) -> ScenarioConfig:
    """Create a default txt2img scenario from a model's API info."""
    base = model_info["base"]
    name = model_info["name"]
    bd = BASE_DEFAULTS.get(base, BASE_DEFAULTS["sd-1"])

    return ScenarioConfig(
        name=f"{name}_{bd['width']}x{bd['height']}_{bd['steps']}steps",
        type="txt2img",
        base=base,
        model_name=name,
        model_key=model_info.get("key"),
        width=bd["width"],
        height=bd["height"],
        steps=bd["steps"],
        cfg_scale=bd["cfg_scale"],
        guidance=bd.get("guidance"),
        positive_prompt=defaults.positive_prompt,
        negative_prompt=defaults.negative_prompt,
    )
