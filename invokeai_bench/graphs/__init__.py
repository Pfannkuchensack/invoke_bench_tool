"""Graph builder registry — dispatches to the correct builder based on base model type."""

from __future__ import annotations

from typing import Callable, Optional

from invokeai_bench.config import ScenarioConfig
from invokeai_bench.graphs import cogview4, flux, flux2, sd1, sd3, sdxl, z_image

# type alias for graph builder functions
GraphBuilder = Callable[[dict, ScenarioConfig, Optional[dict]], dict]
Img2ImgBuilder = Callable[[dict, ScenarioConfig, str, Optional[dict]], dict]

TXT2IMG_BUILDERS: dict[str, GraphBuilder] = {
    "sd-1": sd1.build_txt2img,
    "sd-2": sd1.build_txt2img,
    "sdxl": sdxl.build_txt2img,
    "flux": flux.build_txt2img,
    "flux2": flux2.build_txt2img,
    "sd-3": sd3.build_txt2img,
    "z-image": z_image.build_txt2img,
    "cogview4": cogview4.build_txt2img,
}

IMG2IMG_BUILDERS: dict[str, Img2ImgBuilder] = {
    "sd-1": sd1.build_img2img,
    "sd-2": sd1.build_img2img,
    "sdxl": sdxl.build_img2img,
    "flux": flux.build_img2img,
    "flux2": flux2.build_img2img,
    "sd-3": sd3.build_img2img,
    "z-image": z_image.build_img2img,
    "cogview4": cogview4.build_img2img,
}

SUPPORTED_BASES = set(TXT2IMG_BUILDERS.keys())


def build_graph(
    model_info: dict,
    scenario: ScenarioConfig,
    image_name: Optional[str] = None,
    submodels: Optional[dict] = None,
) -> dict:
    """Build the correct graph for a given scenario and model type."""
    base = scenario.base

    if scenario.type == "img2img":
        if image_name is None:
            raise ValueError("img2img scenario requires an uploaded image_name")
        builder = IMG2IMG_BUILDERS.get(base)
        if builder is None:
            raise ValueError(f"No img2img graph builder for base '{base}'. Supported: {sorted(IMG2IMG_BUILDERS)}")
        return builder(model_info, scenario, image_name, submodels)
    else:
        builder_t2i = TXT2IMG_BUILDERS.get(base)
        if builder_t2i is None:
            raise ValueError(f"No txt2img graph builder for base '{base}'. Supported: {sorted(TXT2IMG_BUILDERS)}")
        return builder_t2i(model_info, scenario, submodels)
