"""Graph builders for SD1.5 / SD2 models."""

from __future__ import annotations

from invokeai_bench.config import ScenarioConfig
from invokeai_bench.graphs.common import make_edge, make_graph, make_model_id, make_node


def build_txt2img(model_info: dict, scenario: ScenarioConfig, submodels: dict | None = None) -> dict:
    nodes = [
        make_node("model_loader", "main_model_loader", model=make_model_id(model_info)),
        make_node("pos_compel", "compel", prompt=scenario.positive_prompt),
        make_node("neg_compel", "compel", prompt=scenario.negative_prompt),
        make_node(
            "noise", "noise",
            seed=42,
            width=scenario.width,
            height=scenario.height,
        ),
        make_node(
            "denoise", "denoise_latents",
            steps=scenario.steps,
            cfg_scale=scenario.cfg_scale,
            denoising_start=0.0,
            denoising_end=1.0,
            scheduler=scenario.scheduler,
        ),
        make_node("l2i", "l2i", fp32=False, tiled=False),
    ]
    edges = [
        make_edge("model_loader", "clip", "pos_compel", "clip"),
        make_edge("model_loader", "clip", "neg_compel", "clip"),
        make_edge("model_loader", "unet", "denoise", "unet"),
        make_edge("model_loader", "vae", "l2i", "vae"),
        make_edge("pos_compel", "conditioning", "denoise", "positive_conditioning"),
        make_edge("neg_compel", "conditioning", "denoise", "negative_conditioning"),
        make_edge("noise", "noise", "denoise", "noise"),
        make_edge("denoise", "latents", "l2i", "latents"),
    ]
    return make_graph(nodes, edges)


def build_img2img(model_info: dict, scenario: ScenarioConfig, image_name: str, submodels: dict | None = None) -> dict:
    nodes = [
        make_node("model_loader", "main_model_loader", model=make_model_id(model_info)),
        make_node("pos_compel", "compel", prompt=scenario.positive_prompt),
        make_node("neg_compel", "compel", prompt=scenario.negative_prompt),
        make_node(
            "noise", "noise",
            seed=42,
            width=scenario.width,
            height=scenario.height,
        ),
        make_node(
            "i2l", "i2l",
            image={"image_name": image_name},
            fp32=False,
            tiled=False,
        ),
        make_node(
            "denoise", "denoise_latents",
            steps=scenario.steps,
            cfg_scale=scenario.cfg_scale,
            denoising_start=scenario.denoising_start,
            denoising_end=1.0,
            scheduler=scenario.scheduler,
        ),
        make_node("l2i", "l2i", fp32=False, tiled=False),
    ]
    edges = [
        make_edge("model_loader", "clip", "pos_compel", "clip"),
        make_edge("model_loader", "clip", "neg_compel", "clip"),
        make_edge("model_loader", "unet", "denoise", "unet"),
        make_edge("model_loader", "vae", "i2l", "vae"),
        make_edge("model_loader", "vae", "l2i", "vae"),
        make_edge("pos_compel", "conditioning", "denoise", "positive_conditioning"),
        make_edge("neg_compel", "conditioning", "denoise", "negative_conditioning"),
        make_edge("noise", "noise", "denoise", "noise"),
        make_edge("i2l", "latents", "denoise", "latents"),
        make_edge("denoise", "latents", "l2i", "latents"),
    ]
    return make_graph(nodes, edges)
