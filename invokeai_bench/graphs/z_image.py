"""Graph builders for Z-Image models."""

from __future__ import annotations

from invokeai_bench.config import ScenarioConfig
from invokeai_bench.graphs.common import make_edge, make_graph, make_model_id, make_node


def _loader_kwargs(model_info: dict, submodels: dict | None) -> dict:
    sub = submodels or {}
    kwargs = {"model": make_model_id(model_info)}
    if "vae_model" in sub:
        kwargs["vae_model"] = sub["vae_model"]
    if "qwen3_encoder_model" in sub:
        kwargs["qwen3_encoder_model"] = sub["qwen3_encoder_model"]
    if "qwen3_source_model" in sub:
        kwargs["qwen3_source_model"] = sub["qwen3_source_model"]
    return kwargs


def build_txt2img(model_info: dict, scenario: ScenarioConfig, submodels: dict | None = None) -> dict:
    nodes = [
        make_node("model_loader", "z_image_model_loader", **_loader_kwargs(model_info, submodels)),
        make_node(
            "text_encoder", "z_image_text_encoder",
            prompt=scenario.positive_prompt,
        ),
        make_node(
            "denoise", "z_image_denoise",
            width=scenario.width,
            height=scenario.height,
            steps=scenario.steps,
            guidance_scale=scenario.cfg_scale,
            seed=42,
            denoising_start=0.0,
            denoising_end=1.0,
            scheduler=scenario.scheduler,
        ),
        make_node("l2i", "z_image_l2i"),
    ]
    edges = [
        make_edge("model_loader", "qwen3_encoder", "text_encoder", "qwen3_encoder"),
        make_edge("model_loader", "transformer", "denoise", "transformer"),
        make_edge("text_encoder", "conditioning", "denoise", "positive_conditioning"),
        make_edge("model_loader", "vae", "denoise", "vae"),
        make_edge("model_loader", "vae", "l2i", "vae"),
        make_edge("denoise", "latents", "l2i", "latents"),
    ]
    return make_graph(nodes, edges)


def build_img2img(model_info: dict, scenario: ScenarioConfig, image_name: str, submodels: dict | None = None) -> dict:
    nodes = [
        make_node("model_loader", "z_image_model_loader", **_loader_kwargs(model_info, submodels)),
        make_node(
            "text_encoder", "z_image_text_encoder",
            prompt=scenario.positive_prompt,
        ),
        make_node(
            "i2l", "z_image_i2l",
            image={"image_name": image_name},
        ),
        make_node(
            "denoise", "z_image_denoise",
            width=scenario.width,
            height=scenario.height,
            steps=scenario.steps,
            guidance_scale=scenario.cfg_scale,
            seed=42,
            denoising_start=scenario.denoising_start,
            denoising_end=1.0,
            scheduler=scenario.scheduler,
            add_noise=True,
        ),
        make_node("l2i", "z_image_l2i"),
    ]
    edges = [
        make_edge("model_loader", "qwen3_encoder", "text_encoder", "qwen3_encoder"),
        make_edge("model_loader", "transformer", "denoise", "transformer"),
        make_edge("model_loader", "vae", "i2l", "vae"),
        make_edge("model_loader", "vae", "denoise", "vae"),
        make_edge("text_encoder", "conditioning", "denoise", "positive_conditioning"),
        make_edge("i2l", "latents", "denoise", "latents"),
        make_edge("model_loader", "vae", "l2i", "vae"),
        make_edge("denoise", "latents", "l2i", "latents"),
    ]
    return make_graph(nodes, edges)
