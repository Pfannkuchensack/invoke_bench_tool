"""Graph builders for FLUX.1 models."""

from __future__ import annotations

from invokeai_bench.config import ScenarioConfig
from invokeai_bench.graphs.common import make_edge, make_graph, make_model_id, make_node


def build_txt2img(model_info: dict, scenario: ScenarioConfig, submodels: dict | None = None) -> dict:
    """Build a txt2img graph for FLUX.1 models.

    submodels should contain: t5_encoder_model, clip_embed_model, vae_model
    """
    guidance = scenario.guidance if scenario.guidance is not None else 4.0
    sub = submodels or {}

    loader_kwargs = {"model": make_model_id(model_info)}
    if "t5_encoder_model" in sub:
        loader_kwargs["t5_encoder_model"] = sub["t5_encoder_model"]
    if "clip_embed_model" in sub:
        loader_kwargs["clip_embed_model"] = sub["clip_embed_model"]
    if "vae_model" in sub:
        loader_kwargs["vae_model"] = sub["vae_model"]

    nodes = [
        make_node("model_loader", "flux_model_loader", **loader_kwargs),
        make_node(
            "text_encoder", "flux_text_encoder",
            prompt=scenario.positive_prompt,
        ),
        make_node(
            "denoise", "flux_denoise",
            width=scenario.width,
            height=scenario.height,
            num_steps=scenario.steps,
            cfg_scale=scenario.cfg_scale,
            guidance=guidance,
            seed=42,
            denoising_start=0.0,
            denoising_end=1.0,
            scheduler=scenario.scheduler,
        ),
        make_node("vae_decode", "flux_vae_decode"),
    ]
    edges = [
        make_edge("model_loader", "clip", "text_encoder", "clip"),
        make_edge("model_loader", "t5_encoder", "text_encoder", "t5_encoder"),
        make_edge("model_loader", "max_seq_len", "text_encoder", "t5_max_seq_len"),
        make_edge("model_loader", "transformer", "denoise", "transformer"),
        make_edge("text_encoder", "conditioning", "denoise", "positive_text_conditioning"),
        make_edge("model_loader", "vae", "vae_decode", "vae"),
        make_edge("denoise", "latents", "vae_decode", "latents"),
    ]
    return make_graph(nodes, edges)


def build_img2img(model_info: dict, scenario: ScenarioConfig, image_name: str, submodels: dict | None = None) -> dict:
    guidance = scenario.guidance if scenario.guidance is not None else 4.0
    sub = submodels or {}

    loader_kwargs = {"model": make_model_id(model_info)}
    if "t5_encoder_model" in sub:
        loader_kwargs["t5_encoder_model"] = sub["t5_encoder_model"]
    if "clip_embed_model" in sub:
        loader_kwargs["clip_embed_model"] = sub["clip_embed_model"]
    if "vae_model" in sub:
        loader_kwargs["vae_model"] = sub["vae_model"]

    nodes = [
        make_node("model_loader", "flux_model_loader", **loader_kwargs),
        make_node(
            "text_encoder", "flux_text_encoder",
            prompt=scenario.positive_prompt,
        ),
        make_node(
            "i2l", "flux_vae_encode",
            image={"image_name": image_name},
        ),
        make_node(
            "denoise", "flux_denoise",
            width=scenario.width,
            height=scenario.height,
            num_steps=scenario.steps,
            cfg_scale=scenario.cfg_scale,
            guidance=guidance,
            seed=42,
            denoising_start=scenario.denoising_start,
            denoising_end=1.0,
            scheduler=scenario.scheduler,
            add_noise=True,
        ),
        make_node("vae_decode", "flux_vae_decode"),
    ]
    edges = [
        make_edge("model_loader", "clip", "text_encoder", "clip"),
        make_edge("model_loader", "t5_encoder", "text_encoder", "t5_encoder"),
        make_edge("model_loader", "max_seq_len", "text_encoder", "t5_max_seq_len"),
        make_edge("model_loader", "transformer", "denoise", "transformer"),
        make_edge("model_loader", "vae", "i2l", "vae"),
        make_edge("text_encoder", "conditioning", "denoise", "positive_text_conditioning"),
        make_edge("i2l", "latents", "denoise", "latents"),
        make_edge("model_loader", "vae", "vae_decode", "vae"),
        make_edge("denoise", "latents", "vae_decode", "latents"),
    ]
    return make_graph(nodes, edges)
