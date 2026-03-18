"""Graph builders for FLUX.2 Klein models."""

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
        make_node("model_loader", "flux2_klein_model_loader", **_loader_kwargs(model_info, submodels)),
        make_node(
            "text_encoder", "flux2_klein_text_encoder",
            prompt=scenario.positive_prompt,
        ),
        make_node(
            "denoise", "flux2_denoise",
            width=scenario.width,
            height=scenario.height,
            num_steps=scenario.steps,
            cfg_scale=scenario.cfg_scale,
            seed=42,
            denoising_start=0.0,
            denoising_end=1.0,
            scheduler=scenario.scheduler,
        ),
        make_node("vae_decode", "flux2_vae_decode"),
    ]
    edges = [
        make_edge("model_loader", "qwen3_encoder", "text_encoder", "qwen3_encoder"),
        make_edge("model_loader", "max_seq_len", "text_encoder", "max_seq_len"),
        make_edge("model_loader", "transformer", "denoise", "transformer"),
        make_edge("model_loader", "vae", "denoise", "vae"),
        make_edge("text_encoder", "conditioning", "denoise", "positive_text_conditioning"),
        make_edge("model_loader", "vae", "vae_decode", "vae"),
        make_edge("denoise", "latents", "vae_decode", "latents"),
    ]
    return make_graph(nodes, edges)


def build_img2img(model_info: dict, scenario: ScenarioConfig, image_name: str, submodels: dict | None = None) -> dict:
    nodes = [
        make_node("model_loader", "flux2_klein_model_loader", **_loader_kwargs(model_info, submodels)),
        make_node(
            "text_encoder", "flux2_klein_text_encoder",
            prompt=scenario.positive_prompt,
        ),
        make_node(
            "i2l", "flux2_vae_encode",
            image={"image_name": image_name},
        ),
        make_node(
            "denoise", "flux2_denoise",
            width=scenario.width,
            height=scenario.height,
            num_steps=scenario.steps,
            cfg_scale=scenario.cfg_scale,
            seed=42,
            denoising_start=scenario.denoising_start,
            denoising_end=1.0,
            scheduler=scenario.scheduler,
            add_noise=True,
        ),
        make_node("vae_decode", "flux2_vae_decode"),
    ]
    edges = [
        make_edge("model_loader", "qwen3_encoder", "text_encoder", "qwen3_encoder"),
        make_edge("model_loader", "max_seq_len", "text_encoder", "max_seq_len"),
        make_edge("model_loader", "transformer", "denoise", "transformer"),
        make_edge("model_loader", "vae", "denoise", "vae"),
        make_edge("model_loader", "vae", "i2l", "vae"),
        make_edge("text_encoder", "conditioning", "denoise", "positive_text_conditioning"),
        make_edge("i2l", "latents", "denoise", "latents"),
        make_edge("model_loader", "vae", "vae_decode", "vae"),
        make_edge("denoise", "latents", "vae_decode", "latents"),
    ]
    return make_graph(nodes, edges)
