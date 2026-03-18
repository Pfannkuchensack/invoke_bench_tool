"""Graph builders for CogView4 models."""

from __future__ import annotations

from invokeai_bench.config import ScenarioConfig
from invokeai_bench.graphs.common import make_edge, make_graph, make_model_id, make_node


def build_txt2img(model_info: dict, scenario: ScenarioConfig, submodels: dict | None = None) -> dict:
    nodes = [
        make_node("model_loader", "cogview4_model_loader", model=make_model_id(model_info)),
        make_node(
            "pos_encoder", "cogview4_text_encoder",
            prompt=scenario.positive_prompt,
        ),
        make_node(
            "neg_encoder", "cogview4_text_encoder",
            prompt=scenario.negative_prompt,
        ),
        make_node(
            "denoise", "cogview4_denoise",
            width=scenario.width,
            height=scenario.height,
            steps=scenario.steps,
            cfg_scale=scenario.cfg_scale,
            seed=42,
            denoising_start=0.0,
            denoising_end=1.0,
        ),
        make_node("l2i", "cogview4_l2i"),
    ]
    edges = [
        make_edge("model_loader", "glm_encoder", "pos_encoder", "glm_encoder"),
        make_edge("model_loader", "glm_encoder", "neg_encoder", "glm_encoder"),
        make_edge("model_loader", "transformer", "denoise", "transformer"),
        make_edge("pos_encoder", "conditioning", "denoise", "positive_conditioning"),
        make_edge("neg_encoder", "conditioning", "denoise", "negative_conditioning"),
        make_edge("model_loader", "vae", "l2i", "vae"),
        make_edge("denoise", "latents", "l2i", "latents"),
    ]
    return make_graph(nodes, edges)


def build_img2img(model_info: dict, scenario: ScenarioConfig, image_name: str, submodels: dict | None = None) -> dict:
    nodes = [
        make_node("model_loader", "cogview4_model_loader", model=make_model_id(model_info)),
        make_node(
            "pos_encoder", "cogview4_text_encoder",
            prompt=scenario.positive_prompt,
        ),
        make_node(
            "neg_encoder", "cogview4_text_encoder",
            prompt=scenario.negative_prompt,
        ),
        make_node(
            "i2l", "cogview4_i2l",
            image={"image_name": image_name},
        ),
        make_node(
            "denoise", "cogview4_denoise",
            width=scenario.width,
            height=scenario.height,
            steps=scenario.steps,
            cfg_scale=scenario.cfg_scale,
            seed=42,
            denoising_start=scenario.denoising_start,
            denoising_end=1.0,
        ),
        make_node("l2i", "cogview4_l2i"),
    ]
    edges = [
        make_edge("model_loader", "glm_encoder", "pos_encoder", "glm_encoder"),
        make_edge("model_loader", "glm_encoder", "neg_encoder", "glm_encoder"),
        make_edge("model_loader", "transformer", "denoise", "transformer"),
        make_edge("model_loader", "vae", "i2l", "vae"),
        make_edge("pos_encoder", "conditioning", "denoise", "positive_conditioning"),
        make_edge("neg_encoder", "conditioning", "denoise", "negative_conditioning"),
        make_edge("i2l", "latents", "denoise", "latents"),
        make_edge("model_loader", "vae", "l2i", "vae"),
        make_edge("denoise", "latents", "l2i", "latents"),
    ]
    return make_graph(nodes, edges)
