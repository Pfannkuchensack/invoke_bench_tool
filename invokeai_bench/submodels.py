"""Auto-discovery of auxiliary submodels (VAE, T5 Encoder, CLIP Embed, Qwen3 Encoder).

Different model architectures require different auxiliary models:
- FLUX.1: T5Encoder + CLIPEmbed + FLUX VAE (all required)
- FLUX2 Klein: Qwen3 source or standalone Qwen3Encoder + VAE (Diffusers models self-provide)
- Z-Image: Qwen3 source or standalone Qwen3Encoder + FLUX VAE (Diffusers models self-provide)
- SD3: all bundled in the model loader
- CogView4: all bundled in the model loader
- SD1/SD2/SDXL: all bundled in the model loader
"""

from __future__ import annotations

from typing import Any, Optional

from invokeai_bench.client import InvokeAIClient
from invokeai_bench.graphs.common import make_model_id


def _find_one(models: list[dict], **filters: Any) -> Optional[dict]:
    """Find the first model matching all filters."""
    for m in models:
        if all(m.get(k) == v for k, v in filters.items()):
            return m
    return None


def resolve_flux_submodels(client: InvokeAIClient, model_info: dict) -> dict:
    """Find T5Encoder, CLIPEmbed, and FLUX VAE for a FLUX.1 model.

    Returns dict with keys: t5_encoder_model, clip_embed_model, vae_model
    (each is a ModelIdentifierField dict or None).
    """
    # Find T5 Encoder
    t5_models = client.list_models(model_type="t5_encoder")
    t5 = t5_models[0] if t5_models else None

    # Find CLIP Embed
    clip_models = client.list_models(model_type="clip_embed")
    clip = clip_models[0] if clip_models else None

    # Find FLUX VAE
    vae_models = client.list_models(base="flux", model_type="vae")
    vae = vae_models[0] if vae_models else None

    result = {}
    if t5:
        result["t5_encoder_model"] = make_model_id(t5)
    if clip:
        result["clip_embed_model"] = make_model_id(clip)
    if vae:
        result["vae_model"] = make_model_id(vae)

    return result


def resolve_z_image_submodels(client: InvokeAIClient, model_info: dict) -> dict:
    """Find VAE and Qwen3 source for a Z-Image model.

    Strategy:
    1. If the model itself is Diffusers format, use it as qwen3_source_model
    2. Otherwise, find a Diffusers Z-Image model as qwen3_source_model
    3. Fallback: find separate FLUX VAE + Qwen3Encoder models
    """
    model_format = model_info.get("format", "")

    if model_format == "diffusers":
        # Diffusers model can provide its own VAE and Qwen3
        return {"qwen3_source_model": make_model_id(model_info)}

    # Non-diffusers model (GGUF, checkpoint) — need external sources
    # Try to find a Diffusers Z-Image model as source
    all_z_image = client.list_models(base="z-image", model_type="main")
    diffusers_source = _find_one(all_z_image, format="diffusers")

    if diffusers_source:
        return {"qwen3_source_model": make_model_id(diffusers_source)}

    # Fallback: find separate models
    result = {}
    vae_models = client.list_models(base="flux", model_type="vae")
    if vae_models:
        result["vae_model"] = make_model_id(vae_models[0])

    qwen3_models = client.list_models(model_type="qwen3_encoder")
    if qwen3_models:
        result["qwen3_encoder_model"] = make_model_id(qwen3_models[0])

    return result


def resolve_flux2_submodels(client: InvokeAIClient, model_info: dict) -> dict:
    """Find VAE and Qwen3 source for a FLUX2 Klein model.

    Strategy same as Z-Image but looks for Diffusers FLUX2 models.
    """
    model_format = model_info.get("format", "")

    if model_format == "diffusers":
        # Diffusers model can self-provide
        return {"qwen3_source_model": make_model_id(model_info)}

    # Non-diffusers — find a Diffusers FLUX2 source
    all_flux2 = client.list_models(base="flux2", model_type="main")
    diffusers_source = _find_one(all_flux2, format="diffusers")

    if diffusers_source:
        return {"qwen3_source_model": make_model_id(diffusers_source)}

    # Fallback: separate models
    result = {}
    # FLUX2 uses its own 32-channel VAE, try flux2 VAE first, then flux
    vae_models = client.list_models(base="flux2", model_type="vae")
    if not vae_models:
        vae_models = client.list_models(base="flux", model_type="vae")
    if vae_models:
        result["vae_model"] = make_model_id(vae_models[0])

    qwen3_models = client.list_models(model_type="qwen3_encoder")
    if qwen3_models:
        result["qwen3_encoder_model"] = make_model_id(qwen3_models[0])

    return result


# Map base type -> resolver function
SUBMODEL_RESOLVERS = {
    "flux": resolve_flux_submodels,
    "flux2": resolve_flux2_submodels,
    "z-image": resolve_z_image_submodels,
}


def resolve_submodels(client: InvokeAIClient, base: str, model_info: dict) -> dict:
    """Resolve auxiliary submodels for a given base model type.

    Returns a dict of submodel fields to add to the model loader node.
    Returns empty dict if no auxiliary models are needed (SD1/SDXL/SD3/CogView4).
    """
    resolver = SUBMODEL_RESOLVERS.get(base)
    if resolver is None:
        return {}
    return resolver(client, model_info)
