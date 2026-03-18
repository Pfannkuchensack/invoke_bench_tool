"""Shared helpers for building InvokeAI graph payloads."""

from __future__ import annotations

import copy
import uuid
from typing import Any


def make_node(node_id: str, node_type: str, **fields: Any) -> dict:
    """Create a node dict for an InvokeAI graph."""
    return {"id": node_id, "type": node_type, **fields}


def make_edge(src_node: str, src_field: str, dst_node: str, dst_field: str) -> dict:
    """Create an edge dict connecting two nodes."""
    return {
        "source": {"node_id": src_node, "field": src_field},
        "destination": {"node_id": dst_node, "field": dst_field},
    }


def make_graph(nodes: list[dict], edges: list[dict]) -> dict:
    """Assemble nodes and edges into a Graph payload."""
    nodes_dict = {n["id"]: n for n in nodes}
    return {
        "id": str(uuid.uuid4()),
        "nodes": nodes_dict,
        "edges": edges,
    }


def make_model_id(model_info: dict) -> dict:
    """Build a ModelIdentifierField from an API model record."""
    return {
        "key": model_info["key"],
        "hash": model_info.get("hash", "unknown"),
        "name": model_info["name"],
        "base": model_info["base"],
        "type": model_info["type"],
        "submodel_type": None,
    }


def vary_seed(graph: dict, new_seed: int) -> dict:
    """Return a copy of the graph with the seed changed in noise/denoise nodes."""
    g = copy.deepcopy(graph)
    for node in g["nodes"].values():
        if "seed" in node:
            node["seed"] = new_seed
    return g
