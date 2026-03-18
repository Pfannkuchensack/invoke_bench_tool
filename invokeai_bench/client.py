"""HTTP API client for communicating with a running InvokeAI instance."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import httpx


class InvokeAIClient:
    """Thin wrapper around the InvokeAI REST API."""

    def __init__(self, base_url: str = "http://127.0.0.1:9090", token: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    # -- Auth ----------------------------------------------------------------

    def login(self, email: str, password: str) -> str:
        """Authenticate and store the JWT token. Returns the token."""
        resp = self._client.post("/api/v1/auth/login", json={"email": email, "password": password})
        resp.raise_for_status()
        token = resp.json()["access_token"]
        self._client.headers["Authorization"] = f"Bearer {token}"
        return token

    # -- App info ------------------------------------------------------------

    def get_version(self) -> str:
        resp = self._client.get("/api/v1/app/version")
        resp.raise_for_status()
        return resp.json().get("version", "unknown")

    # -- Models --------------------------------------------------------------

    def list_models(self, base: Optional[str] = None, model_type: str = "main") -> list[dict]:
        params: dict[str, Any] = {"model_type": model_type}
        if base:
            params["base_models"] = base
        resp = self._client.get("/api/v2/models/", params=params)
        resp.raise_for_status()
        return resp.json().get("models", [])

    def find_model(self, name: str, base: str) -> dict:
        """Find a model by name and base type.

        Tries exact match first, then case-insensitive, then substring match.
        Raises if not found.
        """
        models = self.list_models(base=base)
        available = [m["name"] for m in models]

        # 1. Exact match
        for m in models:
            if m["name"] == name:
                return m

        # 2. Case-insensitive match
        name_lower = name.lower()
        for m in models:
            if m["name"].lower() == name_lower:
                return m

        # 3. Substring match (name contained in model name or vice versa)
        candidates = [m for m in models if name_lower in m["name"].lower() or m["name"].lower() in name_lower]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Model '{name}' with base '{base}' is ambiguous. "
                f"Matches: {[c['name'] for c in candidates]}. Use the exact name."
            )

        raise ValueError(f"Model '{name}' with base '{base}' not found. Available: {available}")

    def find_model_by_key(self, key: str) -> dict:
        """Find a model by its unique key."""
        resp = self._client.get(f"/api/v2/models/i/{key}")
        resp.raise_for_status()
        return resp.json()

    # -- Cache stats ---------------------------------------------------------

    def get_cache_stats(self) -> dict:
        resp = self._client.get("/api/v2/models/stats")
        resp.raise_for_status()
        return resp.json()

    # -- Image upload --------------------------------------------------------

    def upload_image(self, path: Path | str) -> dict:
        """Upload an image file and return the ImageDTO."""
        path = Path(path)
        with open(path, "rb") as f:
            resp = self._client.post(
                "/api/v1/images/upload",
                files={"file": (path.name, f, "image/png")},
                params={"image_category": "user", "is_intermediate": "true"},
            )
        resp.raise_for_status()
        return resp.json()

    # -- Queue ---------------------------------------------------------------

    def get_queue_status(self) -> dict:
        resp = self._client.get("/api/v1/queue/default/status")
        resp.raise_for_status()
        return resp.json()

    def enqueue_batch(self, graph: dict, runs: int = 1) -> dict:
        """Enqueue a batch with the given graph. Returns EnqueueBatchResult."""
        payload = {
            "batch": {
                "graph": graph,
                "runs": runs,
            },
        }
        resp = self._client.post("/api/v1/queue/default/enqueue_batch", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_batch_status(self, batch_id: str) -> dict:
        resp = self._client.get(f"/api/v1/queue/default/b/{batch_id}/status")
        resp.raise_for_status()
        return resp.json()

    def get_queue_item(self, item_id: int) -> dict:
        resp = self._client.get(f"/api/v1/queue/default/i/{item_id}")
        resp.raise_for_status()
        return resp.json()

    def wait_for_batch(
        self,
        batch_id: str,
        item_ids: list[int],
        poll_interval: float = 1.0,
        timeout: float = 600.0,
    ) -> list[dict]:
        """Poll until all items in the batch are done. Returns list of queue items."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_batch_status(batch_id)
            pending = status.get("pending", 0) + status.get("in_progress", 0)
            if pending == 0:
                break
            time.sleep(poll_interval)
        else:
            raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

        return [self.get_queue_item(item_id) for item_id in item_ids]
