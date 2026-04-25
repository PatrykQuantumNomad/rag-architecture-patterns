"""Dataset manifest loader for the rag-architecture-patterns repo.

Reads JSON manifests under ``<dataset_root>/manifests/``. The loader is
deliberately decoupled from ``shared.config`` — it does NOT call
``get_settings()`` at construction time so that callers can introspect
dataset shape without needing a populated ``.env``.

Manifest files materialize across waves:

* ``papers.json``    — Plan 04 (corpus curation)
* ``figures.json``   — Plan 05 (figure extraction)
* ``videos.json``    — Plan 05 (CC-licensed video curation)
* ``metadata.json``  — Plan 06 (top-level dataset metadata)

When a manifest is absent the loader returns ``[]`` (lists) or ``{}``
(dicts), so consumers can treat "Plan 04 hasn't run yet" as a normal state.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any


class DatasetLoader:
    """Reads dataset manifests with graceful handling of missing files."""

    def __init__(self, dataset_root: Path | str | None = None) -> None:
        # Default to the canonical ``dataset/`` directory at the repo root.
        # We do NOT call get_settings() here — keeping loader usable without
        # a populated .env (e.g., for offline manifest inspection).
        self.dataset_root: Path = Path(dataset_root) if dataset_root else Path("dataset")

    # ---- public API ---------------------------------------------------------

    @cache  # noqa: B019 — instance-level cache; tests use fresh instances
    def papers(self) -> list[dict[str, Any]]:
        return self._read_list("papers.json")

    @cache  # noqa: B019
    def figures(self) -> list[dict[str, Any]]:
        return self._read_list("figures.json")

    @cache  # noqa: B019
    def videos(self) -> list[dict[str, Any]]:
        return self._read_list("videos.json")

    @cache  # noqa: B019
    def metadata(self) -> dict[str, Any]:
        path = self._manifest_path("metadata.json")
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a JSON object for metadata, got {type(data).__name__}"
            )
        return data

    # ---- internals ----------------------------------------------------------

    def _manifest_path(self, name: str) -> Path:
        return self.dataset_root / "manifests" / name

    def _read_list(self, name: str) -> list[dict[str, Any]]:
        path = self._manifest_path(name)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(
                f"{path}: expected a JSON array, got {type(data).__name__}"
            )
        return data
