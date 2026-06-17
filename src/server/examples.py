"""例句库读取。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT


EXAMPLES_JSON = PROJECT_ROOT / "frontend/src/data/practice-examples.json"
PUBLIC_DIR = PROJECT_ROOT / "frontend/public"


@lru_cache(maxsize=1)
def load_examples() -> list[dict[str, Any]]:
    """读取前端共用的例句 JSON，避免前后端维护两份材料。"""
    if not EXAMPLES_JSON.exists():
        return []
    data = json.loads(EXAMPLES_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def find_example(example_id: str) -> dict[str, Any] | None:
    for item in load_examples():
        if item.get("id") == example_id:
            return item
    return None


def example_audio_path(example_id: str) -> Path | None:
    item = find_example(example_id)
    if not item:
        return None

    audio_path = str(item.get("audioPath") or "").lstrip("/")
    if not audio_path:
        return None

    path = (PUBLIC_DIR / audio_path).resolve()
    public_root = PUBLIC_DIR.resolve()
    if public_root not in path.parents and path != public_root:
        return None
    return path if path.exists() else None

