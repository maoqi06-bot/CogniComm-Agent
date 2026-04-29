"""Utilities for inspecting and administering persisted memory files."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RESET_CONFIRMATION_TEXT = "RESET MEMORY"


def load_memory_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load long-term memory metadata, returning an empty shape on failure."""

    if not metadata_path.exists():
        return {"version": "1.0", "updated_at": None, "total_memories": 0, "memories": []}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("memory metadata root must be an object")
        memories = data.get("memories", [])
        if not isinstance(memories, list):
            memories = []
        return {
            "version": data.get("version", "1.0"),
            "updated_at": data.get("updated_at"),
            "total_memories": len(memories),
            "memories": memories,
        }
    except Exception:
        return {"version": "1.0", "updated_at": None, "total_memories": 0, "memories": []}


def filter_memory_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    category: Optional[str] = None,
    source: Optional[str] = None,
    tag: Optional[str] = None,
    query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter memory metadata rows for dashboard display."""

    needle = str(query or "").strip().lower()
    selected_tag = str(tag or "").strip()
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if category and category != "ALL" and row.get("category") != category:
            continue
        if source and source != "ALL" and row.get("source") != source:
            continue
        tags = [str(item) for item in row.get("tags", []) if str(item).strip()]
        if selected_tag and selected_tag != "ALL" and selected_tag not in tags:
            continue
        if needle:
            haystack = " ".join(
                [
                    str(row.get("id", "")),
                    str(row.get("content", "")),
                    str(row.get("category", "")),
                    str(row.get("source", "")),
                    " ".join(tags),
                ]
            ).lower()
            if needle not in haystack:
                continue
        filtered.append(row)
    return filtered


def is_reset_confirmed(text: str) -> bool:
    return str(text or "").strip() == RESET_CONFIRMATION_TEXT


def quarantine_memory_files(
    root: Path,
    *,
    timestamp: Optional[str] = None,
) -> Path:
    """Move current memory artifacts into a timestamped quarantine directory."""

    root = Path(root).resolve()
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_dir = root / "data" / "memory_quarantine" / stamp
    long_term_dir = root / "dm_agent" / "data" / "memory"
    timeline_path = root / "data" / "multi_agent_memory_timeline.jsonl"
    approvals_path = root / "data" / "multi_agent_memory_approvals.json"

    quarantine_memory_dir = quarantine_dir / "dm_agent" / "data" / "memory"
    quarantine_data_dir = quarantine_dir / "data"
    quarantine_memory_dir.mkdir(parents=True, exist_ok=True)
    quarantine_data_dir.mkdir(parents=True, exist_ok=True)

    if long_term_dir.exists():
        for child in list(long_term_dir.iterdir()):
            shutil.move(str(child), str(quarantine_memory_dir / child.name))
    long_term_dir.mkdir(parents=True, exist_ok=True)

    for source_path in [timeline_path, approvals_path]:
        if source_path.exists():
            shutil.move(str(source_path), str(quarantine_data_dir / source_path.name))

    return quarantine_dir


def delete_long_term_memory(memory_id: str, *, storage_path: str = "./dm_agent/data/memory") -> bool:
    from .memory_manager import MemoryManager

    manager = MemoryManager(storage_path=storage_path)
    return bool(manager.memory_store.delete(memory_id))


def delete_long_term_memory_category(
    category: str,
    *,
    storage_path: str = "./dm_agent/data/memory",
) -> int:
    from .long_term_memory import MemoryCategory
    from .memory_manager import MemoryManager

    manager = MemoryManager(storage_path=storage_path)
    return int(manager.memory_store.delete_by_category(MemoryCategory(category)))
