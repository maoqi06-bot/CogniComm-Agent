"""Utilities for merging standalone MCP RAG traces into a main trace.

The normal path is that MCP receives the active trace_id and appends directly to
the main trace file. This merge helper is only a conservative compatibility path
for older standalone MCP traces.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


RAG_METADATA_KEYS = ("retrieved_contexts", "rag_eval_samples", "retrieved_context")
MERGE_METADATA_KEYS = (
    "retrieved_contexts",
    "rag_eval_samples",
    "retrieved_context",
    "config_top_k",
    "config_threshold",
)


def _has_rag_payload(metadata: dict[str, Any]) -> bool:
    return any(metadata.get(key) for key in RAG_METADATA_KEYS)


def _text_tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    ascii_tokens = set(re.findall(r"[a-z0-9_]{2,}", text))
    cjk_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
    return ascii_tokens | cjk_chars


def _candidate_queries(trace_data: dict[str, Any]) -> Iterable[str]:
    metadata = trace_data.get("metadata") or {}
    if metadata.get("question"):
        yield str(metadata["question"])

    for sample in metadata.get("rag_eval_samples") or []:
        if isinstance(sample, dict) and sample.get("question"):
            yield str(sample["question"])

    for node in trace_data.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        method = str(node.get("method") or "")
        if "recall" in method.lower() and node.get("input_data"):
            yield str(node["input_data"])


def _candidate_matches_main(
    trace_id: str,
    main_question: str,
    candidate_data: dict[str, Any],
) -> bool:
    metadata = candidate_data.get("metadata") or {}
    explicit_parent = (
        metadata.get("parent_trace_id")
        or metadata.get("main_trace_id")
        or metadata.get("trace_id")
    )
    if explicit_parent and explicit_parent == trace_id:
        return True

    main_tokens = _text_tokens(main_question)
    if not main_tokens:
        return False

    for query in _candidate_queries(candidate_data):
        query_tokens = _text_tokens(query)
        if not query_tokens:
            continue
        overlap = main_tokens & query_tokens
        # Without an explicit trace link, only merge when the retrieval query
        # clearly belongs to the main user task.
        if len(overlap) >= 2 or (len(overlap) == 1 and len(query_tokens) <= 3):
            return True
    return False


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        print(f"Warning: failed to read trace {path.name}: {exc}")
        return None


def _merge_nodes(
    main_nodes: list[dict[str, Any]],
    other_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = set()
    merged_nodes = []
    for node_list in (main_nodes, other_nodes):
        for node in node_list:
            key = (
                node.get("method"),
                node.get("provider"),
                json.dumps(node.get("input_data"), ensure_ascii=False, sort_keys=True, default=str),
            )
            if key in seen:
                continue
            seen.add(key)
            merged_nodes.append(node)
    return merged_nodes


def merge_rag_trace_to_main(
    trace_id: str,
    main_trace_path: str | Path,
    *,
    expected_rag: bool = True,
    max_time_diff_seconds: float = 120,
) -> None:
    """Merge a matching standalone MCP RAG trace into the main trace.

    Args:
        trace_id: Active main trace id.
        main_trace_path: Path to the main trace file.
        expected_rag: Whether the task decomposition actually required RAG.
        max_time_diff_seconds: Maximum mtime distance for compatibility traces.
    """
    if not expected_rag:
        print("RAG trace merge skipped: task decomposition did not require RAG.")
        return

    trace_dir = Path("data/traces")
    main_path = Path(main_trace_path)
    if not main_path.exists():
        print(f"Warning: main trace does not exist: {main_path}")
        return

    main_data = _load_json(main_path)
    if main_data is None:
        return

    main_meta = main_data.setdefault("metadata", {})
    if _has_rag_payload(main_meta):
        print("Main trace already has RAG context; merge skipped.")
        return

    main_question = str(main_meta.get("question") or "")
    main_mtime = main_path.stat().st_mtime
    candidates: list[tuple[Path, float]] = []

    for path in trace_dir.glob("query_*.json"):
        if path.name == main_path.name:
            continue
        time_diff = abs(path.stat().st_mtime - main_mtime)
        if time_diff > max_time_diff_seconds:
            continue

        candidate_data = _load_json(path)
        if candidate_data is None:
            continue
        candidate_meta = candidate_data.get("metadata") or {}
        if "llm_answer" in candidate_meta or not _has_rag_payload(candidate_meta):
            continue
        if not _candidate_matches_main(trace_id, main_question, candidate_data):
            continue
        candidates.append((path, time_diff))

    if not candidates:
        print(f"No matching MCP RAG trace found for main trace: {main_path.name}")
        return

    candidates.sort(key=lambda item: item[1])
    candidate_path = candidates[0][0]
    other = _load_json(candidate_path)
    if other is None:
        return

    other_meta = other.get("metadata") or {}
    if not _has_rag_payload(other_meta):
        print(f"Warning: candidate trace has no RAG context: {candidate_path.name}")
        return

    main_data["nodes"] = _merge_nodes(main_data.get("nodes", []), other.get("nodes", []))
    for key in MERGE_METADATA_KEYS:
        if key in other_meta and key not in main_meta:
            main_meta[key] = other_meta[key]
    if "question" not in main_meta and "question" in other_meta:
        main_meta["question"] = other_meta["question"]

    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(main_data, f, ensure_ascii=False, indent=2)

    print(f"Merged MCP RAG trace: {candidate_path.name} -> {main_path.name}")
