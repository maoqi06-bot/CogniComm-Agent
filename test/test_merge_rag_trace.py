import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dm_agent.multi_agent._merge_rag_trace import merge_rag_trace_to_main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_non_rag_task_does_not_merge_nearby_rag_trace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace_dir = tmp_path / "data" / "traces"
    main_path = trace_dir / "query_main.json"
    other_path = trace_dir / "query_other.json"

    _write_json(
        main_path,
        {
            "trace_id": "query_main",
            "nodes": [],
            "metadata": {
                "question": "写一个hello world的python代码在task目录中",
                "llm_answer": "created task/hello.py",
            },
        },
    )
    _write_json(
        other_path,
        {
            "trace_id": "query_other",
            "nodes": [{"method": "Hybrid_Recall", "provider": "BM25 + FAISS", "input_data": "ISAC definition"}],
            "metadata": {
                "retrieved_contexts": ["ISAC context"],
                "rag_eval_samples": [{"question": "ISAC Integrated Sensing and Communication definition"}],
            },
        },
    )

    now = 1_700_000_000
    os.utime(main_path, (now, now))
    os.utime(other_path, (now + 1, now + 1))

    merge_rag_trace_to_main("query_main", main_path, expected_rag=False)

    merged = json.loads(main_path.read_text(encoding="utf-8"))
    assert merged["nodes"] == []
    assert "retrieved_contexts" not in merged["metadata"]
    assert "rag_eval_samples" not in merged["metadata"]


def test_expected_rag_still_requires_matching_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace_dir = tmp_path / "data" / "traces"
    main_path = trace_dir / "query_main.json"
    other_path = trace_dir / "query_other.json"

    _write_json(
        main_path,
        {
            "trace_id": "query_main",
            "nodes": [],
            "metadata": {
                "question": "写一个hello world的python代码在task目录中",
                "llm_answer": "created task/hello.py",
            },
        },
    )
    _write_json(
        other_path,
        {
            "trace_id": "query_other",
            "nodes": [{"method": "Hybrid_Recall", "provider": "BM25 + FAISS", "input_data": "ISAC definition"}],
            "metadata": {
                "retrieved_contexts": ["ISAC context"],
                "rag_eval_samples": [{"question": "ISAC Integrated Sensing and Communication definition"}],
            },
        },
    )

    now = 1_700_000_000
    os.utime(main_path, (now, now))
    os.utime(other_path, (now + 1, now + 1))

    merge_rag_trace_to_main("query_main", main_path, expected_rag=True)

    merged = json.loads(main_path.read_text(encoding="utf-8"))
    assert merged["nodes"] == []
    assert "retrieved_contexts" not in merged["metadata"]
    assert "rag_eval_samples" not in merged["metadata"]
