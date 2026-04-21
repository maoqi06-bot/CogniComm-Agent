import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections import answer_relevancy, faithfulness


BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRACE_DIR = BASE_DIR / "data" / "traces"
OUTPUT_FILE = BASE_DIR / "data" / "ragas_report.json"
STATUS_FILE = BASE_DIR / "data" / "ragas_eval_status.json"


def _safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(number):
        return fallback
    return number


def _tokenize(text: Any) -> List[str]:
    """Tokenize English words/numbers and individual CJK chars for stable local metrics."""
    if text is None:
        return []
    raw = str(text).lower()
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", raw)


def _token_overlap(left: Any, right: Any) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    # Contexts are often long paper chunks, so Jaccard would be diluted by chunk length.
    # For retrieval diagnostics we care more about how much of the query is covered.
    return len(left_tokens & right_tokens) / len(left_tokens)


def _clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def _generation_mode(answer: str) -> str:
    raw_markers = ("来源:", "Metadata | File", "[Metadata", "检索到以下内容", "retrieved")
    stripped = str(answer or "").strip()
    if any(marker in stripped for marker in raw_markers):
        return "raw_retrieval_dump"
    return "generated_answer"


def _retrieval_metrics(sample: Dict[str, Any]) -> Dict[str, Any]:
    question = sample.get("question", "")
    contexts = [str(ctx) for ctx in sample.get("contexts") or []]
    raw_scores = sample.get("context_scores") or []
    scores = [
        _safe_float(score)
        for score in raw_scores
        if _safe_float(score) is not None
    ]
    sources = [
        str(source)
        for source in (sample.get("context_sources") or [])
        if source is not None and str(source).strip()
    ]

    overlaps = [_token_overlap(question, ctx) for ctx in contexts]
    overlap_max = max(overlaps) if overlaps else 0.0
    overlap_mean = sum(overlaps) / len(overlaps) if overlaps else 0.0

    avg_score = sum(scores) / len(scores) if scores else None
    max_score = max(scores) if scores else None
    min_score = min(scores) if scores else None

    # Reranker scores in this project are usually already in [0, 1]. Clamp when present.
    score_component = _clamp01(max_score)
    if score_component is None:
        retrieval_quality = overlap_max
        retrieval_quality_source = "query_context_overlap"
    else:
        retrieval_quality = 0.7 * score_component + 0.3 * overlap_max
        retrieval_quality_source = "reranker_score_and_overlap"

    return {
        "context_count": len(contexts),
        "avg_context_score": avg_score,
        "max_context_score": max_score,
        "min_context_score": min_score,
        "source_count": len(set(sources)),
        "context_query_overlap_max": overlap_max,
        "context_query_overlap_mean": overlap_mean,
        "retrieval_quality": _clamp01(retrieval_quality) or 0.0,
        "retrieval_quality_source": retrieval_quality_source,
        "weak_retrieval": (_clamp01(retrieval_quality) or 0.0) < 0.25,
        "generation_mode": _generation_mode(sample.get("answer", "")),
    }


def _fallback_relevancy(question: str, answer: str, contexts: List[str]) -> float:
    """Use a deterministic lexical score when embedding-backed Ragas fails."""
    source_tokens = set(_tokenize(f"{question} {' '.join(str(ctx) for ctx in contexts)}"))
    answer_tokens = set(_tokenize(answer))
    if not source_tokens or not answer_tokens:
        return 0.0
    return len(source_tokens & answer_tokens) / len(source_tokens | answer_tokens)


def _build_ragas_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("RAGAS_LLM_MODEL", "gpt-3.5-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("RAGAS_OPENAI_BASE_URL", "https://sg.uiuiapi.com/v1/"),
        temperature=0,
    )


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    embeddings = OpenAIEmbeddings(
        model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("RAGAS_OPENAI_BASE_URL", "https://sg.uiuiapi.com/v1/"),
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _trace_nodes_by_name(trace_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    nodes = {}
    for node in trace_data.get("nodes", []):
        if isinstance(node, dict):
            key = node.get("method") or node.get("name")
            if key:
                nodes[key] = node
    return nodes


def _extract_samples(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = trace_data.get("metadata", {})
    rag_samples = meta.get("rag_eval_samples") or []
    if isinstance(rag_samples, list) and rag_samples:
        return [
            {
                "question": sample.get("question"),
                "contexts": sample.get("contexts") or [],
                "answer": sample.get("answer"),
                "eval_scope": sample.get("eval_scope", "rag_query"),
                "source": sample.get("source"),
                "context_scores": sample.get("context_scores") or [],
                "context_sources": sample.get("context_sources") or [],
            }
            for sample in rag_samples
            if isinstance(sample, dict)
            and sample.get("question")
            and sample.get("contexts")
            and sample.get("answer")
        ]

    if os.getenv("RAGAS_EVAL_SCOPE", "rag").lower() not in {"task", "agent", "end_to_end"}:
        return []

    nodes = _trace_nodes_by_name(trace_data)

    question = meta.get("question") or nodes.get("Hybrid_Recall", {}).get("input_data", "")
    contexts = meta.get("retrieved_contexts") or []

    recall_node = nodes.get("Hybrid_Recall", {})
    out_data = recall_node.get("output_data", recall_node.get("output_val"))
    if not contexts and isinstance(out_data, list):
        contexts = [
            str(item.get("doc", ""))
            for item in out_data
            if isinstance(item, dict) and item.get("doc")
        ]

    answer = meta.get("llm_answer", "")
    if question and contexts and answer:
        return [
            {
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "eval_scope": "task_end_to_end",
                "source": "trace_fallback",
                "context_scores": [],
                "context_sources": [],
            }
        ]
    return []


def _evaluate_dataset(dataset: Dataset, samples: List[Dict[str, Any]]) -> pd.DataFrame:
    llm = _build_ragas_llm()
    embeddings = _build_ragas_embeddings()

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )
        report_df = result.to_pandas()
    except Exception as exc:
        print(f"[WARN] Ragas full evaluation failed, falling back for relevancy: {exc}")
        try:
            result = evaluate(dataset, metrics=[faithfulness], llm=llm)
            report_df = result.to_pandas()
        except Exception as faith_exc:
            print(f"[WARN] Faithfulness evaluation also failed: {faith_exc}")
            report_df = pd.DataFrame([{} for _ in samples])

    report_df["question"] = [sample["question"] for sample in samples]
    report_df["contexts"] = [sample["contexts"] for sample in samples]
    report_df["answer"] = [sample["answer"] for sample in samples]
    report_df["eval_scope"] = [sample.get("eval_scope", "rag_query") for sample in samples]
    report_df["source"] = [sample.get("source") for sample in samples]
    report_df["context_scores"] = [sample.get("context_scores") or [] for sample in samples]
    report_df["context_sources"] = [sample.get("context_sources") or [] for sample in samples]

    relevancy_values = []
    relevancy_sources = []
    for idx, sample in enumerate(samples):
        ragas_value = None
        if "answer_relevancy" in report_df.columns:
            ragas_value = _safe_float(report_df.at[idx, "answer_relevancy"])
        if ragas_value is None:
            relevancy_values.append(
                _fallback_relevancy(sample["question"], sample["answer"], sample["contexts"])
            )
            relevancy_sources.append("local_fallback")
        else:
            relevancy_values.append(ragas_value)
            relevancy_sources.append("ragas")

    report_df["answer_relevancy"] = relevancy_values
    report_df["answer_relevancy_source"] = relevancy_sources

    retrieval_rows = [_retrieval_metrics(sample) for sample in samples]
    for key in retrieval_rows[0].keys():
        report_df[key] = [row[key] for row in retrieval_rows]
    return report_df


def run_automated_evaluation() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _write_status("running")

    if not TRACE_DIR.exists():
        print(f"[ERROR] Trace directory not found: {TRACE_DIR}")
        _write_status("failed", message=f"Trace directory not found: {TRACE_DIR}")
        return

    samples: List[Dict[str, Any]] = []
    trace_files = sorted(TRACE_DIR.glob("query_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"[INFO] Scanning traces from {TRACE_DIR} ...")

    for trace_file in trace_files[:10]:
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
            trace_samples = _extract_samples(trace_data)
            if trace_samples:
                samples.extend(trace_samples)
                print(f"[OK] Loaded {len(trace_samples)} RAG samples from trace: {trace_file.name}")
            else:
                print(f"[SKIP] Trace lacks RAG query samples: {trace_file.name}")
        except Exception as exc:
            print(f"[ERROR] Failed to parse {trace_file.name}: {exc}")

    if not samples:
        print("[ERROR] No valid RAG samples found. Make sure trace metadata has rag_eval_samples.")
        _write_status("failed", message="No valid RAG samples found.")
        return

    print(f"[INFO] Evaluating {len(samples)} samples ...")
    df_tmp = pd.DataFrame(
        {
            "question": [sample["question"] for sample in samples],
            "contexts": [sample["contexts"] for sample in samples],
            "answer": [sample["answer"] for sample in samples],
        }
    )
    dataset = Dataset.from_dict(df_tmp.to_dict("list"))
    report_df = _evaluate_dataset(dataset, samples)

    report_df.to_json(OUTPUT_FILE, orient="records", force_ascii=False, indent=2)
    print(f"[OK] Ragas report generated: {OUTPUT_FILE}")
    _write_status("finished", samples=len(samples), report_path=str(OUTPUT_FILE))


def _write_status(status: str, **extra: Any) -> None:
    payload = {
        "status": status,
        "pid": os.getpid(),
        "updated_at": pd.Timestamp.now().isoformat(),
        **extra,
    }
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_automated_evaluation()
