import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig


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

    context_relevance_source = "query_context_overlap"
    context_relevance = overlap_max
    context_relevance_mean = overlap_mean
    if scores:
        # Reranker scores are the strongest available signal for question-context relevance.
        context_relevance = _clamp01(max_score) or 0.0
        context_relevance_mean = _clamp01(avg_score) or 0.0
        context_relevance_source = "reranker_score"

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
        "context_relevance": context_relevance,
        "context_relevance_mean": context_relevance_mean,
        "context_relevance_source": context_relevance_source,
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


def _fallback_faithfulness(answer: str, contexts: List[str]) -> float:
    """Approximate groundedness when LLM-backed faithfulness cannot finish."""
    context_tokens = set(_tokenize(" ".join(str(ctx) for ctx in contexts)))
    answer_tokens = set(_tokenize(answer))
    if not context_tokens or not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def _limit_text(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars]


def _ragas_eval_view(sample: Dict[str, Any]) -> Dict[str, Any]:
    context_limit = int(os.getenv("RAGAS_CONTEXT_MAX_CHARS", "2500"))
    answer_limit = int(os.getenv("RAGAS_ANSWER_MAX_CHARS", "4000"))
    max_contexts = int(os.getenv("RAGAS_MAX_CONTEXTS", "5"))
    contexts = [
        _limit_text(ctx, context_limit)
        for ctx in (sample.get("contexts") or [])[:max_contexts]
    ]
    return {
        "question": sample["question"],
        "contexts": contexts,
        "answer": _limit_text(sample["answer"], answer_limit),
    }


def _load_main_config() -> dict:
    """从 config.json 加载主 Agent 的配置，并加载 .env 环境变量"""
    import json
    import os
    from pathlib import Path

    # 加载 .env 文件
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
        except Exception:
            pass

    # 加载 config.json
    config_path = Path(__file__).resolve().parent.parent.parent / "config.json"
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass

    # 根据 provider 获取对应的 API key（与 main.py 保持一致）
    provider = config.get("provider", "deepseek")
    api_key = config.get("api_key", "")

    if not api_key:
        provider_env_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
            "claude": "CLAUDE_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        env_var = provider_env_map.get(provider.lower())
        if env_var:
            api_key = os.getenv(env_var, "")

    config["api_key"] = api_key
    return config


def _build_ragas_llm():
    """使用 llm_factory 创建 Ragas 兼容的 InstructorLLM"""
    _load_main_config()
    model = os.getenv("RAGAS_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    api_key = os.getenv("RAGAS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("RAGAS_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")

    if not api_key:
        raise ValueError("RAGas OpenAI judge requires RAGAS_LLM_API_KEY or OPENAI_API_KEY")

    print(f"[RAGas] LLM - Provider: openai, Model: {model}, Base URL: {base_url or 'official'}")

    import httpx
    from openai import OpenAI
    from ragas.llms import llm_factory

    trust_env_proxy = os.getenv("RAGAS_TRUST_ENV_PROXY", "false").lower() in {"1", "true", "yes", "on"}
    timeout = float(os.getenv("RAGAS_LLM_TIMEOUT", "90"))
    client_kwargs = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": int(os.getenv("RAGAS_LLM_MAX_RETRIES", "1")),
        "http_client": httpx.Client(timeout=timeout, trust_env=trust_env_proxy),
    }
    if base_url:
        client_kwargs["base_url"] = base_url

    return llm_factory(
        model=model,
        provider="openai",
        client=OpenAI(**client_kwargs),
        temperature=0,
        max_tokens=int(os.getenv("RAGAS_LLM_MAX_TOKENS", "8192")),
    )

def _build_ragas_embeddings():
    _load_main_config()
    model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = (
        os.getenv("RAGAS_EMBEDDING_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        os.getenv("RAGAS_EMBEDDING_BASE_URL")
        or os.getenv("RAGAS_OPENAI_BASE_URL")
        or os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    )

    if not api_key:
        raise ValueError("RAGas embeddings require RAGAS_EMBEDDING_API_KEY, EMBEDDING_API_KEY, or OPENAI_API_KEY")

    print(f"[RAGas] Embeddings - Model: {model}, Base URL: {base_url or 'official'}")

    import httpx
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    trust_env_proxy = os.getenv("RAGAS_TRUST_ENV_PROXY", "false").lower() in {"1", "true", "yes", "on"}
    timeout = float(os.getenv("RAGAS_EMBEDDING_TIMEOUT", "90"))
    embeddings = OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        timeout=timeout,
        max_retries=int(os.getenv("RAGAS_EMBEDDING_MAX_RETRIES", "1")),
        http_client=httpx.Client(timeout=timeout, trust_env=trust_env_proxy),
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
    allow_retrieval_answer = os.getenv("RAGAS_ALLOW_RETRIEVAL_ANSWER", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if isinstance(rag_samples, list) and rag_samples:
        synthesis_samples = [
            sample
            for sample in rag_samples
            if isinstance(sample, dict)
            and str(sample.get("source") or "").startswith("rag_agent:synthesis")
        ]
        if synthesis_samples:
            rag_samples = synthesis_samples
        valid_samples = []
        for sample in rag_samples:
            if not isinstance(sample, dict):
                continue
            answer = sample.get("answer") or sample.get("llm_answer")
            if not answer and allow_retrieval_answer:
                answer = sample.get("retrieval_answer")
            if not answer:
                answer = meta.get("llm_answer")
            if not (sample.get("question") and sample.get("contexts") and answer):
                continue
            valid_samples.append(
                {
                    "question": sample.get("question"),
                    "contexts": sample.get("contexts") or [],
                    "answer": answer,
                    "answer_source": (
                        "sample.answer"
                        if sample.get("answer")
                        else "sample.llm_answer"
                        if sample.get("llm_answer")
                        else "metadata.llm_answer"
                        if meta.get("llm_answer")
                        else "sample.retrieval_answer"
                    ),
                    "retrieval_answer": sample.get("retrieval_answer"),
                    "eval_scope": sample.get("eval_scope", "rag_query"),
                    "source": sample.get("source"),
                    "context_scores": sample.get("context_scores") or [],
                    "context_sources": sample.get("context_sources") or [],
                }
            )
        if valid_samples:
            return valid_samples

    # Try to extract from trace with question + llm_answer + contexts from nodes
    nodes = _trace_nodes_by_name(trace_data)
    question = meta.get("question")
    answer = meta.get("llm_answer") or meta.get("answer")

    # Get contexts from metadata or from trace nodes
    contexts = meta.get("retrieved_contexts") or []
    if not contexts:
        # Try retrieved_context (string format) or Hybrid_Recall nodes
        retrieved_context_str = meta.get("retrieved_context")
        if retrieved_context_str and isinstance(retrieved_context_str, str):
            import re
            context_pattern = re.compile(r'\[[\d]+\] 来源:([^\[]+)', re.DOTALL)
            matches = context_pattern.findall(retrieved_context_str)
            if matches:
                contexts = [m.strip() for m in matches]
        # Fallback to Hybrid_Recall output_data
        if not contexts:
            recall_node = nodes.get("Hybrid_Recall", {})
            out_data = recall_node.get("output_data", recall_node.get("output_val"))
            if isinstance(out_data, list):
                contexts = [
                    str(item.get("doc", ""))
                    for item in out_data
                    if isinstance(item, dict) and item.get("doc")
                ]

    # Try to get question from Hybrid_Recall input if not in metadata
    if not question:
        question = nodes.get("Hybrid_Recall", {}).get("input_data", "")
        if question and not isinstance(question, str):
            question = str(question)

    # Try to get answer from Generate/LLM nodes if not in metadata
    if not answer:
        for node_name in ["Generate", "LLM_Response", "Chat"]:
            node = nodes.get(node_name, {})
            answer = node.get("output_data") or node.get("output_val") or ""
            if answer:
                break

    # Return sample if we have all three required fields
    if question and contexts and answer:
        return [
            {
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "answer_source": "metadata_or_node.llm_answer",
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
    faithfulness_metric = Faithfulness(llm=llm)
    answer_relevancy_metric = AnswerRelevancy(
        llm=llm,
        embeddings=embeddings,
        strictness=int(os.getenv("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1")),
    )
    report_df = None

    try:
        # 使用现代 Ragas API
        result = evaluate(
            dataset,
            metrics=[faithfulness_metric, answer_relevancy_metric],
            run_config=RunConfig(
                timeout=float(os.getenv("RAGAS_RUN_TIMEOUT", "90")),
                max_retries=int(os.getenv("RAGAS_RUN_MAX_RETRIES", "1")),
            ),
        )
        report_df = result.to_pandas()
    except Exception as exc:
        print(f"[WARN] Ragas full evaluation failed, falling back for relevancy: {exc}")
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness_metric],
                run_config=RunConfig(
                    timeout=float(os.getenv("RAGAS_RUN_TIMEOUT", "90")),
                    max_retries=int(os.getenv("RAGAS_RUN_MAX_RETRIES", "1")),
                ),
            )
            report_df = result.to_pandas()
        except Exception as faith_exc:
            print(f"[WARN] Faithfulness evaluation also failed: {faith_exc}")

    # 如果 Ragas 完全失败，使用空 DataFrame
    if report_df is None or len(report_df) == 0:
        report_df = pd.DataFrame([{} for _ in samples])

    print(f"[DEBUG] report_df columns: {list(report_df.columns)}")
    print(f"[DEBUG] report_df shape: {report_df.shape}")

    faithfulness_values = []
    faithfulness_sources = []
    for idx, sample in enumerate(samples):
        ragas_value = None
        if "faithfulness" in report_df.columns:
            ragas_value = _safe_float(report_df.at[idx, "faithfulness"])
        if ragas_value is None:
            faithfulness_values.append(
                _fallback_faithfulness(sample["answer"], sample["contexts"])
            )
            faithfulness_sources.append("local_fallback")
        else:
            faithfulness_values.append(ragas_value)
            faithfulness_sources.append("ragas")

    report_df["faithfulness"] = faithfulness_values
    report_df["faithfulness_source"] = faithfulness_sources

    report_df["question"] = [sample["question"] for sample in samples]
    report_df["contexts"] = [sample["contexts"] for sample in samples]
    report_df["answer"] = [sample["answer"] for sample in samples]
    report_df["answer_source"] = [sample.get("answer_source", "unknown") for sample in samples]
    report_df["eval_scope"] = [sample.get("eval_scope", "rag_query") for sample in samples]
    report_df["source"] = [sample.get("source") for sample in samples]
    report_df["context_scores"] = [sample.get("context_scores") or [] for sample in samples]
    report_df["context_sources"] = [sample.get("context_sources") or [] for sample in samples]

    # 处理 answer_relevancy
    relevancy_values = []
    relevancy_sources = []
    raw_relevancy_values = []
    for idx, sample in enumerate(samples):
        ragas_value = None
        if "answer_relevancy" in report_df.columns:
            ragas_value = _safe_float(report_df.at[idx, "answer_relevancy"])
        raw_relevancy_values.append(ragas_value)
        fallback_value = _fallback_relevancy(sample["question"], sample["answer"], sample["contexts"])
        if ragas_value is None:
            relevancy_values.append(fallback_value)
            relevancy_sources.append("local_fallback")
        elif ragas_value == 0.0 and fallback_value > float(os.getenv("RAGAS_ZERO_RELEVANCY_FALLBACK_THRESHOLD", "0.02")):
            relevancy_values.append(fallback_value)
            relevancy_sources.append("ragas_zero_local_fallback")
        else:
            relevancy_values.append(ragas_value)
            relevancy_sources.append("ragas")

    report_df["answer_relevancy_raw"] = raw_relevancy_values
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
    eval_samples = [_ragas_eval_view(sample) for sample in samples]
    df_tmp = pd.DataFrame(
        {
            "question": [sample["question"] for sample in eval_samples],
            "contexts": [sample["contexts"] for sample in eval_samples],
            "answer": [sample["answer"] for sample in eval_samples],
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
