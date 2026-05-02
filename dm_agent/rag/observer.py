import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

from dm_agent.rag.evaluator import (
    _build_ragas_embeddings,
    _build_ragas_llm,
    _ragas_eval_view,
    _retrieval_metrics,
)


class RagasObserver:
    @staticmethod
    def _fallback_relevancy(question: str, answer: str, contexts: Iterable[str]) -> float:
        """A cheap local fallback when the embedding model is unavailable."""
        text = f"{question} {' '.join(str(ctx) for ctx in contexts)}"
        q_tokens = {token for token in text.lower().replace("\n", " ").split() if len(token) > 1}
        a_tokens = {token for token in answer.lower().replace("\n", " ").split() if len(token) > 1}
        if not q_tokens or not a_tokens:
            return 0.0
        return len(q_tokens & a_tokens) / len(q_tokens | a_tokens)

    @staticmethod
    def _safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        if math.isnan(number):
            return fallback
        return number

    @staticmethod
    def _build_llm():
        return _build_ragas_llm()

    @staticmethod
    def _build_embeddings():
        return _build_ragas_embeddings()

    @staticmethod
    def _load_report(report_path: Path) -> List[Dict[str, Any]]:
        if not report_path.exists():
            return []
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    @staticmethod
    def _evaluate_one(sample: Dict[str, Any], timestamp: Any = None) -> Dict[str, Any]:
        question = str(sample.get("question") or "")
        contexts = [str(ctx) for ctx in (sample.get("contexts") or []) if str(ctx).strip()]
        answer = str(sample.get("answer") or "")

        score_dict: Dict[str, Any] = {}
        if question and contexts and answer:
            eval_sample = _ragas_eval_view(
                {"question": question, "contexts": contexts, "answer": answer}
            )
            dataset = Dataset.from_dict(
                {
                    "question": [eval_sample["question"]],
                    "contexts": [eval_sample["contexts"]],
                    "answer": [eval_sample["answer"]],
                }
            )
            llm = RagasObserver._build_llm()
            embeddings = RagasObserver._build_embeddings()
            faithfulness_metric = Faithfulness(llm=llm)
            answer_relevancy_metric = AnswerRelevancy(
                llm=llm,
                embeddings=embeddings,
                strictness=int(os.getenv("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1")),
            )
            try:
                results = evaluate(
                    dataset,
                    metrics=[faithfulness_metric, answer_relevancy_metric],
                    run_config=RunConfig(
                        timeout=float(os.getenv("RAGAS_RUN_TIMEOUT", "90")),
                        max_retries=int(os.getenv("RAGAS_RUN_MAX_RETRIES", "1")),
                    ),
                )
                score_dict = results.to_pandas().iloc[0].to_dict()
            except Exception as exc:
                print(f"[WARN] Ragas full evaluation failed, using relevancy fallback: {exc}")
                try:
                    faith_result = evaluate(
                        dataset,
                        metrics=[faithfulness_metric],
                        run_config=RunConfig(
                            timeout=float(os.getenv("RAGAS_RUN_TIMEOUT", "90")),
                            max_retries=int(os.getenv("RAGAS_RUN_MAX_RETRIES", "1")),
                        ),
                    )
                    score_dict = faith_result.to_pandas().iloc[0].to_dict()
                except Exception as faith_exc:
                    print(f"[WARN] Faithfulness evaluation also failed: {faith_exc}")

        ragas_relevancy = RagasObserver._safe_float(score_dict.get("answer_relevancy"))
        ragas_faithfulness = RagasObserver._safe_float(score_dict.get("faithfulness"))
        fallback_relevancy = RagasObserver._fallback_relevancy(question, answer, contexts)
        if ragas_relevancy is None:
            relevancy = fallback_relevancy
            relevancy_source = "local_fallback"
        elif ragas_relevancy == 0.0 and fallback_relevancy > float(os.getenv("RAGAS_ZERO_RELEVANCY_FALLBACK_THRESHOLD", "0.02")):
            relevancy = fallback_relevancy
            relevancy_source = "ragas_zero_local_fallback"
        else:
            relevancy = ragas_relevancy
            relevancy_source = "ragas"

        entry = {
            "question": question,
            "answer": answer,
            "answer_source": sample.get("answer_source", "unknown"),
            "contexts": contexts,
            "faithfulness": ragas_faithfulness,
            "faithfulness_source": "ragas" if ragas_faithfulness is not None else "unavailable",
            "answer_relevancy_raw": ragas_relevancy,
            "answer_relevancy": relevancy,
            "answer_relevancy_source": relevancy_source,
            "eval_scope": sample.get("eval_scope", "rag_query"),
            "source": sample.get("source"),
            "context_scores": sample.get("context_scores") or [],
            "context_sources": sample.get("context_sources") or [],
            "timestamp": timestamp,
        }
        entry.update(_retrieval_metrics({**sample, "question": question, "contexts": contexts, "answer": answer}))
        return entry

    @staticmethod
    def _samples_from_trace(
        trace_data: Dict[str, Any],
        fallback_question: Optional[str] = None,
        fallback_answer: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        metadata = trace_data.get("metadata", {})
        samples = metadata.get("rag_eval_samples") or []
        if isinstance(samples, list) and samples:
            synthesis_samples = [
                sample
                for sample in samples
                if isinstance(sample, dict)
                and str(sample.get("source") or "").startswith("rag_agent:synthesis")
            ]
            if synthesis_samples:
                samples = synthesis_samples
            allow_retrieval_answer = os.getenv("RAGAS_ALLOW_RETRIEVAL_ANSWER", "false").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            normalized_samples = []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                ragas_answer = sample.get("answer") or sample.get("llm_answer")
                answer_source = "sample.answer" if sample.get("answer") else "sample.llm_answer"
                if not ragas_answer and allow_retrieval_answer:
                    ragas_answer = sample.get("retrieval_answer")
                    answer_source = "sample.retrieval_answer"
                if not ragas_answer:
                    ragas_answer = metadata.get("llm_answer")
                    answer_source = "metadata.llm_answer"
                if not ragas_answer:
                    continue
                normalized_samples.append(
                    {
                        **sample,
                        "answer": ragas_answer,
                        "answer_source": answer_source,
                        "retrieval_answer": sample.get("retrieval_answer"),
                    }
                )
            return normalized_samples

        # Default policy: Ragas is for RAG-query evaluation, not whole-agent scoring.
        # Whole task evaluation is allowed only when explicitly requested.
        if os.getenv("RAGAS_EVAL_SCOPE", "rag").lower() in {"task", "agent", "end_to_end"}:
            contexts = metadata.get("retrieved_contexts") or []
            question = fallback_question or metadata.get("question")
            answer = fallback_answer or metadata.get("llm_answer")
            if question and contexts and answer:
                return [
                    {
                        "question": question,
                        "contexts": contexts,
                        "answer": answer,
                        "eval_scope": "task_end_to_end",
                        "source": "trace_fallback",
                    }
                ]
        return []

    @staticmethod
    def instant_eval(
        trace_id: str,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        report_path: str = "data/ragas_report.json",
    ) -> List[Dict[str, Any]]:
        """Evaluate RAG query samples collected in a trace and append them to the report."""
        trace_file = Path(f"data/traces/{trace_id}.json")
        if not trace_file.exists():
            return []

        with open(trace_file, "r", encoding="utf-8") as f:
            trace_data = json.load(f)

        samples = RagasObserver._samples_from_trace(trace_data, question, answer)
        if not samples:
            print("[INFO] No RAG query samples found for Ragas evaluation.")
            return []

        timestamp = trace_data.get("timestamp")
        new_entries = [
            RagasObserver._evaluate_one(sample, timestamp=timestamp)
            for sample in samples
        ]

        report_p = Path(report_path)
        existing_data = RagasObserver._load_report(report_p)
        existing_data.extend(new_entries)
        report_p.parent.mkdir(parents=True, exist_ok=True)
        with open(report_p, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        avg_faith = [
            entry["faithfulness"]
            for entry in new_entries
            if entry.get("faithfulness") is not None
        ]
        avg_rel = [entry["answer_relevancy"] for entry in new_entries]
        faith_text = f"{sum(avg_faith) / len(avg_faith):.2%}" if avg_faith else "N/A"
        rel_text = f"{sum(avg_rel) / len(avg_rel):.2%}" if avg_rel else "N/A"
        print(
            f"\n[量化反馈] RAG samples: {len(new_entries)}, "
            f"Faithfulness: {faith_text}, Relevancy: {rel_text}"
        )
        return new_entries
