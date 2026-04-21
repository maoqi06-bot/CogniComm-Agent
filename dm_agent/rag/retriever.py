"""高级检索器封装，提供混合检索（BM25 + Dense）、RRF 融合与跨编码器精排功能。"""

import os
import sys
import threading
import time
import numpy as np
from typing import List, Optional, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .vector_store import FAISSVectorStore
from .models import SearchResult, DocumentChunk


def _safe_print(message: str) -> None:
    """Print without crashing on legacy Windows console encodings."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe + "\n")
        sys.stdout.flush()


class CrossEncoderReranker:
    """基于 Cross-Encoder 的精排模型（Reranker）。

    用于对粗排召回的结果进行深度语义打分，大幅提升 Top-K 准确度。
    """

    def __init__(self, model_name: str = 'BAAI/bge-reranker-base'):
        _safe_print(f"[INFO] Loading Reranker model: {model_name} ...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 3) -> List[SearchResult]:
        if not results:
            return []

        # 构造输入对：(Query, Document Content)
        pairs = [[query, res.chunk.content] for res in results]

        # 预测语义匹配分数
        scores = self.model.predict(pairs)

        # 将分数写回并重新排序
        for res, score in zip(results, scores):
            res.score = float(score)  # 覆盖为精排分数

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class LazyReranker:
    """Background-loading proxy for CrossEncoderReranker."""

    def __init__(self, model_name: str, wait_seconds: Optional[float] = None):
        self.model_name = model_name
        self.wait_seconds = (
            wait_seconds
            if wait_seconds is not None
            else float(os.getenv("RAG_RERANKER_WAIT_SECONDS", "10"))
        )
        self._reranker: Optional[CrossEncoderReranker] = None
        self._error: Optional[BaseException] = None
        self._ready = threading.Event()
        self._started = False
        self._lock = threading.Lock()

    def preload(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            thread = threading.Thread(
                target=self._load,
                name=f"reranker-preload-{self.model_name}",
                daemon=True,
            )
            thread.start()

    def _load(self) -> None:
        started_at = time.time()
        try:
            self._reranker = CrossEncoderReranker(model_name=self.model_name)
            elapsed = time.time() - started_at
            _safe_print(f"[OK] Reranker preload finished: {self.model_name} ({elapsed:.2f}s)")
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            _safe_print(f"[WARN] Reranker preload failed: {self.model_name}: {exc}")
        finally:
            self._ready.set()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._reranker is not None

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 3) -> List[SearchResult]:
        self.preload()
        if not self._ready.wait(timeout=self.wait_seconds):
            _safe_print(
                f"[WARN] Reranker still loading in background; skip rerank this time: {self.model_name}. "
                "Tune RAG_RERANKER_WAIT_SECONDS to adjust wait time."
            )
            return results[:top_k]

        if self._error or self._reranker is None:
            _safe_print(f"[WARN] Reranker unavailable; skip rerank: {self._error}")
            return results[:top_k]

        return self._reranker.rerank(query, results, top_k=top_k)


class SharedRerankerRegistry:
    """Per-process registry that prevents duplicate reranker loads."""

    _instances: Dict[str, LazyReranker] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_name: str, *, preload: bool = True) -> LazyReranker:
        with cls._lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = LazyReranker(model_name=model_name)
            reranker = cls._instances[model_name]

        if preload:
            reranker.preload()
        return reranker


def get_shared_reranker(model_name: str, *, preload: bool = True) -> LazyReranker:
    return SharedRerankerRegistry.get(model_name, preload=preload)


class Retriever:
    """基础向量检索器（Dense Retrieval）。"""

    def __init__(self, vector_store: FAISSVectorStore, top_k: int = 3, threshold: Optional[float] = None):
        self.vector_store = vector_store
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query: str, k: Optional[int] = None) -> List[SearchResult]:
        k = k or self.top_k
        return self.vector_store.search(query, k=k, threshold=self.threshold)


class HybridRetriever(Retriever):
    """工业级多路混合检索器（Hybrid Retrieval + RRF + 精排）。

    结合语义向量 (FAISS) 与稀疏检索 (BM25)，使用 RRF 算法融合排序，
    并支持通过 Reranker 进行最终精排。非常适合包含专有名词和公式的科研库。

    Args:
        vector_store (FAISSVectorStore): 底层向量数据库实例。
        reranker (Optional[CrossEncoderReranker]): 精排模型实例。若提供，将执行两段式检索。
        rrf_k (int): RRF 融合算法的平滑常数，默认 60。

    Examples:
        >>> store = FAISSVectorStore(embeddings=embedder)
        >>> reranker = CrossEncoderReranker()
        >>> retriever = HybridRetriever(vector_store=store, reranker=reranker)
        >>> res = retriever.retrieve("ISAC beamforming cvxpy", k=3)
    """

    def __init__(
            self,
            vector_store: FAISSVectorStore,
            reranker: Optional[CrossEncoderReranker] = None,
            rrf_k: int = 60,
            **kwargs
    ):
        super().__init__(vector_store, **kwargs)
        self.reranker = reranker
        self.rrf_k = rrf_k

        self.chunk_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self._build_bm25_index()

    def _build_bm25_index(self):
        """基于当前向量库中的文档，构建 BM25 稀疏索引。"""
        if not hasattr(self.vector_store, 'id_to_chunk') or not self.vector_store.id_to_chunk:
            return

        print("🔨 正在构建 BM25 稀疏检索索引...")
        self.chunk_ids = list(self.vector_store.id_to_chunk.keys())
        # 简单分词，对于中英文混合场景可接入 jieba，此处采用基础按空格拆分
        tokenized_corpus = [
            self.vector_store.id_to_chunk[cid].content.lower().split()
            for cid in self.chunk_ids
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ BM25 索引构建完成。")

    def _sparse_search(self, query: str, k: int) -> List[SearchResult]:
        """执行 BM25 稀疏检索。"""
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # 获取 Top-K 索引
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            cid = self.chunk_ids[idx]
            score = scores[idx]
            if score <= 0:  # 过滤毫无关联的
                continue
            chunk = self.vector_store.get_chunk_by_id(cid)
            if chunk:
                results.append(
                    SearchResult(chunk=chunk, score=float(score), content=chunk.content, metadata=chunk.metadata))
        return results

    def retrieve(self, query: str, k: Optional[int] = None) -> List[SearchResult]:
        """执行 [混合召回 -> RRF融合 -> 重排] 的完整工业级 RAG 链路。"""
        final_k = k or self.top_k
        # 粗排阶段通常需要扩大召回倍数（如召回最终所需数量的 3~5 倍）
        recall_k = final_k * 3

        # 1. 双路粗排召回 (Coarse Recall)
        dense_results = super().retrieve(query, k=recall_k)
        sparse_results = self._sparse_search(query, k=recall_k)

        # 2. RRF 融合 (Reciprocal Rank Fusion)
        rrf_scores: Dict[str, float] = {}

        # 计算稠密检索 RRF 得分
        for rank, res in enumerate(dense_results):
            cid = res.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # 计算稀疏检索 RRF 得分
        for rank, res in enumerate(sparse_results):
            cid = res.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # 合并所有被召回的文档
        all_chunks: Dict[str, DocumentChunk] = {}
        for res in dense_results + sparse_results:
            all_chunks[res.chunk.id] = res.chunk

        # 生成基于 RRF 得分的候选列表
        fused_candidates = []
        for cid, r_score in rrf_scores.items():
            chunk = all_chunks[cid]
            fused_candidates.append(
                SearchResult(chunk=chunk, score=r_score, content=chunk.content, metadata=chunk.metadata)
            )

        fused_candidates.sort(key=lambda x: x.score, reverse=True)
        # 截取粗排 Top 候选送入精排
        coarse_results = fused_candidates[:recall_k]

        # 3. 精排重排 (Fine Ranking / Reranking)
        if self.reranker and coarse_results:
            # Reranker 会用深度语义模型重新打分并截取 final_k
            return self.reranker.rerank(query, coarse_results, top_k=final_k)

        # 如果未开启精排，直接截取 RRF 的 Top-K
        return coarse_results[:final_k]
