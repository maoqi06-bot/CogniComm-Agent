"""FAISS 向量数据库封装模块。"""

import os
import json
import numpy as np
import faiss
from typing import List, Optional, Dict, Any, Union

from .models import DocumentChunk, SearchResult
from .embeddings import BaseEmbeddings


class FAISSVectorStore:
    """
    FAISS 向量存储类，提供向量索引的创建、持久化及高效相似度检索。

    支持多种索引类型（Flat, IVF, HNSW），并自动处理 Cosine 相似度与 L2 距离的转换。

    Args:
        embeddings (BaseEmbeddings): 嵌入模型实例，用于将查询文本向量化。
        index_path (Optional[str]): 索引文件的存储路径前缀。
        dimension (Optional[int]): 向量维度，若为 None 则从 embeddings 中获取。
        index_type (str): FAISS 索引类型，可选 "Flat", "IVF", "HNSW"。
        metric (str): 距离度量方式，可选 "cosine" 或 "l2"。
    """

    def __init__(
            self,
            embeddings: BaseEmbeddings,
            index_path: Optional[str] = None,
            dimension: Optional[int] = None,
            index_type: str = "Flat",
            metric: str = "cosine"
    ):
        self.embeddings = embeddings
        self.dimension = dimension or embeddings.dimension
        self.index_type = index_type
        self.metric = metric
        self.index_path = index_path

        self.index = None
        # 核心映射字典：块ID -> 块对象 (供 BM25 和 Hybrid 访问)
        self.id_to_chunk: Dict[str, DocumentChunk] = {}
        self.position_to_id: List[str] = []

        if index_path and os.path.exists(f"{index_path}.faiss"):
            self.load(index_path)
        else:
            self._create_index()

    def _create_index(self):
        """根据配置创建 FAISS 索引实例。"""
        if self.index_type == "Flat":
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            metric_type = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, metric_type)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: Optional[np.ndarray] = None):
        """将文档块及其向量添加到索引中。"""
        if not chunks:
            return

        if embeddings is None:
            texts = [c.content for c in chunks]
            embeddings = self.embeddings.embed(texts)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype('float32')

        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)

        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(embeddings)

        self.index.add(embeddings)

        for chunk in chunks:
            self.id_to_chunk[chunk.id] = chunk
            self.position_to_id.append(chunk.id)

    def search(
            self,
            query: Union[str, np.ndarray],
            k: int = 5,
            threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """在向量空间中执行 Top-K 近似最近邻搜索 (Dense Search)。"""
        if self.index is None or self.index.ntotal == 0:
            return []

        if isinstance(query, str):
            query_vec = self.embeddings.embed_query(query)
        else:
            query_vec = query

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype('float32')

        if self.metric == "cosine":
            faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.position_to_id):
                continue

            sim = float(score) if self.metric == "cosine" else 1.0 / (1.0 + float(score))

            if threshold is not None and sim < threshold:
                continue

            chunk_id = self.position_to_id[idx]
            chunk = self.id_to_chunk.get(chunk_id)

            if chunk:
                results.append(SearchResult(
                    chunk=chunk,
                    score=sim,
                    content=chunk.content,
                    metadata=chunk.metadata
                ))
        return results

    def save(self, path: Optional[str] = None):
        """将向量索引和块元数据保存到磁盘。"""
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("未指定保存路径")

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        faiss.write_index(self.index, f"{save_path}.faiss")

        chunks_meta = []
        for cid in self.position_to_id:
            chunk = self.id_to_chunk.get(cid)
            if chunk:
                chunks_meta.append({
                    "id": cid,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata
                })

        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "chunks": chunks_meta,
            "position_to_id": self.position_to_id
        }

        with open(f"{save_path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """从磁盘加载索引和元数据。"""
        if not os.path.exists(f"{path}.faiss"):
            raise FileNotFoundError(f"找不到索引文件: {path}.faiss")

        self.index = faiss.read_index(f"{path}.faiss")
        self.index_path = path

        with open(f"{path}.meta.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.dimension = metadata["dimension"]
        self.index_type = metadata["index_type"]
        self.metric = metadata["metric"]
        self.position_to_id = metadata.get("position_to_id", [])

        for item in metadata["chunks"]:
            cid = item["id"]
            chunk = DocumentChunk(
                id=cid,
                document_id=item["document_id"],
                content=item["content"],
                chunk_index=item["chunk_index"],
                metadata=item["metadata"]
            )
            self.id_to_chunk[cid] = chunk

    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息。"""
        return {
            "total_chunks": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
        }

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """根据 ID 快速获取原始文档块。"""
        return self.id_to_chunk.get(chunk_id)