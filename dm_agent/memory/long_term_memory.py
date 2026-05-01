"""长期记忆模块 - 基于 RAG 的持久化记忆存储。

该模块提供长期记忆的存储、检索、更新和删除功能，支持重要性评分和记忆衰减。
"""

from __future__ import annotations

import json
import time
import uuid
import hashlib
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from ..rag.embeddings import BaseEmbeddings, create_embeddings, resolve_embedding_provider
from ..rag.vector_store import FAISSVectorStore
from ..rag.models import DocumentChunk, SearchResult


class MemoryCategory(Enum):
    """记忆类别枚举。"""
    USER_PREFERENCE = "user_preference"      # 用户偏好
    PROJECT_CONTEXT = "project_context"      # 项目上下文
    IMPORTANT_FACT = "important_fact"        # 重要事实
    WORKING_STATE = "working_state"         # 工作状态
    SKILL_KNOWLEDGE = "skill_knowledge"     # 技能知识
    CONVERSATION_SUMMARY = "conversation_summary"  # 对话摘要


class MemoryPriority(Enum):
    """记忆优先级枚举。"""
    CRITICAL = 5   # 关键记忆，永不删除
    HIGH = 4       # 高优先级，降低衰减速度
    NORMAL = 3     # 普通优先级
    LOW = 2        # 低优先级，快速衰减
    EPHEMERAL = 1  # 临时记忆，可被快速清理


@dataclass
class MemoryEntry:
    """记忆条目数据模型。

    Attributes:
        id: 记忆唯一标识符
        content: 记忆内容文本
        category: 记忆类别
        priority: 记忆优先级
        importance_score: 重要性评分 (0.0-1.0)
        access_count: 访问次数
        last_accessed: 最后访问时间戳
        created_at: 创建时间戳
        updated_at: 更新时间戳
        tags: 标签集合
        metadata: 额外元数据
        source: 记忆来源（如 "conversation", "user_input", "system"）
        is_pinned: 是否被固定（固定记忆不会被自动清理）
        decay_factor: 衰减因子，控制记忆随时间的保留程度
    """
    id: str
    content: str
    category: MemoryCategory
    priority: MemoryPriority
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    is_pinned: bool = False
    decay_factor: float = 1.0  # 1.0 = 正常衰减，0.5 = 衰减减半

    def __post_init__(self):
        """初始化后处理。"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if isinstance(self.category, str):
            self.category = MemoryCategory(self.category)
        if isinstance(self.priority, int):
            self.priority = MemoryPriority(self.priority)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category.value,
            "priority": self.priority.value,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "source": self.source,
            "is_pinned": self.is_pinned,
            "decay_factor": self.decay_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """从字典创建实例。"""
        data = data.copy()
        if "tags" in data and isinstance(data["tags"], list):
            data["tags"] = set(data["tags"])
        if "category" in data:
            data["category"] = MemoryCategory(data["category"])
        if "priority" in data:
            data["priority"] = MemoryPriority(data["priority"])
        return cls(**data)

    def calculate_decay_score(self, current_time: Optional[float] = None) -> float:
        """计算衰减后的有效重要性分数。

        Args:
            current_time: 当前时间戳，默认为 time.time()

        Returns:
            衰减后的有效重要性分数 (0.0-1.0)
        """
        if current_time is None:
            current_time = time.time()

        time_elapsed = current_time - self.created_at
        days_elapsed = time_elapsed / (24 * 3600)

        # 基础衰减：每天降低 1%
        base_decay = 0.01 * days_elapsed

        # 优先级影响：高优先级衰减更慢
        priority_modifier = {
            MemoryPriority.CRITICAL: 0.0,   # 永不衰减
            MemoryPriority.HIGH: 0.2,       # 80% 减缓衰减
            MemoryPriority.NORMAL: 0.5,     # 50% 减缓衰减
            MemoryPriority.LOW: 0.8,        # 20% 减缓衰减
            MemoryPriority.EPHEMERAL: 1.0, # 正常衰减
        }[self.priority]

        effective_decay = base_decay * priority_modifier * (2 - self.decay_factor)

        # 固定记忆不衰减
        if self.is_pinned:
            effective_decay = 0.0

        # 访问频率提升：被访问越多，衰减越慢
        access_bonus = min(0.2, self.access_count * 0.02)

        effective_score = (
            self.importance_score
            - effective_decay
            + access_bonus
        )

        return max(0.0, min(1.0, effective_score))


@dataclass
class MemorySearchResult:
    """记忆搜索结果。"""
    entry: MemoryEntry
    score: float
    is_exact_match: bool = False


class LongTermMemoryStore:
    """
    长期记忆存储引擎，基于 FAISS 向量检索。

    提供记忆的添加、检索、更新、删除和智能衰减功能。

    Features:
        - 向量语义检索：基于内容的语义相似度搜索
        - 多维度过滤：支持按类别、优先级、时间范围过滤
        - 智能衰减：根据访问频率和重要性自动降低低价值记忆
        - 固定记忆：支持标记重要记忆永不删除
        - 批量操作：支持批量添加、更新和删除

    Args:
        storage_path: 记忆存储路径
        embeddings: 嵌入模型实例
        dimension: 向量维度（默认从 embeddings 获取）
        max_memories: 最大记忆数量（默认 10000）
        default_category: 默认记忆类别
        default_priority: 默认记忆优先级

    Examples:
        >>> store = LongTermMemoryStore("./memory_store")
        >>> store.add("用户喜欢使用 pytest", category="user_preference")
        >>> results = store.search("测试框架偏好", limit=5)
        >>> store.update("memory_id", importance_score=0.9)
    """

    def __init__(
        self,
        storage_path: str = "./dm_agent/data/memory",
        embeddings: Optional[BaseEmbeddings] = None,
        dimension: Optional[int] = None,
        max_memories: int = 10000,
        default_category: MemoryCategory = MemoryCategory.CONVERSATION_SUMMARY,
        default_priority: MemoryPriority = MemoryPriority.NORMAL,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_path / "memory_index"
        self.metadata_path = self.storage_path / "memory_metadata.json"

        # 加载或创建嵌入模型
        if embeddings is None:
            self.embeddings = create_embeddings(provider=resolve_embedding_provider())
        else:
            self.embeddings = embeddings

        self.dimension = dimension or self.embeddings.dimension
        self.max_memories = max_memories
        self.default_category = default_category
        self.default_priority = default_priority

        # 初始化向量存储
        self.vector_store: Optional[FAISSVectorStore] = None
        self._memory_index: Dict[str, MemoryEntry] = {}
        self._category_index: Dict[MemoryCategory, Set[str]] = {
            cat: set() for cat in MemoryCategory
        }
        self._lock = threading.RLock()

        self._load()
        recovered_count = self._recover_entries_from_vector_store()
        if recovered_count:
            self._save_metadata()

    def _load(self):
        """从磁盘加载记忆数据。"""
        if not self.index_path.exists():
            self.vector_store = FAISSVectorStore(
                embeddings=self.embeddings,
                index_path=str(self.index_path),
                dimension=self.dimension,
            )
            return

        try:
            self.vector_store = FAISSVectorStore(
                embeddings=self.embeddings,
                index_path=str(self.index_path),
            )

            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    for entry_data in metadata.get("memories", []):
                        entry = MemoryEntry.from_dict(entry_data)
                        self._memory_index[entry.id] = entry
                        self._category_index[entry.category].add(entry.id)
        except Exception as e:
            print(f"⚠️ 加载记忆数据失败: {e}，将重新创建")
            self.vector_store = FAISSVectorStore(
                embeddings=self.embeddings,
                index_path=str(self.index_path),
                dimension=self.dimension,
            )
            self._memory_index = {}
            self._category_index = {cat: set() for cat in MemoryCategory}

    def _recover_entries_from_vector_store(self) -> int:
        """Recover entries from vector metadata when memory_metadata.json is stale."""
        if not self.vector_store:
            return 0

        recovered = 0
        for chunk_id, chunk in getattr(self.vector_store, "id_to_chunk", {}).items():
            metadata = dict(chunk.metadata or {})
            memory_id = metadata.get("memory_id") or chunk_id
            if memory_id in self._memory_index:
                continue
            if not self._is_recoverable_vector_memory(chunk.content, metadata):
                continue

            category = self._parse_category(metadata.get("category"))
            priority = self._parse_priority(metadata.get("priority"))
            importance_score = self._safe_float(metadata.get("importance_score"), default=0.5)
            source = str(metadata.get("source") or "recovered_from_vector_store")
            now = time.time()
            created_at = self._safe_float(metadata.get("created_at"), default=now)
            updated_at = self._safe_float(metadata.get("updated_at"), default=created_at)
            entry_metadata = {
                key: value
                for key, value in metadata.items()
                if key not in {
                    "memory_id",
                    "category",
                    "priority",
                    "importance_score",
                    "source",
                    "tags",
                    "created_at",
                    "updated_at",
                    "is_pinned",
                    "decay_factor",
                }
            }

            entry = MemoryEntry(
                id=memory_id,
                content=chunk.content,
                category=category,
                priority=priority,
                importance_score=importance_score,
                tags=self._safe_tags(metadata.get("tags")),
                metadata=entry_metadata,
                source=source,
                is_pinned=bool(metadata.get("is_pinned", False)),
                decay_factor=self._safe_float(metadata.get("decay_factor"), default=1.0),
                created_at=created_at,
                updated_at=updated_at,
                last_accessed=self._safe_float(metadata.get("last_accessed"), default=updated_at),
            )
            self._memory_index[entry.id] = entry
            self._category_index[entry.category].add(entry.id)
            recovered += 1
        return recovered

    def _is_recoverable_vector_memory(self, content: str, metadata: Dict[str, Any]) -> bool:
        text = str(content or "").strip()
        if len(text) < 20:
            return False
        if self._looks_mojibake(text):
            return False
        source = str(metadata.get("source") or "").strip()
        if source == "recovered_from_vector_store":
            return False
        lowered = text.lower()
        low_value_markers = [
            "no relevant knowledge found",
            "long-term memory lookup missed",
            "no relevant user preference memory matched",
            "skipped low-value",
        ]
        return not any(marker in lowered for marker in low_value_markers)

    def _looks_mojibake(self, text: str) -> bool:
        markers = [
            "鐢", "鍩", "妫", "绱", "浠", "璧", "銆", "€", "锛", "涓", "鏂", "娴", "澶",
        ]
        sample = str(text or "")[:1200]
        if not sample:
            return False
        marker_hits = sum(sample.count(marker) for marker in markers)
        return marker_hits >= 4 and (marker_hits / max(1, len(sample))) > 0.015

    def _parse_category(self, raw_value: Any) -> MemoryCategory:
        if isinstance(raw_value, MemoryCategory):
            return raw_value
        try:
            return MemoryCategory(str(raw_value))
        except Exception:
            return self.default_category

    def _parse_priority(self, raw_value: Any) -> MemoryPriority:
        if isinstance(raw_value, MemoryPriority):
            return raw_value
        if isinstance(raw_value, int):
            try:
                return MemoryPriority(raw_value)
            except Exception:
                return self.default_priority
        try:
            text = str(raw_value).strip()
            if text.isdigit():
                return MemoryPriority(int(text))
            return MemoryPriority[text.upper()]
        except Exception:
            return self.default_priority

    def _safe_float(self, raw_value: Any, default: float = 0.5) -> float:
        try:
            return float(raw_value)
        except Exception:
            return default

    def _safe_tags(self, raw_value: Any) -> Set[str]:
        if isinstance(raw_value, (set, list, tuple)):
            return {str(tag) for tag in raw_value if str(tag).strip()}
        if isinstance(raw_value, str) and raw_value.strip():
            return {raw_value.strip()}
        return set()

    def _build_vector_metadata(self, entry: MemoryEntry) -> Dict[str, Any]:
        return {
            "memory_id": entry.id,
            "category": entry.category.value,
            "priority": entry.priority.value,
            "importance_score": entry.importance_score,
            "source": entry.source,
            "tags": list(entry.tags),
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "last_accessed": entry.last_accessed,
            "is_pinned": entry.is_pinned,
            "decay_factor": entry.decay_factor,
            **entry.metadata,
        }

    def _save_metadata(self):
        """保存记忆元数据到磁盘。"""
        memories_data = [entry.to_dict() for entry in self._memory_index.values()]
        metadata = {
            "version": "1.0",
            "updated_at": time.time(),
            "total_memories": len(self._memory_index),
            "memories": memories_data,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _save(self):
        """保存所有数据到磁盘。"""
        with self._lock:
            self.vector_store.save()
            self._save_metadata()

    def add(
        self,
        content: str,
        category: Optional[MemoryCategory] = None,
        priority: Optional[MemoryPriority] = None,
        importance_score: float = 0.5,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "system",
        is_pinned: bool = False,
        decay_factor: float = 1.0,
        memory_id: Optional[str] = None,
    ) -> MemoryEntry:
        """
        添加新记忆。

        Args:
            content: 记忆内容文本
            category: 记忆类别
            priority: 记忆优先级
            importance_score: 重要性评分 (0.0-1.0)
            tags: 标签集合
            metadata: 额外元数据
            source: 记忆来源
            is_pinned: 是否固定
            decay_factor: 衰减因子
            memory_id: 指定记忆 ID（可选，默认自动生成）

        Returns:
            MemoryEntry: 创建的记忆条目
        """
        with self._lock:
            category = category or self.default_category
            priority = priority or self.default_priority
            tags = tags or set()
            metadata = metadata or {}

            entry = MemoryEntry(
                id=memory_id or str(uuid.uuid4()),
                content=content,
                category=category,
                priority=priority,
                importance_score=importance_score,
                tags=tags,
                metadata=metadata,
                source=source,
                is_pinned=is_pinned,
                decay_factor=decay_factor,
            )

            # 生成向量并添加到向量存储
            chunk = DocumentChunk(
                id=entry.id,
                document_id=entry.id,
                content=content,
                chunk_index=0,
                metadata=self._build_vector_metadata(entry),
            )
            self.vector_store.add_chunks([chunk])
            self._memory_index[entry.id] = entry
            self._category_index[entry.category].add(entry.id)

            # 检查容量限制
            self._enforce_capacity()

            self._save()
            return entry

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        category: Optional[MemoryCategory] = None,
        priority: Optional[MemoryPriority] = None,
        importance_score: Optional[float] = None,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_pinned: Optional[bool] = None,
        decay_factor: Optional[float] = None,
        increment_access: bool = False,
    ) -> Optional[MemoryEntry]:
        """
        更新已有记忆。

        Args:
            memory_id: 要更新的记忆 ID
            content: 新内容（可选）
            category: 新类别（可选）
            priority: 新优先级（可选）
            importance_score: 新重要性评分（可选）
            tags: 新标签集合（可选）
            metadata: 新元数据（可选）
            is_pinned: 是否固定（可选）
            decay_factor: 新衰减因子（可选）
            increment_access: 是否增加访问计数

        Returns:
            MemoryEntry: 更新后的记忆条目，失败返回 None
        """
        with self._lock:
            if memory_id not in self._memory_index:
                return None

            entry = self._memory_index[memory_id]
            old_category = entry.category

            # 更新字段
            if content is not None:
                entry.content = content
            if category is not None:
                entry.category = category
            if priority is not None:
                entry.priority = priority
            if importance_score is not None:
                entry.importance_score = max(0.0, min(1.0, importance_score))
            if tags is not None:
                entry.tags = tags
            if metadata is not None:
                entry.metadata = metadata
            if is_pinned is not None:
                entry.is_pinned = is_pinned
            if decay_factor is not None:
                entry.decay_factor = max(0.0, min(2.0, decay_factor))

            if increment_access:
                entry.access_count += 1
                entry.last_accessed = time.time()
                # 访问增加时轻微提升重要性
                entry.importance_score = min(1.0, entry.importance_score + 0.01)

            entry.updated_at = time.time()

            # 更新类别索引
            if category is not None and category != old_category:
                self._category_index[old_category].discard(memory_id)
                self._category_index[category].add(memory_id)

            # 如果内容改变，需要重建向量（简单策略：删除后重新添加）
            if content is not None:
                self._rebuild_vector(memory_id)
            elif memory_id in self.vector_store.id_to_chunk:
                self.vector_store.id_to_chunk[memory_id].metadata = self._build_vector_metadata(entry)

            self._save()
            return entry

    def delete(self, memory_id: str) -> bool:
        """
        删除指定记忆。

        Args:
            memory_id: 要删除的记忆 ID

        Returns:
            bool: 是否成功删除
        """
        with self._lock:
            if memory_id not in self._memory_index:
                return False

            entry = self._memory_index[memory_id]

            # 从索引中移除
            self._category_index[entry.category].discard(memory_id)
            del self._memory_index[memory_id]

            # 从向量存储中删除
            self._rebuild_all_vectors()

            self._save()
            return True

    def delete_by_category(self, category: MemoryCategory) -> int:
        """
        删除指定类别的所有记忆。

        Args:
            category: 要删除的记忆类别

        Returns:
            int: 删除的记忆数量
        """
        with self._lock:
            memory_ids = list(self._category_index[category])
            count = 0
            for memory_id in memory_ids:
                if self.delete(memory_id):
                    count += 1
            return count

    def get(self, memory_id: str, increment_access: bool = True) -> Optional[MemoryEntry]:
        """
        获取指定记忆。

        Args:
            memory_id: 记忆 ID
            increment_access: 是否增加访问计数

        Returns:
            MemoryEntry: 记忆条目，不存在返回 None
        """
        with self._lock:
            entry = self._memory_index.get(memory_id)
            if entry and increment_access:
                entry.access_count += 1
                entry.last_accessed = time.time()
                self._save()
            return entry

    def search(
        self,
        query: str,
        category: Optional[MemoryCategory] = None,
        priority: Optional[MemoryPriority] = None,
        tags: Optional[Set[str]] = None,
        min_importance: float = 0.0,
        limit: int = 10,
        include_decay: bool = True,
    ) -> List[MemorySearchResult]:
        """
        语义搜索记忆。

        Args:
            query: 搜索查询文本
            category: 按类别过滤（可选）
            priority: 按优先级过滤（可选）
            tags: 按标签过滤（必须全部匹配）
            min_importance: 最小重要性评分
            limit: 返回结果数量限制
            include_decay: 是否考虑衰减因素

        Returns:
            List[MemorySearchResult]: 搜索结果列表，按相关性排序
        """
        with self._lock:
            current_time = time.time() if include_decay else None

            # 执行向量检索
            raw_results = self.vector_store.search(query, k=limit * 2)

            results = []
            for result in raw_results:
                memory_id = result.chunk.metadata.get("memory_id") or result.chunk.id
                entry = self._memory_index.get(memory_id)
                if not entry:
                    continue

                # 应用过滤器
                if category and entry.category != category:
                    continue
                if priority and entry.priority != priority:
                    continue
                if tags and not tags.issubset(entry.tags):
                    continue

                # 计算有效分数
                effective_score = entry.calculate_decay_score(current_time) if include_decay else entry.importance_score
                if effective_score < min_importance:
                    continue

                # 混合评分：向量相似度 + 重要性
                # 严格要求向量相似度 > 0.55，确保只返回真正相关的记忆
                if result.score < 0.55:
                    continue
                # 向量相似度为主，重要性为辅
                final_score = result.score * 0.9 + effective_score * 0.1

                results.append(MemorySearchResult(
                    entry=entry,
                    score=final_score,
                    is_exact_match=result.score > 0.9,
                ))

            # 按分数排序
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]

    def get_by_category(
        self,
        category: MemoryCategory,
        limit: Optional[int] = None,
        include_decay: bool = True,
    ) -> List[MemorySearchResult]:
        """
        获取指定类别的所有记忆。

        Args:
            category: 记忆类别
            limit: 返回数量限制
            include_decay: 是否考虑衰减

        Returns:
            List[MemorySearchResult]: 该类别的记忆列表
        """
        with self._lock:
            current_time = time.time() if include_decay else None
            results = []

            for memory_id in self._category_index.get(category, set()):
                entry = self._memory_index.get(memory_id)
                if not entry:
                    continue

                effective_score = entry.calculate_decay_score(current_time) if include_decay else entry.importance_score
                results.append(MemorySearchResult(entry=entry, score=effective_score))

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit] if limit else results

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """
        获取最近访问的记忆。

        Args:
            limit: 返回数量限制

        Returns:
            List[MemoryEntry]: 最近访问的记忆列表
        """
        with self._lock:
            sorted_entries = sorted(
                self._memory_index.values(),
                key=lambda x: x.last_accessed,
                reverse=True,
            )
            return sorted_entries[:limit]

    def get_pinned(self) -> List[MemoryEntry]:
        """
        获取所有固定记忆。

        Returns:
            List[MemoryEntry]: 固定记忆列表
        """
        with self._lock:
            return [entry for entry in self._memory_index.values() if entry.is_pinned]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取记忆存储统计信息。

        Returns:
            Dict: 统计信息字典
        """
        with self._lock:
            current_time = time.time()
            category_counts = {
                cat.value: len(ids) for cat, ids in self._category_index.items()
            }

            priority_counts = {
                f"priority_{p.value}": sum(
                    1 for e in self._memory_index.values() if e.priority == p
                )
                for p in MemoryPriority
            }

            return {
                "total_memories": len(self._memory_index),
                "max_capacity": self.max_memories,
                "utilization_rate": len(self._memory_index) / self.max_memories,
                "category_distribution": category_counts,
                "priority_distribution": priority_counts,
                "pinned_count": sum(1 for e in self._memory_index.values() if e.is_pinned),
                "avg_importance": np.mean([
                    e.importance_score for e in self._memory_index.values()
                ]) if self._memory_index else 0.0,
                "avg_access_count": np.mean([
                    e.access_count for e in self._memory_index.values()
                ]) if self._memory_index else 0.0,
                "oldest_memory_age_days": (
                    (current_time - min(e.created_at for e in self._memory_index.values())) / 86400
                ) if self._memory_index else 0,
            }

    def consolidate(self, similarity_threshold: float = 0.85) -> int:
        """
        合并相似记忆，减少冗余。

        Args:
            similarity_threshold: 相似度阈值，超过此值认为可以合并

        Returns:
            int: 合并的记忆对数
        """
        with self._lock:
            if len(self._memory_index) < 2:
                return 0

            merged_count = 0
            to_merge: List[tuple] = []

            entries = list(self._memory_index.values())
            for i, entry_a in enumerate(entries):
                for entry_b in entries[i + 1:]:
                    if entry_a.category != entry_b.category:
                        continue

                    results = self.vector_store.search(entry_a.content, k=1)
                    if results and results[0].chunk.id == entry_b.id:
                        similarity = results[0].score
                        if similarity >= similarity_threshold:
                            to_merge.append((entry_a, entry_b, similarity))

            for entry_a, entry_b, similarity in to_merge:
                # 保留更重要或更近更新的
                if entry_a.importance_score > entry_b.importance_score or (
                    entry_a.importance_score == entry_b.importance_score
                    and entry_a.updated_at > entry_b.updated_at
                ):
                    keeper, discarded = entry_a, entry_b
                else:
                    keeper, discarded = entry_b, entry_a

                # 合并标签和元数据
                keeper.tags.update(discarded.tags)
                keeper.metadata.update(discarded.metadata)
                keeper.access_count += discarded.access_count
                keeper.importance_score = max(keeper.importance_score, discarded.importance_score)
                keeper.updated_at = time.time()

                self.delete(discarded.id)
                merged_count += 1

            return merged_count

    def cleanup_low_value(self, min_score: float = 0.1) -> int:
        """
        清理低价值记忆。

        Args:
            min_score: 最低分数阈值，低于此值的非固定记忆将被删除

        Returns:
            int: 清理的记忆数量
        """
        with self._lock:
            current_time = time.time()
            to_delete = []

            for entry in self._memory_index.values():
                if entry.is_pinned:
                    continue
                if entry.priority == MemoryPriority.CRITICAL:
                    continue

                score = entry.calculate_decay_score(current_time)
                if score < min_score:
                    to_delete.append(entry.id)

            for memory_id in to_delete:
                self.delete(memory_id)

            return len(to_delete)

    def _enforce_capacity(self):
        """强制执行容量限制。"""
        if len(self._memory_index) <= self.max_memories:
            return

        # 删除最低价值的非固定记忆
        current_time = time.time()
        candidates = [
            (entry.id, entry.calculate_decay_score(current_time))
            for entry in self._memory_index.values()
            if not entry.is_pinned and entry.priority != MemoryPriority.CRITICAL
        ]

        candidates.sort(key=lambda x: x[1])
        excess = len(self._memory_index) - self.max_memories

        for memory_id, _ in candidates[:excess]:
            self.delete(memory_id)

    def _rebuild_vector(self, memory_id: str):
        """重建指定记忆的向量表示。"""
        if memory_id not in self._memory_index:
            return

        self._rebuild_all_vectors()
        return

        entry = self._memory_index[memory_id]

        # 简单策略：直接从内容重建
        chunk = DocumentChunk(
            id=entry.id,
            document_id=entry.id,
            content=entry.content,
            chunk_index=0,
            metadata=self._build_vector_metadata(entry),
        )

        # 重新生成向量
        vectors = self.embeddings.embed([entry.content])
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # 在 id_to_chunk 中更新
        self.vector_store.id_to_chunk[memory_id] = chunk

    def _rebuild_all_vectors(self):
        """Rebuild the vector index from the authoritative memory index."""
        index_type = getattr(self.vector_store, "index_type", "Flat")
        metric = getattr(self.vector_store, "metric", "cosine")
        self.vector_store = FAISSVectorStore(
            embeddings=self.embeddings,
            index_path=str(self.index_path),
            dimension=self.dimension,
            index_type=index_type,
            metric=metric,
        )
        chunks = [
            DocumentChunk(
                id=entry.id,
                document_id=entry.id,
                content=entry.content,
                chunk_index=0,
                metadata=self._build_vector_metadata(entry),
            )
            for entry in self._memory_index.values()
        ]
        if chunks:
            self.vector_store.add_chunks(chunks)

    def export_memories(self, filepath: Optional[str] = None) -> str:
        """
        导出所有记忆为 JSON 文件。

        Args:
            filepath: 导出路径（可选，默认生成带时间戳的文件名）

        Returns:
            str: 导出文件的路径
        """
        with self._lock:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = str(self.storage_path / f"memory_export_{timestamp}.json")

            export_data = {
                "version": "1.0",
                "exported_at": time.time(),
                "total_memories": len(self._memory_index),
                "memories": [entry.to_dict() for entry in self._memory_index.values()],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            return filepath

    def import_memories(self, filepath: str, merge: bool = True) -> int:
        """
        从 JSON 文件导入记忆。

        Args:
            filepath: 导入文件路径
            merge: 是否与现有记忆合并（True=合并，False=替换）

        Returns:
            int: 导入的记忆数量
        """
        with self._lock:
            with open(filepath, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            memories = import_data.get("memories", [])
            count = 0

            if not merge:
                # 清空现有记忆
                self._memory_index.clear()
                for cat in self._category_index:
                    self._category_index[cat].clear()

            for mem_data in memories:
                entry = MemoryEntry.from_dict(mem_data)
                if entry.id not in self._memory_index:
                    self._memory_index[entry.id] = entry
                    self._category_index[entry.category].add(entry.id)
                    count += 1

            if count > 0:
                self._save()

            return count
