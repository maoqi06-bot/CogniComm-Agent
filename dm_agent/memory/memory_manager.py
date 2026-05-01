"""记忆管理器 - 协调短期记忆压缩与长期记忆的存取。

该模块负责：
1. 从对话历史中自动提取值得长期保存的信息
2. 管理短期记忆（上下文压缩）与长期记忆（RAG存储）的交互
3. 在每次对话开始时检索相关长期记忆增强上下文
4. 自动进行记忆整合和清理
"""

from __future__ import annotations

import json
import re
import time
import hashlib
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field

from ..clients.base_client import BaseLLMClient
from ..prompts.memory_prompts import (
    build_memory_extraction_prompt,
    build_memory_resolution_prompt,
)
from .long_term_memory import (
    LongTermMemoryStore,
    MemoryEntry,
    MemoryCategory,
    MemoryPriority,
    MemorySearchResult,
)
from .context_compressor import ContextCompressor


@dataclass
class MemoryRetrievalResult:
    """记忆检索结果，包含检索到的记忆和增强后的上下文。"""
    memories: List[MemorySearchResult]
    enhanced_context: str
    categories_used: Set[MemoryCategory]


class MemoryExtractionPattern:
    """记忆提取模式定义。"""

    # 用户偏好相关的关键词
    PREFERENCE_PATTERNS = [
        r"我喜欢(.+?)[，,。;；]",
        r"我更(喜欢|倾向于|习惯用)(.+?)[，,。;；]",
        r"(使用|用|选|偏好)(.+?)(更|比较)?(好|方便|习惯)",
        r"请(.+?)一直|总是|一直用",
    ]

    # 项目上下文相关
    PROJECT_PATTERNS = [
        r"项目[名名称叫](.+?)[，,。;；]",
        r"正在开发(.+?)[，,。;；]",
        r"当前(.+?)目录",
        r"工作[在空间目录](.+?)[，,。;；]",
    ]

    # 重要事实相关
    FACT_PATTERNS = [
        r"关键[的是]|重要的是|必须记住",
        r"记住(.+?)[，,。;；]",
        r"这个(.+?)不能改|不要动|保留",
    ]

    # 错误和解决方案
    ERROR_PATTERNS = [
        r"之前(出|有)错.*?(原因|因为|是由于)(.+?)[。]",
        r"(Bug|错误|问题)[:：](.+?)[。]",
        r"解决方案是[:：](.+?)[。]",
    ]


class MemoryManager:
    """
    记忆管理器 - 协调短期与长期记忆的核心组件。

    MemoryManager 是整个记忆系统的核心，它：
    1. 桥接 ContextCompressor（短期记忆）和 LongTermMemoryStore（长期记忆）
    2. 自动从对话历史中提取值得长期保存的信息
    3. 在新对话开始时检索相关长期记忆增强上下文
    4. 管理记忆的优先级、衰减和清理

    Architecture:
        User Input → Context Compressor → [if worth saving] → Long Term Memory Store
                    ↑
        LLM Response ← Enhanced Context ← Memory Retrieval ←

    Args:
        memory_store: 长期记忆存储实例
        llm_client: LLM 客户端（用于记忆提取和摘要生成）
        context_compressor: 上下文压缩器实例
        config: 记忆管理配置

    Configuration Options:
        auto_extract_enabled: 是否自动提取记忆（默认 True）
        extraction_threshold: 自动提取的阈值（默认 0.6）
        retrieval_top_k: 检索返回的最多记忆数（默认 5）
        consolidation_interval: 整合间隔（小时，默认 24）
        cleanup_interval: 清理间隔（小时，默认 168/周）
        enable_importance_boost: 是否启用重要性提升（默认 True）
    """

    def __init__(
        self,
        memory_store: Optional[LongTermMemoryStore] = None,
        llm_client: Optional[BaseLLMClient] = None,
        context_compressor: Optional[ContextCompressor] = None,
        config: Optional[Dict[str, Any]] = None,
        storage_path: str = "./dm_agent/data/memory",
    ):
        self.config = config or {}
        self.storage_path = storage_path

        # 初始化长期记忆存储
        if memory_store is None:
            self.memory_store = LongTermMemoryStore(storage_path=storage_path)
        else:
            self.memory_store = memory_store

        self.llm_client = llm_client
        self.context_compressor = context_compressor

        # 配置参数
        self.auto_extract_enabled = self.config.get("auto_extract_enabled", True)
        self.extraction_threshold = self.config.get("extraction_threshold", 0.6)
        self.retrieval_top_k = self.config.get("retrieval_top_k", 5)
        self.consolidation_interval = self.config.get("consolidation_interval", 24)  # hours
        self.cleanup_interval = self.config.get("cleanup_interval", 168)  # hours (1 week)
        self.smart_memory_update_enabled = self.config.get("smart_memory_update_enabled", True)
        self.memory_resolution_top_k = self.config.get("memory_resolution_top_k", 5)
        self.memory_resolution_min_similarity = self.config.get("memory_resolution_min_similarity", 0.72)
        self.memory_resolution_temperature = self.config.get("memory_resolution_temperature", 0.0)

        # 状态跟踪
        self._last_consolidation = time.time()
        self._last_cleanup = time.time()
        self._session_memory_ids: List[str] = []  # 本次会话添加的记忆 ID

        # 自定义提取回调
        self._extraction_callbacks: List[Callable[[str, str], Optional[MemoryEntry]]] = []

        # 检索缓存
        self._retrieval_cache: Dict[str, MemoryRetrievalResult] = {}
        self._cache_max_size = 50

    _OPERATIONAL_FAILURE_TERMS = (
        "gbk",
        "unicodeencodeerror",
        "unicodedecodeerror",
        "emoji",
        "mcp_wireless-rag",
        "mcp-wireless-rag",
        "rag \u5de5\u5177",
        "\u7f16\u7801\u9519\u8bef",
        "\u7f16\u7801\u5f02\u5e38",
        "\u68c0\u7d22\u672a\u80fd\u6267\u884c",
        "\u65e0\u6cd5\u6267\u884c",
        "\u5de5\u5177\u8f93\u51fa",
        "\u73af\u5883\u7f16\u7801",
        "timeout",
        "timed out",
        "超时",
        "rate limit",
        "限流",
        "429",
        "500",
        "503",
        "connection refused",
        "连接拒绝",
        "temporary failure",
        "临时失败",
        "traceback",
        "debug",
        "stack trace",
        "调试",
        "日志",
        "tempfile",
        "tmp",
        "临时",
        "cache",
        "缓存",
    )

    _OPERATIONAL_FAILURE_SOURCES = (
        "mcp",
        "rag",
        "tool",
        "run_python",
        "run_shell",
        "\u5de5\u5177",
        "\u68c0\u7d22",
        "search_memory",
        "read_file",
        "write_file",
        "execute",
    )

    # ==================== Public API ====================

    def retrieve_for_context(
        self,
        current_task: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        categories: Optional[List[MemoryCategory]] = None,
        limit: Optional[int] = None,
    ) -> MemoryRetrievalResult:
        """
        为当前任务检索相关长期记忆。

        这是最常用的方法，在每次新任务开始时调用，
        返回的增强上下文可以直接拼接到 system prompt 或用户提示中。

        Args:
            current_task: 当前任务描述
            conversation_history: 最近的对话历史（可选，用于更精确的检索）
            categories: 要检索的特定类别（可选，None 表示全部）
            limit: 返回记忆数量限制

        Returns:
            MemoryRetrievalResult: 包含检索结果和增强上下文的封装对象
        """
        limit = limit or self.retrieval_top_k

        # 检查缓存
        cache_key = f"{current_task}:{categories}:{limit}"
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]

        all_results: List[MemorySearchResult] = []

        if categories:
            # 按指定类别检索
            for cat in categories:
                results = self.memory_store.search(
                    query=current_task,
                    category=cat,
                    limit=limit,
                    include_decay=True,
                )
                all_results.extend(results)
        else:
            # 全品类检索
            # 1. 先检索所有相关记忆（提高阈值）
            results = self.memory_store.search(
                query=current_task,
                limit=limit * 2,
                include_decay=True,
            )
            all_results.extend(results)

            # 2. 补充检索用户偏好（总是很重要）- 限制数量
            preference_results = self.memory_store.search(
                query=current_task,
                category=MemoryCategory.USER_PREFERENCE,
                limit=1,  # 减少为1条
                include_decay=False,
            )
            all_results.extend(preference_results)

        # 去重（基于记忆 ID）
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if self._is_low_value_operational_memory(result.entry):
                continue
            if result.entry.id not in seen_ids:
                seen_ids.add(result.entry.id)
                unique_results.append(result)

        # 按分数重新排序
        unique_results.sort(key=lambda x: x.score, reverse=True)
        unique_results = unique_results[:limit]

        # 更新访问计数
        for result in unique_results:
            self.memory_store.update(result.entry.id, increment_access=True)

        # 构建增强上下文
        enhanced_context = self._build_enhanced_context(unique_results, current_task)

        categories_used = set(r.entry.category for r in unique_results)

        result = MemoryRetrievalResult(
            memories=unique_results,
            enhanced_context=enhanced_context,
            categories_used=categories_used,
        )

        # 缓存结果
        if len(self._retrieval_cache) >= self._cache_max_size:
            # 删除最旧的缓存
            oldest_key = next(iter(self._retrieval_cache))
            del self._retrieval_cache[oldest_key]
        self._retrieval_cache[cache_key] = result

        return result

    def add_memory(
        self,
        content: str,
        category: MemoryCategory,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        importance_score: float = 0.5,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "manual",
        is_pinned: bool = False,
    ) -> MemoryEntry:
        """
        手动添加一条记忆。

        Args:
            content: 记忆内容
            category: 记忆类别
            priority: 优先级
            importance_score: 重要性评分
            tags: 标签
            metadata: 元数据
            source: 来源
            is_pinned: 是否固定

        Returns:
            MemoryEntry: 创建的记忆条目
        """
        entry = self.add_or_update_memory(
            content=content,
            category=category,
            priority=priority,
            importance_score=importance_score,
            tags=tags,
            metadata=metadata,
            source=source,
            is_pinned=is_pinned,
        )
        return entry

    def add_or_update_memory(
        self,
        content: str,
        category: MemoryCategory,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        importance_score: float = 0.5,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "manual",
        is_pinned: bool = False,
        decay_factor: float = 1.0,
    ) -> MemoryEntry:
        """Add a memory, or intelligently update/replace a similar existing one."""
        tags = set(tags or set())
        metadata = dict(metadata or {})
        candidate_entry = MemoryEntry(
            id="",
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
        if self._is_low_value_operational_memory(candidate_entry):
            return candidate_entry

        if not self.smart_memory_update_enabled:
            return self._create_memory(candidate_entry)

        candidates = self._find_similar_memories(candidate_entry)
        exact_duplicate = self._find_exact_duplicate(candidate_entry, candidates)
        if exact_duplicate:
            updated = self.memory_store.update(exact_duplicate.id, increment_access=True)
            return updated or exact_duplicate

        if not candidates:
            return self._create_memory(candidate_entry)

        decision = self.resolve_memory_conflict(candidate_entry, candidates)
        action = str(decision.get("action") or "create_new").strip().lower()

        if action == "ignore":
            target = self._get_decision_target(decision, candidates)
            if target:
                updated = self.memory_store.update(target.id, increment_access=True)
                return updated or target
            return candidate_entry

        if action in {"update_existing", "replace_existing"}:
            target = self._get_decision_target(decision, candidates)
            if target:
                return self._apply_memory_resolution(
                    target=target,
                    incoming=candidate_entry,
                    decision=decision,
                    action=action,
                )

        return self._create_memory(candidate_entry)

    def resolve_memory_conflict(
        self,
        new_memory: MemoryEntry,
        candidates: List[MemorySearchResult],
    ) -> Dict[str, Any]:
        """Ask the LLM how to handle a new memory relative to similar existing memories."""
        if not self.llm_client:
            return {"action": "create_new", "reason": "No LLM client available."}

        prompt = build_memory_resolution_prompt(
            new_memory=self._serialize_memory_for_prompt(new_memory),
            candidates=[
                {
                    **self._serialize_memory_for_prompt(result.entry),
                    "similarity_score": result.score,
                }
                for result in candidates
            ],
        )
        try:
            response = self.llm_client.respond(
                [{"role": "user", "content": prompt}],
                temperature=self.memory_resolution_temperature,
            )
            json_str = self._extract_json_from_response(response)
            if not json_str:
                return {"action": "create_new", "reason": "LLM returned no parseable JSON."}
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return {"action": "create_new", "reason": "LLM response was not a JSON object."}
            if data.get("action") not in {"create_new", "update_existing", "replace_existing", "ignore"}:
                data["action"] = "create_new"
            return data
        except Exception as e:
            return {"action": "create_new", "reason": f"Memory resolution failed: {e}"}

    def extract_and_store(
        self,
        conversation_history: List[Dict[str, str]],
        current_task: Optional[str] = None,
        force_extract: bool = False,
    ) -> List[MemoryEntry]:
        """
        从对话历史中自动提取值得保存的信息并存储。

        Args:
            conversation_history: 对话历史
            current_task: 当前任务（可选，用于上下文）
            force_extract: 是否强制提取（即使未达到阈值）

        Returns:
            List[MemoryEntry]: 提取并存储的记忆列表
        """
        if not self.auto_extract_enabled and not force_extract:
            return []

        extracted = []

        # 1. 基于规则的快速提取
        rule_extracted = self._extract_by_rules(conversation_history, current_task)
        extracted.extend(rule_extracted)

        # 2. 基于 LLM 的深度提取（如果有 LLM 客户端）
        if self.llm_client and len(conversation_history) >= 3:
            llm_extracted = self._extract_by_llm(conversation_history, current_task)
            extracted.extend(llm_extracted)

        # 3. 去重（基于内容哈希）
        seen_hashes = set()
        unique_extracted = []
        for entry in extracted:
            if self._is_low_value_operational_memory(entry):
                continue
            content_hash = hashlib.md5(entry.content.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_extracted.append(entry)

        # 4. 存储唯一记忆
        stored = []
        for entry in unique_extracted:
            stored_entry = self.add_or_update_memory(
                content=entry.content,
                category=entry.category,
                priority=entry.priority,
                importance_score=entry.importance_score,
                tags=entry.tags,
                metadata=entry.metadata,
                source=f"extracted_from_{entry.source}",
                is_pinned=entry.is_pinned,
                decay_factor=entry.decay_factor,
            )
            stored.append(stored_entry)

        return stored

    def update_memory_importance(
        self,
        memory_id: str,
        task_completed_successfully: bool,
        user_feedback: Optional[str] = None,
    ) -> Optional[MemoryEntry]:
        """
        根据任务完成情况更新相关记忆的重要性。

        Args:
            memory_id: 记忆 ID
            task_completed_successfully: 任务是否成功完成
            user_feedback: 用户反馈（可选）

        Returns:
            MemoryEntry: 更新后的记忆
        """
        entry = self.memory_store.get(memory_id)
        if not entry:
            return None

        # 根据任务完成情况调整重要性
        if task_completed_successfully:
            # 成功完成 -> 提升重要性
            new_score = min(1.0, entry.importance_score + 0.15)
        else:
            # 失败 -> 小幅降低
            new_score = max(0.1, entry.importance_score - 0.05)

        return self.memory_store.update(
            memory_id,
            importance_score=new_score,
        )

    def consolidate_memories(self) -> Dict[str, int]:
        """
        执行记忆整合和清理。

        Returns:
            Dict: 包含整合和清理统计的字典
        """
        current_time = time.time()

        stats = {
            "consolidated_pairs": 0,
            "cleaned_memories": 0,
            "total_memories_before": len(self.memory_store._memory_index),
        }

        # 1. 合并相似记忆
        if current_time - self._last_consolidation >= self.consolidation_interval * 3600:
            stats["consolidated_pairs"] = self.memory_store.consolidate()
            self._last_consolidation = current_time

        # 2. 清理低价值记忆
        if current_time - self._last_cleanup >= self.cleanup_interval * 3600:
            stats["cleaned_memories"] = self.memory_store.cleanup_low_value()
            self._last_cleanup = current_time

        stats["total_memories_after"] = len(self.memory_store._memory_index)
        return stats

    def get_session_memories(self) -> List[MemoryEntry]:
        """
        获取本次会话中添加的所有记忆。

        Returns:
            List[MemoryEntry]: 会话记忆列表
        """
        return [
            self.memory_store.get(mid)
            for mid in self._session_memory_ids
            if self.memory_store.get(mid) is not None
        ]

    def clear_session_memories(self) -> int:
        """
        清除本次会话中添加的记忆（但不删除已存储的重要记忆）。

        Returns:
            int: 清除的记忆数量
        """
        cleared = len(self._session_memory_ids)
        self._session_memory_ids.clear()
        return cleared

    def register_extraction_callback(
        self,
        callback: Callable[[str, str], Optional[MemoryEntry]],
    ):
        """
        注册自定义记忆提取回调。

        Args:
            callback: 回调函数，接收 (content, source) 返回 MemoryEntry 或 None
        """
        self._extraction_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取记忆系统统计信息。

        Returns:
            Dict: 统计信息
        """
        store_stats = self.memory_store.get_statistics()
        return {
            "store": store_stats,
            "session": {
                "memories_added": len(self._session_memory_ids),
                "auto_extract_enabled": self.auto_extract_enabled,
            },
            "config": {
                "extraction_threshold": self.extraction_threshold,
                "retrieval_top_k": self.retrieval_top_k,
                "consolidation_interval_hours": self.consolidation_interval,
                "cleanup_interval_hours": self.cleanup_interval,
                "smart_memory_update_enabled": self.smart_memory_update_enabled,
                "memory_resolution_top_k": self.memory_resolution_top_k,
                "memory_resolution_min_similarity": self.memory_resolution_min_similarity,
            },
        }

    # ==================== Private Methods ====================

    def _create_memory(self, entry: MemoryEntry) -> MemoryEntry:
        stored = self.memory_store.add(
            content=entry.content,
            category=entry.category,
            priority=entry.priority,
            importance_score=entry.importance_score,
            tags=entry.tags,
            metadata=entry.metadata,
            source=entry.source,
            is_pinned=entry.is_pinned,
            decay_factor=entry.decay_factor,
        )
        self._remember_session_id(stored.id)
        self._retrieval_cache.clear()
        return stored

    def _remember_session_id(self, memory_id: str) -> None:
        if memory_id and memory_id not in self._session_memory_ids:
            self._session_memory_ids.append(memory_id)

    def _find_similar_memories(self, entry: MemoryEntry) -> List[MemorySearchResult]:
        try:
            results = self.memory_store.search(
                query=entry.content,
                category=entry.category,
                limit=self.memory_resolution_top_k,
                include_decay=False,
            )
        except Exception:
            return []
        filtered: List[MemorySearchResult] = []
        for result in results:
            if self._is_low_value_operational_memory(result.entry):
                continue
            if result.score >= self.memory_resolution_min_similarity:
                filtered.append(result)
        return filtered

    def _find_exact_duplicate(
        self,
        entry: MemoryEntry,
        candidates: List[MemorySearchResult],
    ) -> Optional[MemoryEntry]:
        normalized = self._normalize_memory_text(entry.content)
        for result in candidates:
            if (
                result.entry.category == entry.category
                and self._normalize_memory_text(result.entry.content) == normalized
            ):
                return result.entry
        return None

    def _normalize_memory_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower())

    def _get_decision_target(
        self,
        decision: Dict[str, Any],
        candidates: List[MemorySearchResult],
    ) -> Optional[MemoryEntry]:
        target_id = str(decision.get("target_memory_id") or "").strip()
        if target_id:
            for result in candidates:
                if result.entry.id == target_id:
                    return result.entry
            try:
                return self.memory_store.get(target_id, increment_access=False)
            except TypeError:
                return self.memory_store.get(target_id)
            except Exception:
                return None
        return candidates[0].entry if candidates else None

    def _apply_memory_resolution(
        self,
        target: MemoryEntry,
        incoming: MemoryEntry,
        decision: Dict[str, Any],
        action: str,
    ) -> MemoryEntry:
        reason = str(decision.get("reason") or "").strip()
        updated_content = str(decision.get("updated_content") or "").strip()
        if not updated_content:
            updated_content = incoming.content if action == "replace_existing" else target.content

        final_category = self._parse_category_value(decision.get("category"), target.category)
        final_importance = self._safe_score(
            decision.get("importance_score"),
            max(target.importance_score, incoming.importance_score),
        )
        final_tags = set(target.tags or set()).union(incoming.tags or set())
        if isinstance(decision.get("tags"), list):
            final_tags.update(str(tag) for tag in decision["tags"] if str(tag).strip())

        history_item = {
            "action": action,
            "previous_content": target.content,
            "incoming_content": incoming.content,
            "incoming_source": incoming.source,
            "reason": reason,
            "resolved_at": time.time(),
        }
        metadata = dict(target.metadata or {})
        history = list(metadata.get("memory_update_history") or [])
        history.append(history_item)
        metadata["memory_update_history"] = history
        metadata["last_merge_reason"] = reason
        metadata["resolved_from_memory_ids"] = sorted(
            set(metadata.get("resolved_from_memory_ids") or []) | {incoming.id}
        )
        if action == "replace_existing":
            metadata["superseded_by"] = target.id

        updated = self.memory_store.update(
            target.id,
            content=updated_content,
            category=final_category,
            priority=max(target.priority, incoming.priority, key=lambda item: item.value),
            importance_score=final_importance,
            tags=final_tags,
            metadata=metadata,
            is_pinned=bool(decision.get("is_pinned", target.is_pinned or incoming.is_pinned)),
            decay_factor=min(target.decay_factor, incoming.decay_factor),
        )
        resolved = updated or target
        self._remember_session_id(resolved.id)
        self._retrieval_cache.clear()
        return resolved

    def _serialize_memory_for_prompt(self, entry: MemoryEntry) -> Dict[str, Any]:
        return {
            "id": entry.id,
            "content": entry.content,
            "category": entry.category.value,
            "priority": entry.priority.value,
            "importance_score": entry.importance_score,
            "tags": sorted(entry.tags or set()),
            "metadata": entry.metadata or {},
            "source": entry.source,
            "is_pinned": entry.is_pinned,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }

    def _parse_category_value(self, value: Any, default: MemoryCategory) -> MemoryCategory:
        if not value:
            return default
        try:
            return MemoryCategory(str(value))
        except Exception:
            return default

    def _safe_score(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return max(0.0, min(1.0, float(default)))

    def _is_low_value_operational_memory(self, entry: MemoryEntry) -> bool:
        """Return True for transient tool/runtime failures that should not become long-term memory."""
        content = (entry.content or "").strip()
        if not content:
            return True

        text_parts = [content]
        if entry.metadata:
            text_parts.extend(str(value) for value in entry.metadata.values())
        if entry.tags:
            text_parts.extend(str(tag) for tag in entry.tags)
        haystack = " ".join(text_parts).lower()

        has_failure_term = any(term.lower() in haystack for term in self._OPERATIONAL_FAILURE_TERMS)
        has_tool_context = any(term.lower() in haystack for term in self._OPERATIONAL_FAILURE_SOURCES)
        if has_failure_term and has_tool_context:
            return True

        error_tags = {"error", "failure", "failed", "troubleshooting", "bug"}
        normalized_tags = {str(tag).lower() for tag in entry.tags or set()}
        if error_tags.intersection(normalized_tags):
            if any(term in haystack for term in ("gbk", "unicode", "emoji", "\u7f16\u7801")):
                return True

        return False

    def _build_enhanced_context(
        self,
        results: List[MemorySearchResult],
        current_task: str,
    ) -> str:
        """
        根据检索结果构建增强上下文。

        Args:
            results: 记忆检索结果
            current_task: 当前任务

        Returns:
            str: 格式化后的增强上下文
        """
        if not results:
            return ""

        # 只包含向量相似度 > 0.65 的记忆，避免低相关记忆污染上下文
        high_quality_results = [r for r in results if r.score > 0.65]
        if not high_quality_results:
            return ""  # 没有足够相关的记忆，返回空

        lines = ["\n\n=== 相关历史记忆 ==="]

        # 按类别分组
        by_category: Dict[MemoryCategory, List[MemorySearchResult]] = {}
        for result in high_quality_results:
            cat = result.entry.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)

        # 生成各部分内容
        for category, category_results in by_category.items():
            lines.append(f"\n【{self._get_category_display_name(category)}】")
            for result in category_results[:2]:  # 每类最多2条
                entry = result.entry
                lines.append(f"- {entry.content}")
                if entry.tags:
                    lines.append(f"  [标签: {', '.join(list(entry.tags)[:3])}]")

        lines.append("\n=== 历史记忆结束 ===\n")
        return "\n".join(lines)

    def _get_category_display_name(self, category: MemoryCategory) -> str:
        """获取类别的中文显示名称。"""
        names = {
            MemoryCategory.USER_PREFERENCE: "用户偏好",
            MemoryCategory.PROJECT_CONTEXT: "项目上下文",
            MemoryCategory.IMPORTANT_FACT: "重要事实",
            MemoryCategory.WORKING_STATE: "工作状态",
            MemoryCategory.SKILL_KNOWLEDGE: "技能知识",
            MemoryCategory.CONVERSATION_SUMMARY: "对话摘要",
        }
        return names.get(category, category.value)

    def _extract_by_rules(
        self,
        conversation_history: List[Dict[str, str]],
        current_task: Optional[str],
    ) -> List[MemoryEntry]:
        """
        基于规则从对话历史中提取记忆。

        Args:
            conversation_history: 对话历史
            current_task: 当前任务

        Returns:
            List[MemoryEntry]: 提取的记忆列表
        """
        extracted = []

        # 合并所有对话内容
        all_content = "\n".join([
            msg.get("content", "") for msg in conversation_history
        ])

        # 提取用户偏好
        for pattern in MemoryExtractionPattern.PREFERENCE_PATTERNS:
            matches = re.finditer(pattern, all_content)
            for match in matches:
                extracted.append(MemoryEntry(
                    id="",
                    content=match.group(0),
                    category=MemoryCategory.USER_PREFERENCE,
                    priority=MemoryPriority.HIGH,
                    importance_score=0.7,
                    source="rule_extraction",
                    tags={"preference", "user"},
                ))

        # 提取项目上下文
        for pattern in MemoryExtractionPattern.PROJECT_PATTERNS:
            matches = re.finditer(pattern, all_content)
            for match in matches:
                extracted.append(MemoryEntry(
                    id="",
                    content=match.group(0),
                    category=MemoryCategory.PROJECT_CONTEXT,
                    priority=MemoryPriority.NORMAL,
                    importance_score=0.6,
                    source="rule_extraction",
                    tags={"project", "context"},
                ))

        # 提取重要事实
        for pattern in MemoryExtractionPattern.FACT_PATTERNS:
            matches = re.finditer(pattern, all_content)
            for match in matches:
                extracted.append(MemoryEntry(
                    id="",
                    content=match.group(0),
                    category=MemoryCategory.IMPORTANT_FACT,
                    priority=MemoryPriority.HIGH,
                    importance_score=0.8,
                    source="rule_extraction",
                    tags={"important", "fact"},
                    is_pinned=True,  # 重要事实默认固定
                ))

        # 提取错误和解决方案
        for pattern in MemoryExtractionPattern.ERROR_PATTERNS:
            matches = re.finditer(pattern, all_content)
            for match in matches:
                extracted.append(MemoryEntry(
                    id="",
                    content=match.group(0),
                    category=MemoryCategory.SKILL_KNOWLEDGE,
                    priority=MemoryPriority.HIGH,
                    importance_score=0.75,
                    source="rule_extraction",
                    tags={"error", "solution", "troubleshooting"},
                ))

        return extracted

    def _extract_by_llm(
        self,
        conversation_history: List[Dict[str, str]],
        current_task: Optional[str],
    ) -> List[MemoryEntry]:
        """
        基于 LLM 从对话历史中深度提取记忆。

        Args:
            conversation_history: 对话历史
            current_task: 当前任务

        Returns:
            List[MemoryEntry]: 提取的记忆列表
        """
        if not self.llm_client:
            return []

        # 构建提取提示
        history_text = self._format_conversation_for_extraction(conversation_history)
        extraction_prompt = build_memory_extraction_prompt(history_text, current_task)

        try:
            messages = [{"role": "user", "content": extraction_prompt}]
            response = self.llm_client.respond(messages, temperature=0.3)

            # 解析响应
            json_str = self._extract_json_from_response(response)
            if json_str:
                data = json.loads(json_str)
                if isinstance(data, list):
                    return [
                        MemoryEntry(
                            id="",
                            content=item.get("content", ""),
                            category=MemoryCategory(item.get("category", "conversation_summary")),
                            priority=MemoryPriority.NORMAL,
                            importance_score=item.get("importance_score", 0.5),
                            tags=set(item.get("tags", [])),
                            metadata={"extraction_reason": item.get("reason", "")},
                            source="llm_extraction",
                        )
                        for item in data
                        if item.get("content")
                    ]
        except Exception as e:
            print(f"⚠️ LLM 记忆提取失败: {e}")

        return []

    def _format_conversation_for_extraction(
        self,
        conversation_history: List[Dict[str, str]],
        max_turns: int = 10,
    ) -> str:
        """
        格式化对话历史用于提取。

        Args:
            conversation_history: 对话历史
            max_turns: 最大使用的对话轮数

        Returns:
            str: 格式化后的对话文本
        """
        lines = []
        for msg in conversation_history[-max_turns:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:500]  # 截断过长的内容
            lines.append(f"[{role}]: {content}")

        return "\n".join(lines)

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """
        从 LLM 响应中提取 JSON 字符串。

        Args:
            response: LLM 响应文本

        Returns:
            str: 提取的 JSON 字符串，失败返回 None
        """
        # 尝试直接解析
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            pass

        # 尝试提取代码块中的 JSON
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\[\s*\{[\s\S]*\}\s*\]',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                potential_json = match.group(1) if match.lastindex else match.group(0)
                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    continue

        return None


def create_memory_manager(
    storage_path: str = "./dm_agent/data/memory",
    llm_client: Optional[BaseLLMClient] = None,
    config: Optional[Dict[str, Any]] = None,
) -> MemoryManager:
    """
    创建记忆管理器的工厂函数。

    Args:
        storage_path: 记忆存储路径
        llm_client: LLM 客户端（可选）
        config: 配置参数（可选）

    Returns:
        MemoryManager: 配置好的记忆管理器实例
    """
    return MemoryManager(
        memory_store=None,  # 内部创建
        llm_client=llm_client,
        context_compressor=None,
        config=config or {},
        storage_path=storage_path,
    )
