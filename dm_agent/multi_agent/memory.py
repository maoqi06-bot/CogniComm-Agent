"""Memory coordination for multi-agent execution.

This module does not replace the single-agent memory system. It provides a
thin coordination layer for multi-agent runs:

- shared short-term task memory across all sub-agents
- private short-term working memory per sub-agent
- optional bridge to the existing long-term MemoryManager
"""

from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentMemoryPolicy:
    """Read/write policy for one multi-agent role."""

    read_long_term: bool = True
    write_long_term: bool = False
    write_shared_short_term: bool = True
    write_private_short_term: bool = True
    write_task_summary: bool = False
    allowed_long_term_kinds: List[str] = field(default_factory=list)


@dataclass
class MemoryWriteTemplate:
    """Long-term memory write template for a domain or memory kind."""

    category: str = "conversation_summary"
    priority: str = "normal"
    importance_score: float = 0.55
    tags: List[str] = field(default_factory=list)
    source: str = "multi_agent"
    title: str = "Multi-agent memory"


@dataclass
class MultiAgentMemoryConfig:
    enabled: bool = True
    long_term_enabled: bool = True
    auto_store_task_summary: bool = True
    async_long_term_writes: bool = True
    shared_recent_limit: int = 12
    private_recent_limit: int = 6
    context_char_limit: int = 6000
    timeline_path: str = "./data/multi_agent_memory_timeline.jsonl"
    approval_path: str = "./data/multi_agent_memory_approvals.json"
    record_long_term_lookup_events: bool = True
    human_approval_required_kinds: List[str] = field(
        default_factory=lambda: ["engineering_experience"]
    )
    engineering_experience_min_chars: int = 160
    user_preference_min_chars: int = 12
    user_preference_lookup_limit: int = 5
    user_preference_max_items: int = 2
    user_preference_min_score: float = 0.62
    user_preference_context_char_limit: int = 1000
    engineering_experience_keywords: List[str] = field(default_factory=lambda: [
        "code",
        "file",
        "test",
        "pytest",
        "python",
        "module",
        "function",
        "implemented",
        "created",
        "modified",
        "validated",
        "passed",
        "代码",
        "文件",
        "测试",
        "实现",
        "修改",
        "运行",
        "通过",
    ])
    user_preference_keywords: List[str] = field(default_factory=lambda: [
        "prefer",
        "preferred",
        "preference",
        "like",
        "usually",
        "always",
        "default",
        "markdown",
        "report",
        "structured",
        "code",
        "python",
        "test",
        "pytest",
        "rag",
        "mcp",
        "simulation",
        "wireless",
        "isac",
        "ris",
        "hbf",
        "偏好",
        "喜欢",
        "习惯",
        "默认",
        "结构化",
        "报告",
        "文档",
        "代码",
        "测试",
        "仿真",
        "检索",
        "无线",
        "通感",
    ])
    policies: Dict[str, AgentMemoryPolicy] = field(default_factory=dict)
    write_templates: Dict[str, MemoryWriteTemplate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        default_policies = {
            "orchestrator": AgentMemoryPolicy(
                read_long_term=True,
                write_long_term=True,
                write_task_summary=True,
                allowed_long_term_kinds=["task_summary", "user_preference"],
            ),
            "rag_agent": AgentMemoryPolicy(
                read_long_term=True,
                write_long_term=False,
                allowed_long_term_kinds=[],
            ),
            "code_agent": AgentMemoryPolicy(
                read_long_term=True,
                write_long_term=True,
                allowed_long_term_kinds=[
                    "engineering_experience",
                    "debugging_lesson",
                    "implementation_pattern",
                    "simulation_result",
                ],
            ),
        }
        for name, policy in default_policies.items():
            self.policies.setdefault(name, policy)

        default_templates = {
            "task_summary": MemoryWriteTemplate(
                category="conversation_summary",
                priority="normal",
                importance_score=0.55,
                tags=["multi_agent", "task_summary"],
                title="Multi-agent task summary",
            ),
            "user_preference": MemoryWriteTemplate(
                category="user_preference",
                priority="high",
                importance_score=0.72,
                tags=["multi_agent", "user_preference"],
                title="User preference",
            ),
            "engineering_experience": MemoryWriteTemplate(
                category="skill_knowledge",
                priority="normal",
                importance_score=0.6,
                tags=["multi_agent", "code_agent", "engineering"],
                title="CodeAgent engineering experience",
            ),
            "research_note": MemoryWriteTemplate(
                category="skill_knowledge",
                priority="normal",
                importance_score=0.55,
                tags=["multi_agent", "rag_agent", "research"],
                title="RAGAgent research note",
            ),
            "debugging_lesson": MemoryWriteTemplate(
                category="skill_knowledge",
                priority="normal",
                importance_score=0.62,
                tags=["multi_agent", "code_agent", "debugging"],
                title="CodeAgent debugging lesson",
            ),
            "implementation_pattern": MemoryWriteTemplate(
                category="skill_knowledge",
                priority="normal",
                importance_score=0.64,
                tags=["multi_agent", "code_agent", "implementation_pattern"],
                title="CodeAgent implementation pattern",
            ),
            "simulation_result": MemoryWriteTemplate(
                category="skill_knowledge",
                priority="normal",
                importance_score=0.58,
                tags=["multi_agent", "code_agent", "simulation"],
                title="CodeAgent simulation result",
            ),
        }
        for name, template in default_templates.items():
            self.write_templates.setdefault(name, template)


@dataclass
class MemoryEvent:
    agent_name: str
    kind: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class MultiAgentMemoryHub:
    """Task-scoped memory hub shared by Orchestrator, RAGAgent, and CodeAgent."""

    def __init__(
        self,
        *,
        memory_manager: Optional[Any] = None,
        config: Optional[MultiAgentMemoryConfig] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.memory_manager = memory_manager
        self.config = config or MultiAgentMemoryConfig()
        self.logger = logger
        self._shared_events: List[MemoryEvent] = []
        self._private_events: Dict[str, List[MemoryEvent]] = {}
        self._current_task_id: Optional[str] = None
        self._current_task: str = ""
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="multi-agent-memory")
            if self.config.async_long_term_writes
            else None
        )

    def start_task(self, task_id: str, task: str) -> None:
        self._current_task_id = task_id
        self._current_task = task
        self._shared_events.clear()
        self._private_events.clear()
        self.add_event(
            "orchestrator",
            "task_start",
            task,
            shared=True,
            metadata={"task_id": task_id},
        )
        self._capture_user_preferences(task)

    def add_event(
        self,
        agent_name: str,
        kind: str,
        content: str,
        *,
        shared: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.config.enabled or not content:
            return
        policy = self._policy_for(agent_name)
        if shared and not policy.write_shared_short_term:
            return
        if not shared and not policy.write_private_short_term:
            return
        event = MemoryEvent(
            agent_name=agent_name,
            kind=kind,
            content=str(content)[: self.config.context_char_limit],
            metadata=metadata or {},
        )
        if shared:
            self._shared_events.append(event)
            self._shared_events = self._shared_events[-self.config.shared_recent_limit :]
        else:
            bucket = self._private_events.setdefault(agent_name, [])
            bucket.append(event)
            self._private_events[agent_name] = bucket[-self.config.private_recent_limit :]
        self._persist_event(event, shared=shared)

    def build_context(self, task: str, *, agent_name: Optional[str] = None) -> str:
        """Build short-term plus optional long-term memory context for an agent."""

        if not self.config.enabled:
            return ""

        sections: List[str] = []
        effective_agent = agent_name or "orchestrator"
        policy = self._policy_for(effective_agent)
        long_term_bundle = self._retrieve_long_term_bundle(task) if policy.read_long_term else {}
        long_term = str(long_term_bundle.get("general_context", "") or "")
        preference_text = str(long_term_bundle.get("preference_context", "") or "")
        if policy.read_long_term:
            self._record_long_term_lookup(
                agent_name=effective_agent,
                task=task,
                hit=bool(long_term or preference_text),
                context_chars=len(long_term) + len(preference_text),
                preference_hit=bool(preference_text),
                preference_count=int(long_term_bundle.get("preference_count", 0) or 0),
            )
        if preference_text:
            sections.append("### Relevant user preferences\n" + preference_text)
        if long_term:
            sections.append("### Long-term memory\n" + long_term)

        shared = self._format_events(self._shared_events[-self.config.shared_recent_limit :])
        if shared:
            sections.append("### Shared task memory\n" + shared)

        if agent_name:
            private = self._format_events(
                self._private_events.get(agent_name, [])[-self.config.private_recent_limit :]
            )
            if private:
                sections.append(f"### Private working memory for {agent_name}\n" + private)

        context = "\n\n".join(sections)
        if len(context) > self.config.context_char_limit:
            context = context[: self.config.context_char_limit] + "\n...[memory context truncated]"
        return context

    def record_subtask_result(self, task: Any) -> None:
        summary = self._summarize_result(getattr(task, "result", None))
        if not summary:
            return
        self.add_event(
            getattr(task, "agent_name", "") or "unknown_agent",
            "subtask_result",
            f"{getattr(task, 'id', 'subtask')} [{getattr(getattr(task, 'type', None), 'value', '')}]: {summary}",
            shared=True,
            metadata={
                "subtask_id": getattr(task, "id", None),
                "status": getattr(task, "status", None),
            },
        )

    def store_task_summary(
        self,
        *,
        original_task: str,
        final_answer: str,
        completed_count: int,
        failed_count: int,
    ) -> None:
        if (
            not self.config.enabled
            or not self.config.long_term_enabled
            or not self.config.auto_store_task_summary
            or not self.memory_manager
        ):
            return
        policy = self._policy_for("orchestrator")
        if not policy.write_long_term or not policy.write_task_summary:
            return
        content = (
            f"Multi-agent task summary: {original_task}\n"
            f"Completed subtasks: {completed_count}, failed subtasks: {failed_count}.\n"
            f"Final answer preview: {final_answer[:1200]}"
        )
        self.store_agent_memory(
            "orchestrator",
            "task_summary",
            content,
            metadata={
                "task_id": self._current_task_id,
                "completed_count": completed_count,
                "failed_count": failed_count,
            },
        )

    def store_agent_memory(
        self,
        agent_name: str,
        memory_kind: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a role-approved long-term memory item."""

        if (
            not self.config.enabled
            or not self.config.long_term_enabled
            or not self.memory_manager
            or not content
        ):
            return
        policy = self._policy_for(agent_name)
        if not policy.write_long_term or memory_kind not in policy.allowed_long_term_kinds:
            return
        if not self._passes_quality_filter(memory_kind, content, metadata or {}):
            self._persist_event(
                MemoryEvent(
                    agent_name=agent_name,
                    kind="long_term_write_skipped",
                    content=f"Skipped low-value {memory_kind} memory.",
                    metadata={"memory_kind": memory_kind, "reason": "quality_filter"},
                ),
                shared=True,
            )
            return
        template = self.config.write_templates.get(memory_kind, MemoryWriteTemplate())
        write_metadata = dict(metadata or {})
        write_metadata.update(
            {
                "task_id": self._current_task_id,
                "agent_name": agent_name,
                "memory_kind": memory_kind,
            }
        )
        body = f"{template.title}: {content}"
        if memory_kind in self.config.human_approval_required_kinds:
            approval_id = self._queue_approval(
                agent_name=agent_name,
                memory_kind=memory_kind,
                content=body,
                template=template,
                metadata=write_metadata,
            )
            self._persist_event(
                MemoryEvent(
                    agent_name=agent_name,
                    kind="long_term_write_pending_approval",
                    content=f"Queued {memory_kind} memory for human approval.",
                    metadata={
                        "memory_kind": memory_kind,
                        "approval_id": approval_id,
                    },
                ),
                shared=True,
            )
            return
        self._submit_long_term_write(template, body, write_metadata)

    def _write_long_term(
        self,
        template: MemoryWriteTemplate,
        content: str,
        metadata: Dict[str, Any],
    ) -> None:
        try:
            from ..memory.long_term_memory import MemoryCategory, MemoryPriority

            self.memory_manager.add_memory(
                content=content,
                category=MemoryCategory(template.category),
                priority=MemoryPriority[template.priority.upper()],
                importance_score=template.importance_score,
                tags=set(template.tags),
                metadata=metadata,
                source=template.source,
            )
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"Failed to store multi-agent memory summary: {exc}")

    def _retrieve_long_term(self, task: str) -> str:
        if not self.config.long_term_enabled or not self.memory_manager:
            return ""
        try:
            result = self.memory_manager.retrieve_for_context(current_task=task)
            return getattr(result, "enhanced_context", "") or ""
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"Multi-agent memory retrieval failed: {exc}")
            return ""

    def _retrieve_long_term_bundle(self, task: str) -> Dict[str, Any]:
        if not self.config.long_term_enabled or not self.memory_manager:
            return {}
        try:
            from ..memory.long_term_memory import MemoryCategory

            general_result = self.memory_manager.retrieve_for_context(
                current_task=task,
                categories=[
                    MemoryCategory.PROJECT_CONTEXT,
                    MemoryCategory.IMPORTANT_FACT,
                    MemoryCategory.WORKING_STATE,
                    MemoryCategory.SKILL_KNOWLEDGE,
                    MemoryCategory.CONVERSATION_SUMMARY,
                ],
            )
            preference_results = self._retrieve_relevant_preferences(task)
            return {
                "general_context": getattr(general_result, "enhanced_context", "") or "",
                "preference_context": self._format_preference_context(preference_results),
                "preference_count": len(preference_results),
            }
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"Multi-agent memory bundle retrieval failed: {exc}")
            return {
                "general_context": self._retrieve_long_term(task),
                "preference_context": "",
                "preference_count": 0,
            }

    def _retrieve_relevant_preferences(self, task: str) -> List[Any]:
        try:
            from ..memory.long_term_memory import MemoryCategory
        except Exception:
            return []

        store = getattr(self.memory_manager, "memory_store", None)
        if not store:
            return []

        try:
            raw_results = store.search(
                task,
                category=MemoryCategory.USER_PREFERENCE,
                limit=max(1, self.config.user_preference_lookup_limit),
                include_decay=False,
            )
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"User preference search failed: {exc}")
            return []

        filtered = [result for result in raw_results if self._is_relevant_preference(task, result)]
        filtered.sort(key=lambda item: item.score, reverse=True)
        return filtered[: self.config.user_preference_max_items]

    def _is_relevant_preference(self, task: str, result: Any) -> bool:
        score = float(getattr(result, "score", 0.0) or 0.0)
        if score >= self.config.user_preference_min_score:
            return True

        task_tokens = self._extract_preference_keywords(task)
        if not task_tokens:
            return False

        entry = getattr(result, "entry", None)
        if not entry:
            return False

        pref_tokens = self._extract_preference_keywords(getattr(entry, "content", ""))
        entry_tags = {
            str(tag).strip().lower()
            for tag in (getattr(entry, "tags", None) or set())
            if str(tag).strip()
        }
        overlap = task_tokens & (pref_tokens | entry_tags)
        return bool(overlap) and score >= max(0.35, self.config.user_preference_min_score - 0.18)

    def _extract_preference_keywords(self, text: str) -> set[str]:
        lowered = str(text or "").lower()
        tokens = set(re.findall(r"[a-z0-9_+-]{3,}", lowered))
        for keyword in self.config.user_preference_keywords:
            key = str(keyword).strip().lower()
            if key and key in lowered:
                tokens.add(key)
        return tokens

    def _format_preference_context(self, results: List[Any]) -> str:
        if not results:
            return ""
        lines: List[str] = []
        for result in results:
            entry = getattr(result, "entry", None)
            if not entry:
                continue
            content = str(getattr(entry, "content", "") or "").strip()
            if not content:
                continue
            lines.append(f"- {content}")
            lines.append(f"  [relevance={float(getattr(result, 'score', 0.0) or 0.0):.2f}]")
        text = "\n".join(lines).strip()
        if len(text) > self.config.user_preference_context_char_limit:
            text = text[: self.config.user_preference_context_char_limit] + "\n...[preferences truncated]"
        return text

    def _record_long_term_lookup(
        self,
        *,
        agent_name: str,
        task: str,
        hit: bool,
        context_chars: int,
        preference_hit: bool = False,
        preference_count: int = 0,
    ) -> None:
        if not self.config.record_long_term_lookup_events:
            return
        kind = "long_term_memory_hit" if hit else "long_term_memory_miss"
        self._persist_event(
            MemoryEvent(
                agent_name=agent_name,
                kind=kind,
                content=(
                    "Long-term memory lookup hit."
                    if hit
                    else "Long-term memory lookup missed."
                ),
                metadata={
                    "query_preview": str(task or "")[:300],
                    "context_chars": context_chars,
                    "preference_hit": preference_hit,
                    "preference_count": preference_count,
                },
            ),
            shared=False,
        )
        preference_kind = (
            "user_preference_memory_hit" if preference_hit else "user_preference_memory_miss"
        )
        self._persist_event(
            MemoryEvent(
                agent_name=agent_name,
                kind=preference_kind,
                content=(
                    "Relevant user preference memory matched."
                    if preference_hit
                    else "No relevant user preference memory matched."
                ),
                metadata={
                    "query_preview": str(task or "")[:300],
                    "preference_count": preference_count,
                },
            ),
            shared=False,
        )

    def build_replay(self, task_id: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        """Build a replay payload for debugging memory usage."""

        selected_task_id = task_id or self._current_task_id
        events = self.load_timeline(task_id=selected_task_id, limit=limit)
        return {
            "task_id": selected_task_id,
            "current_task": self._current_task if selected_task_id == self._current_task_id else "",
            "event_count": len(events),
            "events": events,
            "policies": {
                name: {
                    "read_long_term": policy.read_long_term,
                    "write_long_term": policy.write_long_term,
                    "write_shared_short_term": policy.write_shared_short_term,
                    "write_private_short_term": policy.write_private_short_term,
                    "write_task_summary": policy.write_task_summary,
                    "allowed_long_term_kinds": list(policy.allowed_long_term_kinds),
                }
                for name, policy in self.config.policies.items()
            },
        }

    def load_timeline(self, task_id: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        path = Path(self.config.timeline_path)
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if task_id and row.get("task_id") != task_id:
                        continue
                    rows.append(row)
            return rows[-limit:]
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"Failed to load multi-agent memory timeline: {exc}")
            return []

    def _passes_quality_filter(
        self,
        memory_kind: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> bool:
        if memory_kind == "user_preference":
            return self._is_explicit_preference(str(content or ""))
        if memory_kind != "engineering_experience":
            return True
        if metadata.get("cancelled"):
            return False
        text = str(content or "").strip()
        if len(text) < self.config.engineering_experience_min_chars:
            return False
        lowered = text.lower()
        weak_markers = [
            "not initialized",
            "failed",
            "error",
            "cancelled",
            "无法完成",
            "未完成",
            "失败",
            "报错",
        ]
        if any(marker in lowered for marker in weak_markers):
            return False
        return any(keyword.lower() in lowered for keyword in self.config.engineering_experience_keywords)

    def _capture_user_preferences(self, task: str) -> None:
        for preference in self._extract_explicit_user_preferences(task):
            self.store_agent_memory(
                "orchestrator",
                "user_preference",
                preference,
                metadata={
                    "task_id": self._current_task_id,
                    "captured_from": "task_start",
                },
            )

    def _extract_explicit_user_preferences(self, text: str) -> List[str]:
        content = str(text or "").strip()
        if len(content) < self.config.user_preference_min_chars:
            return []

        patterns = [
            r"(我(?:更)?喜欢[^。；\n]+)",
            r"(我偏好[^。；\n]+)",
            r"(我习惯[^。；\n]+)",
            r"(请以后[^。；\n]+)",
            r"(默认[^。；\n]+)",
            r"(I\s+(?:prefer|usually|always)\s+[^.\n;]+)",
            r"(Please\s+(?:default|prefer)\s+[^.\n;]+)",
        ]
        hits: List[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, content, flags=re.IGNORECASE):
                snippet = match.group(1).strip(" \t\r\n:：,，。；;")
                normalized = snippet.lower()
                if len(snippet) < self.config.user_preference_min_chars or normalized in seen:
                    continue
                seen.add(normalized)
                hits.append(snippet)
        return hits

    def _is_explicit_preference(self, text: str) -> bool:
        return bool(self._extract_explicit_user_preferences(text))

    def _policy_for(self, agent_name: str) -> AgentMemoryPolicy:
        if agent_name in self.config.policies:
            return self.config.policies[agent_name]
        short_name = agent_name.replace("-", "_").lower()
        if short_name in self.config.policies:
            return self.config.policies[short_name]
        return AgentMemoryPolicy(read_long_term=True)

    def _submit_long_term_write(
        self,
        template: MemoryWriteTemplate,
        content: str,
        metadata: Dict[str, Any],
    ) -> None:
        if self._executor:
            self._executor.submit(self._write_long_term, template, content, metadata)
        else:
            self._write_long_term(template, content, metadata)

    def _queue_approval(
        self,
        *,
        agent_name: str,
        memory_kind: str,
        content: str,
        template: MemoryWriteTemplate,
        metadata: Dict[str, Any],
    ) -> str:
        approval_id = f"mem_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        record = {
            "approval_id": approval_id,
            "status": "pending",
            "created_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_id": self._current_task_id,
            "agent_name": agent_name,
            "memory_kind": memory_kind,
            "content": content,
            "template": {
                "category": template.category,
                "priority": template.priority,
                "importance_score": template.importance_score,
                "tags": list(template.tags),
                "source": template.source,
                "title": template.title,
            },
            "metadata": metadata,
        }
        path = Path(self.config.approval_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records: List[Dict[str, Any]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    records = loaded
            except Exception:
                records = []
        records.append(record)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return approval_id

    def _persist_event(self, event: MemoryEvent, *, shared: bool) -> None:
        try:
            path = Path(self.config.timeline_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "task_id": self._current_task_id,
                "agent_name": event.agent_name,
                "kind": event.kind,
                "content": event.content,
                "metadata": event.metadata,
                "created_at": event.created_at,
                "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.created_at)),
                "shared": shared,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"Failed to persist multi-agent memory event: {exc}")

    def _format_events(self, events: List[MemoryEvent]) -> str:
        lines: List[str] = []
        for event in events:
            lines.append(f"- [{event.agent_name}/{event.kind}] {event.content}")
        return "\n".join(lines)

    def _summarize_result(self, result: Any, max_chars: int = 1600) -> str:
        if result is None:
            return ""
        if isinstance(result, dict):
            text = result.get("answer") or result.get("result") or result.get("message") or str(result)
        else:
            text = str(result)
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "...[truncated]"
        return text
