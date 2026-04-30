"""记忆管理工具 - 供 Agent 使用的长期记忆操作工具。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from ..tools.base import Tool
from .long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority
from .memory_manager import MemoryManager, MemoryRetrievalResult


class AddMemoryTool(Tool):
    """
    添加长期记忆工具。

    当 Agent 需要记住重要信息（如用户偏好、项目约束、关键决策）时使用。
    添加后的记忆可以被后续对话检索。

    使用场景：
    - 用户明确说明的偏好（如"我更喜欢用 pytest"）
    - 重要的项目约束（如"这个项目必须用 Python 3.9"）
    - 关键决策（如"我们决定用 SQLite 作为数据库"）
    """

    name = "add_memory"
    description = """添加一条长期记忆。Agent 可以使用此工具记住重要的用户偏好、项目约束、关键决策等信息。

参数：
- content (必需): 要记住的内容，应简洁明确，1-2 句话最佳
- category (必需): 记忆类别，可选值：
  * user_preference - 用户偏好（如喜欢的工具、编码风格）
  * project_context - 项目上下文（如项目名称、技术栈）
  * important_fact - 重要事实或约束条件
  * skill_knowledge - 技能知识或经验
  * working_state - 当前工作状态
  * conversation_summary - 对话摘要
- importance_score (可选): 重要性评分 0.0-1.0，默认 0.5
- tags (可选): 标签列表，帮助检索
- is_pinned (可选): 是否固定，固定记忆不会被自动清理
"""

    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self.memory_manager = memory_manager

    def execute(self, action_input: Dict[str, Any]) -> str:
        try:
            # 解析参数
            content = action_input.get("content", "")
            if not content:
                return "错误：必须提供记忆内容（content）"

            category_str = action_input.get("category", "conversation_summary")
            try:
                category = MemoryCategory(category_str)
            except ValueError:
                valid_cats = [c.value for c in MemoryCategory]
                return f"错误：无效的类别 '{category_str}'，可选值：{valid_cats}"

            importance_score = float(action_input.get("importance_score", 0.5))
            importance_score = max(0.0, min(1.0, importance_score))

            tags = set(action_input.get("tags", []))
            if isinstance(tags, list):
                tags = set(tags)

            is_pinned = bool(action_input.get("is_pinned", False))

            # 添加记忆
            entry = self.memory_manager.add_memory(
                content=content,
                category=category,
                importance_score=importance_score,
                tags=tags,
                is_pinned=is_pinned,
                source="agent_tool",
            )

            return f"✅ 已添加记忆 [ID: {entry.id}]\n类别: {entry.category.value}\n内容: {entry.content}\n重要性: {entry.importance_score:.2f}"

        except Exception as e:
            return f"❌ 添加记忆失败: {e}"


class SearchMemoryTool(Tool):
    """
    搜索长期记忆工具。

    Agent 在执行任务时可以检索相关记忆，增强对用户偏好和项目上下文的理解。

    使用场景：
    - 开始新任务前检索相关上下文
    - 查询用户是否有特定偏好
    - 了解项目的技术栈和约束
    """

    name = "search_memory"
    description = """搜索长期记忆。根据查询文本在记忆库中检索相关内容。

参数：
- query (必需): 搜索查询，应描述想要查找的记忆主题
- category (可选): 按类别过滤，可选值同 add_memory
- limit (可选): 返回结果数量，默认 5
"""

    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self.memory_manager = memory_manager

    def execute(self, action_input: Dict[str, Any]) -> str:
        try:
            query = action_input.get("query", "")
            if not query:
                return "错误：必须提供搜索查询（query）"

            category_str = action_input.get("category")
            category = None
            if category_str:
                try:
                    category = MemoryCategory(category_str)
                except ValueError:
                    valid_cats = [c.value for c in MemoryCategory]
                    return f"错误：无效的类别 '{category_str}'，可选值：{valid_cats}"

            limit = int(action_input.get("limit", 5))
            limit = max(1, min(20, limit))

            # 检索记忆
            results = self.memory_manager.memory_store.search(
                query=query,
                category=category,
                limit=limit,
                include_decay=True,
            )

            # 过滤低价值操作记忆
            if hasattr(self.memory_manager, '_is_low_value_operational_memory'):
                filtered_results = []
                seen_ids = set()
                for result in results:
                    if result.entry.id in seen_ids:
                        continue
                    if self.memory_manager._is_low_value_operational_memory(result.entry):
                        continue
                    seen_ids.add(result.entry.id)
                    filtered_results.append(result)
                results = filtered_results

            if not results:
                return f"未找到与 '{query}' 相关的记忆"

            # 格式化输出
            lines = [f"🔍 找到 {len(results)} 条相关记忆：\n"]
            for i, result in enumerate(results, 1):
                entry = result.entry
                lines.append(f"{i}. [{entry.category.value}] {entry.content}")
                lines.append(f"   重要性: {result.score:.2f} | 访问: {entry.access_count}次")
                if entry.tags:
                    lines.append(f"   标签: {', '.join(list(entry.tags)[:3])}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ 搜索记忆失败: {e}"


class UpdateMemoryTool(Tool):
    """
    更新长期记忆工具。

    用于更新记忆的重要性、标签等内容，或标记记忆为固定/取消固定。

    使用场景：
    - 用户反馈某偏好很重要，需要提升重要性
    - 任务完成后更新相关记忆的重要性
    - 标记重要记忆为固定
    """

    name = "update_memory"
    description = """更新已存在的长期记忆。

参数：
- memory_id (必需): 要更新的记忆 ID
- content (可选): 新内容
- importance_score (可选): 新的重要性评分 0.0-1.0
- tags (可选): 新的标签列表
- is_pinned (可选): 是否固定
"""

    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self.memory_manager = memory_manager

    def execute(self, action_input: Dict[str, Any]) -> str:
        try:
            memory_id = action_input.get("memory_id", "")
            if not memory_id:
                return "错误：必须提供记忆 ID（memory_id）"

            # 获取现有记忆
            existing = self.memory_manager.memory_store.get(memory_id)
            if not existing:
                return f"错误：未找到 ID 为 '{memory_id}' 的记忆"

            # 构建更新参数
            update_kwargs = {}
            if "content" in action_input:
                update_kwargs["content"] = action_input["content"]
            if "importance_score" in action_input:
                score = float(action_input["importance_score"])
                update_kwargs["importance_score"] = max(0.0, min(1.0, score))
            if "tags" in action_input:
                tags = action_input["tags"]
                if isinstance(tags, list):
                    tags = set(tags)
                update_kwargs["tags"] = tags
            if "is_pinned" in action_input:
                update_kwargs["is_pinned"] = bool(action_input["is_pinned"])

            if not update_kwargs:
                return "错误：未提供任何要更新的内容"

            # 执行更新
            updated = self.memory_manager.memory_store.update(memory_id, **update_kwargs)
            if not updated:
                return "❌ 更新失败"

            return f"✅ 已更新记忆 [ID: {memory_id}]\n新内容: {updated.content}\n重要性: {updated.importance_score:.2f}\n固定: {'是' if updated.is_pinned else '否'}"

        except Exception as e:
            return f"❌ 更新记忆失败: {e}"


class DeleteMemoryTool(Tool):
    """
    删除长期记忆工具。

    用于删除不再需要的记忆或过时的信息。

    使用场景：
    - 用户明确要求删除某条记忆
    - 发现记忆内容已过时或不准确
    """

    name = "delete_memory"
    description = """删除一条长期记忆。

参数：
- memory_id (必需): 要删除的记忆 ID
"""

    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self.memory_manager = memory_manager

    def execute(self, action_input: Dict[str, Any]) -> str:
        try:
            memory_id = action_input.get("memory_id", "")
            if not memory_id:
                return "错误：必须提供记忆 ID（memory_id）"

            # 获取记忆内容用于确认
            existing = self.memory_manager.memory_store.get(memory_id, increment_access=False)
            if not existing:
                return f"错误：未找到 ID 为 '{memory_id}' 的记忆"

            content_preview = existing.content[:50] + "..." if len(existing.content) > 50 else existing.content

            # 执行删除
            success = self.memory_manager.memory_store.delete(memory_id)
            if success:
                return f"✅ 已删除记忆 [ID: {memory_id}]\n内容预览: {content_preview}"
            else:
                return "❌ 删除失败"

        except Exception as e:
            return f"❌ 删除记忆失败: {e}"


class ListMemoriesTool(Tool):
    """
    列出记忆工具。

    查看当前记忆库中的记忆，支持按类别和优先级筛选。
    """

    name = "list_memories"
    description = """列出记忆库中的记忆。

参数：
- category (可选): 按类别过滤
- limit (可选): 返回数量限制，默认 20
- sort_by (可选): 排序方式，可选 'importance'（默认）、'recent'、'access'
"""

    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self.memory_manager = memory_manager

    def execute(self, action_input: Dict[str, Any]) -> str:
        try:
            category_str = action_input.get("category")
            category = None
            if category_str:
                try:
                    category = MemoryCategory(category_str)
                except ValueError:
                    valid_cats = [c.value for c in MemoryCategory]
                    return f"错误：无效的类别 '{category_str}'，可选值：{valid_cats}"

            limit = int(action_input.get("limit", 20))
            limit = max(1, min(100, limit))

            sort_by = action_input.get("sort_by", "importance")

            # 获取记忆
            if category:
                results = self.memory_manager.memory_store.get_by_category(
                    category, limit=limit
                )
            else:
                # 全量获取并排序
                all_entries = list(self.memory_manager.memory_store._memory_index.values())

                if sort_by == "importance":
                    all_entries.sort(key=lambda x: x.importance_score, reverse=True)
                elif sort_by == "recent":
                    all_entries.sort(key=lambda x: x.updated_at, reverse=True)
                elif sort_by == "access":
                    all_entries.sort(key=lambda x: x.access_count, reverse=True)

                results = all_entries[:limit]

            if not results:
                category_name = category.value if category else "所有"
                return f"记忆库为空（类别：{category_name}）"

            # 格式化输出
            lines = [f"📚 当前记忆库（{len(results)} 条）：\n"]
            for i, result in enumerate(results, 1):
                if hasattr(result, "entry"):
                    entry = result.entry
                else:
                    entry = result

                lines.append(f"{i}. [ID: {entry.id[:8]}...]")
                lines.append(f"   [{entry.category.value}] {entry.content[:60]}...")
                lines.append(f"   重要性: {entry.importance_score:.2f} | 固定: {'是' if entry.is_pinned else '否'}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ 列出记忆失败: {e}"


class GetMemoryStatsTool(Tool):
    """
    获取记忆统计信息工具。

    查看记忆库的统计信息，包括各类别数量、总容量使用情况等。
    """

    name = "get_memory_stats"
    description = """获取记忆系统的统计信息。

无需参数，返回记忆库的统计摘要。
"""

    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self.memory_manager = memory_manager

    def execute(self, action_input: Dict[str, Any]) -> str:
        try:
            stats = self.memory_manager.get_statistics()
            store = stats["store"]
            session = stats["session"]

            lines = ["📊 记忆系统统计：\n"]
            lines.append(f"总记忆数: {store['total_memories']} / {store['max_capacity']}")
            lines.append(f"容量使用率: {store['utilization_rate']:.1%}")
            lines.append(f"固定记忆: {store['pinned_count']} 条")
            lines.append(f"平均重要性: {store['avg_importance']:.2f}")
            lines.append(f"平均访问次数: {store['avg_access_count']:.1f}")
            lines.append("")

            lines.append("📂 类别分布：")
            for cat, count in store["category_distribution"].items():
                lines.append(f"  - {cat}: {count} 条")

            lines.append("")
            lines.append(f"🔧 本会话添加: {session['memories_added']} 条")
            lines.append(f"⚙️ 自动提取: {'已启用' if session['auto_extract_enabled'] else '已禁用'}")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ 获取统计失败: {e}"


def create_memory_tools(memory_manager: MemoryManager) -> List[Tool]:
    """
    创建所有记忆管理工具。

    Args:
        memory_manager: 记忆管理器实例

    Returns:
        List[Tool]: 工具列表
    """
    return [
        AddMemoryTool(memory_manager),
        SearchMemoryTool(memory_manager),
        UpdateMemoryTool(memory_manager),
        DeleteMemoryTool(memory_manager),
        ListMemoriesTool(memory_manager),
        GetMemoryStatsTool(memory_manager),
    ]


def get_memory_tool_names() -> List[str]:
    """获取所有记忆工具的名称列表。"""
    return [
        "add_memory",
        "search_memory",
        "update_memory",
        "delete_memory",
        "list_memories",
        "get_memory_stats",
    ]
