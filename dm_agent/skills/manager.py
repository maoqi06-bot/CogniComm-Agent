"""技能管理器"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from .builtin.base_rag_skill import BaseRAGSkill
except ImportError:
    BaseRAGSkill = None
from .base import BaseSkill, ConfigSkill
from .selector import SkillSelector

if TYPE_CHECKING:
    from ..clients.base_client import BaseLLMClient
    from ..tools.base import Tool


class SkillManager:
    """技能管理器，负责技能的注册、加载、选择和激活。

    仿照 MCPManager 模式设计。
    """

    def __init__(
        self,
        *,
        mcp_manager: Optional["MCPManager"] = None,  # [新增] 接收管理器
        max_active_skills: int = 3,
        min_keyword_score: float = 0.05,
        enable_llm_fallback: bool = False,
        llm_client: Optional["BaseLLMClient"] = None,
    ) -> None:
        self.mcp_manager = mcp_manager
        self.skills: Dict[str, BaseSkill] = {}
        self.active_skills: List[str] = []
        self._selector = SkillSelector(
            max_active_skills=max_active_skills,
            min_keyword_score=min_keyword_score,
            enable_llm_fallback=enable_llm_fallback,
            llm_client=llm_client,
        )

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load_builtin_skills(self) -> int:
        """从 builtin 包加载内置技能，返回加载数量。"""
        from .builtin import get_builtin_skills

        count = 0
        for skill in get_builtin_skills():
            meta = skill.get_metadata()
            self.skills[meta.name] = skill
            count += 1
        return count

    def load_custom_skills(self, directory: str | Path | None = None) -> int:
        """从 JSON 加载技能（兼容 ConfigSkill 与 BaseRAGSkill）。

        注意：带有 skill_name 字段的 JSON 由 GenericMCPSkill (load_custom_skills_method2) 处理，
        此方法只处理旧格式（使用 name 字段）的自定义技能。
        """
        if directory is None:
            directory = Path(__file__).parent / "custom"
        else:
            directory = Path(directory)

        if not directory.is_dir(): return 0

        count = 0
        for json_file in sorted(directory.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # 跳过 GenericMCPSkill 格式的 JSON（使用 skill_name 字段）
                if "skill_name" in config:
                    continue

                # 关键：根据 type 字段决定实例化哪个类
                if config.get("type") == "rag" and BaseRAGSkill is not None:
                    skill = BaseRAGSkill(config=config)
                elif config.get("type") == "rag":
                    continue
                else:
                    skill = ConfigSkill(config)  # 原有的工具调用技能

                meta = skill.get_metadata()
                self.skills[meta.name] = skill
                count += 1
            except Exception as e:
                print(f"⚠ 加载 {json_file.name} 失败: {e}")
        return count

    def load_custom_skills_method2(self, directory: str | Path | None = None) -> int:
        """
        [Method 2] 纯配置驱动：读取 JSON 并通过 GenericMCPSkill 实例化。
        """
        if directory is None:
            directory = Path(__file__).parent / "custom"
        directory = Path(directory)

        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
            return 0

        count = 0
        # 导入刚才写的通用类
        from .builtin.generic_mcp_skill import GenericMCPSkill

        for json_file in sorted(directory.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # 实例化并传入 mcp_manager (假设实例已持有)
                skill = GenericMCPSkill(config=config, mcp_manager=self.mcp_manager)

                meta = skill.get_metadata()
                self.skills[meta.name] = skill
                count += 1
                print(f"✅ 成功加载 MCP 技能配置: {meta.display_name}")
            except Exception as e:
                print(f"❌ 加载 {json_file.name} 失败: {e}")
        return count

    def load_all(self) -> int:
        """加载全部技能（内置 + 自定义），返回总数。"""
        builtin_count = self.load_builtin_skills()
        custom_count = self.load_custom_skills_method2() + self.load_custom_skills()  # MCP挂载和Agent系统内置两种挂在方法
        total = builtin_count + custom_count

        # --- 自动生成数据库的关键逻辑 ---
        print(f"🔍 正在扫描 {total} 个技能的索引状态...")
        for name, skill in self.skills.items():
            # 如果是 BaseRAGSkill 或其子类
            if BaseRAGSkill is not None and isinstance(skill, BaseRAGSkill):
                # 检查目录是否存在
                if not skill.builtin_dir.exists():
                    print(f"⚠️ 警告: {name} 的原始数据目录不存在: {skill.builtin_dir}")
                    continue

                # 触发初始化 (内部会判断 index 是否存在，不存在则解析文档生成向量库)
                success = skill._ensure_initialized()
                if success:
                    stats = skill._vector_store.get_stats()
                    print(f"✅ 技能 [{name}] 索引就绪 (包含 {stats['total_chunks']} 个知识分块)")

        return total

    # ------------------------------------------------------------------
    # 选择与激活
    # ------------------------------------------------------------------

    def select_skills_for_task(self, task: str) -> List[str]:
        """根据任务自动选择技能，返回选中的技能名称列表。"""
        return self._selector.select(task, self.skills)

    def activate_skills(self, names: List[str]) -> None:
        """激活指定技能。"""
        self.deactivate_all()
        for name in names:
            skill = self.skills.get(name)
            if skill:
                skill.on_activate()
                self.active_skills.append(name)

    def deactivate_all(self) -> None:
        """停用所有已激活的技能。"""
        for name in self.active_skills:
            skill = self.skills.get(name)
            if skill:
                skill.on_deactivate()
        self.active_skills = []

    # ------------------------------------------------------------------
    # 获取激活技能的内容
    # ------------------------------------------------------------------

    def get_active_prompt_additions(self) -> str:
        """返回所有激活技能的 prompt 合并文本。"""
        parts: List[str] = []
        for name in self.active_skills:
            skill = self.skills.get(name)
            if skill:
                addition = skill.get_prompt_addition()
                if addition:
                    meta = skill.get_metadata()
                    parts.append(f"\n\n## 专家技能：{meta.display_name}\n{addition}")
        return "".join(parts)

    def get_active_tools(self) -> List["Tool"]:
        """返回所有激活技能的工具合并列表。"""
        tools: List["Tool"] = []
        for name in self.active_skills:
            skill = self.skills.get(name)
            if skill:
                tools.extend(skill.get_tools())
        return tools

    # ------------------------------------------------------------------
    # 信息查询
    # ------------------------------------------------------------------

    def get_all_skill_info(self) -> List[Dict[str, Any]]:
        """返回所有技能摘要信息。"""
        info_list: List[Dict[str, Any]] = []
        for name, skill in self.skills.items():
            meta = skill.get_metadata()
            tools = skill.get_tools()
            info_list.append({
                "name": meta.name,
                "display_name": meta.display_name,
                "description": meta.description,
                "keywords": meta.keywords,
                "priority": meta.priority,
                "version": meta.version,
                "tools_count": len(tools),
                "is_active": name in self.active_skills,
                "is_builtin": not isinstance(skill, ConfigSkill),
            })
        return info_list
