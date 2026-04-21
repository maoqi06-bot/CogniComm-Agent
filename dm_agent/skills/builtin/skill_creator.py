"""技能创建专家技能"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import BaseSkill, SkillMetadata
from ...tools.base import Tool


class SkillCreatorSkill(BaseSkill):
    """技能创建专家，提供创建和优化Agent技能的指导。"""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="skill_creator",
            display_name="技能创建专家",
            description="提供创建和优化Agent技能的指导。当用户需要创建新技能、更新现有技能或构建可重用能力包时使用。",
            keywords=["skill", "create", "build", "new", "技能", "创建", "构建", "设计", "模板", "guide", "指导", "expert"],
            patterns=[r"create\s+skill", r"new\s+skill", r"build\s+skill", r"技能创建", r"设计技能"],
            priority=5,
            version="1.0.0",
        )

    def get_prompt_addition(self) -> str:
        return (
            "你现在是技能创建专家，擅长设计和实现Agent技能。请遵循以下原则：\n\n"
            "1. **简洁性**：上下文窗口是公共资源。技能与系统提示、对话历史、其他技能元数据和用户请求共享上下文窗口。\n\n"
            "2. **默认假设**：Agent已经非常智能。只添加Agent没有的信息。质疑每条信息：\"Agent真的需要这个解释吗？\" 和 \"这段文字是否值得其token成本？\"\n\n"
            "3. **渐进式披露**：使用三级加载系统高效管理上下文：\n"
            "   - 元数据（名称+描述）- 始终在上下文中（约100字）\n"
            "   - SKILL.md正文 - 技能触发时加载（<5k字）\n"
            "   - 捆绑资源 - 按需加载（脚本可不读入上下文直接执行）\n\n"
            "4. **技能结构**：每个技能包含必需的SKILL.md文件和可选的捆绑资源：\n"
            "   - scripts/ - 可执行代码（Python/Bash等）\n"
            "   - references/ - 文档和参考材料\n"
            "   - assets/ - 输出中使用的文件（模板、图标、字体等）\n\n"
            "5. **设计模式**：\n"
            "   - **工作流模式**：适用于顺序处理流程\n"
            "   - **任务模式**：适用于工具集合\n"
            "   - **参考/指南模式**：适用于标准或规范\n"
            "   - **能力模式**：适用于集成系统\n\n"
            "6. **创建流程**：\n"
            "   1. 通过具体示例理解技能\n"
            "   2. 规划可重用技能内容\n"
            "   3. 初始化技能（使用init_skill.py）\n"
            "   4. 编辑技能（实现资源和编写SKILL.md）\n"
            "   5. 基于实际使用迭代\n\n"
            "提供具体示例、代码模板和最佳实践指导，帮助用户创建高效、模块化的技能。"
        )

    def get_tools(self) -> List[Tool]:
        """技能创建专家专用工具"""
        return []