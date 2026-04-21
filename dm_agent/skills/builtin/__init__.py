"""内置技能注册"""

from __future__ import annotations

from typing import List

# 1. 导入你刚刚重写后的 WirelessCommSkill (子类)
from .wirelessCommSkill import WirelessCommSkill
from .wirelessComm_expert import WirelessCommSkill_2
# 2. 如果你依然想保留一个通用的、支持手动 load_documents 的基础 RAG 技能
from .base_rag_skill import BaseRAGSkill
from ..base import BaseSkill
from .python_expert import PythonExpertSkill
from .db_expert import DatabaseExpertSkill
from .frontend_dev import FrontendDevSkill
from .skill_creator import SkillCreatorSkill


def get_builtin_skills() -> List[BaseSkill]:
    """返回所有内置技能实例列表。"""

    # # 定义一个基础 RAG 技能的配置，用于处理通用文档加载
    # base_rag_config = {
    #     "skill_id": "general_rag",
    #     "display_name": "通用知识库",
    #     "description": "用于加载和检索本地通用文档的工具。",
    #     "keywords": ["资料", "文档", "查找", "搜寻"],
    #     "index_subdir": "general_idx",
    #     "data_subdir": "builtin_knowledge",  # 对应你之前的 data/builtin_knowledge 目录
    #     "priority": 50
    # }

    return [
        # 原有的工具类技能
        PythonExpertSkill(),
        DatabaseExpertSkill(),
        FrontendDevSkill(),
        SkillCreatorSkill(),

        # 3. 注册你的无线通信专家 (Python 子类)6
        # 它会自动去 data/indices/wireless_idx 找向量库
        WirelessCommSkill(),  # 通过RAG-MCP服务器加载
        WirelessCommSkill_2(),

        # 4. (可选) 注册一个基础 RAG 技能
        # BaseRAGSkill(config=base_rag_config),
    ]