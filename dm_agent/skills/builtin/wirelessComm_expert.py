"""
无线通信专家技能 (Python 子类实现)。
继承自 BaseRAGSkill，复用工业级混合检索与重排链路。
"""

from .base_rag_skill import BaseRAGSkill
from typing import Dict, Any


class WirelessCommSkill_2(BaseRAGSkill):
    """
    专注于无线通信 (ISAC, MIMO, UAV) 的 RAG 技能。

    该类通过配置 index_subdir 和 data_subdir，
    自动挂载位于 dm_agent/data/knowledge_base/wirelessComm 下的 PDF、代码和文档。
    """

    def __init__(self, **kwargs):
        # 1. 定义该领域的专用配置字典
        config = {
            "skill_id": "wireless_comm",
            "display_name": "无线通信专家库",
            "description": (
                "提供 ISAC (集成感知与通信), 6G, MIMO 理论以及轨迹优化 (SCA) 相关的专业文献检索。"
                "同时支持提供对应领域的高质量 Python/MATLAB 代码实现方案。"
            ),
            "keywords": ["ISAC", "波束成形", "信道估计", "轨迹优化", "MIMO", "UAV通信", "SCA", "CVX"],
            "patterns": [
                r"波束成形", r"信道", r"轨迹优化", r"通信算法",
                r"ISAC", r"MIMO", r"UAV", r"凸优化"
            ],
            "priority": 5,  # 提高优先级，确保通信相关问题优先命中
            "index_subdir": "wireless_idx",  # 向量索引存放路径
            "data_subdir": "wirelessComm",  # 原始文档存放路径
            "use_hybrid": True,  # 启用 BM25 + 向量混合检索
            "use_reranker": True,  # 启用 Cross-Encoder 精排提升准确度
            "chunk_size": 1200,  # 针对论文公式微调分块大小
            "chunk_overlap": 200
        }

        # 2. 调用父类的初始化，BaseRAGSkill 会处理剩下的所有工作
        super().__init__(config=config, **kwargs)

    # 提示：如果你需要对无线通信领域的 Prompt 做特殊定制，
    # 可以选择性重写 get_prompt_addition，否则直接用父类的即可。