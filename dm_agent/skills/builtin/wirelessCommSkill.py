"""
无线通信专家技能 (基于 MCP RAG 中台驱动)。
不再继承 BaseRAGSkill，直接定义元数据并调用远程 RAG 服务。
"""

from typing import List, Dict, Any, Optional
from ..base import BaseSkill, SkillMetadata
from ...tools.base import Tool
import json


class WirelessCommSkill(BaseSkill):
    """无线通信专家技能：专注于 ISAC, MIMO, UAV 轨迹优化等领域"""

    def __init__(self, mcp_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.mcp_manager = mcp_manager
        self.rag_server_name = "rag_service"  # 对应 mcp_config.json 中的 RAG 服务 Key
        self._initialized_remote = False

        # 定义该领域连接 RAG 中台所需的具体配置
        self.domain_config = {
            "skill_id": "wireless_comm",
            "index_subdir": "wireless_idx",
            "data_subdir": "wirelessComm",
            "top_k": 3,
            "threshold": 0.4,
            "use_hybrid": True,
            "use_reranker": True
        }

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="wireless_comm_expert",
            display_name="无线通信专家",
            description="提供 ISAC (感知通信一体化)、MIMO、UAV 轨迹优化 (SCA) 相关的专业文献检索与算法指导",
            keywords=[
                "ISAC", "波束成形", "信道估计", "轨迹优化", "MIMO", "UAV通信",
                "SCA", "CVX", "凸优化", "信号处理", "无线资源管理"
            ],
            patterns=[
                r"波束成形", r"信道", r"轨迹优化", r"通信算法",
                r"ISAC", r"MIMO", r"UAV", r"凸优化", r"信号处理"
            ],
            priority=5,
            version="1.1.0",
        )

    def get_prompt_addition(self) -> str:
        return (
            "你现在具备无线通信专家能力。在处理相关学术或工程任务时请遵循以下原则：\n"
            "1. 优先检索专业文献库，确保物理层建模（如信道增益、路径损耗）的准确性\n"
            "2. 在讨论优化问题时，明确目标函数、约束条件以及是否为凸问题\n"
            "3. 推荐算法时区分场景：如针对轨迹优化的 SCA (Successive Convex Approximation)\n"
            "4. 代码实现优先推荐 MATLAB (CVX) 或 Python (PyTorch/CVXPY) 方案\n"
            "5. 使用 search_wireless_knowledge 工具获取最前沿的论文背景和参数设定 如果只是检索和解释专业知识或者概念，就不需要在通过default tools中的工具在进行搜索，我们只使用search_wireless_knowledge来进行检索和回答。并且如没有在专业知识库中寻找到相关知识，直接说不知道即可。如果有写代码或者其他代码需求才通过default tools中的工具进行检索\n"
        )

    def _ensure_rag_connected(self) -> bool:
        """静默确保远程 RAG 服务器已加载本领域知识"""
        if self._initialized_remote:
            return True
        if not self.mcp_manager:
            return False

        client = self.mcp_manager.clients.get(self.rag_server_name)
        if client and client.is_running():
            # 通过 MCP 协议下发初始化指令，将 RAG 系统连接至无线通信数据库
            init_payload = json.dumps(self.domain_config)
            client.call_tool("initialize_expert_context", {"config_json": init_payload})
            self._initialized_remote = True
            return True
        return False

    def _wireless_knowledge_runner(self, arguments: Dict[str, Any]) -> str:
        """工具运行器：转发请求至 MCP RAG 服务器"""
        if not self._ensure_rag_connected():
            return "❌ 远程 RAG 检索服务未启动或连接失败。"

        query = arguments.get("query", "")
        trace_id = arguments.get("trace_id") or getattr(self, "current_trace_id", None)  # 支持从 Agent 环境穿透 tid

        client = self.mcp_manager.clients.get(self.rag_server_name)
        print(f"🔍 [技能] 准备调用 MCP，trace_id = {trace_id}")
        # 调用远程 RAG 服务器的通用搜索接口
        return client.call_tool("search", {
            "query": query,
            "trace_id": trace_id
        })

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search",
                description=(
                    "在无线通信专业文献库中搜索知识。"
                    "参数：{\"query\": \"搜索词\", \"trace_id\": \"可选追踪ID\"}。"
                    "适用于查询 ISAC 建模、MIMO 算法、UAV 轨迹优化等具体学术问题。"
                ),
                runner=self._wireless_knowledge_runner,
            )
        ]