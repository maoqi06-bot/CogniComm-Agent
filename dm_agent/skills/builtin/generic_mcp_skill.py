from typing import List, Dict, Any, Optional
from ..base import BaseSkill, SkillMetadata
from ...tools.base import Tool
import json


class GenericMCPSkill(BaseSkill):
    """
    通用 MCP 驱动技能：只需提供 JSON 配置即可实例化。
    支持全链路 Trace ID 穿透，确保 Dashboard 能够正常显示耗时。
    """

    def __init__(self, config: Dict[str, Any], mcp_manager=None):
        super().__init__()
        self.config = config
        self.mcp_manager = mcp_manager
        self.rag_server_name = config.get("rag_server_name", "rag_service")
        self._initialized_remote = False

        # 核心：由 main.py 在运行时动态注入，用于追踪
        self.current_trace_id = None

    def get_metadata(self) -> SkillMetadata:
        """从 JSON 动态映射元数据"""
        return SkillMetadata(
            name=self.config["skill_name"],
            display_name=self.config["display_name"],
            description=self.config["description"],
            keywords=self.config.get("keywords", []),
            patterns=self.config.get("patterns", []),
            priority=self.config.get("priority", 5),
            version=self.config.get("version", "1.0.0"),
        )

    def get_prompt_addition(self) -> str:
        """从 JSON 获取该领域的 System Prompt 增强"""
        return self.config.get("prompt_addition", "")

    def _ensure_rag_connected(self) -> bool:
        """确保远程 RAG 服务器加载了 JSON 中指定的领域索引"""
        if self._initialized_remote:
            return True
        if not self.mcp_manager:
            return False

        client = self.mcp_manager.clients.get(self.rag_server_name)
        if client and client.is_running():
            # 下发 JSON 中的 domain_config (包含 index_subdir 等)
            init_payload = json.dumps(self.config["domain_config"])
            client.call_tool("initialize_expert_context", {"config_json": init_payload})
            self._initialized_remote = True
            return True
        return False

    def _universal_tool_runner(self, arguments: Dict[str, Any]) -> str:
        """通用的工具运行器：转发请求并穿透 Trace ID"""
        if not self._ensure_rag_connected():
            return "❌ 远程 RAG 检索服务未启动或连接失败。"

        query = arguments.get("query", "")
        # 优先级：注入的 tid > 参数带入的 tid
        tid = self.current_trace_id or arguments.get("trace_id")

        client = self.mcp_manager.clients.get(self.rag_server_name)
        return client.call_tool("search_expert_knowledge", {
            "query": query,
            "trace_id": tid
        })

    def get_tools(self) -> List[Tool]:
        """从 JSON 动态生成工具定义"""
        t_def = self.config.get("tool_definition", {})
        return [
            Tool(
                name=t_def.get("name", f"search_{self.config['skill_name']}"),
                description=t_def.get("description", "专业知识检索工具"),
                runner=self._universal_tool_runner,
            )
        ]