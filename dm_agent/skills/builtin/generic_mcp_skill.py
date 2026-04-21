from typing import List, Dict, Any
from ..base import BaseSkill, SkillMetadata
from ...tools.base import Tool
import json


class GenericMCPSkill(BaseSkill):
    """
    Generic MCP-driven RAG skill.

    All MCP-driven domain skills should point to the same shared RAG MCP
    server. Domain-specific differences belong in metadata, prompt additions,
    and `domain_config`, not in separate MCP server processes.
    """

    DEFAULT_RAG_SERVER_NAME = "wireless-rag"

    def __init__(self, config: Dict[str, Any], mcp_manager=None):
        super().__init__()
        self.config = config
        self.mcp_manager = mcp_manager
        self.rag_server_name = config.get("rag_server_name", self.DEFAULT_RAG_SERVER_NAME)
        self._initialized_remote = False
        self.current_trace_id = None

    def get_metadata(self) -> SkillMetadata:
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
        return self.config.get("prompt_addition", "")

    def _domain_config(self) -> Dict[str, Any]:
        domain_config = dict(self.config.get("domain_config", {}))
        if not domain_config:
            raise ValueError("Missing required field `domain_config` in GenericMCPSkill config.")
        if not domain_config.get("skill_id"):
            domain_config["skill_id"] = self.config.get("skill_name")
        if not domain_config.get("display_name"):
            domain_config["display_name"] = self.config.get("display_name", domain_config["skill_id"])
        return domain_config

    def _ensure_rag_connected(self) -> bool:
        if self._initialized_remote:
            return True
        if not self.mcp_manager:
            return False

        client = self.mcp_manager.clients.get(self.rag_server_name)
        if client and client.is_running():
            domain_config = self._domain_config()
            init_payload = json.dumps(domain_config, ensure_ascii=False)
            client.call_tool(
                "initialize_expert_context",
                {
                    "skill_id": domain_config["skill_id"],
                    "config_json": init_payload,
                },
            )
            self._initialized_remote = True
            return True
        return False

    def _universal_tool_runner(self, arguments: Dict[str, Any]) -> str:
        if not self._ensure_rag_connected():
            return "Remote RAG service is unavailable or failed to initialize."

        query = arguments.get("query", "")
        tid = self.current_trace_id or arguments.get("trace_id")
        domain_config = self._domain_config()
        client = self.mcp_manager.clients.get(self.rag_server_name)

        return client.call_tool(
            "search",
            {
                "skill_id": domain_config["skill_id"],
                "query": query,
                "trace_id": tid,
            },
        )

    def get_tools(self) -> List[Tool]:
        t_def = self.config.get("tool_definition", {})
        return [
            Tool(
                name=t_def.get("name", f"{self.config['skill_name']}_search"),
                description=t_def.get(
                    "description",
                    "Search specialized knowledge through the shared RAG MCP server.",
                ),
                runner=self._universal_tool_runner,
            )
        ]
