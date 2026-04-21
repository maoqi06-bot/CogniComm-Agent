"""
Wireless communication expert skill backed by the shared RAG MCP server.

All MCP-driven domain skills should use the same shared RAG MCP server.
Domain-specific differences belong in metadata, prompt text, and
domain-specific data/index configuration.
"""

from typing import List, Dict, Any
from ..base import BaseSkill, SkillMetadata
from ...tools.base import Tool
import json


class WirelessCommSkill(BaseSkill):
    """Wireless communication expert skill for ISAC, RIS, MIMO, and UAV topics."""

    DEFAULT_RAG_SERVER_NAME = "wireless-rag"

    def __init__(self, mcp_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.mcp_manager = mcp_manager
        self.rag_server_name = self.DEFAULT_RAG_SERVER_NAME
        self._initialized_remote = False
        self.current_trace_id = None
        self.domain_config = {
            "skill_id": "wireless_comm",
            "display_name": "Wireless Communication Knowledge Base",
            "index_subdir": "wireless_idx",
            "data_subdir": "wirelessComm",
            "top_k": 3,
            "threshold": 0.4,
            "use_hybrid": True,
            "use_reranker": True,
        }

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="wireless_comm_expert",
            display_name="Wireless Communication Expert",
            description=(
                "Provides domain retrieval and academic background support for "
                "ISAC, RIS, MIMO, UAV, beamforming, and related wireless topics."
            ),
            keywords=["ISAC", "RIS", "MIMO", "UAV", "beamforming", "wireless communication"],
            patterns=[r"ISAC", r"RIS", r"MIMO", r"beamforming", r"trajectory optimization"],
            priority=5,
            version="1.1.0",
        )

    def get_prompt_addition(self) -> str:
        return (
            "You now have wireless communication expert augmentation.\n"
            "1. For ISAC, RIS, MIMO, UAV, beamforming, and trajectory optimization topics, "
            "prefer the wireless communication knowledge base first.\n"
            "2. For pure knowledge retrieval and explanation tasks, rely on the wireless "
            "RAG results before using general reasoning.\n"
            "3. If the task also needs coding, continue with the default code tools after "
            "grounding the answer in retrieved results.\n"
        )

    def _ensure_rag_connected(self) -> bool:
        if self._initialized_remote:
            return True
        if not self.mcp_manager:
            return False

        client = self.mcp_manager.clients.get(self.rag_server_name)
        if client and client.is_running():
            init_payload = json.dumps(self.domain_config, ensure_ascii=False)
            client.call_tool(
                "initialize_expert_context",
                {
                    "skill_id": self.domain_config["skill_id"],
                    "config_json": init_payload,
                },
            )
            self._initialized_remote = True
            return True
        return False

    def _wireless_knowledge_runner(self, arguments: Dict[str, Any]) -> str:
        if not self._ensure_rag_connected():
            return "Remote RAG service is unavailable or failed to initialize."

        query = arguments.get("query", "")
        trace_id = arguments.get("trace_id") or getattr(self, "current_trace_id", None)
        client = self.mcp_manager.clients.get(self.rag_server_name)
        return client.call_tool(
            "search",
            {
                "skill_id": self.domain_config["skill_id"],
                "query": query,
                "trace_id": trace_id,
            },
        )

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="wireless_comm_search",
                description=(
                    "Search the wireless communication knowledge base for ISAC, RIS, MIMO, "
                    "UAV, beamforming, and related material."
                ),
                runner=self._wireless_knowledge_runner,
            )
        ]
