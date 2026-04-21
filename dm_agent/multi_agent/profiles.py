"""Configuration profiles for specialized sub-agents.

Profiles keep tool, prompt, skill, and MCP wiring close to the agent role
instead of baking every option into the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentProfile:
    """Lightweight role profile shared by all multi-agent workers."""

    name: str
    prompt: Optional[str] = None
    tools: List[Any] = field(default_factory=list)
    skills: List[Any] = field(default_factory=list)
    mcp_tools: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_enabled: bool = True
    memory_policy: Optional[Any] = None
    memory_write_templates: Dict[str, Any] = field(default_factory=dict)

    def merged_tools(self, base_tools: Optional[List[Any]] = None) -> List[Any]:
        """Return base tools plus profile-local tools without duplicate names."""

        merged: List[Any] = []
        seen: set[str] = set()
        for tool in list(base_tools or []) + list(self.tools) + list(self.mcp_tools):
            name = getattr(tool, "name", None) or repr(tool)
            if name in seen:
                continue
            seen.add(name)
            merged.append(tool)
        return merged


@dataclass
class RAGAgentProfile(AgentProfile):
    """Profile for a full RAG-chain research/domain agent."""

    name: str = "rag"
    top_k: int = 5
    synthesis_temperature: float = 0.2
    domain_style: str = "research"
    long_term_memory_enabled: bool = True


@dataclass
class CodeAgentProfile(AgentProfile):
    """Profile for an implementation/testing agent."""

    name: str = "code"
    allow_rag_tools: bool = False
    use_docker: bool = True
    code_style: str = "engineering"
    long_term_memory_enabled: bool = True
