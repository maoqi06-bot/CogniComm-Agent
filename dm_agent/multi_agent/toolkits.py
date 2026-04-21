"""Tool routing helpers for multi-agent profiles."""

from __future__ import annotations

from typing import Any, Iterable, List, Tuple


def tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "").lower()


def is_rag_tool(tool: Any) -> bool:
    """Heuristic for routing MCP/domain tools to RAGAgent by default."""

    name = tool_name(tool)
    description = str(getattr(tool, "description", "") or "").lower()
    return (
        "rag" in name
        or "retriev" in name
        or "search" in name
        or "vector" in name
        or "知识" in description
        or "检索" in description
    )


def split_mcp_tools(tools: Iterable[Any]) -> Tuple[List[Any], List[Any]]:
    """Split MCP tools into RAG-oriented and code-oriented groups."""

    rag_tools: List[Any] = []
    code_tools: List[Any] = []
    for tool in tools or []:
        if is_rag_tool(tool):
            rag_tools.append(tool)
        else:
            code_tools.append(tool)
    return rag_tools, code_tools


def filter_tools_by_profile(tools: Iterable[Any], profile: Any) -> List[Any]:
    """Filter MCP/tools for a profile when the profile declares tool names.

    If no allowlist is present, the original tools are returned. This keeps
    existing behavior while allowing JSON profiles to bind selected MCP tools to
    one sub-agent.
    """

    metadata = getattr(profile, "metadata", {}) or {}
    allowed = set()
    for key in ("mcp_tool_names", "mcp_tools", "tool_names"):
        allowed.update(str(item).lower() for item in metadata.get(key, []) if item)
    if not allowed:
        return list(tools or [])
    filtered: List[Any] = []
    for tool in tools or []:
        name = tool_name(tool)
        if name in allowed or str(getattr(tool, "name", "") or "") in allowed:
            filtered.append(tool)
    return filtered
