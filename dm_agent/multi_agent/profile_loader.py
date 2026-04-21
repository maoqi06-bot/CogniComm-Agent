"""JSON-driven profile loading for multi-agent roles.

This module lets users extend multi-agent behavior by dropping JSON profile
files into a configured directory. It intentionally does not change the
single-agent skill system; it only maps declarative profile settings to the
multi-agent RAGAgent/CodeAgent profiles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .domain_profiles import build_domain_profiles
from .memory import AgentMemoryPolicy, MemoryWriteTemplate
from .profiles import AgentProfile, CodeAgentProfile, RAGAgentProfile


DEFAULT_PROFILE_DIR = Path("configs") / "multi_agent" / "profiles"


def load_profiles_for_task(
    task: str,
    *,
    domain: Optional[str] = None,
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
) -> Dict[str, AgentProfile]:
    """Load built-in and user-defined profiles for a task.

    Selection order:
    1. built-in domain presets when ``domain`` is provided;
    2. JSON profiles whose aliases/keywords match ``task`` or ``domain``;
    3. later matching files override earlier settings role by role.
    """

    profiles: Dict[str, AgentProfile] = build_domain_profiles(domain or "")
    for config in _iter_matching_profile_configs(task, domain=domain, profile_dir=profile_dir):
        profiles.update(_profiles_from_config(config))
    return profiles


def _iter_matching_profile_configs(
    task: str,
    *,
    domain: Optional[str],
    profile_dir: str | Path,
) -> Iterable[Dict[str, Any]]:
    directory = Path(profile_dir)
    if not directory.is_dir():
        return []

    task_text = (task or "").lower()
    domain_text = (domain or "").lower()
    matched: list[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            continue

        if not isinstance(config, dict):
            continue
        if _matches_config(config, task_text=task_text, domain_text=domain_text):
            config.setdefault("_source_path", str(path))
            matched.append(config)
    return matched


def _matches_config(config: Dict[str, Any], *, task_text: str, domain_text: str) -> bool:
    if config.get("enabled", True) is False:
        return False

    names = [
        str(config.get("domain", "")),
        str(config.get("name", "")),
        *[str(item) for item in config.get("aliases", [])],
        *[str(item) for item in config.get("auto_activate_keywords", [])],
    ]
    lowered = [item.lower() for item in names if item]
    if domain_text and domain_text in lowered:
        return True
    return any(item and item in task_text for item in lowered)


def _profiles_from_config(config: Dict[str, Any]) -> Dict[str, AgentProfile]:
    profiles: Dict[str, AgentProfile] = {}
    raw_profiles = config.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        return profiles

    for role, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, dict):
            continue
        role_key = str(role).lower()
        if role_key == "rag":
            profiles["rag"] = _rag_profile_from_dict(raw_profile, config)
        elif role_key == "code":
            profiles["code"] = _code_profile_from_dict(raw_profile, config)
        else:
            profiles[role_key] = _agent_profile_from_dict(raw_profile, config)
    return profiles


def _agent_profile_from_dict(raw: Dict[str, Any], root: Dict[str, Any]) -> AgentProfile:
    return AgentProfile(
        name=str(raw.get("name") or "agent"),
        prompt=raw.get("prompt"),
        metadata=_metadata_from_dict(raw, root),
        memory_enabled=bool(raw.get("memory_enabled", True)),
        memory_policy=_memory_policy_from_dict(raw.get("memory_policy")),
        memory_write_templates=_memory_templates_from_dict(raw.get("memory_write_templates")),
    )


def _rag_profile_from_dict(raw: Dict[str, Any], root: Dict[str, Any]) -> RAGAgentProfile:
    return RAGAgentProfile(
        name=str(raw.get("name") or "rag_agent"),
        prompt=raw.get("prompt"),
        metadata=_metadata_from_dict(raw, root),
        memory_enabled=bool(raw.get("memory_enabled", True)),
        memory_policy=_memory_policy_from_dict(raw.get("memory_policy")),
        memory_write_templates=_memory_templates_from_dict(raw.get("memory_write_templates")),
        top_k=int(raw.get("top_k", 5)),
        synthesis_temperature=float(raw.get("synthesis_temperature", 0.2)),
        domain_style=str(raw.get("domain_style") or root.get("domain") or "research"),
        long_term_memory_enabled=bool(raw.get("long_term_memory_enabled", True)),
    )


def _code_profile_from_dict(raw: Dict[str, Any], root: Dict[str, Any]) -> CodeAgentProfile:
    return CodeAgentProfile(
        name=str(raw.get("name") or "code_agent"),
        prompt=raw.get("prompt"),
        metadata=_metadata_from_dict(raw, root),
        memory_enabled=bool(raw.get("memory_enabled", True)),
        memory_policy=_memory_policy_from_dict(raw.get("memory_policy")),
        memory_write_templates=_memory_templates_from_dict(raw.get("memory_write_templates")),
        allow_rag_tools=bool(raw.get("allow_rag_tools", False)),
        use_docker=bool(raw.get("use_docker", True)),
        code_style=str(raw.get("code_style") or "engineering"),
        long_term_memory_enabled=bool(raw.get("long_term_memory_enabled", True)),
    )


def _metadata_from_dict(raw: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(raw.get("metadata") or {})
    for key in ["skills", "skill_names", "mcp_tools", "mcp_tool_names", "tool_names"]:
        if key in raw:
            metadata[key] = list(raw.get(key) or [])
    metadata.setdefault("domain", root.get("domain"))
    metadata.setdefault("profile_source", root.get("_source_path"))
    return metadata


def _memory_policy_from_dict(raw: Any) -> Optional[AgentMemoryPolicy]:
    if not isinstance(raw, dict):
        return None
    return AgentMemoryPolicy(
        read_long_term=bool(raw.get("read_long_term", True)),
        write_long_term=bool(raw.get("write_long_term", False)),
        write_shared_short_term=bool(raw.get("write_shared_short_term", True)),
        write_private_short_term=bool(raw.get("write_private_short_term", True)),
        write_task_summary=bool(raw.get("write_task_summary", False)),
        allowed_long_term_kinds=list(raw.get("allowed_long_term_kinds") or []),
    )


def _memory_templates_from_dict(raw: Any) -> Dict[str, MemoryWriteTemplate]:
    if not isinstance(raw, dict):
        return {}
    templates: Dict[str, MemoryWriteTemplate] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            continue
        templates[str(name)] = MemoryWriteTemplate(
            category=str(value.get("category", "skill_knowledge")),
            priority=str(value.get("priority", "normal")),
            importance_score=float(value.get("importance_score", 0.55)),
            tags=list(value.get("tags") or []),
            source=str(value.get("source", "multi_agent")),
            title=str(value.get("title", f"Multi-agent {name}")),
        )
    return templates
