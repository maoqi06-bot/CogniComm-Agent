"""Preset multi-agent profiles for common research/application domains."""

from __future__ import annotations

from typing import Dict

from .memory import AgentMemoryPolicy, MemoryWriteTemplate
from .profiles import AgentProfile, CodeAgentProfile, RAGAgentProfile


def build_domain_profiles(domain: str) -> Dict[str, AgentProfile]:
    """Return role profiles for a known domain.

    These presets are intentionally lightweight. They customize memory policy,
    memory write templates, and domain style while preserving the same
    Orchestrator/RAGAgent/CodeAgent implementation.
    """

    key = (domain or "").strip().lower()
    if key in {"wireless", "wireless_comm", "isac", "通信", "无线通信"}:
        return _wireless_profiles()
    if key in {"medical", "medicine", "healthcare", "医学", "医疗"}:
        return _regulated_research_profiles(
            domain_style="medical_research",
            tags=["multi_agent", "medical", "research"],
        )
    if key in {"legal", "law", "法律"}:
        return _regulated_research_profiles(
            domain_style="legal_analysis",
            tags=["multi_agent", "legal", "research"],
        )
    if key in {"finance", "financial", "金融"}:
        return _regulated_research_profiles(
            domain_style="financial_analysis",
            tags=["multi_agent", "finance", "research"],
        )
    if key in {"ai", "ml", "machine_learning", "deep_learning", "人工智能", "机器学习", "深度学习"}:
        return _technical_research_profiles(
            domain_style="ai_research",
            tags=["multi_agent", "ai", "machine_learning", "research"],
            code_tags=["multi_agent", "ai", "experiment", "code_agent"],
        )
    if key in {"robotics", "robot", "control", "机器人", "控制"}:
        return _technical_research_profiles(
            domain_style="robotics_research",
            tags=["multi_agent", "robotics", "control", "research"],
            code_tags=["multi_agent", "robotics", "simulation", "code_agent"],
        )
    if key in {"education", "teaching", "learning", "教育", "教学"}:
        return _technical_research_profiles(
            domain_style="education_research",
            tags=["multi_agent", "education", "teaching", "research"],
            code_tags=["multi_agent", "education", "tooling", "code_agent"],
        )
    if key in {"data_science", "analytics", "data", "数据科学", "数据分析"}:
        return _technical_research_profiles(
            domain_style="data_science",
            tags=["multi_agent", "data_science", "analytics", "research"],
            code_tags=["multi_agent", "data_science", "pipeline", "code_agent"],
        )
    if key in {"cybersecurity", "security", "安全", "网络安全"}:
        return _regulated_research_profiles(
            domain_style="cybersecurity_analysis",
            tags=["multi_agent", "cybersecurity", "research"],
        )
    return _generic_profiles()


def _generic_profiles() -> Dict[str, AgentProfile]:
    return {
        "rag": RAGAgentProfile(
            name="rag_agent",
            domain_style="research",
            memory_policy=AgentMemoryPolicy(read_long_term=True, write_long_term=False),
        ),
        "code": CodeAgentProfile(
            name="code_agent",
            memory_policy=AgentMemoryPolicy(
                read_long_term=True,
                write_long_term=True,
                allowed_long_term_kinds=[
                    "engineering_experience",
                    "debugging_lesson",
                    "implementation_pattern",
                    "simulation_result",
                ],
            ),
        ),
    }


def _wireless_profiles() -> Dict[str, AgentProfile]:
    return {
        "rag": RAGAgentProfile(
            name="rag_agent",
            domain_style="wireless_research",
            top_k=6,
            memory_policy=AgentMemoryPolicy(read_long_term=True, write_long_term=False),
            memory_write_templates={
                "research_note": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.6,
                    tags=["multi_agent", "wireless", "isac", "rag_agent"],
                    title="Wireless research note",
                )
            },
        ),
        "code": CodeAgentProfile(
            name="code_agent",
            memory_policy=AgentMemoryPolicy(
                read_long_term=True,
                write_long_term=True,
                allowed_long_term_kinds=[
                    "engineering_experience",
                    "debugging_lesson",
                    "implementation_pattern",
                    "simulation_result",
                ],
            ),
            memory_write_templates={
                "engineering_experience": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.65,
                    tags=["multi_agent", "wireless", "simulation", "code_agent"],
                    title="Wireless simulation engineering experience",
                ),
                "simulation_result": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.62,
                    tags=["multi_agent", "wireless", "isac", "simulation_result"],
                    title="Wireless/ISAC simulation result",
                ),
                "implementation_pattern": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.66,
                    tags=["multi_agent", "wireless", "implementation_pattern"],
                    title="Wireless implementation pattern",
                ),
            },
        ),
    }


def _regulated_research_profiles(domain_style: str, tags: list[str]) -> Dict[str, AgentProfile]:
    return {
        "rag": RAGAgentProfile(
            name="rag_agent",
            domain_style=domain_style,
            top_k=6,
            synthesis_temperature=0.1,
            memory_policy=AgentMemoryPolicy(read_long_term=True, write_long_term=False),
            memory_write_templates={
                "research_note": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.6,
                    tags=tags,
                    title=f"{domain_style} note",
                )
            },
        ),
        "code": CodeAgentProfile(
            name="code_agent",
            memory_policy=AgentMemoryPolicy(
                read_long_term=True,
                write_long_term=True,
                allowed_long_term_kinds=[
                    "engineering_experience",
                    "debugging_lesson",
                    "implementation_pattern",
                    "simulation_result",
                ],
            ),
            memory_write_templates={
                "engineering_experience": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.58,
                    tags=tags + ["code_agent"],
                    title=f"{domain_style} engineering experience",
                )
            },
        ),
    }


def _technical_research_profiles(
    domain_style: str,
    tags: list[str],
    code_tags: list[str],
) -> Dict[str, AgentProfile]:
    return {
        "rag": RAGAgentProfile(
            name="rag_agent",
            domain_style=domain_style,
            top_k=6,
            synthesis_temperature=0.15,
            memory_policy=AgentMemoryPolicy(read_long_term=True, write_long_term=False),
            memory_write_templates={
                "research_note": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.6,
                    tags=tags,
                    title=f"{domain_style} research note",
                )
            },
        ),
        "code": CodeAgentProfile(
            name="code_agent",
            memory_policy=AgentMemoryPolicy(
                read_long_term=True,
                write_long_term=True,
                allowed_long_term_kinds=[
                    "engineering_experience",
                    "debugging_lesson",
                    "implementation_pattern",
                    "simulation_result",
                ],
            ),
            memory_write_templates={
                "engineering_experience": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.62,
                    tags=code_tags,
                    title=f"{domain_style} engineering experience",
                ),
                "debugging_lesson": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.64,
                    tags=code_tags + ["debugging"],
                    title=f"{domain_style} debugging lesson",
                ),
                "implementation_pattern": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.66,
                    tags=code_tags + ["implementation_pattern"],
                    title=f"{domain_style} implementation pattern",
                ),
                "simulation_result": MemoryWriteTemplate(
                    category="skill_knowledge",
                    priority="normal",
                    importance_score=0.6,
                    tags=code_tags + ["simulation_result"],
                    title=f"{domain_style} simulation or experiment result",
                ),
            },
        ),
    }
