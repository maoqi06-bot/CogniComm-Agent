"""数据库专家技能"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import BaseSkill, SkillMetadata
from ...tools.base import Tool


_SQL_COMMON_ISSUES = {
    "SELECT *": "避免使用 SELECT *，显式列出需要的列以提高性能和可维护性",
    "无 WHERE": "缺少 WHERE 子句可能导致全表扫描，确认是否需要过滤条件",
    "无索引提示": "对于 JOIN 和 WHERE 中的列，建议创建合适的索引",
    "N+1 查询": "检测到可能的 N+1 查询模式，考虑使用 JOIN 或批量查询",
    "子查询优化": "考虑将相关子查询改写为 JOIN 以提升性能",
    "LIKE 前缀通配符": "LIKE '%xxx' 无法使用索引，考虑全文索引或调整查询模式",
}


def _sql_review_runner(arguments: Dict[str, Any]) -> str:
    """审查 SQL 语句，检测常见问题并给出优化建议"""
    sql = arguments.get("sql", "").strip()
    if not sql:
        return "请提供要审查的 SQL 语句。参数：{\"sql\": \"你的 SQL 语句\"}"

    sql_upper = sql.upper()
    findings: List[str] = []

    if "SELECT *" in sql_upper:
        findings.append(f"⚠ {_SQL_COMMON_ISSUES['SELECT *']}")

    if "SELECT" in sql_upper and "WHERE" not in sql_upper and "INSERT" not in sql_upper:
        findings.append(f"⚠ {_SQL_COMMON_ISSUES['无 WHERE']}")

    if "JOIN" in sql_upper or "WHERE" in sql_upper:
        findings.append(f"💡 {_SQL_COMMON_ISSUES['无索引提示']}")

    if "LIKE" in sql_upper and "'%" in sql:
        findings.append(f"⚠ {_SQL_COMMON_ISSUES['LIKE 前缀通配符']}")

    if not findings:
        return "✅ SQL 语句看起来没有明显问题。建议进一步使用 EXPLAIN 分析执行计划。"

    result = "## SQL 审查结果\n\n"
    result += "\n".join(f"- {f}" for f in findings)
    result += "\n\n### 通用建议\n"
    result += "- 使用 EXPLAIN / EXPLAIN ANALYZE 查看执行计划\n"
    result += "- 确保 WHERE 和 JOIN 条件列上有合适的索引\n"
    result += "- 考虑使用参数化查询防止 SQL 注入"
    return result


class DatabaseExpertSkill(BaseSkill):
    """数据库专家技能"""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="db_expert",
            display_name="数据库专家",
            description="提供 SQL 最佳实践、数据库设计、ORM 使用和性能优化指导",
            keywords=[
                "sql", "mysql", "postgresql", "sqlite", "数据库", "索引",
                "orm", "sqlalchemy", "django orm", "查询优化", "事务",
                "迁移", "migration", "表设计", "mongodb", "redis",
            ],
            patterns=[
                r"\bSELECT\b",
                r"\bCREATE\s+TABLE\b",
                r"\bINSERT\s+INTO\b",
                r"\bALTER\s+TABLE\b",
                r"\.sql\b",
            ],
            priority=5,
            version="1.0.0",
        )

    def get_prompt_addition(self) -> str:
        return (
            "你现在具备数据库专家能力。在处理数据库相关任务时请遵循以下原则：\n"
            "1. 编写高效的 SQL 查询，避免全表扫描\n"
            "2. 正确设计表结构，遵循范式或合理的反范式\n"
            "3. 合理使用索引，注意复合索引的列顺序\n"
            "4. 使用事务保证数据一致性\n"
            "5. 使用参数化查询防止 SQL 注入\n"
            "6. 使用 sql_review 工具审查 SQL 语句的常见问题\n"
        )

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="sql_review",
                description=(
                    "审查 SQL 语句，检测常见问题并给出优化建议。"
                    "参数：{\"sql\": \"要审查的 SQL 语句\"}"
                ),
                runner=_sql_review_runner,
            )
        ]
