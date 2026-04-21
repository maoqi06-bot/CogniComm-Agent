"""上下文压缩器 - 增强版（引入科研锚点与草稿本逻辑）"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..clients.base_client import BaseLLMClient


class ContextCompressor:
    """
    每 N 轮对话自动压缩上下文 (增强科研任务支持)

    新增功能：
    1. 科研锚点保留：自动识别并保护数学公式 ($...$)、物理参数定义。
    2. 进度锚点：强制保留最新的 Task Planner 状态。
    """

    def __init__(
            self, client: Optional[BaseLLMClient] = None, compress_every: int = 5, keep_recent: int = 3
    ):
        self.client = client
        self.compress_every = compress_every
        self.keep_recent = keep_recent
        self.turn_count = 0

    def should_compress(self, history: List[Dict[str, str]]) -> bool:
        user_messages = [msg for msg in history if msg.get("role") == "user"]
        self.turn_count = len(user_messages)
        return self.turn_count >= self.compress_every

    def compress(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not history:
            return []

        system_messages = [msg for msg in history if msg.get("role") == "system"]
        non_system = [msg for msg in history if msg.get("role") != "system"]

        recent_messages = (
            non_system[-self.keep_recent * 2:]
            if len(non_system) > self.keep_recent * 2
            else non_system
        )

        middle_messages = (
            non_system[: -self.keep_recent * 2] if len(non_system) > self.keep_recent * 2 else []
        )

        compressed_middle = []
        if middle_messages:
            # 调用增强后的提取逻辑
            summary = self._extract_key_information(middle_messages)
            # 这里的角色设定为 user 并在内容中标记为“草稿本/摘要”，引导 Agent 重点关注
            compressed_middle = [{"role": "user", "content": f"📝 [科研草稿本 & 历史摘要]\n{summary}"}]

        result = system_messages + compressed_middle + recent_messages
        self.turn_count = len([msg for msg in result if msg.get("role") == "user"])

        return result

    def _extract_key_information(self, messages: List[Dict[str, str]]) -> str:
        """
        增强型提取：增加了对科研参数、数学公式和执行进度的锚点保护
        """
        key_info = []

        # 1. 【科研锚点】提取数学公式和核心参数定义 (如 Nt=8, P_max=30dBm)
        research_anchors = set()
        for msg in messages:
            content = msg.get("content", "")
            # 提取 LaTeX 公式 $...$ 或 $$...$$
            formulas = re.findall(r"\${1,2}.*?\${1,2}", content)
            research_anchors.update(formulas)

            # 提取常见的物理参数定义模式 (变量名 = 数字/矩阵)
            params = re.findall(r"([a-zA-Z]_[a-zA-Z0-9]+|[a-zA-Z]{1,2})\s*=\s*\d+\.?\d*", content)
            for p in params:
                # 只保留可能具有物理意义的参数名
                if p not in ['i', 'j', 'x', 'y']:
                    research_anchors.add(p)

        if research_anchors:
            key_info.append(f"🧬 关键科研锚点（公式/参数）：\n{', '.join(list(research_anchors)[:15])}")

        # 2. 【进度锚点】提取最新的任务进度 (Planner 状态)
        latest_plan_status = ""
        for msg in reversed(messages):
            content = msg.get("content", "")
            if "计划进度" in content or "步骤" in content:
                # 尝试提取最近的一个进度摘要
                plan_match = re.search(r"(计划进度：\d+/\d+.*?\n)", content)
                if plan_match:
                    latest_plan_status = plan_match.group(1).strip()
                    break

        if latest_plan_status:
            key_info.append(f"📍 当前任务进度：{latest_plan_status}")

        # 3. 提取文件路径 (保留原逻辑并增强)
        file_paths = set()
        for msg in messages:
            content = msg.get("content", "")
            paths = re.findall(r"(?:path|文件|读取|创建|编辑)[:：]\s*([^\s,，;；\n]+\.[a-zA-Z0-9]+)", content)
            file_paths.update(paths)
        if file_paths:
            key_info.append(f"📁 涉及文件：{', '.join(sorted(file_paths))}")

        # 4. 提取错误信息 (保留原逻辑)
        errors = []
        for msg in messages:
            content = msg.get("content", "")
            if any(kw in content.lower() for kw in ["错误", "error", "失败", "traceback"]):
                lines = [l.strip() for l in content.split("\n") if
                         any(kw in l.lower() for kw in ["error", "traceback"])]
                errors.extend(lines[:1])  # 科研任务中通常只需要保留最近的一个致命错误
        if errors:
            key_info.append(f"⚠️ 遗留问题/错误：\n" + "\n".join(set(errors)))

        # 5. 提取工具调用 (保留原逻辑)
        tools_used = set()
        for msg in messages:
            content = msg.get("content", "")
            if "执行工具" in content:
                tool_match = re.search(r"执行工具\s+(\w+)", content)
                if tool_match: tools_used.add(tool_match.group(1))
        if tools_used:
            key_info.append(f"🛠️ 已调用的工具：{', '.join(sorted(tools_used))}")

        if not key_info:
            return f"进行了 {len(messages)} 轮对话，正在处理科研代码任务。"

        return "\n\n".join(key_info)

    def get_compression_stats(self, original: List[Dict[str, str]], compressed: List[Dict[str, str]]) -> Dict[str, Any]:
        return {
            "original_messages": len(original),
            "compressed_messages": len(compressed),
            "compression_ratio": (1 - len(compressed) / len(original) if len(original) > 0 else 0),
            "saved_messages": len(original) - len(compressed),
        }