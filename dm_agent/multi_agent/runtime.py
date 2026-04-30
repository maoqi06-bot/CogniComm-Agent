"""多 Agent 编排系统

架构：
- OrchestratorAgent: 主协调 Agent，负责接收用户任务、分解任务、路由到专业 Agent、聚合结果
- RAGAgent: 知识检索 Agent，使用 RAG 技能进行专业领域知识查询
- CodeAgent: 代码执行 Agent，执行代码、文件操作等
- DockerRunner: Docker 隔离执行环境
"""

from __future__ import annotations

import json
import os
import shlex
import time
import uuid
import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from abc import ABC, abstractmethod

from ..core.agent import ReactAgent, Step
from ..skills.manager import SkillManager
from ..skills.builtin.base_rag_skill import BaseRAGSkill
from ..clients.base_client import BaseLLMClient
from ..utils.logger import get_logger, AgentLogger
from ..utils.security import ResourceManager, RateLimitConfig
from .profiles import AgentProfile, CodeAgentProfile, RAGAgentProfile
from .prompts import build_multi_agent_code_prompt, build_rag_synthesis_prompt
from .toolkits import filter_tools_by_profile, split_mcp_tools
from .memory import AgentMemoryPolicy, MemoryWriteTemplate, MultiAgentMemoryConfig, MultiAgentMemoryHub


class TaskType(Enum):
    """任务类型枚举"""
    KNOWLEDGE_QUERY = "knowledge_query"      # 知识查询（需要 RAG）
    CODE_EXECUTION = "code_execution"        # 代码执行
    ANALYSIS = "analysis"                     # 分析任务（需要代码+RAG）
    GENERAL = "general"                       # 通用任务


@dataclass
class SubTask:
    """子任务定义"""
    id: str
    type: TaskType
    description: str
    dependencies: List[str] = field(default_factory=list)  # 依赖的子任务 ID
    priority: int = 0
    result: Optional[Any] = None
    status: str = "pending"  # pending, running, completed, partial, failed
    error: Optional[str] = None
    agent_name: str = ""  # 执行该子任务的 Agent
    cancel_event: Any = field(default_factory=threading.Event, repr=False)


@dataclass
class TaskDecomposition:
    """任务分解结果"""
    original_task: str
    sub_tasks: List[SubTask]
    requires_rag: bool = False
    requires_code: bool = False
    execution_plan: str = ""


# ============================================================
# Agent 基类
# ============================================================

class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(
        self,
        name: str,
        client: Optional[BaseLLMClient] = None,
        logger: Optional[AgentLogger] = None,
    ):
        self.name = name
        self.client = client
        self.logger = logger or get_logger(f"agent.{name}")
        self._active = False

    @abstractmethod
    def process(self, task: SubTask) -> Any:
        """处理子任务"""
        pass

    def activate(self):
        self._active = True
        self.logger.info(f"Agent {self.name} activated")

    def deactivate(self):
        self._active = False
        self.logger.info(f"Agent {self.name} deactivated")

    @property
    def is_active(self) -> bool:
        return self._active


# ============================================================
# RAG Agent
# ============================================================

class RAGAgent(BaseAgent):
    """知识检索 Agent

    使用 RAG 技能进行专业领域知识查询。
    支持多领域知识库并行检索。
    """

    def __init__(
        self,
        name: str = "rag_agent",
        client: Optional[BaseLLMClient] = None,
        skill_manager: Optional[SkillManager] = None,
        logger: Optional[AgentLogger] = None,
        profile: Optional[RAGAgentProfile] = None,
        mcp_tools: Optional[List[Any]] = None,
        memory_hub: Optional[MultiAgentMemoryHub] = None,
    ):
        super().__init__(name, client, logger)
        self.profile = profile or RAGAgentProfile()
        self.skill_manager = skill_manager
        self._mcp_tools = list(mcp_tools or []) + list(self.profile.mcp_tools)
        self.memory_hub = memory_hub
        self._rag_skills: Dict[str, BaseRAGSkill] = {}
        self._lock = threading.Lock()

    def register_rag_skill(self, skill: BaseRAGSkill):
        """注册 RAG 技能"""
        with self._lock:
            meta = skill.get_metadata()
            self._rag_skills[meta.name] = skill
            self.logger.info(f"Registered RAG skill: {meta.display_name}")

    def get_available_domains(self) -> List[str]:
        """获取可用的知识库领域"""
        return list(self._rag_skills.keys())

    def _build_context_block(self, results: List[Dict[str, Any]], max_chars: int = 1200) -> str:
        context_lines = []
        for idx, item in enumerate(results, start=1):
            source = item.get("source") or "unknown"
            score = item.get("score", 0)
            content = str(item.get("content", ""))[:max_chars]
            context_lines.append(
                f"[{idx}] source={source}; score={score}\n{content}"
            )
        return "\n\n".join(context_lines)

    def _synthesize_answer(
        self,
        query: str,
        results: List[Dict[str, Any]],
        *,
        stop_event: Optional[threading.Event] = None,
    ) -> str:
        """Generate a RAG answer from retrieved contexts for multi-agent mode."""
        if stop_event and stop_event.is_set():
            return ""

        top_k = max(1, int(getattr(self.profile, "top_k", 5)))
        used_results = results[:top_k]
        if not self.client:
            sources = ", ".join(sorted({str(item.get("source", "unknown")) for item in used_results}))
            return f"已检索到相关资料，但当前 RAG Agent 未配置 LLM，无法完成综合生成。主要来源：{sources}"

        context_block = self._build_context_block(used_results)
        messages = build_rag_synthesis_prompt(
            query=query,
            contexts_text=context_block,
            domain=self.name,
            style=getattr(self.profile, "domain_style", "research"),
        )
        temperature = float(getattr(self.profile, "synthesis_temperature", 0.2))

        try:
            return self.client.respond(messages, temperature=temperature).strip()
        except Exception as exc:
            self.logger.warning(f"RAG synthesis failed, using concise fallback: {exc}")
            sources = ", ".join(sorted({str(item.get("source", "unknown")) for item in used_results}))
            return f"检索已完成，但综合生成失败：{exc}。可用来源包括：{sources}"


    def process(self, task: SubTask) -> Dict[str, Any]:
        """处理知识查询任务"""
        self.logger.info(f"Processing knowledge query: {task.description[:100]}...")

        if task.cancel_event.is_set():
            return {
                "success": False,
                "error": "Task cancelled before RAG retrieval",
                "results": [],
                "cancelled": True,
            }

        if not self._rag_skills:
            return {
                "success": False,
                "error": "No RAG skills available",
                "results": [],
            }

        results: List[Dict[str, Any]] = []
        query = task.description
        tracer = None
        memory_context = ""
        if self.memory_hub and getattr(self.profile, "memory_enabled", True):
            memory_context = self.memory_hub.build_context(query, agent_name=self.name)
            self.memory_hub.add_event(
                self.name,
                "query",
                query,
                shared=False,
                metadata={"subtask_id": task.id},
            )

        # 尝试所有已注册的 RAG 技能
        for skill_name, skill in self._rag_skills.items():
            if task.cancel_event.is_set():
                return {
                    "success": False,
                    "error": "Task cancelled during RAG retrieval",
                    "results": results,
                    "cancelled": True,
                }
            try:
                skill_tracer = getattr(skill, "_tracer", None)
                if tracer is None:
                    tracer = skill_tracer
                node = None
                # 确保技能已初始化
                if hasattr(skill, '_ensure_initialized'):
                    skill._ensure_initialized()

                # 执行检索
                if hasattr(skill, '_retriever') and skill._retriever:
                    if skill_tracer:
                        node = skill_tracer.add_node("Hybrid_RAGAgent_Query", skill.get_metadata().display_name, input_val=query)
                    search_results = skill._retriever.retrieve(query, k=3)
                    if skill_tracer and search_results:
                        skill_tracer.metadata.setdefault("retrieved_contexts", [])
                        skill_tracer.metadata["retrieved_contexts"].extend([res.content for res in search_results])
                    if skill_tracer and node:
                        skill_tracer.end_node(
                            node,
                            output_val=[
                                {
                                    "doc": res.content[:500],
                                    "score": res.score,
                                    "source": res.metadata.get("file_name", "unknown"),
                                }
                                for res in (search_results or [])
                            ],
                        )
                    if search_results:
                        for res in search_results:
                            results.append({
                                "skill": skill.get_metadata().display_name,
                                "content": res.content,
                                "score": res.score,
                                "source": res.metadata.get("file_name", "unknown"),
                            })
            except Exception as e:
                self.logger.warning(f"RAG search failed for {skill_name}: {e}")

        if not results:
            return {
                "success": True,
                "error": None,
                "results": [],
                "message": "No relevant knowledge found",
            }

        # 按相关性排序
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:5]

        if task.cancel_event.is_set():
            return {
                "success": False,
                "error": "Task cancelled before RAG synthesis",
                "results": top_results,
                "cancelled": True,
            }

        synthesis_node = None
        if tracer:
            synthesis_node = tracer.add_node("RAG_Synthesis", self.name, input_val=query)
        synthesis_query = query
        if memory_context:
            synthesis_query = (
                f"{query}\n\n"
                "Multi-agent memory context for synthesis:\n"
                f"{memory_context}"
            )
        answer = self._synthesize_answer(synthesis_query, top_results, stop_event=task.cancel_event)
        if tracer and synthesis_node:
            tracer.end_node(synthesis_node, output_val=answer[:2000])

        if tracer and answer:
            tracer.metadata.setdefault("rag_eval_samples", []).append({
                "question": query,
                "contexts": [item["content"] for item in top_results],
                "answer": answer,
                "eval_scope": "rag_query",
                "source": "rag_agent:synthesis",
                "context_scores": [float(item.get("score", 0) or 0) for item in top_results],
                "context_sources": [item.get("source") or "unknown" for item in top_results],
            })

        if self.memory_hub:
            self.memory_hub.add_event(
                self.name,
                "answer",
                answer,
                shared=True,
                metadata={"subtask_id": task.id, "total_found": len(results)},
            )

        return {
            "success": True,
            "result": answer,
            "answer": answer,
            "results": top_results,  # 返回 Top 5
            "total_found": len(results),
            "generation_mode": "generated_answer",
        }

    def query(self, query: str, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """直接查询接口

        Args:
            query: 查询文本
            domains: 指定查询的知识库领域，None 表示所有领域

        Returns:
            查询结果字典
        """
        task = SubTask(
            id=f"rag_{uuid.uuid4().hex[:6]}",
            type=TaskType.KNOWLEDGE_QUERY,
            description=query,
        )

        if domains:
            # 只在指定领域查询
            with self._lock:
                original_skills = dict(self._rag_skills)
                self._rag_skills = {k: v for k, v in original_skills.items() if k in domains}

            try:
                result = self.process(task)
            finally:
                with self._lock:
                    self._rag_skills = original_skills
        else:
            result = self.process(task)

        return result


# ============================================================
# Code Agent
# ============================================================

class CodeAgent(BaseAgent):
    """代码执行 Agent

    使用 ReactAgent 进行代码编写、调试、执行等任务。
    支持 RAG 工具注入和 Docker 隔离执行。
    """

    def __init__(
        self,
        name: str = "code_agent",
        client: Optional[BaseLLMClient] = None,
        tools: Optional[List[Any]] = None,
        max_steps: int = 30,
        logger: Optional[AgentLogger] = None,
        resource_manager: Optional[ResourceManager] = None,
        skill_manager: Optional[Any] = None,
        docker_runner: Optional["DockerRunner"] = None,
        allow_rag_tools: bool = True,
        profile: Optional[CodeAgentProfile] = None,
        memory_hub: Optional[MultiAgentMemoryHub] = None,
    ):
        super().__init__(name, client, logger)
        self.profile = profile or CodeAgentProfile(allow_rag_tools=allow_rag_tools)
        self.max_steps = int(os.getenv("CODE_AGENT_MAX_STEPS", str(max_steps)))
        self.resource_manager = resource_manager
        self._react_agent: Optional[ReactAgent] = None
        self._tools = self.profile.merged_tools(tools or [])
        self._skill_manager = skill_manager
        self._docker_runner = docker_runner
        self._allow_rag_tools = bool(getattr(self.profile, "allow_rag_tools", allow_rag_tools))
        self.memory_hub = memory_hub

        if client:
            self._init_react_agent()

    def _init_react_agent(self):
        """初始化 ReactAgent，注入 RAG 工具并包装 Docker 执行"""
        if not self.client or not self._tools:
            return

        tools_to_register = list(self._tools)

        # 1. 单 Agent/兼容模式可从 skill_manager 注入 RAG 工具。
        # 多 Agent 模式下由独立 RAGAgent 完整负责 retrieve + synthesize，CodeAgent 不再挂载 RAG 工具。
        if self._skill_manager and self._allow_rag_tools:
            from ..skills.builtin.base_rag_skill import BaseRAGSkill
            rag_tools_count = 0
            for skill_name, skill in getattr(self._skill_manager, "skills", {}).items():
                if isinstance(skill, BaseRAGSkill):
                    for tool in skill.get_tools():
                        tools_to_register.append(tool)
                        rag_tools_count += 1
                    # 注入 prompt 增强
                    prompt_addition = skill.get_prompt_addition()
                    self.logger.info(f"Registered RAG skill tool: {skill.get_metadata().name}")
            if rag_tools_count > 0:
                self.logger.info(f"Injected {rag_tools_count} RAG tools from skill_manager")
        elif self._skill_manager and not self._allow_rag_tools:
            self.logger.info("RAG tools are disabled for CodeAgent; use RAGAgent for knowledge tasks")

        # 2. 如果有 DockerRunner，替换 run_python 工具为 Docker 版本
        if self._docker_runner:
            tools_to_register = self._inject_docker_tools(tools_to_register)
            self.logger.info("Docker execution enabled for code tools")

        system_prompt = None
        if self.profile.prompt:
            system_prompt = self.profile.prompt
        if not self._allow_rag_tools:
            system_prompt = system_prompt or self._build_multi_agent_code_prompt(tools_to_register)

        self._react_agent = ReactAgent(
            client=self.client,
            tools=tools_to_register,
            max_steps=self.max_steps,
            system_prompt=system_prompt,
            logger=self.logger,
            agent_id=self.name,
            resource_manager=self.resource_manager,
            enable_long_term_memory=False,  # Code Agent 禁用长期记忆
        )
        self.logger.info(f"ReactAgent initialized with {len(tools_to_register)} tools")

    def _normalize_docker_requirements(self, raw_requirements: Any) -> List[str]:
        """Normalize optional Docker Python package requirements."""
        if raw_requirements is None:
            return []
        if isinstance(raw_requirements, str):
            return [item.strip() for item in raw_requirements.splitlines() if item.strip()]
        if isinstance(raw_requirements, list):
            return [str(item).strip() for item in raw_requirements if str(item).strip()]
        return []

    def _infer_python_requirements(self, code: str, explicit_requirements: List[str]) -> List[str]:
        """Infer common scientific Python dependencies used by generated task scripts."""
        requirements = list(dict.fromkeys(explicit_requirements))
        text = str(code or "")
        import_to_package = {
            "numpy": "numpy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "scipy": "scipy",
        }
        for module_name, package_name in import_to_package.items():
            patterns = (f"import {module_name}", f"from {module_name} ")
            if any(pattern in text for pattern in patterns) and package_name not in requirements:
                requirements.append(package_name)
        return requirements

    def _infer_shell_requirements(self, command: str, explicit_requirements: List[str]) -> List[str]:
        """Infer package requirements for simple `python path.py` shell commands."""
        requirements = list(dict.fromkeys(explicit_requirements))
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            parts = command.split()

        for index, part in enumerate(parts[:-1]):
            executable = part.strip("\"'").replace("\\", "/").split("/")[-1].lower()
            if executable not in {"python", "python3"}:
                continue
            script = parts[index + 1].strip("\"'")
            if not script.endswith(".py"):
                continue
            script_path = os.path.join(os.getcwd(), script)
            if not os.path.exists(script_path):
                continue
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    requirements = self._infer_python_requirements(f.read(), requirements)
            except Exception:
                continue
        return requirements

    def _build_multi_agent_code_prompt(self, tools: List[Any]) -> str:
        return build_multi_agent_code_prompt(tools)
        tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        tool_names = ", ".join(tool.name for tool in tools)
        return f"""# ROLE: Multi-Agent CodeAgent

你是多 Agent 系统中的 CodeAgent，只负责代码、文件、实验、测试和技术文档落地。

## 职责边界
1. 不要执行 RAG 检索。RAG 检索和生成由独立 RAGAgent 完成。
2. 不要调用任何知识检索、专家检索、`*_search`、`retrieve` 或未出现在工具列表中的工具。
3. 如果任务描述里包含“上游子任务结果”，那就是 RAGAgent 或其它 Agent 已经提供的上下文。你必须优先使用这些上游结果继续工作。
4. 如果上游知识不足，直接在最终结果里说明缺口，不要尝试调用不存在的 RAG 工具。
5. 所有新建代码、报告、图表和中间结果默认放在 `task/` 目录。
6. 运行 Python 时使用 `run_python`；在多 Agent Docker 模式下该工具会在容器中执行，并可访问 `/workspace` 下的项目代码。
7. 需要结束当前子任务时，调用 `task_complete` 或 `finish`。

## 可用工具
{tool_lines}

## 工具约束
`action` 必须严格是以下工具之一：
{tool_names}

## 响应格式
必须只返回可被 `json.loads` 解析的 JSON 对象：
{{"thought": "当前判断和下一步", "action": "工具名", "action_input": {{...}}}}

如果已经可以回答或交付当前子任务，使用：
{{"thought": "已经完成当前子任务", "action": "task_complete", "action_input": {{"message": "结果摘要"}}}}
"""

    def _inject_docker_tools(self, tools: List[Any]) -> List[Any]:
        """将执行类工具替换为 Docker 隔离版本"""
        docker_runner = self._docker_runner

        def docker_run_python(arguments: Dict[str, Any]) -> str:
            """在 Docker 容器中执行 Python 代码"""
            code = arguments.get("code")
            path_value = arguments.get("path")
            requirements = self._normalize_docker_requirements(arguments.get("requirements"))

            if isinstance(code, str) and code.strip():
                result = docker_runner.execute(
                    code=code,
                    language="python",
                    requirements=self._infer_python_requirements(code, requirements),
                )
            elif isinstance(path_value, str) and path_value.strip():
                # 读取文件内容
                try:
                    with open(path_value, "r", encoding="utf-8") as f:
                        code = f.read()
                    result = docker_runner.execute(
                        code=code,
                        language="python",
                        requirements=self._infer_python_requirements(code, requirements),
                    )
                except FileNotFoundError:
                    return f"文件未找到: {path_value}"
                except Exception as e:
                    return f"读取文件失败: {e}"
            else:
                return "run_python 工具需要 'code' 或 'path' 参数。"

            if result.get("success"):
                output = result.get("stdout", "")
                meta = (
                    f"executed_in={result.get('executed_in', 'docker')}; "
                    f"returncode={result.get('returncode', 0)}"
                )
                return "\n".join(part for part in [output, meta] if part).strip()
            else:
                stderr = result.get("stderr", "")
                error = result.get("error", "")
                raise RuntimeError(f"Docker Python execution failed: {error}\n{stderr}".strip())

        def docker_run_shell(arguments: Dict[str, Any]) -> str:
            """在 Docker 容器的 /workspace 中执行 shell 命令"""
            command = arguments.get("command")
            if not isinstance(command, str) or not command.strip():
                return "run_shell 工具需要 'command' 参数。"
            requirements = self._normalize_docker_requirements(arguments.get("requirements"))
            requirements = self._infer_shell_requirements(command, requirements)
            result = docker_runner.execute(code=command, language="shell", requirements=requirements)
            if result.get("success"):
                output = result.get("stdout", "")
                stderr = result.get("stderr", "")
                meta = (
                    f"executed_in={result.get('executed_in', 'docker')}; "
                    f"returncode={result.get('returncode', 0)}"
                )
                return "\n".join(part for part in [output, f"stderr:\n{stderr}" if stderr else "", meta] if part).strip()
            stderr = result.get("stderr", "")
            error = result.get("error", "")
            raise RuntimeError(f"Docker shell execution failed: {error}\n{stderr}".strip())

        def docker_run_tests(arguments: Dict[str, Any]) -> str:
            """在 Docker 容器中运行测试。"""
            framework = arguments.get("framework", "pytest")
            test_path = arguments.get("test_path", ".")
            if not isinstance(test_path, str) or not test_path.strip():
                test_path = "."
            if framework == "unittest":
                command = f"python -m unittest discover -s {test_path}"
                requirements = []
            else:
                verbose = " -v" if arguments.get("verbose", False) else ""
                command = f"python -m pytest{verbose} {test_path}"
                requirements = ["pytest"]
            return docker_run_shell({"command": command, "requirements": requirements})

        from ..tools.base import Tool
        new_tools = []
        for tool in tools:
            if tool.name == "run_python":
                # 替换为 Docker 版本
                new_tools.append(Tool(
                    name="run_python",
                    description=(
                        "[Docker] Execute Python code in isolated container. "
                        "Optional action_input.requirements may list pip packages; common scientific imports are auto-detected. "
                        + tool.description
                    ),
                    runner=docker_run_python,
                ))
            elif tool.name == "run_shell":
                new_tools.append(Tool(
                    name="run_shell",
                    description=(
                        "[Docker] Execute a shell command in isolated container under /workspace. "
                        "Optional action_input.requirements may list pip packages. "
                        + tool.description
                    ),
                    runner=docker_run_shell,
                ))
            elif tool.name == "run_tests":
                new_tools.append(Tool(
                    name="run_tests",
                    description="[Docker] Run tests in isolated container under /workspace. " + tool.description,
                    runner=docker_run_tests,
                ))
            else:
                new_tools.append(tool)

        return new_tools

    def add_tools(self, tools: List[Any]):
        """添加工具"""
        self._tools.extend(tools)
        if self._react_agent:
            for tool in tools:
                self._react_agent.tools[tool.name] = tool
        self.logger.info(f"Added {len(tools)} tools, total: {len(self._tools)}")

    def process(self, task: SubTask) -> Dict[str, Any]:
        """Process a code/execution task."""
        self.logger.info(f"Processing code task: {task.description[:100]}...")

        if not self._react_agent:
            return {
                "success": False,
                "error": "ReactAgent not initialized. Provide client and tools.",
                "result": None,
            }

        try:
            self._react_agent.reset_conversation()
            task_description = task.description
            if self.memory_hub and getattr(self.profile, "memory_enabled", True):
                memory_context = self.memory_hub.build_context(task.description, agent_name=self.name)
                if memory_context:
                    task_description = (
                        f"{task.description}\n\n"
                        "Multi-agent memory context:\n"
                        f"{memory_context}\n\n"
                        "Use the memory context only when it is relevant to this code task."
                    )
                self.memory_hub.add_event(
                    self.name,
                    "task_start",
                    task.description,
                    shared=False,
                    metadata={"subtask_id": task.id},
                )

            result = self._react_agent.run(
                task_description,
                max_steps=self.max_steps,
                stop_event=task.cancel_event,
            )

            if self.memory_hub:
                self.memory_hub.add_event(
                    self.name,
                    "result",
                    result.get("final_answer", ""),
                    shared=True,
                    metadata={"subtask_id": task.id, "cancelled": result.get("cancelled", False)},
                )
                self.memory_hub.store_agent_memory(
                    self.name,
                    "engineering_experience",
                    result.get("final_answer", ""),
                    metadata={
                        "subtask_id": task.id,
                        "task": task.description[:500],
                        "cancelled": result.get("cancelled", False),
                    },
                )

            return {
                "success": not result.get("cancelled", False),
                "result": result["final_answer"],
                "steps_count": len(result.get("steps", [])),
                "metrics": self._react_agent.get_metrics(),
                "cancelled": result.get("cancelled", False),
            }
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "result": None,
            }


    def execute(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """直接执行代码

        Args:
            code: 要执行的代码
            timeout: 超时时间（秒）

        Returns:
            执行结果
        """
        task = SubTask(
            id=f"code_{uuid.uuid4().hex[:6]}",
            type=TaskType.CODE_EXECUTION,
            description=code,
        )
        return self.process(task)


# ============================================================
# Docker 运行器
# ============================================================

class DockerRunner:
    """Docker 隔离执行器

    在 Docker 容器中安全执行代码，避免对宿主机的风险。
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 300,
        memory_limit: str = "512m",
        cpu_limit: str = "0.5",
        workspace_dir: Optional[str] = None,
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.workspace_dir = os.path.abspath(workspace_dir or os.getcwd())
        self._logger = get_logger("docker.runner")

    def health_check(self) -> Dict[str, Any]:
        import shutil
        import subprocess
        import sys

        docker_path = shutil.which("docker")
        diagnostics = {
            "available": False,
            "docker_path": docker_path,
            "python_executable": sys.executable,
            "workspace_dir": self.workspace_dir,
        }
        if not docker_path:
            diagnostics["error"] = "docker executable not found in PATH"
            return diagnostics

        try:
            result = subprocess.run(
                [docker_path, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception as exc:
            diagnostics["error"] = str(exc)
            return diagnostics

        diagnostics["returncode"] = result.returncode
        diagnostics["stdout"] = result.stdout.strip()
        diagnostics["stderr"] = result.stderr.strip()
        if result.returncode != 0:
            diagnostics["available"] = False
            diagnostics["error"] = result.stderr.strip() or result.stdout.strip() or "docker --version failed"
            return diagnostics

        try:
            daemon_result = subprocess.run(
                [docker_path, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception as exc:
            diagnostics["available"] = False
            diagnostics["error"] = f"docker daemon check failed: {exc}"
            return diagnostics

        diagnostics["daemon_returncode"] = daemon_result.returncode
        diagnostics["daemon_stdout"] = daemon_result.stdout.strip()
        diagnostics["daemon_stderr"] = daemon_result.stderr.strip()
        diagnostics["available"] = daemon_result.returncode == 0
        if daemon_result.returncode != 0:
            diagnostics["error"] = (
                daemon_result.stderr.strip()
                or daemon_result.stdout.strip()
                or "docker daemon is not reachable"
            )
        return diagnostics

    def execute(
        self,
        code: str,
        language: str = "python",
        requirements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """在 Docker 容器中执行代码

        Args:
            code: 要执行的代码
            language: 编程语言
            requirements: 依赖包列表

        Returns:
            执行结果字典
        """
        import shutil
        import subprocess
        import tempfile
        import os

        self._logger.info(f"Executing {language} code in Docker container")

        # 创建临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            if language == "python":
                code_file = os.path.join(tmpdir, "main.py")
                with open(code_file, "w", encoding="utf-8") as f:
                    f.write(code)

                # 创建 requirements.txt
                if requirements:
                    req_file = os.path.join(tmpdir, "requirements.txt")
                    with open(req_file, "w") as f:
                        f.write("\n".join(requirements))

                # Docker 命令
                docker_cmd = [
                    "docker", "run", "--rm",
                    "--memory", self.memory_limit,
                    "--cpus", self.cpu_limit,
                    "-v", f"{tmpdir}:/code",
                    "-v", f"{self.workspace_dir}:/workspace",
                    "-w", "/workspace",
                    "-e", "PYTHONPATH=/workspace",
                    "-e", "PYTHONIOENCODING=utf-8",
                    self.image,
                ]

                if requirements:
                    docker_cmd.extend(["bash", "-c", "pip install -q -r /code/requirements.txt && python /code/main.py"])
                else:
                    docker_cmd.extend(["python", "/code/main.py"])

            elif language == "javascript":
                code_file = os.path.join(tmpdir, "main.js")
                with open(code_file, "w", encoding="utf-8") as f:
                    f.write(code)

                docker_cmd = [
                    "docker", "run", "--rm",
                    "--memory", self.memory_limit,
                    "--cpus", self.cpu_limit,
                    "-v", f"{tmpdir}:/code",
                    "-v", f"{self.workspace_dir}:/workspace",
                    "-w", "/workspace",
                    "-e", "PYTHONIOENCODING=utf-8",
                    "node:20-slim",
                    "node", "/code/main.js",
                ]
            elif language == "shell":
                shell_code = code
                if requirements:
                    req_file = os.path.join(tmpdir, "requirements.txt")
                    with open(req_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(requirements))
                    shell_code = f"pip install -q -r /code/requirements.txt && {code}"
                docker_cmd = [
                    "docker", "run", "--rm",
                    "--memory", self.memory_limit,
                    "--cpus", self.cpu_limit,
                    "-v", f"{tmpdir}:/code",
                    "-v", f"{self.workspace_dir}:/workspace",
                    "-w", "/workspace",
                    "-e", "PYTHONPATH=/workspace",
                    "-e", "PYTHONIOENCODING=utf-8",
                    self.image,
                    "bash", "-lc", shell_code,
                ]
            else:
                return {
                    "success": False,
                    "error": f"Unsupported language: {language}",
                    "stdout": "",
                    "stderr": "",
                }

            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                )

                return {
                    "success": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "executed_in": "docker",
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Execution timed out after {self.timeout}s",
                    "stdout": "",
                    "stderr": "",
                    "timeout": True,
                }
            except FileNotFoundError:
                docker_path = shutil.which("docker")
                return {
                    "success": False,
                    "error": "Docker executable not found in PATH.",
                    "stdout": "",
                    "stderr": "",
                    "docker_path": docker_path,
                    "executed_in": "docker",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "stdout": "",
                    "stderr": "",
                }


# ============================================================
# 任务分解器
# ============================================================

class TaskDecomposer:
    """任务分解器

    使用 LLM 将复杂任务分解为可并行执行的子任务。
    """

    def __init__(
        self,
        client: BaseLLMClient,
        logger: Optional[AgentLogger] = None,
    ):
        self.client = client
        self.logger = logger or get_logger("task_decomposer")

    def _parse_llm_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的 JSON 响应，支持 markdown 包裹和纯文本格式"""
        if not response or not response.strip():
            return None

        text = response.strip()

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试去掉 markdown 代码块包裹
        import re
        # 匹配 ```json ... ``` 或 ``` ... ```
        code_block_patterns = [
            r'```json\s*([\s\S]+?)\s*```',
            r'```\s*([\s\S]+?)\s*```',
        ]
        for pattern in code_block_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # 尝试提取 JSON 对象（兼容不完整的 markdown）
        try:
            start = text.find("{")
            if start >= 0:
                parsed, _ = json.JSONDecoder().raw_decode(text[start:])
                return parsed
        except json.JSONDecodeError:
            pass

        return None

    def decompose(self, task: str) -> TaskDecomposition:
        """分解任务

        Args:
            task: 原始任务描述

        Returns:
            TaskDecomposition: 分解结果
        """
        prompt = f"""你是一个任务分解专家。请将以下任务分解为多个可以并行执行的子任务。

任务：{task}

请以 JSON 格式返回分解结果：
{{
    "requires_rag": true/false,  // 是否需要知识库检索
    "requires_code": true/false,  // 是否需要代码执行
    "sub_tasks": [
        {{
            "id": "task_1",
            "type": "knowledge_query|code_execution|analysis|general",
            "description": "子任务描述",
            "dependencies": [],  // 依赖的子任务 ID 列表
            "priority": 0-10
        }}
    ],
    "execution_plan": "执行计划说明"
}}

规则：
1. 如果任务需要专业知识（RAG），必须包含知识查询子任务
2. 如果任务需要计算或代码执行，必须包含代码执行子任务
3. 独立的子任务之间没有依赖关系，可以并行执行
4. 有依赖关系的子任务必须指定依赖
5. 如果代码、分析或文档子任务需要使用专业知识，必须依赖相应的 knowledge_query 子任务，不要让 Code Agent 重复检索
6. knowledge_query 子任务由独立 RAG Agent 完成完整的“检索 + 生成”链路；code_execution 子任务只负责代码、文件、实验和测试
7. 返回的 JSON 必须只包含上述字段，不要有其他内容
"""

        try:
            response = self.client.respond([{"role": "user", "content": prompt}])
            data = self._parse_llm_json_response(response)

            if data is None:
                raise ValueError(f"无法解析 LLM 响应为 JSON: {response[:200]}")

            sub_tasks = []
            for task_data in data.get("sub_tasks", []):
                task_type_name = task_data["type"].upper()
                if task_type_name not in TaskType.__members__:
                    task_type_name = "GENERAL"
                task_type = TaskType[task_type_name]
                sub_tasks.append(SubTask(
                    id=task_data["id"],
                    type=task_type,
                    description=task_data["description"],
                    dependencies=task_data.get("dependencies", []),
                    priority=task_data.get("priority", 0),
                ))

            return TaskDecomposition(
                original_task=task,
                sub_tasks=sub_tasks,
                requires_rag=data.get("requires_rag", False),
                requires_code=data.get("requires_code", False),
                execution_plan=data.get("execution_plan", ""),
            )
        except Exception as e:
            self.logger.error(f"Task decomposition failed: {e}, response={response[:500] if 'response' in dir() else 'N/A'}", exc_info=True)
            # 回退：创建知识查询 + 代码执行两个独立任务
            return TaskDecomposition(
                original_task=task,
                sub_tasks=[
                    SubTask(
                        id="rag_task",
                        type=TaskType.KNOWLEDGE_QUERY,
                        description=f"检索任务相关专业知识：{task}",
                        priority=10,
                    ),
                    SubTask(
                        id="code_task",
                        type=TaskType.CODE_EXECUTION,
                        description=task,
                        dependencies=["rag_task"],
                        priority=5,
                    ),
                ],
                requires_rag=True,
                requires_code=True,
                execution_plan="Fallback: 分别执行知识检索和代码执行任务",
            )


# ============================================================
# 任务调度器
# ============================================================

class TaskScheduler:
    """任务调度器

    根据任务依赖关系和资源可用性调度执行子任务。
    支持并行执行无依赖的子任务。
    """

    def __init__(
        self,
        logger: Optional[AgentLogger] = None,
        max_parallel: int = 3,
        task_timeout: int = 300,
    ):
        self.logger = logger or get_logger("task_scheduler")
        self.max_parallel = max_parallel
        self._task_timeout = task_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)

    def schedule(self, tasks: List[SubTask]) -> List[List[SubTask]]:
        """调度任务分组

        返回多轮执行批次，每批内的任务可以并行执行。

        Args:
            tasks: 子任务列表

        Returns:
            分批次的任务列表
        """
        # 构建依赖图
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: 0 for t in tasks}

        for task in tasks:
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.id] += 1

        # Kahn 算法分层
        batches: List[List[SubTask]] = []
        remaining = set(task_map.keys())

        while remaining:
            # 找出所有入度为 0 的任务
            ready = [task_map[tid] for tid in remaining if in_degree[tid] == 0]

            if not ready:
                # 可能有循环依赖，回退到按优先级选择
                self.logger.warning("Circular dependency detected, using fallback scheduling")
                ready = [task_map[tid] for tid in remaining][:1]

            batches.append(ready)

            # 更新入度
            for task in ready:
                remaining.remove(task.id)
                for tid, t in task_map.items():
                    if task.id in t.dependencies:
                        in_degree[tid] -= 1

        self.logger.info(f"Scheduled {len(tasks)} tasks into {len(batches)} batches")
        return batches

    def execute_batch(
        self,
        batch: List[SubTask],
        agent_map: Dict[TaskType, BaseAgent],
    ) -> List[SubTask]:
        """并行执行一批任务

        Args:
            batch: 任务批次
            agent_map: 任务类型到 Agent 的映射

        Returns:
            执行结果列表
        """
        futures: Dict[str, Future] = {}

        for task in batch:
            agent = agent_map.get(task.type)
            if agent:
                task.status = "running"
                task.agent_name = getattr(agent, "name", "") or task.agent_name
                future = self._executor.submit(self._execute_task, task, agent)
                futures[task.id] = future
            else:
                task.status = "failed"
                task.error = f"No agent available for task type: {task.type}"
                self.logger.warning(f"No agent for task type: {task.type}")

        # 收集结果
        for task_id, future in futures.items():
            task = next(t for t in batch if t.id == task_id)
            try:
                result = future.result(timeout=self._task_timeout)
                task.result = result

                # 处理空结果或 None
                if not isinstance(result, dict):
                    task.status = "failed"
                    task.error = f"Unexpected result type: {type(result).__name__}"
                    self.logger.error(f"Task {task_id} returned unexpected type: {type(result).__name__}")
                elif result.get("success"):
                    task.status = "completed"
                else:
                    task.status = "failed"
                    error_msg = result.get("error")
                    task.error = str(error_msg) if error_msg is not None else "Unknown error"
                    self.logger.warning(f"Task {task_id} failed: {task.error}")

            except TimeoutError:
                task.cancel_event.set()
                future.cancel()
                task.status = "partial"
                task.error = f"Task timeout after {self._task_timeout}s"
                task.result = {
                    "success": False,
                    "partial": True,
                    "error": task.error,
                    "message": (
                        "Subtask timed out before an explicit task_complete call. "
                        "Any files already written by this subtask may still be usable by downstream tasks."
                    ),
                }
                self.logger.error(f"Task {task_id} timed out after {self._task_timeout}s")
            except Exception as e:
                task.status = "failed"
                task.error = str(e) or type(e).__name__
                self.logger.error(f"Task {task_id} execution failed: {e}", exc_info=True)

        return batch

    def _execute_task(self, task: SubTask, agent: BaseAgent) -> Dict[str, Any]:
        """执行单个任务"""
        return agent.process(task)


# ============================================================
# 结果聚合器
# ============================================================

class ResultAggregator:
    """结果聚合器

    将多个子任务的结果聚合成最终答案。
    """

    def __init__(
        self,
        client: Optional[BaseLLMClient] = None,
        logger: Optional[AgentLogger] = None,
    ):
        self.client = client
        self.logger = logger or get_logger("result_aggregator")

    def aggregate(
        self,
        original_task: str,
        sub_results: List[SubTask],
    ) -> str:
        """聚合结果

        Args:
            original_task: 原始任务
            sub_results: 子任务结果列表

        Returns:
            聚合后的最终答案
        """
        if not self.client:
            # 无 LLM 时，直接拼接结果
            return self._simple_aggregate(sub_results)

        # 构建聚合 prompt
        context_parts = []
        for task in sub_results:
            if task.status in {"completed", "partial"} and task.result:
                context_parts.append(
                    f"### {task.type.value} [{task.status}] - {task.description}\n"
                    f"{self._format_task_result(task.result)}"
                )

        context = "\n\n".join(context_parts)

        prompt = f"""你是一个任务汇总专家。请根据以下子任务结果，生成完整的最终答案。

原始任务：{original_task}

子任务结果：
{context}

请生成最终答案，要求：
1. 整合所有子任务的结果
2. 回答原始问题
3. 如果某些子任务失败，说明原因
4. 答案要完整、准确、有条理
"""

        try:
            response = self.client.respond([{"role": "user", "content": prompt}])
            return response
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}", exc_info=True)
            return self._simple_aggregate(sub_results)

    def _simple_aggregate(self, sub_results: List[SubTask]) -> str:
        """简单拼接结果"""
        parts = []
        for task in sub_results:
            if task.status in {"completed", "partial"} and task.result:
                parts.append(self._format_task_result(task.result))
            elif task.status == "failed":
                parts.append(f"[{task.type.value} failed]: {task.error}")

        return "\n\n".join(parts) if parts else "No results available"

    def _format_task_result(self, result: Any, max_chars: int = 4000) -> str:
        """Return a compact result summary for aggregation prompts."""
        if isinstance(result, dict):
            if "result" in result and result["result"] is not None:
                text = str(result["result"])
            elif "results" in result:
                lines = []
                for item in result.get("results", [])[:3]:
                    if isinstance(item, dict):
                        content = str(item.get("content", ""))
                        source = item.get("source") or item.get("domain") or item.get("file_name")
                        prefix = f"[{source}] " if source else ""
                        lines.append(f"- {prefix}{content[:1200]}")
                    else:
                        lines.append(f"- {str(item)[:1200]}")
                text = "\n".join(lines)
            elif "stdout" in result:
                text = str(result.get("stdout", ""))
            elif "message" in result:
                text = str(result["message"])
            else:
                text = json.dumps(
                    {k: v for k, v in result.items() if k not in {"steps", "metrics"}},
                    ensure_ascii=False,
                    default=str,
                )
        else:
            text = str(result)

        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]"
        return text


# ============================================================
# 主协调 Agent
# ============================================================

class OrchestratorAgent:
    """主协调 Agent

    整合所有专业 Agent，统一接收用户任务并返回最终答案。

    使用流程：
    1. 接收用户任务
    2. 使用 TaskDecomposer 分解任务
    3. 使用 TaskScheduler 调度执行
    4. 使用各个专业 Agent 执行子任务
    5. 使用 ResultAggregator 聚合结果
    6. 返回最终答案
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        *,
        code_tools: Optional[List[Any]] = None,
        skill_manager: Optional[SkillManager] = None,
        resource_manager: Optional[ResourceManager] = None,
        max_parallel: int = 3,
        use_docker: bool = True,
        docker_image: str = "python:3.11-slim",
        logger: Optional[AgentLogger] = None,
        agent_profiles: Optional[Dict[str, AgentProfile]] = None,
        mcp_tools: Optional[List[Any]] = None,
        memory_manager: Optional[Any] = None,
        memory_config: Optional[MultiAgentMemoryConfig] = None,
        enable_memory: bool = True,
    ):
        self.llm_client = llm_client
        self.logger = logger or get_logger("orchestrator")
        self.resource_manager = resource_manager
        self.skill_manager = skill_manager
        self.agent_profiles = agent_profiles or {}
        resolved_memory_config = memory_config or MultiAgentMemoryConfig(enabled=enable_memory)
        for profile_name, profile in self.agent_profiles.items():
            if getattr(profile, "memory_policy", None):
                resolved_memory_config.policies[profile_name] = profile.memory_policy
                resolved_memory_config.policies[getattr(profile, "name", profile_name)] = profile.memory_policy
            for template_name, template in getattr(profile, "memory_write_templates", {}).items():
                resolved_memory_config.write_templates[template_name] = template
        if enable_memory and memory_manager is None:
            try:
                from ..memory import MemoryManager

                memory_manager = MemoryManager(llm_client=llm_client)
            except Exception as exc:
                self.logger.warning(f"Multi-agent long-term memory disabled: {exc}")
        self.memory_hub = MultiAgentMemoryHub(
            memory_manager=memory_manager,
            config=resolved_memory_config,
            logger=self.logger,
        )
        rag_mcp_tools, code_mcp_tools = split_mcp_tools(mcp_tools or [])

        rag_profile = self.agent_profiles.get("rag")
        if rag_profile is None:
            rag_profile = RAGAgentProfile(mcp_tools=rag_mcp_tools)
        elif isinstance(rag_profile, RAGAgentProfile):
            rag_profile.mcp_tools.extend(filter_tools_by_profile(rag_mcp_tools, rag_profile))

        code_profile = self.agent_profiles.get("code")
        if code_profile is None:
            code_profile = CodeAgentProfile(
                allow_rag_tools=False,
                use_docker=use_docker,
                mcp_tools=code_mcp_tools,
            )
        elif isinstance(code_profile, CodeAgentProfile):
            code_profile.mcp_tools.extend(filter_tools_by_profile(code_mcp_tools, code_profile))

        # 初始化组件
        self.decomposer = TaskDecomposer(llm_client, self.logger)
        task_timeout = int(os.getenv("MULTI_AGENT_TASK_TIMEOUT", "900"))
        self.scheduler = TaskScheduler(self.logger, max_parallel, task_timeout=task_timeout)
        self.aggregator = ResultAggregator(llm_client, self.logger)

        # 初始化专业 Agent
        self.rag_agent = RAGAgent(
            client=llm_client,
            skill_manager=skill_manager,
            logger=self.logger,
            profile=rag_profile if isinstance(rag_profile, RAGAgentProfile) else None,
            mcp_tools=rag_mcp_tools,
            memory_hub=self.memory_hub,
        )
        # Docker 运行器（可选）
        self.use_docker = use_docker
        self.docker_runner = DockerRunner(image=docker_image, workspace_dir=os.getcwd()) if use_docker else None

        self.code_agent = CodeAgent(
            client=llm_client,
            tools=code_tools or [],
            logger=self.logger,
            resource_manager=resource_manager,
            skill_manager=skill_manager,
            docker_runner=self.docker_runner,
            allow_rag_tools=False,
            profile=code_profile if isinstance(code_profile, CodeAgentProfile) else None,
            memory_hub=self.memory_hub,
        )

        # Agent 到任务类型的映射
        self._agent_map: Dict[TaskType, BaseAgent] = {
            TaskType.KNOWLEDGE_QUERY: self.rag_agent,
            TaskType.CODE_EXECUTION: self.code_agent,
            TaskType.ANALYSIS: self.code_agent,
            TaskType.GENERAL: self.code_agent,
        }

        # 注册 RAG 技能
        if skill_manager:
            self._register_rag_skills(skill_manager)

        self._task_count = 0

    def _inject_trace(self, tracer: Any, trace_id: str) -> None:
        """Inject the same trace anchor into every skill used by sub-agents."""
        if not self.skill_manager:
            return
        for skill in getattr(self.skill_manager, "skills", {}).values():
            setattr(skill, "current_trace_id", trace_id)
            if hasattr(skill, "_tracer"):
                setattr(skill, "_tracer", tracer)

    def _clear_trace(self) -> None:
        if not self.skill_manager:
            return
        for skill in getattr(self.skill_manager, "skills", {}).values():
            if hasattr(skill, "_tracer"):
                setattr(skill, "_tracer", None)
            if hasattr(skill, "current_trace_id"):
                setattr(skill, "current_trace_id", None)

    def _register_rag_skills(self, skill_manager: SkillManager):
        """从 SkillManager 注册 RAG 技能"""
        from ..skills.builtin.base_rag_skill import BaseRAGSkill

        allowed_skills = {
            str(item)
            for item in (getattr(self.rag_agent.profile, "metadata", {}) or {}).get("skills", [])
            if item
        }
        allowed_skills.update(
            str(item)
            for item in (getattr(self.rag_agent.profile, "metadata", {}) or {}).get("skill_names", [])
            if item
        )
        for name, skill in skill_manager.skills.items():
            meta = skill.get_metadata()
            if allowed_skills and name not in allowed_skills and meta.name not in allowed_skills:
                continue
            if isinstance(skill, BaseRAGSkill):
                self.rag_agent.register_rag_skill(skill)

        self.logger.info(f"Registered {len(self.rag_agent.get_available_domains())} RAG domains")

    def add_rag_skill(self, skill: BaseRAGSkill):
        """动态添加 RAG 技能"""
        self.rag_agent.register_rag_skill(skill)

    def add_code_tools(self, tools: List[Any]):
        """动态添加工具"""
        self.code_agent.add_tools(tools)

    def _dependency_context_for_task(
        self,
        task: SubTask,
        completed_tasks: Dict[str, SubTask],
        max_chars: int = 6000,
    ) -> str:
        if not task.dependencies:
            return ""

        parts = []
        for dep_id in task.dependencies:
            dep = completed_tasks.get(dep_id)
            if not dep:
                continue
            if dep.status in {"completed", "partial"} and dep.result:
                parts.append(
                    f"### Upstream result: {dep.id} ({dep.type.value}, {dep.status})\n"
                    f"Task: {dep.description}\n"
                    f"{self.aggregator._format_task_result(dep.result, max_chars=2500)}"
                )
            elif dep.status == "failed":
                parts.append(
                    f"### Upstream result: {dep.id} ({dep.type.value}, failed)\n"
                    f"Task: {dep.description}\n"
                    f"Error: {dep.error}"
                )

        if not parts:
            return ""

        context = "\n\n".join(parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...[upstream context truncated]"
        return context

    def _attach_dependency_context(self, task: SubTask, completed_tasks: Dict[str, SubTask]) -> None:
        context = self._dependency_context_for_task(task, completed_tasks)
        if not context or "上游子任务结果：" in task.description:
            return

        task.description = (
            f"{task.description}\n\n"
            "上游子任务结果：\n"
            f"{context}\n\n"
            "请优先使用这些上游结果完成当前子任务；不要重复执行已经完成的检索。"
        )

    def run(self, task: str, *, trace: bool = True) -> Dict[str, Any]:
        """执行任务

        Args:
            task: 用户任务描述
            trace: 是否返回执行追踪信息

        Returns:
            执行结果字典
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        self._task_count += 1

        self.logger.info(f"[{task_id}] Starting task: {task[:100]}...")
        self.logger.set_context(task_id=task_id)
        self.memory_hub.start_task(task_id, task)
        tracer = None
        trace_id = None

        try:
            if trace:
                from ..rag.observability import TraceManager

                tracer = TraceManager()
                trace_id = tracer.start_trace(trace_type="Query")
                tracer.metadata["question"] = task
                tracer.metadata["task_id"] = task_id
                self._inject_trace(tracer, trace_id)

            # 1. 任务分解
            self.logger.info("Decomposing task...")
            decomposition = self.decomposer.decompose(task)

            self.logger.info(
                f"Task decomposed into {len(decomposition.sub_tasks)} sub-tasks, "
                f"rag={decomposition.requires_rag}, code={decomposition.requires_code}"
            )

            # 2. 任务调度
            batches = self.scheduler.schedule(decomposition.sub_tasks)

            # 3. 执行批次
            all_results: List[SubTask] = []
            completed_tasks: Dict[str, SubTask] = {}
            task_status: Dict[str, str] = {}
            for i, batch in enumerate(batches):
                self.logger.info(f"Executing batch {i+1}/{len(batches)} with {len(batch)} tasks")
                runnable = []
                skipped = []
                for sub_task in batch:
                    failed_deps = [
                        dep for dep in sub_task.dependencies
                        if task_status.get(dep) == "failed"
                    ]
                    if failed_deps:
                        sub_task.status = "failed"
                        sub_task.error = f"Skipped because dependencies failed: {', '.join(failed_deps)}"
                        skipped.append(sub_task)
                        self.logger.warning(f"Skipping {sub_task.id}: {sub_task.error}")
                    else:
                        self._attach_dependency_context(sub_task, completed_tasks)
                        runnable.append(sub_task)

                results = self.scheduler.execute_batch(runnable, self._agent_map) if runnable else []
                batch_results = skipped + results
                for sub_task in batch_results:
                    task_status[sub_task.id] = sub_task.status
                    self.memory_hub.record_subtask_result(sub_task)
                    if sub_task.status in {"completed", "partial"}:
                        completed_tasks[sub_task.id] = sub_task
                all_results.extend(batch_results)

            # 4. 聚合结果
            self.logger.info("Aggregating results...")
            final_answer = self.aggregator.aggregate(task, all_results)
            if tracer:
                tracer.metadata["llm_answer"] = final_answer
                tracer.finish_and_save()

            duration = time.time() - start_time
            self.logger.info(f"[{task_id}] Task completed in {duration:.2f}s")
            completed_count = sum(1 for t in all_results if t.status == "completed")
            partial_count = sum(1 for t in all_results if t.status == "partial")
            failed_count = sum(1 for t in all_results if t.status == "failed")
            self.memory_hub.store_task_summary(
                original_task=task,
                final_answer=final_answer,
                completed_count=completed_count,
                failed_count=failed_count,
            )

            if failed_count == 0 and partial_count == 0:
                overall_status = "success"
            elif completed_count > 0 or partial_count > 0:
                overall_status = "partial_success"
            else:
                overall_status = "failed"

            result = {
                "success": overall_status == "success",
                "overall_status": overall_status,
                "task_id": task_id,
                "final_answer": final_answer,
                "duration": duration,
                "sub_tasks_count": len(all_results),
                "completed_count": completed_count,
                "partial_count": partial_count,
                "failed_count": failed_count,
            }
            if trace_id:
                result["rag_trace_id"] = trace_id

            if trace:
                result["trace"] = {
                    "decomposition": {
                        "original_task": task,
                        "requires_rag": decomposition.requires_rag,
                        "requires_code": decomposition.requires_code,
                        "execution_plan": decomposition.execution_plan,
                    },
                    "sub_tasks": [
                        {
                            "id": t.id,
                            "type": t.type.value,
                            "description": t.description,
                            "status": t.status,
                            "error": t.error,
                        }
                        for t in all_results
                    ],
                }
                result["memory_replay"] = self.memory_hub.build_replay(task_id=task_id)

            return result

        except Exception as e:
            if tracer:
                tracer.metadata["llm_answer"] = f"任务失败：{e}"
                tracer.finish_and_save()
            self.logger.error(f"[{task_id}] Task failed: {e}", exc_info=True)
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "rag_trace_id": trace_id,
                "duration": time.time() - start_time,
            }
        finally:
            self._clear_trace()

    def query_knowledge(
        self,
        query: str,
        domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """直接查询知识库（不需要任务分解）"""
        return self.rag_agent.query(query, domains)

    def execute_code(
        self,
        code: str,
        use_docker: bool = True,
    ) -> Dict[str, Any]:
        """直接执行代码"""
        if use_docker and self.docker_runner:
            return self.docker_runner.execute(code)
        return self.code_agent.execute(code)

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "active_tasks": self._task_count,
            "available_domains": self.rag_agent.get_available_domains(),
            "code_tools_count": len(self.code_agent._tools),
            "docker_enabled": self.use_docker,
            "resource_manager_active": self.resource_manager is not None,
        }
