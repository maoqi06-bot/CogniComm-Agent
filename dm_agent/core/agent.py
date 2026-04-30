"""由 LLM API 驱动的 ReAct 风格智能体。"""

from __future__ import annotations

import json
import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager

from ..clients.base_client import BaseLLMClient, LLMCallMetrics
from ..tools.base import Tool
from ..prompts import build_code_agent_prompt
from ..memory.context_compressor import ContextCompressor
from ..memory.memory_manager import MemoryManager
from ..memory.memory_tools import create_memory_tools
from ..utils.logger import get_logger, AgentLogger, setup_logging
from ..utils.security import ResourceManager
from .planner import TaskPlanner, PlanStep


@dataclass
class Step:
    """表示智能体的一个推理步骤。"""

    thought: str                 # 智能体的思考过程
    action: str                  # 要执行的动作/工具名称
    action_input: Any            # 动作的输入参数
    observation: str             # 执行动作后的观察结果
    raw: str = ""                # 原始响应内容


@dataclass
class AgentMetrics:
    """Agent 执行指标"""
    task_id: str = ""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_steps: int = 0
    total_duration: float = 0.0
    start_time: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    @property
    def avg_steps(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.total_steps / self.total_tasks


class ReactAgent:
    """
    ReAct Agent 实现了推理(Reasoning)和行动(Action)的循环模式，允许智能体通过与环境交互来解决问题。
    它结合了任务规划、上下文压缩、长期记忆管理等功能，提供了一个完整的智能体执行框架。

    Attributes:
        client (BaseLLMClient): 用于与大语言模型通信的客户端
        tools (Dict[str, Tool]): 可用工具的字典映射，键为工具名称
        tools_list (List[Tool]): 工具列表，用于规划器初始化
        max_steps (int): 最大执行步骤数
        temperature (float): LLM生成文本的温度参数
        system_prompt (str): 系统提示词
        step_callback (Optional[Callable[[int, Step], None]]): 步骤执行回调函数
        enable_planning (bool): 是否启用任务规划功能
        enable_compression (bool): 是否启用上下文压缩功能
        conversation_history (List[Dict[str, str]]): 对话历史记录
        planner (Optional[TaskPlanner]): 任务规划器实例
        compressor (Optional[ContextCompressor]): 上下文压缩器实例
        memory_manager (Optional[MemoryManager]): 长期记忆管理器
        enable_long_term_memory (bool): 是否启用长期记忆功能
    """

    def __init__(
        self,
        client: BaseLLMClient,
        tools: List[Tool],
        *,
        max_steps: int = 200,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        step_callback: Optional[Callable[[int, Step], None]] = None,   # 步骤回调函数
        enable_planning: bool = True,      # 是否启用规划
        enable_compression: bool = True,   # 是否启用上下文压缩
        skill_manager: Optional[Any] = None,  # 技能管理器
        memory_manager: Optional[MemoryManager] = None,  # 长期记忆管理器
        enable_long_term_memory: bool = True,  # 是否启用长期记忆
        logger: Optional[AgentLogger] = None,  # 日志记录器
        agent_id: Optional[str] = None,  # Agent ID
resource_manager: Optional[ResourceManager] = None,  # 资源管理器
    ) -> None:
        """
        初始化 ReactAgent 实例

        Args:
            client (BaseLLMClient): LLM客户端实例
            tools (List[Tool]): 可用工具列表
            max_steps (int, optional): 最大执行步骤数，默认为200
            temperature (float, optional): LLM生成文本的温度参数，默认为0.0
            system_prompt (Optional[str], optional): 系统提示词，默认为None，将使用默认构建的提示词
            step_callback (Optional[Callable[[int, Step], None]], optional):
                步骤执行回调函数，可用于实时监控执行过程，默认为None
            enable_planning (bool, optional): 是否启用任务规划功能，默认为True
            enable_compression (bool, optional): 是否启用上下文压缩功能，默认为True
            memory_manager (MemoryManager, optional): 长期记忆管理器实例
            enable_long_term_memory (bool, optional): 是否启用长期记忆功能，默认为True
            logger (AgentLogger, optional): 日志记录器，默认自动创建
            agent_id (str, optional): Agent 标识符
            resource_manager (ResourceManager, optional): 资源管理器，默认自动创建

        Raises:
            ValueError: 当提供的工具列表为空时抛出异常

        Examples:
            >>> from dm_agent.clients import OpenAIClient
            >>> from dm_agent.tools import default_tools
            >>>
            >>> client = OpenAIClient(api_key="your-api-key")
            >>> tools = default_tools()
            >>> agent = ReactAgent(client, tools, max_steps=50)
            >>> result = agent.run("分析项目代码结构")
        """
        if not tools:
            raise ValueError("必须为 ReactAgent 提供至少一个工具。")
        self.client = client

        # 日志记录器
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.logger = logger or get_logger(f"agent.{self.agent_id}")
        self.logger.set_context(agent_id=self.agent_id)

        # 指标统计
        self.metrics = AgentMetrics()

        self.tools = {tool.name: tool for tool in tools}
        self.tools_list = tools  # 保留工具列表用于规划器
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_raw_chars = int(os.getenv("AGENT_MAX_RAW_CHARS", "4000"))
        self.max_observation_chars = int(os.getenv("AGENT_MAX_OBSERVATION_CHARS", "6000"))
        self.max_repeat_actions = int(os.getenv("AGENT_MAX_REPEAT_ACTIONS", "4"))
        self.max_parse_failures = int(os.getenv("AGENT_MAX_PARSE_FAILURES", "3"))
        self.system_prompt = system_prompt or build_code_agent_prompt(tools)
        self.step_callback = step_callback
        # 多轮对话历史记录
        self.conversation_history: List[Dict[str, str]] = []

        # 规划器
        self.enable_planning = enable_planning
        self.planner = TaskPlanner(client, tools) if enable_planning else None

        # 上下文压缩器（每 5 轮对话压缩一次）
        self.enable_compression = enable_compression
        self.compressor = ContextCompressor(client, compress_every=5, keep_recent=3) if enable_compression else None

        # 技能管理器
        self.skill_manager = skill_manager
        self._base_system_prompt = self.system_prompt
        self._base_tools = dict(self.tools)

        # 长期记忆管理器
        self.enable_long_term_memory = enable_long_term_memory
        if enable_long_term_memory:
            if memory_manager is None:
                from ..memory import MemoryManager
                memory_manager = MemoryManager(llm_client=client)
            self.memory_manager = memory_manager
            # 注册记忆工具
            memory_tools = create_memory_tools(memory_manager)
            for tool in memory_tools:
                self.tools[tool.name] = tool
            # 在系统提示中添加记忆使用指导
            memory_guidance = """

=== 长期记忆使用指南 ===
你可以使用以下工具管理长期记忆：
- add_memory: 添加需要记住的重要信息
- search_memory: 搜索相关记忆以获取上下文
- update_memory: 更新现有记忆的重要性或内容
- delete_memory: 删除过时或不准确的记忆
- list_memories: 查看当前记忆库
- get_memory_stats: 查看记忆系统统计

当用户表达偏好、做出重要决策或遇到有价值的问题解决方案时，主动使用 add_memory 保存。
"""
            self.system_prompt += memory_guidance

        # 资源管理器
        if resource_manager is None:
            from ..utils.security import get_resource_manager
            resource_manager = get_resource_manager()
        self.resource_manager = resource_manager

        # 注册健康检查回调
        self.resource_manager.health.record_request(True)

    def run(self, task: str, *, max_steps: Optional[int] = None, stop_event: Optional[Any] = None) -> Dict[str, Any]:
        """
        执行指定任务

        该方法实现了完整的ReAct循环，包括任务规划、推理、行动和观察等阶段。它支持上下文压缩以
        控制token消耗，并提供回调机制用于监控执行过程。长期记忆功能会在任务开始时自动检索
        相关记忆，增强对用户偏好和项目上下文的理解。

        Args:
            task (str): 要执行的任务描述
            max_steps (Optional[int], optional): 覆盖默认的最大步骤数

        Returns:
            result (Dict[str, Any]): 包含最终答案和执行步骤的字典
                    - final_answer (str): 任务执行的最终结果
                    - steps (List[Dict]): 执行的所有步骤信息列表

        Raises:
            ValueError: 当任务不是非空字符串时抛出异常

        Examples:
            >>> result = agent.run("帮我分析项目的代码结构")
            >>> print(result["final_answer"])
            '已成功分析项目代码结构...'
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("任务必须是非空字符串。")

        # 生成任务 ID 并设置日志上下文
        task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.logger.set_context(task_id=task_id)

        # 更新指标
        self.metrics.total_tasks += 1
        self.metrics.task_id = task_id
        self.metrics.start_time = datetime.now()

        self.logger.info(f"[START] New task: {task[:100]}...", extra={"task_id": task_id, "task_preview": task[:200]})

        # steps: 只放Step类型对象的一个列表
        steps: List[Step] = []
        limit = max_steps or self.max_steps # 获取最大步骤数
        action_counts: Dict[str, int] = {}
        parse_failures = 0

        # 技能自动选择
        if self.skill_manager:
            self._apply_skills_for_task(task)

        # 长期记忆：任务开始前检索相关记忆
        enhanced_context = ""
        if self.enable_long_term_memory and hasattr(self, 'memory_manager'):
            try:
                retrieval_result = self.memory_manager.retrieve_for_context(
                    current_task=task,
                    conversation_history=self.conversation_history[-10:] if self.conversation_history else None,
                )
                if retrieval_result.enhanced_context:
                    enhanced_context = retrieval_result.enhanced_context
                    self.logger.info(
                        f"Retrieved {len(retrieval_result.memories)} relevant memories",
                        extra={"memory_count": len(retrieval_result.memories), "categories": [r.entry.category.value for r in retrieval_result.memories]}
                    )
            except Exception as e:
                self.logger.warning(f"Memory retrieval failed: {e}", exc_info=True)

        # 第一步：生成计划（如果启用）
        plan : List[PlanStep] = []
        if self.enable_planning and self.planner:
            try:
                plan = self.planner.plan(task)
                if plan:
                    plan_text = self.planner.get_progress()
                    self.logger.info(f"Generated execution plan with {len(plan)} steps")
            except Exception as e:
                self.logger.warning(f"Plan generation failed: {e}, using normal mode")

        # 添加新任务到对话历史
        task_prompt : str = self._build_user_prompt(task, steps, plan, enhanced_context)
        self.conversation_history.append({"role": "user", "content": task_prompt})

        for step_num in range(1, limit + 1):
            if stop_event is not None and stop_event.is_set():
                return self._cancelled_result(task, steps, "任务收到取消信号，已停止继续执行。")

            # 第二步：压缩上下文（如果需要）
            messages_to_send = [{"role": "system", "content": self.system_prompt}] + self.conversation_history
            if self.enable_compression and self.compressor:
                if self.compressor.should_compress(self.conversation_history):
                    self.logger.info("Compressing conversation history")
                    original_history = self.conversation_history
                    compressed_history = self.compressor.compress(self.conversation_history)
                    self.conversation_history = compressed_history
                    messages_to_send = [{"role": "system", "content": self.system_prompt}] + compressed_history

                    # 显示压缩统计
                    stats = self.compressor.get_compression_stats(
                        original_history, compressed_history
                    )
                    self.logger.info(
                        f"Compression: ratio={stats['compression_ratio']:.1%}, saved={stats['saved_messages']} messages"
                    )

            # 获取 AI 响应
            self.logger.debug(f"Sending {len(messages_to_send)} messages to LLM, last preview: {messages_to_send[-1]['content'][:80]}...")

            step_start_time = time.time()
            try:
                raw = self.client.respond(messages_to_send, temperature=self.temperature)
                elapsed = time.time() - step_start_time
                self.logger.log_llm_call(model=self.client.model, tokens_used=0, duration=elapsed, step=step_num)
                self.logger.info(f"Step {step_num}: LLM responded in {elapsed:.2f}s, response length: {len(raw)}")
            except Exception as e:
                self.logger.error(f"LLM call failed: {e}", exc_info=True)
                self.resource_manager.health.record_request(False)
                self.resource_manager.health.record_llm_call(False)
                raise

            if stop_event is not None and stop_event.is_set():
                return self._cancelled_result(task, steps, "任务在 LLM 响应后收到取消信号，已跳过后续工具调用。")

            raw_for_history = self._compact_text(raw, self.max_raw_chars)
            # 将 AI 响应添加到历史记录，但限制长度，避免长代码/长 JSON 反复进入上下文
            self.conversation_history.append({"role": "assistant", "content": raw_for_history})
            try:
                parsed = self._parse_agent_response(raw)
            except ValueError as exc:
                parse_failures += 1
                observation = (
                    f"解析智能体响应失败：{exc}。请只返回一个可被 json.loads 解析的 JSON 对象；"
                    "不要输出 Markdown 代码块；长代码必须作为 JSON 字符串正确转义，或先写入较小文件片段。"
                )
                step = Step(
                    thought="",
                    action="error",
                    action_input={},
                    observation=observation,
                    raw=raw_for_history,
                )
                steps.append(step)
                self.logger.warning(f"Step {step_num}: Failed to parse response: {exc}")

                if parse_failures >= self.max_parse_failures:
                    final_answer = f"连续 {parse_failures} 次无法解析模型输出，已提前停止以避免死循环。最后错误：{exc}"
                    self.metrics.failed_tasks += 1
                    self.metrics.total_steps += len(steps)
                    self.metrics.total_duration += (datetime.now() - self.metrics.start_time).total_seconds()
                    self.logger.error(f"[FAILED] Consecutive parse failures reached {parse_failures}")
                    return {"final_answer": final_answer, "steps": [step.__dict__ for step in steps]}

                # 将错误观察添加到历史记录
                self.conversation_history.append({"role": "user", "content": f"观察：{observation}"})

                if self.step_callback:
                    self.step_callback(step_num, step)
                continue

            # 获取动作、thought 和输入
            parse_failures = 0
            action = parsed.get("action", "").strip()
            thought = parsed.get("thought", "").strip()
            action_input = parsed.get("action_input")

            if stop_event is not None and stop_event.is_set():
                return self._cancelled_result(task, steps, "任务在解析动作后收到取消信号，已停止。")

            signature = self._action_signature(action, action_input)
            action_counts[signature] = action_counts.get(signature, 0) + 1
            if action_counts[signature] > self.max_repeat_actions:
                observation = (
                    f"检测到重复工具调用超过 {self.max_repeat_actions} 次：{action}。"
                    "为避免死循环，已提前停止。请拆分任务、缩小读取范围，或直接提交当前最佳结果。"
                )
                step = Step(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    raw=raw_for_history,
                )
                steps.append(step)
                self.metrics.failed_tasks += 1
                self.metrics.total_steps += len(steps)
                self.metrics.total_duration += (datetime.now() - self.metrics.start_time).total_seconds()
                self.logger.error(f"[FAILED] Repeated action guard tripped for {action}")
                return {"final_answer": observation, "steps": [step.__dict__ for step in steps]}

            # 检查是否完成
            if action == "finish":
                final = self._format_final_answer(action_input)
                step = Step(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation="<finished>",
                    raw=raw_for_history,
                )
                steps.append(step)
                self.logger.log_execution("TASK_COMPLETED", action="finish", step=step_num, task_id=task_id)
                self.resource_manager.health.record_request(True)
                self.resource_manager.health.record_llm_call(True)

                # 添加完成标记到历史记录
                self.conversation_history.append({"role": "user", "content": f"任务完成：{final}"})

                if self.step_callback:
                    self.step_callback(step_num, step)

                # 更新指标
                self.metrics.successful_tasks += 1
                self.metrics.total_steps += len(steps)
                self.metrics.total_duration += (datetime.now() - self.metrics.start_time).total_seconds()

                self.logger.info(f"[COMPLETE] Task finished successfully in {len(steps)} steps")
                return {"final_answer": final, "steps": [step.__dict__ for step in steps]}

            # 检查工具
            tool = self.tools.get(action)
            if tool is None:
                observation = f"未知工具 '{action}'。"
                step = Step(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    raw=raw_for_history,
                )
                steps.append(step)
                self.logger.warning(f"Step {step_num}: Unknown tool '{action}'")

                # 将观察结果添加到历史记录
                self.conversation_history.append({"role": "user", "content": f"观察：{observation}"})

                if self.step_callback:
                    self.step_callback(step_num, step)
                continue

            # task_complete 工具可以接受字符串或空参数
            tool_start_time = time.time()
            tool_success = True
            if action == "task_complete":
                if action_input is None:
                    action_input = {}
                elif isinstance(action_input, str):
                    action_input = {"message": action_input}
                elif not isinstance(action_input, dict):
                    action_input = {}
                try:
                    observation = tool.execute(action_input)
                except Exception as exc:  # noqa: BLE001 - 将工具错误传递给 LLM
                    observation = f"工具执行失败：{exc}"
                    tool_success = False
            elif action_input is None:
                observation = "工具参数缺失（action_input 为 null）。"
                tool_success = False
            elif not isinstance(action_input, dict):
                observation = "工具参数必须是 JSON 对象。"
                tool_success = False
            else:
                try:
                    observation = tool.execute(action_input)
                except Exception as exc:  # noqa: BLE001 - 将工具错误传递给 LLM
                    observation = f"工具执行失败：{exc}"
                    tool_success = False

            if stop_event is not None and stop_event.is_set():
                return self._cancelled_result(task, steps, "任务在工具执行后收到取消信号，已停止。")

            tool_duration = time.time() - tool_start_time
            self.logger.log_tool_call(tool_name=action, success=tool_success, duration=tool_duration, step=step_num)
            self.logger.log_execution("TOOL_EXECUTED", tool=action, success=tool_success, duration=tool_duration, step=step_num)

            observation_for_history = self._compact_text(str(observation), self.max_observation_chars)
            step = Step(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation_for_history,
                raw=raw_for_history,
            )
            steps.append(step)

            # 更新计划进度（如果有计划）
            if plan and self.planner:
                # 查找当前步骤对应的计划步骤
                for plan_step in plan:
                    if plan_step.action == action and not plan_step.completed:
                        self.planner.mark_completed(plan_step.step_number, observation_for_history)
                        break

            # 将工具执行结果添加到历史记录
            tool_info = (
                f"执行工具 {action}，输入：{json.dumps(action_input, ensure_ascii=False)}\n"
                f"观察：{observation_for_history}"
            )
            self.conversation_history.append({"role": "user", "content": tool_info})

            # 调用回调函数实时输出步骤
            if self.step_callback:
                self.step_callback(step_num, step)

            # 检查是否调用了 task_complete 工具
            if action == "task_complete" and not observation.startswith("工具执行失败"):
                # 任务成功完成：提取并存储记忆
                if self.enable_long_term_memory and hasattr(self, 'memory_manager'):
                    self._extract_and_store_memories(task, success=True)

                self.metrics.successful_tasks += 1
                self.metrics.total_steps += len(steps)
                self.metrics.total_duration += (datetime.now() - self.metrics.start_time).total_seconds()
                self.logger.info(f"[COMPLETE] Task completed via tool in {len(steps)} steps")

                return {
                    "final_answer": observation,
                    "steps": [step.__dict__ for step in steps],
                }

        # 达到步骤限制：也尝试提取记忆
        if self.enable_long_term_memory and hasattr(self, 'memory_manager'):
            self._extract_and_store_memories(task, success=False)

        self.metrics.failed_tasks += 1
        self.metrics.total_steps += len(steps)
        self.metrics.total_duration += (datetime.now() - self.metrics.start_time).total_seconds()
        self.logger.error(f"[FAILED] Task reached max steps ({limit}) without completion")

        return {
            "final_answer": "达到步骤限制但未完成。",
            "steps": [step.__dict__ for step in steps],
        }

    def _extract_and_store_memories(self, task: str, success: bool) -> None:
        """
        从对话历史中提取并存储重要记忆。

        使用后台线程执行，不阻塞主流程。
        """
        if not hasattr(self, 'memory_manager') or not self.memory_manager:
            return

        # 在后台线程执行记忆提取，避免阻塞主流程
        def _background_extract():
            try:
                extracted = self.memory_manager.extract_and_store(
                    conversation_history=self.conversation_history,
                    current_task=task,
                )
                if extracted:
                    self.logger.info(f"[Background] Extracted {len(extracted)} memories from conversation", extra={"memory_count": len(extracted)})
            except Exception as e:
                self.logger.warning(f"[Background] Memory extraction failed: {e}", exc_info=True)

        import threading
        thread = threading.Thread(target=_background_extract, daemon=True)
        thread.start()

    def _apply_skills_for_task(self, task: str) -> None:
        """根据任务自动选择并激活相关技能。"""
        # 恢复基础状态，避免上一次任务的技能残留
        self.system_prompt = self._base_system_prompt
        self.tools = dict(self._base_tools)

        # 自动选择
        selected = self.skill_manager.select_skills_for_task(task)
        if not selected:
            self.skill_manager.deactivate_all()
            return

        # 激活选中技能
        self.skill_manager.activate_skills(selected)

        # 追加技能 prompt
        prompt_addition = self.skill_manager.get_active_prompt_additions()
        if prompt_addition:
            self.system_prompt += prompt_addition

        # 合并技能工具
        skill_tools = self.skill_manager.get_active_tools()
        for tool in skill_tools:
            self.tools[tool.name] = tool

        # 打印激活信息
        display_names = []
        for name in selected:
            skill = self.skill_manager.skills.get(name)
            if skill:
                display_names.append(skill.get_metadata().display_name)
        if display_names:
            self.logger.info(f"Activated skills: {', '.join(display_names)}", extra={"skills": display_names})

    def _cancelled_result(self, task: str, steps: List[Step], message: str) -> Dict[str, Any]:
        if self.enable_long_term_memory and hasattr(self, 'memory_manager'):
            self._extract_and_store_memories(task, success=False)

        self.metrics.failed_tasks += 1
        self.metrics.total_steps += len(steps)
        if self.metrics.start_time:
            self.metrics.total_duration += (datetime.now() - self.metrics.start_time).total_seconds()
        self.logger.warning(f"[CANCELLED] {message}")
        return {
            "final_answer": message,
            "steps": [step.__dict__ for step in steps],
            "cancelled": True,
        }

    def _build_user_prompt(self, task: str, steps: List[Step], plan: List[PlanStep] = None, enhanced_context: str = "") -> str:
        """
        构建用户提示词

        Args:
            task (str): 当前任务描述
            steps (List[Step]): 已执行的步骤列表
            plan (List[PlanStep], optional): 执行计划
            enhanced_context (str, optional): 长期记忆检索到的增强上下文

        Returns:
            prompt (str): 构建好的用户提示词字符串
        """
        lines : List[str] = [f"任务：{task.strip()}"]

        # 添加长期记忆检索到的上下文
        if enhanced_context:
            lines.append(enhanced_context)

        # 如果有计划，添加到提示中
        if plan:
            lines.append("\n执行计划：")
            for plan_step in plan:
                status = "✓" if plan_step.completed else "○"
                lines.append(f"{status} 步骤 {plan_step.step_number}: {plan_step.action} - {plan_step.reason}")

        if steps:
            lines.append("\n之前的步骤：")
            for index, step in enumerate(steps, start=1):
                lines.append(f"步骤 {index} 思考：{step.thought}")
                lines.append(f"步骤 {index} 动作：{step.action}")
                lines.append(f"步骤 {index} 输入：{json.dumps(step.action_input, ensure_ascii=False)}")
                lines.append(f"步骤 {index} 观察：{step.observation}")
        lines.append(
            "\n用 JSON 对象回应：{\"thought\": string, \"action\": string, \"action_input\": object|string}。"
        )
        return "\n".join(lines)

    @staticmethod
    def _compact_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        half = max_chars // 2
        omitted = len(text) - max_chars
        return (
            f"{text[:half]}\n\n"
            f"... <truncated {omitted} chars to keep agent context bounded> ...\n\n"
            f"{text[-half:]}"
        )

    @staticmethod
    def _normalize_for_signature(value: Any) -> Any:
        if isinstance(value, dict):
            normalized: Dict[str, Any] = {}
            for key, item in sorted(value.items()):
                if isinstance(item, str) and len(item) > 500:
                    digest = hashlib.sha256(item.encode("utf-8", errors="ignore")).hexdigest()
                    normalized[key] = f"<str:{len(item)}:{digest}>"
                else:
                    normalized[key] = ReactAgent._normalize_for_signature(item)
            return normalized
        if isinstance(value, list):
            return [ReactAgent._normalize_for_signature(item) for item in value]
        return value

    @classmethod
    def _action_signature(cls, action: str, action_input: Any) -> str:
        payload = {
            "action": action,
            "action_input": cls._normalize_for_signature(action_input),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8", errors="ignore")).hexdigest()
        return f"{action}:{digest}"

    def _parse_agent_response(self, raw: str) -> Dict[str, Any]:
        """
        解析智能体响应
        
        Args:
            raw (str): 智能体的原始响应字符串
            
        Returns:
            parsed (Dict[str, Any]): 解析后的JSON对象
            
        Raises:
            ValueError: 当响应不是有效的JSON时抛出异常
        """
        candidate = raw.strip()
        if not candidate:
            raise ValueError("模型返回空响应。")
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
                candidate = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            # 处理 "Extra data" 错误：可能有多个 JSON 对象
            if "Extra data" in str(e):
                # 找到第一个完整的 JSON 对象
                start = candidate.find('{"')
                if start == -1:
                    start = candidate.find('{')
                depth = 0
                end = start
                for i, c in enumerate(candidate[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if start != -1 and end > start:
                    snippet = candidate[start:end]
                    try:
                        parsed = json.loads(snippet)
                    except json.JSONDecodeError:
                        raise ValueError(f"响应包含多个 JSON 对象，无法解析: {e}")
                else:
                    raise ValueError(f"响应不是有效的 JSON: {e}")
            else:
                start = candidate.find("{")
                end = candidate.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise ValueError(f"响应不是有效的 JSON: {e}")
                snippet = candidate[start : end + 1]
                parsed = json.loads(snippet)
        if not isinstance(parsed, dict):
            raise ValueError("智能体响应的 JSON 必须是对象。")
        # Compatibility for older prompts that asked models to return
        # {"tool": "...", "args": {...}}. The runtime protocol is
        # {"action": "...", "action_input": {...}}.
        if "action" not in parsed and "tool" in parsed:
            parsed["action"] = parsed.get("tool")
        if "action_input" not in parsed and "args" in parsed:
            parsed["action_input"] = parsed.get("args")
        action = parsed.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("智能体响应缺少有效 action；请返回非空的 action 字段。")
        return parsed

    def reset_conversation(self) -> None:
        """重置对话历史
        
        清空所有对话历史记录，为新任务做准备。
        """
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史
        
        Returns:
            conversation_history (List[Dict[str, str]]): 对话历史记录的副本
        """
        return self.conversation_history.copy()

    @staticmethod
    def _format_final_answer(action_input: Any) -> str:
        """
        格式化最终答案

        Args:
            action_input (Any): finish动作的输入参数

        Returns:
            answer (str): 格式化后的最终答案字符串
        """
        if isinstance(action_input, str):
            return action_input
        if isinstance(action_input, dict) and "answer" in action_input:
            value = action_input["answer"]
            if isinstance(value, str):
                return value
        return json.dumps(action_input, ensure_ascii=False)

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取 Agent 执行指标

        Returns:
            Dict: 包含 Agent 指标和 LLM 客户端指标的字典
        """
        agent_metrics = {
            "agent_id": self.agent_id,
            "total_tasks": self.metrics.total_tasks,
            "successful_tasks": self.metrics.successful_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "success_rate": self.metrics.success_rate,
            "total_steps": self.metrics.total_steps,
            "avg_steps": self.metrics.avg_steps,
            "total_duration": self.metrics.total_duration,
        }

        llm_metrics = self.client.get_metrics() if hasattr(self.client, 'get_metrics') else {}

        # 资源管理器状态
        resource_status = {}
        if hasattr(self, 'resource_manager') and self.resource_manager:
            resource_status = self.resource_manager.get_status()

        return {
            "agent": agent_metrics,
            "llm": llm_metrics,
            "resource": resource_status,
        }

    def reset_metrics(self):
        """重置所有指标"""
        self.metrics = AgentMetrics()
        if hasattr(self.client, 'reset_metrics'):
            self.client.reset_metrics()
