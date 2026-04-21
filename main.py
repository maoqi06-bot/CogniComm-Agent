"""LLM 驱动的 ReAct 智能体的 CLI 入口点。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from dm_agent.rag.observability import TraceManager  # 确保路径正确

from dotenv import load_dotenv

from dm_agent import (
    LLMError,
    ReactAgent,
    Tool,
    create_llm_client,
    default_tools,
    PROVIDER_DEFAULTS,
    # P2: Multi-Agent
    OrchestratorAgent,
    load_profiles_for_task,
    setup_logging,
)
from dm_agent.mcp import MCPManager, load_mcp_config
from dm_agent.skills import SkillManager

# 尝试导入 colorama 用于彩色输出
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # 如果没有 colorama，定义空的颜色常量
    class Fore:
        GREEN = ""
        YELLOW = ""
        RED = ""
        CYAN = ""
        MAGENTA = ""
        BLUE = ""

    class Style:
        BRIGHT = ""
        RESET_ALL = ""


@dataclass
class Config:
    """运行时配置"""
    api_key: str
    provider: str = "openai"  # deepseek/openai/claude/gemini
    model: str = "gpt-5" # deepseek-chat/gpt-5/claude-sonnet-4-5/gemini-2.5-flash
    base_url: str = "https://sg.uiuiapi.com/v1/" # https://api.deepseek.com/v1/chat/completions
    max_steps: int = 10
    temperature: float = 0.15
    show_steps: bool = False


# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


def load_config_from_file() -> Dict[str, Any]:
    """从配置文件加载设置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ 配置文件加载失败：{e}，使用默认设置{Style.RESET_ALL}")
    return {}


def save_config_to_file(config: Config) -> None:
    """保存配置到文件"""
    try:
        config_data = {
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
            "max_steps": config.max_steps,
            "temperature": config.temperature,
            "show_steps": config.show_steps,
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"{Fore.GREEN}✓ 配置已保存{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}✗ 配置保存失败：{e}{Style.RESET_ALL}")


def get_api_key_for_provider(provider: str) -> str | None:
    """根据提供商获取对应的 API 密钥"""
    provider_env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "CLAUDE_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = provider_env_map.get(provider.lower())
    return os.getenv(env_var) if env_var else None


def has_usable_api_key(provider: str) -> bool:
    """Return True when the provider has a non-placeholder API key in env."""
    value = get_api_key_for_provider(provider)
    if not value:
        return False
    lowered = value.strip().lower()
    return not (
        lowered.startswith("your_")
        or lowered.endswith("_here")
        or lowered in {"", "none", "null"}
    )


def resolve_default_provider(saved_config: Dict[str, Any]) -> str:
    """Choose the default provider.

    Priority:
    1. saved config
    2. explicit DEFAULT_PROVIDER env
    3. Config default
    """
    saved_provider = saved_config.get("provider")
    if saved_provider:
        return saved_provider

    env_default = os.getenv("DEFAULT_PROVIDER", "").strip().lower()
    if env_default:
        return env_default

    return Config.provider


def resolve_default_model(saved_config: Dict[str, Any], provider: str) -> str:
    if saved_config.get("model"):
        return saved_config["model"]
    return PROVIDER_DEFAULTS.get(provider, {}).get("model", Config.model)


def resolve_default_base_url(saved_config: Dict[str, Any], provider: str) -> str:
    if saved_config.get("base_url"):
        return saved_config["base_url"]
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL", "").strip()
    return PROVIDER_DEFAULTS.get(provider, {}).get("base_url", Config.base_url)


def apply_runtime_provider_env(config: Config) -> None:
    """Keep runtime env in sync with the selected provider config."""
    provider = config.provider.lower()
    if provider == "openai":
        if config.api_key:
            os.environ["OPENAI_API_KEY"] = config.api_key
        if config.base_url:
            os.environ["OPENAI_BASE_URL"] = config.base_url
        os.environ.setdefault("OPENAI_API_STYLE", "auto")
    os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-ada-002")
    if os.getenv("OPENAI_API_KEY") and not os.getenv("EMBEDDING_API_KEY"):
        os.environ["EMBEDDING_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    if os.getenv("OPENAI_BASE_URL") and not os.getenv("EMBEDDING_BASE_URL"):
        os.environ["EMBEDDING_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "")


def mask_secret(value: str, *, prefix: int = 6, suffix: int = 4) -> str:
    if not value:
        return "(empty)"
    if len(value) <= prefix + suffix:
        return "*" * len(value)
    return f"{value[:prefix]}...{value[-suffix:]}"


def resolve_docker_path() -> Optional[str]:
    import shutil

    configured = os.getenv("DOCKER_PATH", "").strip().strip('"')
    if configured and os.path.exists(configured):
        docker_dir = os.path.dirname(configured)
        current_path = os.environ.get("PATH", "")
        if docker_dir and docker_dir not in current_path.split(os.pathsep):
            os.environ["PATH"] = docker_dir + os.pathsep + current_path
        return configured

    docker_path = shutil.which("docker")
    if docker_path:
        return docker_path

    candidate_dirs = [
        r"C:\Program Files\Docker\Docker\resources\bin",
        r"C:\Program Files\Docker\Docker\resources",
        r"C:\ProgramData\DockerDesktop\version-bin",
    ]
    for directory in candidate_dirs:
        candidate = os.path.join(directory, "docker.exe")
        if os.path.exists(candidate):
            current_path = os.environ.get("PATH", "")
            if directory not in current_path.split(os.pathsep):
                os.environ["PATH"] = directory + os.pathsep + current_path
            return candidate
    return None


def detect_key_source(config: Config) -> str:
    provider = config.provider.lower()
    env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "CLAUDE_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = env_map.get(provider)
    env_value = os.getenv(env_var or "", "")
    if env_var and env_value and config.api_key == env_value:
        return f"env:{env_var}"
    return "cli/config"


def run_multi_agent_preflight(config: Config, client: Any, use_docker: bool) -> tuple[bool, bool]:
    """Run lightweight runtime checks for multi-agent mode."""
    import subprocess

    print(f"{Fore.CYAN}      运行环境: {sys.executable}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}      Provider: {config.provider} | Model: {config.model}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}      Base URL: {config.base_url or '(provider default)'}{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}      API Key 来源: {detect_key_source(config)} | "
        f"Key: {mask_secret(config.api_key)}{Style.RESET_ALL}"
    )
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model = os.getenv("EMBEDDING_MODEL", "")
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
    if embedding_model:
        print(
            f"{Fore.CYAN}      Embedding Provider: {embedding_provider} | "
            f"Embedding Model: {embedding_model}{Style.RESET_ALL}"
        )
        if embedding_base_url:
            print(f"{Fore.CYAN}      Embedding Base URL: {embedding_base_url}{Style.RESET_ALL}")
    if config.provider.lower() == "openai":
        print(f"{Fore.CYAN}      API Style: {os.getenv('OPENAI_API_STYLE', 'auto')}{Style.RESET_ALL}")

    llm_ready = True
    try:
        client.respond([{"role": "user", "content": "Reply with OK only."}])
        print(f"{Fore.GREEN}      ✓ LLM 鉴权预检通过{Style.RESET_ALL}")
    except Exception as exc:  # noqa: BLE001
        llm_ready = False
        print(f"{Fore.RED}      ✗ LLM 鉴权预检失败: {exc}{Style.RESET_ALL}")

    docker_enabled = use_docker
    if use_docker:
        docker_path = resolve_docker_path()
        if not docker_path:
            docker_enabled = False
            print(f"{Fore.YELLOW}      ⚠ 未在当前 PATH 中发现 docker，可见性检查失败，自动降级为非 Docker 模式{Style.RESET_ALL}")
        else:
            try:
                result = subprocess.run(
                    [docker_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0:
                    version = (result.stdout or result.stderr).strip()
                    print(f"{Fore.GREEN}      ✓ Docker 预检通过: {docker_path} ({version}){Style.RESET_ALL}")
                else:
                    docker_enabled = False
                    details = (result.stderr or result.stdout).strip()
                    print(f"{Fore.YELLOW}      ⚠ Docker 预检失败: {details or 'docker --version returned non-zero'}，自动降级为非 Docker 模式{Style.RESET_ALL}")
            except Exception as exc:  # noqa: BLE001
                docker_enabled = False
                print(f"{Fore.YELLOW}      ⚠ Docker 预检异常: {exc}，自动降级为非 Docker 模式{Style.RESET_ALL}")

    return llm_ready, docker_enabled


def format_multi_agent_status(result: Dict[str, Any]) -> str:
    status = result.get("overall_status")
    if status == "success":
        return "成功"
    if status == "partial_success":
        return "部分成功"
    if status == "failed":
        return "失败"
    return "成功" if result.get("success") else "失败"


def parse_args(argv: Any) -> argparse.Namespace:
    # 先加载配置文件中的默认值
    saved_config = load_config_from_file()

    # 创建函数解析器
    parser = argparse.ArgumentParser(description="运行基于 LLM 的 ReAct 智能体来完成任务描述。")

    parser.add_argument("task", nargs="?", help="智能体要完成的自然语言任务。")

    # 推断默认提供商，避免在未显式配置时意外切换到新的 provider
    default_provider = resolve_default_provider(saved_config)
    default_model = resolve_default_model(saved_config, default_provider)
    default_base_url = resolve_default_base_url(saved_config, default_provider)

    # 根据提供商获取对应的 API 密钥
    default_api_key = get_api_key_for_provider(default_provider)

    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=default_api_key,
        help="API 密钥（默认使用环境变量）。",
    )
    parser.add_argument(
        "--provider",
        default=default_provider,
        help=f"LLM 提供商 (deepseek/openai/claude/gemini，默认自动推断，当前：{default_provider})。",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="模型标识符（默认根据提供商选择）。",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=default_base_url,
        help="API 基础 URL（可选，使用提供商默认值）。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=saved_config.get("max_steps", Config.max_steps),
        help=f"放弃前的最大推理/工具步骤数（默认：{Config.max_steps}）。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=saved_config.get("temperature", Config.temperature),
        help=f"模型的采样温度（默认：{Config.temperature}）。",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        default=saved_config.get("show_steps", False),
        help="打印智能体执行的中间 ReAct 步骤。",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启动交互式菜单模式。",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        default=False,
        help="使用多 Agent 协作系统（P2）执行任务。",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="多 Agent 模式最大并行任务数（默认：3）。",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        default=True,
        help="多 Agent 模式使用 Docker 隔离执行代码。",
    )
    parser.add_argument(
        "--no-docker",
        action="store_false",
        dest="use_docker",
        help="多 Agent 模式禁用 Docker 隔离执行代码。",
    )
    return parser.parse_args(argv)


def print_separator(char: str = "=", length: int = 70) -> None:
    """打印分隔线"""
    print(f"{Fore.CYAN}{char * length}{Style.RESET_ALL}")


def print_header(text: str) -> None:
    """打印标题"""
    print_separator()
    print(f"{Fore.GREEN}{Style.BRIGHT}{text.center(70)}{Style.RESET_ALL}")
    print_separator()


def print_welcome() -> None:
    """打印欢迎界面"""
    print("\n")
    print_header("CogniComm-Agent")
    print(f"{Fore.YELLOW}欢迎使用 LLM 驱动的 CogniComm-Agent 智能体系统！{Style.RESET_ALL}")

    # 显示配置文件状态
    if os.path.exists(CONFIG_FILE):
        print(f"{Fore.GREEN}✓ 已加载配置文件: config.json{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}* 使用默认配置 (max_steps=100, temperature=0.15){Style.RESET_ALL}")
    print()


def print_menu() -> None:
    """打印主菜单"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}主菜单：{Style.RESET_ALL}")
    print(f"{Fore.GREEN}  1.{Style.RESET_ALL} 执行新任务")
    print(f"{Fore.GREEN}  2.{Style.RESET_ALL} 多轮对话模式")
    print(f"{Fore.GREEN}  3.{Style.RESET_ALL} 查看可用工具列表")
    print(f"{Fore.GREEN}  4.{Style.RESET_ALL} 配置设置")
    print(f"{Fore.GREEN}  5.{Style.RESET_ALL} 查看可用技能列表")
    print(f"{Fore.GREEN}  6.{Style.RESET_ALL} P2: 多 Agent 协作模式")
    print(f"{Fore.GREEN}  7.{Style.RESET_ALL} 退出程序")
    print()


def show_tools(tools: List[Tool]) -> None:
    """显示可用工具列表"""
    print_separator("-")
    print(f"{Fore.CYAN}{Style.BRIGHT}可用工具列表：{Style.RESET_ALL}\n")

    for idx, tool in enumerate(tools, start=1):
        print(f"{Fore.GREEN}{idx}. {tool.name}{Style.RESET_ALL}")
        print(f"   {Fore.YELLOW}描述：{Style.RESET_ALL}{tool.description}")
        print()

    print_separator("-")


def show_skills(skill_manager: SkillManager) -> None:
    """显示可用技能列表"""
    print_separator("-")
    print(f"{Fore.CYAN}{Style.BRIGHT}可用技能列表：{Style.RESET_ALL}\n")

    skills_info = skill_manager.get_all_skill_info()
    if not skills_info:
        print(f"{Fore.YELLOW}暂无可用技能{Style.RESET_ALL}")
    else:
        for idx, info in enumerate(skills_info, start=1):
            status = f"{Fore.GREEN}[激活]{Style.RESET_ALL}" if info["is_active"] else ""
            source = "内置" if info["is_builtin"] else "自定义"
            print(f"{Fore.GREEN}{idx}. {info['display_name']}{Style.RESET_ALL} {status}")
            print(f"   {Fore.YELLOW}标识：{Style.RESET_ALL}{info['name']}")
            print(f"   {Fore.YELLOW}描述：{Style.RESET_ALL}{info['description']}")
            print(f"   {Fore.YELLOW}类型：{Style.RESET_ALL}{source}  {Fore.YELLOW}版本：{Style.RESET_ALL}{info['version']}  {Fore.YELLOW}专用工具：{Style.RESET_ALL}{info['tools_count']} 个")
            print(f"   {Fore.YELLOW}关键词：{Style.RESET_ALL}{', '.join(info['keywords'][:8])}{'...' if len(info['keywords']) > 8 else ''}")
            print()

    print_separator("-")


def configure_settings(config: Config) -> None:
    """配置设置"""
    print_separator("-")
    print(f"{Fore.CYAN}{Style.BRIGHT}当前配置：{Style.RESET_ALL}\n")
    print(f"  提供商：{Fore.YELLOW}{config.provider}{Style.RESET_ALL}")
    print(f"  模型：{Fore.YELLOW}{config.model}{Style.RESET_ALL}")
    print(f"  Base URL：{Fore.YELLOW}{config.base_url}{Style.RESET_ALL}")
    print(f"  最大步骤数：{Fore.YELLOW}{config.max_steps}{Style.RESET_ALL}")
    print(f"  温度：{Fore.YELLOW}{config.temperature}{Style.RESET_ALL}")
    print(f"  显示步骤：{Fore.YELLOW}{'是' if config.show_steps else '否'}{Style.RESET_ALL}")
    print()

    print(f"{Fore.CYAN}选择要修改的设置（直接回车跳过）：{Style.RESET_ALL}\n")

    config_changed = False

    # 修改提供商
    provider_input = input(f"LLM 提供商 (deepseek/openai/claude/gemini) [{config.provider}]: ").strip().lower()
    if provider_input and provider_input in ["deepseek", "openai", "claude", "gemini"]:
        if provider_input != config.provider:
            # 尝试获取新提供商的 API 密钥
            new_api_key = get_api_key_for_provider(provider_input)
            if not new_api_key:
                print(f"{Fore.RED}✗ 未找到 {provider_input.upper()}_API_KEY 环境变量{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}请在 .env 文件中配置 {provider_input.upper()}_API_KEY{Style.RESET_ALL}")
            else:
                config.provider = provider_input
                config.api_key = new_api_key  # 更新 API 密钥
                # 自动更新默认模型和 base_url
                defaults = PROVIDER_DEFAULTS.get(provider_input, {})
                config.model = defaults.get("model", config.model)
                config.base_url = defaults.get("base_url", config.base_url)
                config_changed = True
                print(f"{Fore.GREEN}✓ 已更新提供商为 {provider_input}，模型和 URL 已自动调整{Style.RESET_ALL}")
    elif provider_input and provider_input not in ["deepseek", "openai", "claude", "gemini"]:
        print(f"{Fore.RED}✗ 无效的提供商{Style.RESET_ALL}")

    # 修改模型
    model_input = input(f"模型名称 [{config.model}]: ").strip()
    if model_input:
        config.model = model_input
        config_changed = True
        print(f"{Fore.GREEN}✓ 已更新模型为 {model_input}{Style.RESET_ALL}")

    # 修改 Base URL
    base_url_input = input(f"Base URL [{config.base_url}]: ").strip()
    if base_url_input:
        config.base_url = base_url_input
        config_changed = True
        print(f"{Fore.GREEN}✓ 已更新 Base URL 为 {base_url_input}{Style.RESET_ALL}")

    # 修改最大步骤数
    try:
        max_steps_input = input(f"最大步骤数 [{config.max_steps}]: ").strip()
        if max_steps_input:
            new_max_steps = int(max_steps_input)
            if new_max_steps > 0:
                config.max_steps = new_max_steps
                config_changed = True
                print(f"{Fore.GREEN}✓ 已更新最大步骤数为 {new_max_steps}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ 最大步骤数必须大于 0{Style.RESET_ALL}")
    except ValueError:
        print(f"{Fore.RED}✗ 无效的数字{Style.RESET_ALL}")

    # 修改温度
    try:
        temp_input = input(f"温度 (0.0-2.0) [{config.temperature}]: ").strip()
        if temp_input:
            new_temp = float(temp_input)
            if 0.0 <= new_temp <= 2.0:
                config.temperature = new_temp
                config_changed = True
                print(f"{Fore.GREEN}✓ 已更新温度为 {new_temp}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ 温度必须在 0.0 到 2.0 之间{Style.RESET_ALL}")
    except ValueError:
        print(f"{Fore.RED}✗ 无效的数字{Style.RESET_ALL}")

    # 修改显示步骤
    show_steps_input = input(f"显示步骤 (y/n) [{'y' if config.show_steps else 'n'}]: ").strip().lower()
    if show_steps_input in ['y', 'yes', '是']:
        if not config.show_steps:
            config.show_steps = True
            config_changed = True
        print(f"{Fore.GREEN}✓ 已启用显示步骤{Style.RESET_ALL}")
    elif show_steps_input in ['n', 'no', '否']:
        if config.show_steps:
            config.show_steps = False
            config_changed = True
        print(f"{Fore.GREEN}✓ 已禁用显示步骤{Style.RESET_ALL}")

    # 保存配置
    if config_changed:
        print()
        save_choice = input(f"{Fore.CYAN}是否保存为永久配置？(y/n) [y]: {Style.RESET_ALL}").strip().lower()
        if save_choice in ['', 'y', 'yes', '是']:
            save_config_to_file(config)

    print_separator("-")


def display_result(result: Dict[str, Any], show_steps: bool = False) -> None:
    """格式化显示任务结果"""
    print_separator("-")

    if show_steps and result.get("steps"):
        print(f"{Fore.CYAN}{Style.BRIGHT}执行步骤：{Style.RESET_ALL}\n")
        for idx, step in enumerate(result.get("steps", []), start=1):
            print(f"{Fore.MAGENTA}步骤 {idx}:{Style.RESET_ALL}")
            print(f"  {Fore.YELLOW}思考：{Style.RESET_ALL}{step.get('thought')}")
            print(f"  {Fore.YELLOW}动作：{Style.RESET_ALL}{step.get('action')}")
            action_input = step.get('action_input')
            if action_input:
                print(f"  {Fore.YELLOW}输入：{Style.RESET_ALL}{json.dumps(action_input, ensure_ascii=False)}")
            print(f"  {Fore.YELLOW}观察：{Style.RESET_ALL}{step.get('observation')}")
            print()

    print(f"{Fore.GREEN}{Style.BRIGHT}最终答案：{Style.RESET_ALL}\n")
    final_answer = result.get("final_answer", "")
    print(final_answer)
    print()
    print_separator("-")


def create_step_callback(show_steps: bool):
    """创建步骤回调函数，用于实时打印 agent 执行状态"""
    def callback(step_num: int, step: Any) -> None:
        if show_steps:
            print(f"\n{Fore.MAGENTA}{Style.BRIGHT}[步骤 {step_num}]{Style.RESET_ALL}")
            print(f"  {Fore.YELLOW}思考：{Style.RESET_ALL}{step.thought}")
            print(f"  {Fore.YELLOW}动作：{Style.RESET_ALL}{step.action}")
            if step.action_input:
                print(f"  {Fore.YELLOW}输入：{Style.RESET_ALL}{json.dumps(step.action_input, ensure_ascii=False)}")
            print(f"  {Fore.YELLOW}观察：{Style.RESET_ALL}{step.observation}")
        else:
            # 即使不显示详细步骤，也显示简要进度
            print(f"{Fore.CYAN}[步骤 {step_num}] {step.action}{Style.RESET_ALL}", end=" ", flush=True)
            if step.action == "finish" or step.action == "task_complete":
                print(f"{Fore.GREEN}✓{Style.RESET_ALL}")
            elif step.action == "error":
                print(f"{Fore.RED}✗{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}✓{Style.RESET_ALL}")

    return callback


def multi_turn_conversation(config: Config, tools: List[Tool], skill_manager: SkillManager | None = None) -> None:
    """多轮对话模式"""
    print_separator("-")
    print(f"{Fore.CYAN}{Style.BRIGHT}多轮对话模式{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}进入多轮对话模式，智能体会记住之前的所有对话内容{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}输入 'exit' 退出对话模式，输入 'reset' 重置对话历史{Style.RESET_ALL}\n")
    print_separator("-")

    try:
        # 创建客户端和智能体
        client = create_llm_client(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
        )
        step_callback = create_step_callback(config.show_steps)

        agent = ReactAgent(
            client,
            tools,
            max_steps=config.max_steps,
            temperature=config.temperature,
            step_callback=step_callback,
            skill_manager=skill_manager,
        )

        conversation_count = 0

        while True:
            print(f"\n{Fore.CYAN}[对话 {conversation_count + 1}]{Style.RESET_ALL}")
            task = input(f"{Fore.YELLOW}请输入任务（exit 退出，reset 重置历史）：{Style.RESET_ALL}\n> ").strip()

            if not task:
                print(f"{Fore.RED}✗ 任务描述不能为空{Style.RESET_ALL}")
                continue

            if task.lower() == "exit":
                print(f"\n{Fore.YELLOW}退出多轮对话模式{Style.RESET_ALL}")
                break

            if task.lower() == "reset":
                agent.reset_conversation()
                conversation_count = 0
                print(f"{Fore.GREEN}✓ 对话历史已重置{Style.RESET_ALL}")
                continue

            try:
                print(f"\n{Fore.CYAN}正在执行任务...{Style.RESET_ALL}\n")
                print_separator("-")

                # 执行任务
                result = agent.run(task)
                conversation_count += 1

                # 显示最终结果
                print(f"\n{Fore.GREEN}{Style.BRIGHT}最终答案：{Style.RESET_ALL}\n")
                print(result.get("final_answer", ""))
                print()
                print_separator("-")

            except LLMError as e:
                print(f"\n{Fore.RED}{Style.BRIGHT}✗ API 错误：{Style.RESET_ALL}{e}")
                print_separator("-")
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}退出多轮对话模式{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}{Style.BRIGHT}✗ 发生错误：{Style.RESET_ALL}{e}")
                print_separator("-")

    except Exception as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}✗ 初始化错误：{Style.RESET_ALL}{e}")
        print_separator("-")


def execute_task(config: Config, tools: List[Tool], skill_manager: SkillManager | None = None) -> None:
    """执行任务"""
    print_separator("-")
    print(f"{Fore.CYAN}{Style.BRIGHT}执行新任务{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}请输入任务描述（输入完成后按回车）：{Style.RESET_ALL}")

    task = input("> ").strip()

    if not task:
        print(f"{Fore.RED}✗ 任务描述不能为空{Style.RESET_ALL}")
        return

    try:
        # 创建客户端和智能体
        client = create_llm_client(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
        )

        # 创建步骤回调函数
        step_callback = create_step_callback(config.show_steps)

        # 1. 显式创建全局唯一的 tracer
        tracer = TraceManager()
        tid = tracer.start_trace(trace_type="Query")
        tracer.metadata["question"] = task  # 立即存入问题
        print(f"DEBUG: 当前技能字典长度: {len(skill_manager.skills)}")
        print(f"DEBUG: 技能字典内容: {list(skill_manager.skills.keys())}")

        # --- [新增] 2. 手动将 tracer 注入技能 (无需修改 ReactAgent 类) ---
        # 遍历所有技能，将全局 tracer 注入进去
        for skill_id, skill_instance in skill_manager.skills.items():
            if hasattr(skill_instance, "_tracer"):  # 检查是否有这个属性
                skill_instance._tracer = tracer
                print(f"✅ 已将全局 Tracer 注入技能: {skill_id}")
            # 注入 Trace ID (字符串)，供 MCP 转发使用
            skill_instance.current_trace_id = tid

            # 注入 Tracer 对象 (实例)，供本地记录使用 (如果技能有本地逻辑)
            skill_instance._tracer = tracer

            # 打印确认 (现在一定会有输出了)
            print(f"🔗 Trace 锚点已同步: [{skill_id}] -> {tid}")

        agent = ReactAgent(
            client,
            tools,
            max_steps=config.max_steps,
            temperature=config.temperature,
            step_callback=step_callback,
            skill_manager=skill_manager,
        )

        print(f"\n{Fore.CYAN}正在执行任务...{Style.RESET_ALL}\n")
        print_separator("-")

        # 执行任务
        result = agent.run(task)
        # 4. 提取答案并保存
        # 注意：确保这里拿到了真正的字符串答案
        final_res_text = ""
        if isinstance(result, dict):
            final_res_text = result.get("final_answer", "")
        else:
            final_res_text = str(result)

        if tracer:
            tracer.metadata["llm_answer"] = final_res_text
            save_path = tracer.finish_and_save()  # 统一在这里保存！
            print(f"📊 Trace 已保存至: {save_path}")

        print(f"{Fore.CYAN}Ragas 评估已改为 Dashboard 手动触发，不再阻塞主任务流程。{Style.RESET_ALL}")

        # 显示最终结果
        print(f"\n{Fore.GREEN}{Style.BRIGHT}最终答案：{Style.RESET_ALL}\n")
        print(result.get("final_answer", ""))
        print()
        print_separator("-")

    except LLMError as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}✗ API 错误：{Style.RESET_ALL}{e}")
        print_separator("-")
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}任务已被用户中断{Style.RESET_ALL}")
        print_separator("-")
    except Exception as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}✗ 发生错误：{Style.RESET_ALL}{e}")
        print_separator("-")


def interactive_mode(config: Config) -> int:
    """交互式菜单模式"""
    print_welcome()

    # 初始化 MCP 管理器
    mcp_config = load_mcp_config()
    mcp_manager = MCPManager(mcp_config)

    # 启动所有启用的 MCP 服务器
    print(f"{Fore.CYAN}正在加载 MCP 服务器...{Style.RESET_ALL}")
    started_count = mcp_manager.start_all()
    if started_count > 0:
        print(f"{Fore.GREEN}✓ 成功启动 {started_count} 个 MCP 服务器{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}ℹ 未启用 MCP 服务器{Style.RESET_ALL}")

    # 获取包含 MCP 工具的工具列表
    mcp_tools = mcp_manager.get_tools()
    tools = default_tools(include_mcp=True, mcp_tools=mcp_tools)

    if mcp_tools:
        print(f"{Fore.GREEN}✓ 加载了 {len(mcp_tools)} 个 MCP 工具{Style.RESET_ALL}")

    # 初始化技能管理器
    skill_manager = SkillManager()
    skill_count = skill_manager.load_all()
    if skill_count > 0:
        print(f"{Fore.GREEN}✓ 加载了 {skill_count} 个技能{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}* 未加载任何技能{Style.RESET_ALL}")

    try:
        while True:
            try:
                print_menu()
                choice = input(f"{Fore.CYAN}请选择操作 (1-6): {Style.RESET_ALL}").strip()

                if choice == "1":
                    # 执行新任务
                    execute_task(config, tools, skill_manager)

                elif choice == "2":
                    # 多轮对话模式
                    multi_turn_conversation(config, tools, skill_manager)

                elif choice == "3":
                    # 查看工具列表
                    show_tools(tools)

                elif choice == "4":
                    # 配置设置
                    configure_settings(config)

                elif choice == "5":
                    # 查看技能列表
                    show_skills(skill_manager)

                elif choice == "6":
                    # P2: 多 Agent 协作模式
                    print(f"\n{Fore.CYAN}请输入多 Agent 协作模式的任务：{Style.RESET_ALL}")
                    task = input("> ").strip()
                    if task:
                        run_multi_agent_task(config, task, mcp_tools=mcp_tools)

                elif choice == "7":
                    # 退出程序
                    print(f"\n{Fore.YELLOW}感谢使用！再见！{Style.RESET_ALL}\n")
                    return 0

                else:
                    print(f"{Fore.RED}✗ 无效的选择，请输入 1-7{Style.RESET_ALL}")

            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}感谢使用！再见！{Style.RESET_ALL}\n")
                return 0
            except EOFError:
                print(f"\n\n{Fore.YELLOW}感谢使用！再见！{Style.RESET_ALL}\n")
                return 0
            except Exception as e:
                print(f"\n{Fore.RED}{Style.BRIGHT}✗ 发生错误：{Style.RESET_ALL}{e}\n")

    finally:
        # 清理 MCP 资源
        print(f"{Fore.CYAN}正在关闭 MCP 服务器...{Style.RESET_ALL}")
        mcp_manager.stop_all()
        print(f"{Fore.GREEN}✓ MCP 服务器已关闭{Style.RESET_ALL}")


def run_single_task(config: Config, task: str) -> int:
    """运行单个任务（命令行模式）"""
    apply_runtime_provider_env(config)
    # 初始化 MCP
    mcp_config = load_mcp_config()
    mcp_manager = MCPManager(mcp_config)

    try:
        # 启动 MCP 服务器
        started_count = mcp_manager.start_all()
        if started_count > 0:
            print(f"{Fore.GREEN}✓ 启动了 {started_count} 个 MCP 服务器{Style.RESET_ALL}")

        # 获取工具
        mcp_tools = mcp_manager.get_tools()
        tools = default_tools(include_mcp=True, mcp_tools=mcp_tools)

        # 初始化技能管理器
        skill_manager = SkillManager(mcp_manager=mcp_tools)
        skill_manager.load_all()

        client = create_llm_client(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
        )

        # 创建步骤回调函数
        step_callback = create_step_callback(config.show_steps)

        agent = ReactAgent(
            client,
            tools,
            max_steps=config.max_steps,
            temperature=config.temperature,
            step_callback=step_callback,
            skill_manager=skill_manager,
        )

        print(f"\n{Fore.CYAN}正在执行任务：{Style.RESET_ALL}{task}\n")
        print_separator()

        result = agent.run(task)

        # 显示最终结果
        print(f"\n{Fore.GREEN}{Style.BRIGHT}最终答案：{Style.RESET_ALL}\n")
        print(result.get("final_answer", ""))
        print()
        print_separator()

        return 0

    except LLMError as e:
        print(f"{Fore.RED}{Style.BRIGHT}✗ API 错误：{Style.RESET_ALL}{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"{Fore.RED}{Style.BRIGHT}✗ 发生错误：{Style.RESET_ALL}{e}", file=sys.stderr)
        return 1
    finally:
# 清理 MCP 资源
        mcp_manager.stop_all()


def run_multi_agent_task(
    config: Config,
    task: str,
    max_parallel: int = 3,
    use_docker: bool = True,
    mcp_tools: Optional[List[Tool]] = None,
) -> int:
    """使用多 Agent 协作系统运行任务（P2）"""
    import re
    import time
    from datetime import datetime

    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== P2 多 Agent 协作系统 ==={Style.RESET_ALL}")
    print(f"最大并行任务数: {max_parallel}")
    print(f"Docker 隔离执行: {'启用' if use_docker else '禁用'}")
    print()

    apply_runtime_provider_env(config)

    # 初始化 LLM 客户端
    print(f"{Fore.CYAN}[1/5] 初始化 LLM 客户端...{Style.RESET_ALL}")
    client = create_llm_client(
        provider=config.provider,
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
    )
    llm_ready, effective_use_docker = run_multi_agent_preflight(config, client, use_docker)
    if not llm_ready:
        return 1

    # 初始化技能管理器
    print(f"{Fore.CYAN}[2/5] 加载技能管理器...{Style.RESET_ALL}")
    skill_manager = SkillManager(llm_client=client)
    skill_count = skill_manager.load_all()
    print(f"      已加载 {skill_count} 个技能")

    # 获取代码工具
    print(f"{Fore.CYAN}[3/5] 加载工具...{Style.RESET_ALL}")
    tools = default_tools()
    print(f"      已加载 {len(tools)} 个工具")
    agent_profiles = load_profiles_for_task(task)
    if agent_profiles:
        profile_names = ", ".join(sorted(agent_profiles.keys()))
        print(f"      已加载多 Agent Profile: {profile_names}")

    # 创建 OrchestratorAgent
    print(f"{Fore.CYAN}[4/5] 创建 OrchestratorAgent...{Style.RESET_ALL}")
    orchestrator = OrchestratorAgent(
        llm_client=client,
        code_tools=tools,
        skill_manager=skill_manager,
        max_parallel=max_parallel,
        use_docker=effective_use_docker,
        mcp_tools=mcp_tools or [],
        agent_profiles=agent_profiles,
    )

    # 执行任务
    print(f"{Fore.CYAN}[5/5] 执行任务...{Style.RESET_ALL}")
    print_separator()

    start_time = time.time()
    try:
        result = orchestrator.run(task, trace=True)
        duration = time.time() - start_time

        print_separator()
        print()
        print(f"{Fore.GREEN}任务完成!{Style.RESET_ALL}")
        print(f"  耗时: {duration:.2f}s")
        print(f"  状态: {format_multi_agent_status(result)}")
        partial_count = result.get("partial_count", 0)
        print(
            f"  子任务: {result.get('completed_count', 0)} 完成"
            f" / {partial_count} 部分完成"
            f" / {result.get('failed_count', 0)} 失败"
            f" / 共 {result.get('sub_tasks_count', 0)}"
        )

        # 保存结果到 task 文件夹。每次运行使用独立目录，避免不同任务互相覆盖。
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = result.get("task_id", "task")
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", task).strip("_").lower()[:48]
        run_name = "_".join(part for part in (timestamp, task_id, slug) if part)
        output_dir = os.path.join("task", run_name)
        os.makedirs(output_dir, exist_ok=True)

        # 保存 Markdown 文档
        md_file = os.path.join(output_dir, f"{task_id}_report.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# 多 Agent 任务执行报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**任务**: {task}\n\n")
            f.write(f"**执行时间**: {duration:.2f} 秒\n\n")
            f.write(f"**状态**: {format_multi_agent_status(result)}\n\n")
            f.write(f"**完成子任务**: {result.get('completed_count', 0)} / {result.get('sub_tasks_count', 0)}\n\n")
            f.write(f"**部分完成子任务**: {result.get('partial_count', 0)}\n\n")
            f.write(f"**失败子任务**: {result.get('failed_count', 0)}\n\n")
            f.write("---\n\n## 分析结果\n\n")
            f.write(result.get("final_answer", "无结果"))
            f.write("\n\n---\n\n## 执行追踪\n\n")
            if result.get("trace"):
                decomp = result["trace"].get("decomposition", {})
                f.write(f"### 任务分解\n\n")
                f.write(f"- 需要 RAG: {'是' if decomp.get('requires_rag') else '否'}\n")
                f.write(f"- 需要代码: {'是' if decomp.get('requires_code') else '否'}\n")
                f.write(f"- 执行计划: {decomp.get('execution_plan', 'N/A')}\n\n")
                f.write(f"### 子任务\n\n")
                for st in result["trace"].get("sub_tasks", []):
                    status = st.get("status")
                    if status == "completed":
                        status_icon = "[OK]"
                    elif status == "partial":
                        status_icon = "[PARTIAL]"
                    else:
                        status_icon = "[FAIL]"
                    f.write(f"- {status_icon} [{st.get('type')}] {st.get('description')[:60]}...\n")
                    if st.get("error"):
                        f.write(f"  - 错误/说明: {st.get('error')}\n")

        print(f"\n{Fore.GREEN}报告已保存: {md_file}{Style.RESET_ALL}")

        if result.get("rag_trace_id"):
            print(f"{Fore.CYAN}Ragas 评估已改为 Dashboard 手动触发，不再阻塞主任务流程。{Style.RESET_ALL}")

        # 显示最终答案
        print(f"\n{Fore.GREEN}{Style.BRIGHT}最终答案：{Style.RESET_ALL}\n")
        print(result.get("final_answer", ""))
        print()
        print_separator()

        return 0 if result.get("overall_status") == "success" else 1

    except Exception as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}✗ 执行错误：{Style.RESET_ALL}{e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main(argv: Any = None) -> int:
    """主入口函数"""
    load_dotenv()
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # 如果没有提供 API 密钥，尝试根据提供商获取
    if not args.api_key:
        args.api_key = get_api_key_for_provider(args.provider)

    # 检查 API 密钥
    if not args.api_key:
        print(f"{Fore.RED}✗ 缺少 API 密钥。{Style.RESET_ALL}", file=sys.stderr)
        print(f"请提供 --api-key 或设置环境变量 {args.provider.upper()}_API_KEY。", file=sys.stderr)
        return 2

    # 获取提供商的默认配置
    provider_defaults = PROVIDER_DEFAULTS.get(args.provider, {})

    # 如果没有指定 base_url，使用提供商默认值
    if not args.base_url:
        args.base_url = provider_defaults.get("base_url", "https://api.deepseek.com")

    # 如果模型是默认的 deepseek-chat 但提供商不是 deepseek，更新模型
    if args.model == "deepseek-chat" and args.provider != "deepseek":
        args.model = provider_defaults.get("model", args.model)

    # 创建配置
    config = Config(
        api_key=args.api_key,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        max_steps=args.max_steps,
        temperature=args.temperature,
        show_steps=args.show_steps,
    )

    # 如果提供了任务参数，直接执行任务
    if args.task:
        # 如果启用了多 Agent 模式
        if args.multi_agent:
            return run_multi_agent_task(config, args.task, args.max_parallel, args.use_docker)
        return run_single_task(config, args.task)

    # 如果指定了交互模式或没有提供任务，进入交互式菜单
    if args.interactive or not args.task:
        return interactive_mode(config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
