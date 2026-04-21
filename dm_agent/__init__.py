"""DM-Agent - 基于 ReAct 的多模型智能体系统

一个支持多种 LLM API (DeepSeek、OpenAI、Claude、Gemini) 的 ReAct 智能体实现。
"""

from .core import ReactAgent, Step
from .clients import (
    BaseLLMClient,
    LLMError,
    DeepSeekClient,
    OpenAIClient,
    ClaudeClient,
    GeminiClient,
    create_llm_client,
    PROVIDER_DEFAULTS,
)
from .tools import Tool, default_tools
from .prompts import build_code_agent_prompt
from .skills import BaseSkill, ConfigSkill, SkillMetadata, SkillManager
from .memory import (
    ContextCompressor,
    LongTermMemoryStore,
    MemoryEntry,
    MemoryCategory,
    MemoryPriority,
    MemoryManager,
    MemoryRetrievalResult,
    create_memory_manager,
    create_memory_tools,
)
from .utils import (
    setup_logging,
    get_logger,
    retry_on_api_error,
    with_timeout,
    TimeoutError,
    # P1: Resource Management & Security
    ResourceManager,
    RateLimitConfig,
    SecurityConfig,
    SecurityLevel,
    SemaphoreManager,
    SecureShellExecutor,
    HealthChecker,
    HealthStatus,
    setup_resource_manager,
    get_resource_manager,
)
from .multi_agent import (
    AgentMemoryPolicy,
    AgentProfile,
    OrchestratorAgent,
    RAGAgent,
    RAGAgentProfile,
    CodeAgent,
    CodeAgentProfile,
    DockerRunner,
    MemoryEvent,
    MemoryWriteTemplate,
    MultiAgentMemoryConfig,
    MultiAgentMemoryHub,
    TaskDecomposer,
    TaskScheduler,
    ResultAggregator,
    TaskType,
    SubTask,
    TaskDecomposition,
    BaseAgent,
    build_domain_profiles,
    load_profiles_for_task,
)

__version__ = "1.7.0"

__all__ = [
    # Core
    "ReactAgent",
    "Step",
    # Clients
    "BaseLLMClient",
    "LLMError",
    "DeepSeekClient",
    "OpenAIClient",
    "ClaudeClient",
    "GeminiClient",
    "create_llm_client",
    "PROVIDER_DEFAULTS",
    # Tools
    "Tool",
    "default_tools",
    # Prompts
    "build_code_agent_prompt",
    # Skills
    "BaseSkill",
    "ConfigSkill",
    "SkillMetadata",
    "SkillManager",
    # Memory
    "ContextCompressor",
    "LongTermMemoryStore",
    "MemoryEntry",
    "MemoryCategory",
    "MemoryPriority",
    "MemoryManager",
    "MemoryRetrievalResult",
    "create_memory_manager",
    "create_memory_tools",
    # Utils
    "setup_logging",
    "get_logger",
    "retry_on_api_error",
    "with_timeout",
    "TimeoutError",
    # P1: Resource Management & Security
    "ResourceManager",
    "RateLimitConfig",
    "SecurityConfig",
    "SecurityLevel",
    "SemaphoreManager",
    "SecureShellExecutor",
    "HealthChecker",
    "HealthStatus",
    "setup_resource_manager",
    "get_resource_manager",
    # P2: Multi-Agent Orchestration
    "AgentMemoryPolicy",
    "AgentProfile",
    "OrchestratorAgent",
    "RAGAgent",
    "RAGAgentProfile",
    "CodeAgent",
    "CodeAgentProfile",
    "DockerRunner",
    "MemoryEvent",
    "MemoryWriteTemplate",
    "MultiAgentMemoryConfig",
    "MultiAgentMemoryHub",
    "TaskDecomposer",
    "TaskScheduler",
    "ResultAggregator",
    "TaskType",
    "SubTask",
    "TaskDecomposition",
    "BaseAgent",
    "build_domain_profiles",
    "load_profiles_for_task",
]
