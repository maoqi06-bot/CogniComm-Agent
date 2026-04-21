"""Public exports for the multi-agent package.

Implementation lives in focused modules so the package entry point stays small.
"""

from .profiles import AgentProfile, CodeAgentProfile, RAGAgentProfile
from .domain_profiles import build_domain_profiles
from .profile_loader import load_profiles_for_task
from .memory import (
    AgentMemoryPolicy,
    MemoryEvent,
    MemoryWriteTemplate,
    MultiAgentMemoryConfig,
    MultiAgentMemoryHub,
)
from .runtime import (
    BaseAgent,
    CodeAgent,
    DockerRunner,
    OrchestratorAgent,
    RAGAgent,
    ResultAggregator,
    SubTask,
    TaskDecomposition,
    TaskDecomposer,
    TaskScheduler,
    TaskType,
)

__all__ = [
    "AgentProfile",
    "AgentMemoryPolicy",
    "BaseAgent",
    "build_domain_profiles",
    "load_profiles_for_task",
    "CodeAgent",
    "CodeAgentProfile",
    "DockerRunner",
    "MemoryEvent",
    "MemoryWriteTemplate",
    "MultiAgentMemoryConfig",
    "MultiAgentMemoryHub",
    "OrchestratorAgent",
    "RAGAgent",
    "RAGAgentProfile",
    "ResultAggregator",
    "SubTask",
    "TaskDecomposition",
    "TaskDecomposer",
    "TaskScheduler",
    "TaskType",
]
