"""记忆与上下文管理模块

该模块提供短期记忆（上下文压缩）和长期记忆（RAG 存储）的完整解决方案。

短期记忆 (Short-term Memory):
    - ContextCompressor: 对话历史压缩，减少 token 消耗

长期记忆 (Long-term Memory):
    - LongTermMemoryStore: 基于 FAISS 的向量记忆存储，支持 CRUD 操作
    - MemoryManager: 协调短期与长期记忆的核心管理器
    - MemoryEntry: 记忆条目数据模型
    - MemoryCategory: 记忆类别枚举
    - MemoryPriority: 记忆优先级枚举

记忆管理工具:
    - create_memory_tools: 创建供 Agent 使用的记忆操作工具
    - get_memory_tool_names: 获取工具名称列表
"""

from .context_compressor import ContextCompressor
from .long_term_memory import (
    LongTermMemoryStore,
    MemoryEntry,
    MemoryCategory,
    MemoryPriority,
    MemorySearchResult,
)
from .memory_manager import (
    MemoryManager,
    MemoryRetrievalResult,
    create_memory_manager,
)
from .memory_tools import (
    create_memory_tools,
    get_memory_tool_names,
)

__all__ = [
    # 短期记忆
    "ContextCompressor",
    # 长期记忆存储
    "LongTermMemoryStore",
    "MemoryEntry",
    "MemoryCategory",
    "MemoryPriority",
    "MemorySearchResult",
    # 记忆管理器
    "MemoryManager",
    "MemoryRetrievalResult",
    "create_memory_manager",
    # 记忆工具
    "create_memory_tools",
    "get_memory_tool_names",
]