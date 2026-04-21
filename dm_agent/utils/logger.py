"""日志系统模块

提供统一的日志配置，支持：
1. 分级日志输出（DEBUG/INFO/WARNING/ERROR/CRITICAL）
2. 文件 + 控制台双输出
3. 结构化日志格式
4. 自动日志轮转
5. 上下文感知日志（自动添加任务ID、Agent ID等）
"""

from __future__ import annotations

import os
import sys
import logging
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

# 上下文变量，用于存储请求级别的信息
_request_id: ContextVar[str] = ContextVar('request_id', default='')
_agent_id: ContextVar[str] = ContextVar('agent_id', default='')
_task_id: ContextVar[str] = ContextVar('task_id', default='')


@dataclass
class LogConfig:
    """日志配置"""
    # 基本配置
    log_dir: str = "./logs"
    log_level: str = "INFO"
    log_file: str = "dm_agent.log"
    error_file: str = "errors.log"

    # 格式化配置
    use_json: bool = False
    include_timestamp: bool = True
    include_caller_info: bool = True

    # 轮转配置
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    when: str = "midnight"  # 按天轮转

    # 输出配置
    console_output: bool = True
    file_output: bool = True
    error_file_output: bool = True

    # 性能配置
    buffer_size: int = 100  # 日志缓冲区大小
    flush_level: int = logging.ERROR  # 缓冲区刷新级别


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def __init__(self, include_caller_info: bool = True):
        super().__init__()
        self.include_caller_info = include_caller_info

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加上下文信息
        request_id = _request_id.get()
        agent_id = _agent_id.get()
        task_id = _task_id.get()

        if request_id:
            log_data["request_id"] = request_id
        if agent_id:
            log_data["agent_id"] = agent_id
        if task_id:
            log_data["task_id"] = task_id

        # 添加调用者信息
        if self.include_caller_info:
            log_data["caller"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # 添加自定义字段
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（用于控制台）"""

    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m',
    }

    def __init__(self, fmt: Optional[str] = None):
        super().__init__(fmt)
        self.default_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.formatter = logging.Formatter(self.default_fmt)

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        record.levelname = f"{color}{record.levelname}{reset}"
        record.name = f"{color}{record.name}{reset}"

        return self.formatter.format(record)


class AgentLogger:
    """
    Agent 专用日志记录器

    提供上下文感知的日志记录，自动附加请求/任务信息。
    """

    def __init__(
        self,
        name: str = "dm_agent",
        config: Optional[LogConfig] = None,
    ):
        self.name = name
        self.config = config or LogConfig()
        self._logger: Optional[logging.Logger] = None
        self._setup_logger()

    def _setup_logger(self):
        """设置日志记录器"""
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(getattr(logging, self.config.log_level.upper()))
        self._logger.handlers.clear()

        # 确保日志目录存在
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 文件处理器 - 主日志
        if self.config.file_output:
            file_handler = RotatingFileHandler(
                filename=str(log_dir / self.config.log_file),
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count,
                encoding='utf-8',
            )
            file_handler.setLevel(logging.DEBUG)

            if self.config.use_json:
                file_handler.setFormatter(StructuredFormatter(
                    include_caller_info=self.config.include_caller_info
                ))
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))

            self._logger.addHandler(file_handler)

        # 文件处理器 - 错误日志
        if self.config.error_file_output:
            error_handler = RotatingFileHandler(
                filename=str(log_dir / self.config.error_file),
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count,
                encoding='utf-8',
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self._logger.addHandler(error_handler)

        # 控制台处理器
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.log_level.upper()))

            if self.config.use_json:
                console_handler.setFormatter(StructuredFormatter(
                    include_caller_info=False
                ))
            else:
                console_handler.setFormatter(ColoredFormatter())

            self._logger.addHandler(console_handler)

    def set_context(
        self,
        request_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        """设置日志上下文"""
        if request_id:
            _request_id.set(request_id)
        if agent_id:
            _agent_id.set(agent_id)
        if task_id:
            _task_id.set(task_id)

    def clear_context(self):
        """清除日志上下文"""
        _request_id.set('')
        _agent_id.set('')
        _task_id.set('')

    def debug(self, message: str, **kwargs):
        self._logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        self._logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        self._logger.warning(message, extra=kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        self._logger.error(message, exc_info=exc_info, extra=kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs):
        self._logger.critical(message, exc_info=exc_info, extra=kwargs)

    def exception(self, message: str, **kwargs):
        """记录异常，自动包含堆栈信息"""
        self._logger.exception(message, extra=kwargs)

    def log_execution(self, action: str, **details):
        """记录执行步骤"""
        self.info(f"[EXEC] {action}", extra={"execution_details": details})

    def log_llm_call(self, model: str, tokens_used: int, duration: float, **kwargs):
        """记录 LLM 调用"""
        self.info(
            f"[LLM] Model: {model}, Tokens: {tokens_used}, Duration: {duration:.2f}s",
            extra={"llm_stats": {"model": model, "tokens": tokens_used, "duration": duration, **kwargs}}
        )

    def log_tool_call(self, tool_name: str, success: bool, duration: float, **kwargs):
        """记录工具调用"""
        level = logging.INFO if success else logging.ERROR
        self._logger.log(
            level,
            f"[TOOL] {tool_name} - {'SUCCESS' if success else 'FAILED'} ({duration:.2f}s)",
            extra={"tool_stats": {"name": tool_name, "success": success, "duration": duration, **kwargs}}
        )


# 全局日志实例
_global_logger: Optional[AgentLogger] = None


def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    use_json: bool = False,
    **kwargs
) -> AgentLogger:
    """
    设置全局日志系统

    Args:
        log_dir: 日志目录
        log_level: 日志级别
        use_json: 是否使用 JSON 格式
        **kwargs: 其他 LogConfig 参数

    Returns:
        AgentLogger: 配置好的日志记录器
    """
    global _global_logger

    config = LogConfig(
        log_dir=log_dir,
        log_level=log_level,
        use_json=use_json,
        **kwargs
    )

    _global_logger = AgentLogger(config=config)
    return _global_logger


def get_logger(name: Optional[str] = None) -> AgentLogger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称（可选）

    Returns:
        AgentLogger: 日志记录器实例
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = setup_logging()

    if name and name != _global_logger.name:
        # 返回新的命名的 logger
        return AgentLogger(name=name, config=_global_logger.config)

    return _global_logger


def log_function_calls(logger: Optional[AgentLogger] = None):
    """
    函数调用日志装饰器

    Args:
        logger: 日志记录器（可选）

    Examples:
        @log_function_calls()
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger()

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            logger.debug(f"[ENTER] {func_name}(args={args}, kwargs={kwargs})")

            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"[EXIT] {func_name} -> {type(result).__name__} ({duration:.3f}s)")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.exception(f"[ERROR] {func_name} failed after {duration:.3f}s: {e}")
                raise

        return wrapper
    return decorator


class LogCapture:
    """日志捕获上下文管理器，用于测试"""

    def __init__(self, logger_name: str = "dm_agent", level: int = logging.INFO):
        self.logger_name = logger_name
        self.level = level
        self.records: list = []
        self.handler: Optional[logging.Handler] = None

    def __enter__(self):
        self.records = []
        self.handler = logging.Handler()
        self.handler.setLevel(self.level)
        self.handler.emit = lambda record: self.records.append(record)

        logger = logging.getLogger(self.logger_name)
        logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            logger = logging.getLogger(self.logger_name)
            logger.removeHandler(self.handler)

    def get_messages(self) -> list:
        """获取捕获的日志消息"""
        return [record.getMessage() for record in self.records]

    def has_message(self, substring: str) -> bool:
        """检查是否包含指定消息"""
        return any(substring in msg for msg in self.get_messages())
