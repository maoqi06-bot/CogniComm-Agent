"""资源管理与安全模块

提供：
1. 资源限流（并发控制、超时控制）
2. 安全沙箱（命令白名单、危险命令检测）
3. 健康检查（系统状态监控）
"""

from __future__ import annotations

import os
import sys
import time
import signal
import subprocess
import threading
import platform
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from contextlib import contextmanager

import psutil


class SecurityLevel(Enum):
    """安全级别"""
    NONE = "none"           # 无限制
    BASIC = "basic"         # 基础限制
    STRICT = "strict"       # 严格限制


# ============================================================
# 资源限流
# ============================================================

@dataclass
class RateLimitConfig:
    """限流配置"""
    max_concurrent_llm: int = 5           # 最大并发 LLM 调用
    max_concurrent_tools: int = 10        # 最大并发工具执行
    llm_timeout: float = 60.0            # LLM 调用超时（秒）
    tool_timeout: float = 30.0          # 工具执行超时（秒）
    shell_timeout: float = 60.0         # Shell 命令超时（秒）
    daily_token_limit: int = 0           # 每日 token 限额（0 = 无限制）


class SemaphoreManager:
    """
    信号量管理器 - 控制并发资源访问
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._llm_semaphore = threading.Semaphore(config.max_concurrent_llm)
        self._tool_semaphore = threading.Semaphore(config.max_concurrent_tools)

        # 统计
        self._llm_active = 0
        self._tool_active = 0
        self._llm_lock = threading.Lock()
        self._tool_lock = threading.Lock()

        # Token 统计
        self._daily_tokens = 0
        self._last_reset = datetime.now()

    @contextmanager
    def llm_resource(self):
        """LLM 资源上下文"""
        with self._llm_semaphore:
            with self._llm_lock:
                self._llm_active += 1
            try:
                yield
            finally:
                with self._llm_lock:
                    self._llm_active -= 1

    @contextmanager
    def tool_resource(self):
        """工具资源上下文"""
        with self._tool_semaphore:
            with self._tool_lock:
                self._tool_active += 1
            try:
                yield
            finally:
                with self._tool_lock:
                    self._tool_active -= 1

    def add_tokens(self, count: int):
        """添加 token 消耗"""
        # 检查是否需要重置
        now = datetime.now()
        if (now - self._last_reset).days >= 1:
            self._daily_tokens = 0
            self._last_reset = now

        self._daily_tokens += count

    def check_token_limit(self, additional: int = 0) -> bool:
        """检查是否超过 token 限额"""
        if self.config.daily_token_limit == 0:
            return True
        return (self._daily_tokens + additional) <= self.config.daily_token_limit

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "llm_active": self._llm_active,
            "llm_available": self.config.max_concurrent_llm - self._llm_active,
            "llm_max": self.config.max_concurrent_llm,
            "tool_active": self._tool_active,
            "tool_available": self.config.max_concurrent_tools - self._tool_active,
            "tool_max": self.config.max_concurrent_tools,
            "daily_tokens": self._daily_tokens,
            "daily_token_limit": self.config.daily_token_limit,
        }


# ============================================================
# 超时控制
# ============================================================

class TimeoutError(Exception):
    """超时异常"""
    pass


def run_with_timeout(
    func: Callable,
    timeout: float,
    args: tuple = (),
    kwargs: dict = None,
    default: Any = None,
) -> Any:
    """
    带超时的函数执行

    Args:
        func: 要执行的函数
        timeout: 超时时间（秒）
        args: 位置参数
        kwargs: 关键字参数
        default: 超时时返回的默认值

    Returns:
        函数返回值或默认值

    Raises:
        TimeoutError: 超时时抛出（如果 default=None）
    """
    if kwargs is None:
        kwargs = {}

    result = [None]
    exception = [None]
    finished = threading.Event()

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
        finally:
            finished.set()

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    finished.wait(timeout)

    if thread.is_alive():
        # 超时
        if default is not None:
            return default
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout}s")

    if exception[0]:
        raise exception[0]

    return result[0]


# ============================================================
# 安全沙箱
# ============================================================

@dataclass
class SecurityConfig:
    """安全配置"""
    level: SecurityLevel = SecurityLevel.BASIC
    allowed_commands: Set[str] = field(default_factory=set)
    blocked_patterns: List[str] = field(default_factory=list)
    allowed_paths: Set[str] = field(default_factory=set)
    blocked_paths: Set[str] = field(default_factory=set)
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    enable_path_traversal_check: bool = True


class SecureShellExecutor:
    """
    安全 Shell 命令执行器

    Features:
    - 命令白名单
    - 危险命令黑名单
    - 路径遍历攻击检测
    - 输出长度限制
    """

    # 默认允许的命令
    DEFAULT_ALLOWED = {
        "python", "python3", "pip", "pytest", "unittest",
        "git", "ls", "cat", "head", "tail", "grep", "find",
        "wc", "sort", "uniq", "awk", "sed",
        "node", "npm", "npx",
    }

    # 危险命令模式
    DEFAULT_BLOCKED_PATTERNS = [
        r"rm\s+-rf\s+/",           # 删除根目录
        r"rm\s+-rf\s+\*",          # 删除所有文件
        r"format\s+[a-z]:",       # 格式化磁盘
        r">\s*/dev/sd",           # 直接写入设备
        r"dd\s+if=.*of=/dev/",     # dd 到设备
        r";\s*rm\s+",             # 注入 rm
        r"&\s*rm\s+",             # 后台 rm
        r"\|\s*rm\s+",            # 管道 rm
        r"curl.*\|.*sh",           # curl pipe sh
        r"wget.*\|.*sh",           # wget pipe sh
        r"eval\s+.*\$",           # eval 注入
        r"exec\s+.*\$",           # exec 注入
    ]

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._setup_defaults()

    def _setup_defaults(self):
        """设置默认值"""
        if not self.config.allowed_commands:
            self.config.allowed_commands = set(self.DEFAULT_ALLOWED)

        if not self.config.blocked_patterns:
            self.config.blocked_patterns = list(self.DEFAULT_BLOCKED_PATTERNS)

    def validate_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        验证命令是否安全

        Args:
            command: 要验证的命令

        Returns:
            (is_safe, error_message)
        """
        if not command or not command.strip():
            return False, "Empty command"

        # 检查危险模式
        for pattern in self.config.blocked_patterns:
            import re
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command matches blocked pattern: {pattern}"

        # 检查路径遍历
        if self.config.enable_path_traversal_check:
            if ".." in command or "/.." in command or "\\.." in command:
                return False, "Path traversal detected"

        # 白名单检查（如果配置了）
        if self.config.level == SecurityLevel.STRICT and self.config.allowed_commands:
            cmd_parts = command.strip().split()
            if cmd_parts:
                cmd = os.path.basename(cmd_parts[0])  # 只检查基础命令
                if cmd not in self.config.allowed_commands:
                    return False, f"Command '{cmd}' not in whitelist"

        return True, None

    def execute(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: Optional[str] = None,
        max_output_size: int = 1024 * 1024,  # 1MB
    ) -> Dict[str, Any]:
        """
        安全执行 Shell 命令

        Args:
            command: 要执行的命令
            timeout: 超时时间
            cwd: 工作目录
            max_output_size: 最大输出大小

        Returns:
            执行结果字典

        Raises:
            ValueError: 命令不安全
            TimeoutError: 执行超时
        """
        # 验证命令
        is_safe, error = self.validate_command(command)
        if not is_safe:
            return {
                "success": False,
                "error": f"Security Error: {error}",
                "returncode": -1,
                "stdout": "",
                "stderr": "",
            }

        try:
            # 执行命令
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # 限制输出大小
            stdout = result.stdout[:max_output_size]
            stderr = result.stderr[:max_output_size]

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": len(result.stdout) > max_output_size,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
                "returncode": -1,
                "stdout": "",
                "stderr": "",
                "timeout": True,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "returncode": -1,
                "stdout": "",
                "stderr": "",
            }


# ============================================================
# 健康检查
# ============================================================

@dataclass
class HealthStatus:
    """健康状态"""
    status: str = "healthy"  # healthy, degraded, unhealthy
    timestamp: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    系统健康检查器

    监控：
    - Agent 状态
    - LLM 调用状态
    - 资源使用情况
    - 错误率
    """

    def __init__(
        self,
        max_error_rate: float = 0.3,        # 最大错误率（超过此值标记为不健康）
        max_memory_mb: float = 1024,         # 最大内存使用（MB）
        min_success_rate: float = 0.5,       # 最小成功率
    ):
        self.start_time = datetime.now()
        self.max_error_rate = max_error_rate
        self.max_memory_mb = max_memory_mb
        self.min_success_rate = min_success_rate

        self._request_count = 0
        self._error_count = 0
        self._llm_call_count = 0
        self._llm_error_count = 0
        self._lock = threading.Lock()

    def record_request(self, success: bool):
        """记录请求"""
        with self._lock:
            self._request_count += 1
            if not success:
                self._error_count += 1

    def record_llm_call(self, success: bool):
        """记录 LLM 调用"""
        with self._lock:
            self._llm_call_count += 1
            if not success:
                self._llm_error_count += 1

    def check(self) -> HealthStatus:
        """
        执行健康检查

        Returns:
            HealthStatus: 健康状态
        """
        details = {}

        # 计算运行时间
        uptime = (datetime.now() - self.start_time).total_seconds()

        # 计算错误率
        error_rate = 0.0
        if self._request_count > 0:
            error_rate = self._error_count / self._request_count

        # 计算成功率
        success_rate = 1.0 - error_rate

        # 计算 LLM 错误率
        llm_error_rate = 0.0
        if self._llm_call_count > 0:
            llm_error_rate = self._llm_error_count / self._llm_call_count

        # 获取内存使用
        memory_mb = 0.0
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
        except Exception:
            pass

        # 获取系统状态
        cpu_percent = 0.0
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except Exception:
            pass

        # 组装详情
        details.update({
            "uptime_seconds": uptime,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": error_rate,
            "success_rate": success_rate,
            "llm_calls": self._llm_call_count,
            "llm_errors": self._llm_error_count,
            "llm_error_rate": llm_error_rate,
            "memory_mb": round(memory_mb, 2),
            "cpu_percent": round(cpu_percent, 2),
        })

        # 判断状态
        status = "healthy"

        if error_rate > self.max_error_rate:
            status = "unhealthy"
            details["reason"] = f"Error rate {error_rate:.1%} exceeds threshold {self.max_error_rate:.1%}"
        elif success_rate < self.min_success_rate:
            status = "degraded"
            details["reason"] = f"Success rate {success_rate:.1%} below threshold {self.min_success_rate:.1%}"
        elif memory_mb > self.max_memory_mb:
            status = "degraded"
            details["reason"] = f"Memory usage {memory_mb:.1f}MB exceeds threshold {self.max_memory_mb}MB"

        return HealthStatus(
            status=status,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            details=details,
        )

    def reset_stats(self):
        """重置统计"""
        with self._lock:
            self._request_count = 0
            self._error_count = 0
            self._llm_call_count = 0
            self._llm_error_count = 0


# ============================================================
# 资源管理器
# ============================================================

class ResourceManager:
    """
    统一资源管理器

    整合限流、超时、安全沙箱和健康检查。
    """

    def __init__(
        self,
        rate_limit_config: Optional[RateLimitConfig] = None,
        security_config: Optional[SecurityConfig] = None,
        health_config: Optional[Dict[str, Any]] = None,
    ):
        self.rate_limiter = SemaphoreManager(rate_limit_config or RateLimitConfig())
        self.security = SecureShellExecutor(security_config or SecurityConfig())
        self.health = HealthChecker(**(health_config or {}))

    def get_status(self) -> Dict[str, Any]:
        """获取完整状态"""
        health = self.health.check()
        return {
            "health": {
                "status": health.status,
                "details": health.details,
            },
            "rate_limit": self.rate_limiter.get_stats(),
        }

    def is_healthy(self) -> bool:
        """检查是否健康"""
        return self.health.check().status == "healthy"


# ============================================================
# 便捷函数
# ============================================================

# 全局资源管理器实例
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """获取全局资源管理器"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
    return _global_resource_manager


def setup_resource_manager(
    rate_limit_config: Optional[RateLimitConfig] = None,
    security_config: Optional[SecurityConfig] = None,
) -> ResourceManager:
    """设置全局资源管理器"""
    global _global_resource_manager
    _global_resource_manager = ResourceManager(rate_limit_config, security_config)
    return _global_resource_manager
