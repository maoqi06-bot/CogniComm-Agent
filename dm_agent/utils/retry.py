"""异常处理与重试机制模块

提供：
1. LLM API 调用重试装饰器（指数退避）
2. 工具执行超时装饰器
3. 统一的异常类型定义
"""

from __future__ import annotations

import time
import functools
import signal
from typing import Callable, Any, Optional, TypeVar, Tuple, Type
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T')


class RetryStrategy(Enum):
    """重试策略枚举"""
    FIXED = "fixed"           # 固定间隔
    EXPONENTIAL = "exponential"  # 指数退避
    LINEAR = "linear"         # 线性增长


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3           # 最大尝试次数
    initial_delay: float = 1.0       # 初始延迟（秒）
    max_delay: float = 30.0          # 最大延迟（秒）
    multiplier: float = 2.0          # 延迟倍增因子
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


class AgentError(Exception):
    """Agent 基础异常类"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class LLMError(AgentError):
    """LLM 调用相关异常"""
    pass


class ToolExecutionError(AgentError):
    """工具执行异常"""
    pass


class TimeoutError(AgentError):
    """超时异常"""
    pass


class RateLimitError(LLMError):
    """API 限流异常"""
    pass


class AuthenticationError(LLMError):
    """认证失败异常"""
    pass


def retry_on_api_error(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    multiplier: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
        LLMError,
    ),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    LLM API 调用重试装饰器

    使用指数退避策略，自动处理临时性错误。

    Args:
        max_attempts: 最大尝试次数
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        multiplier: 延迟倍增因子
        retryable_exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数 (exception, attempt_number)

    Returns:
        装饰后的函数

    Examples:
        @retry_on_api_error(max_attempts=3)
        def call_llm(messages):
            return llm_client.respond(messages)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except AuthenticationError:
                    # 认证错误不重试，直接抛出
                    raise

                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        break

                    # 计算延迟时间
                    if multiplier > 0:
                        delay = min(initial_delay * (multiplier ** (attempt - 1)), max_delay)
                        # 添加随机抖动（避免惊群效应）
                        import random
                        jitter = delay * 0.1 * random.random()
                        delay = delay + jitter
                    else:
                        delay = initial_delay

                    # 调用回调
                    if on_retry:
                        on_retry(e, attempt)

                    # 等待后重试
                    time.sleep(delay)

                except Exception as e:
                    # 非预期异常
                    last_exception = e
                    break

            # 所有重试都失败
            if last_exception:
                raise LLMError(
                    f"Failed after {max_attempts} attempts: {str(last_exception)}",
                    original_error=last_exception
                )

        return wrapper
    return decorator


def with_timeout(
    timeout: float,
    default_return: Any = None,
    on_timeout: Optional[Callable[[], Any]] = None,
):
    """
    超时控制装饰器

    使用 signal 实现 Unix 系统超时，Windows 下使用线程实现。

    Args:
        timeout: 超时时间（秒）
        default_return: 超时时返回的默认值
        on_timeout: 超时时执行的回调函数

    Returns:
        装饰后的函数

    Examples:
        @with_timeout(30, default_return="timeout")
        def execute_shell(command):
            return subprocess.run(command, shell=True)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import platform

            if platform.system() == "Windows":
                return _timeout_windows(func, args, kwargs, timeout, default_return, on_timeout)
            else:
                return _timeout_unix(func, args, kwargs, timeout, default_return, on_timeout)

        return wrapper
    return decorator


def _timeout_unix(
    func: Callable[..., T],
    args: tuple,
    kwargs: dict,
    timeout: float,
    default_return: Any,
    on_timeout: Optional[Callable[[], Any]],
) -> T:
    """Unix 系统超时实现（使用 signal）"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout}s")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout) + 1)  # 多给 1 秒

    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        if on_timeout:
            return on_timeout()
        return default_return
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return result


def _timeout_windows(
    func: Callable[..., T],
    args: tuple,
    kwargs: dict,
    timeout: float,
    default_return: Any,
    on_timeout: Optional[Callable[[], Any]],
) -> T:
    """Windows 超时实现（使用线程）"""
    import threading

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
        if on_timeout:
            return on_timeout()
        return default_return

    if exception[0]:
        raise exception[0]

    return result[0]


class RetryContext:
    """
    可管理的重试上下文

    用于需要精细控制重试逻辑的场景。
    """

    def __init__(
        self,
        name: str,
        config: Optional[RetryConfig] = None,
        logger: Optional[Any] = None,
    ):
        self.name = name
        self.config = config or RetryConfig()
        self.logger = logger
        self.attempt = 0
        self.total_delay = 0.0

    def calculate_delay(self) -> float:
        """计算下次重试的延迟"""
        if self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (self.config.multiplier ** self.attempt)
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.initial_delay * (self.attempt + 1)
        else:
            delay = self.config.initial_delay

        return min(delay, self.config.max_delay)

    def should_retry(self, exception: Exception) -> bool:
        """判断是否应该重试"""
        self.attempt += 1

        if self.attempt >= self.config.max_attempts:
            return False

        return isinstance(exception, self.config.retryable_exceptions)

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        执行函数，自动重试

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            LLMError: 所有重试都失败后抛出
        """
        last_error = None

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e

                if not self.should_retry(e):
                    raise LLMError(
                        f"[{self.name}] Failed after {self.attempt} attempts: {e}",
                        original_error=e
                    )

                delay = self.calculate_delay()
                self.total_delay += delay

                if self.logger:
                    self.logger.warning(
                        f"[{self.name}] Attempt {self.attempt} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                time.sleep(delay)

    def get_stats(self) -> dict:
        """获取重试统计"""
        return {
            "name": self.name,
            "total_attempts": self.attempt,
            "total_delay": self.total_delay,
        }
