"""LLM 客户端基类定义。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field

from ..utils.logger import get_logger, AgentLogger
from ..utils.retry import (
    LLMError as AgentLLMError,
    RetryConfig,
    RetryStrategy,
)

# 为了向后兼容，重新导出
LLMError = AgentLLMError


@dataclass
class LLMCallMetrics:
    """LLM 调用指标"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_duration: float = 0.0
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_duration(self) -> float:
        if self.successful_calls == 0:
            return 0.0
        return self.total_duration / self.successful_calls


class BaseLLMClient(ABC):
    """LLM 客户端的抽象基类，带重试和日志支持。"""

    # 可重试的异常类型
    RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        logger: Optional[AgentLogger] = None,
    ) -> None:
        if not api_key:
            raise ValueError("LLM 客户端需要 API 密钥。")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self._logger = logger or get_logger(f"llm.{self.__class__.__name__}")

        # 指标统计
        self.metrics = LLMCallMetrics()

    def _calculate_retry_delay(self, attempt: int) -> float:
        """计算重试延迟（指数退避 + 抖动）"""
        delay = min(
            self.retry_delay * (2 ** (attempt - 1)),
            self.max_retry_delay
        )
        # 添加 10% 随机抖动
        import random
        jitter = delay * 0.1 * random.random()
        return delay + jitter

    def _should_retry(self, exception: Exception) -> bool:
        """判断异常是否应该重试"""
        # 认证错误不重试
        if "auth" in str(exception).lower() or "401" in str(exception):
            return False
        if "invalid" in str(exception).lower() and "token" in str(exception).lower():
            return False

        # 检查是否是可重试异常
        return isinstance(exception, self.RETRYABLE_EXCEPTIONS)

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        **extra: Any,
    ) -> Dict[str, Any]:
        """发送聊天式补全请求到 LLM API。

        Args:
            messages: 消息列表，每个消息包含 role 和 content
            **extra: 额外的参数（如 temperature, max_tokens 等）

        Returns:
            API 响应的字典
        """
        pass

    @abstractmethod
    def extract_text(self, data: Dict[str, Any]) -> str:
        """从 API 响应中提取文本内容。

        Args:
            data: API 响应的字典

        Returns:
            提取的文本内容
        """
        pass

    def respond(self, messages: List[Dict[str, str]], **extra: Any) -> str:
        """返回补全响应的文本部分，带重试机制。

        Args:
            messages: 消息列表
            **extra: 额外的参数

        Returns:
            提取的文本响应

        Raises:
            LLMError: 所有重试都失败后抛出
        """
        last_error = None
        start_time = time.time()

        for attempt in range(1, self.max_retries + 1):
            try:
                self._logger.debug(
                    f"LLM API call attempt {attempt}/{self.max_retries}",
                    extra={"attempt": attempt, "model": self.model}
                )

                data = self.complete(messages, **extra)
                self.metrics.total_calls += 1
                self.metrics.successful_calls += 1

                # 记录耗时
                duration = time.time() - start_time
                self.metrics.total_duration += duration

                # 尝试提取 token 使用量
                if hasattr(data, 'usage') or (isinstance(data, dict) and 'usage' in data):
                    usage = data.get('usage', {}) or {}
                    tokens = usage.get('total_tokens', 0)
                    self.metrics.total_tokens += tokens

                    self._logger.log_llm_call(
                        model=self.model,
                        tokens_used=tokens,
                        duration=duration,
                        attempt=attempt
                    )
                else:
                    self._logger.log_llm_call(
                        model=self.model,
                        tokens_used=0,
                        duration=duration,
                        attempt=attempt
                    )

                return self.extract_text(data)

            except Exception as e:
                last_error = e
                self.metrics.failed_calls += 1
                self.metrics.last_error = str(e)

                if attempt < self.max_retries and self._should_retry(e):
                    delay = self._calculate_retry_delay(attempt)
                    self._logger.warning(
                        f"LLM API call failed: {e}. Retrying in {delay:.1f}s...",
                        extra={"attempt": attempt, "error": str(e)}
                    )
                    time.sleep(delay)
                else:
                    self._logger.error(
                        f"LLM API call failed after {attempt} attempts: {e}",
                        exc_info=True
                    )
                    break

        # 所有重试都失败
        raise AgentLLMError(
            f"LLM call failed after {self.max_retries} attempts: {last_error}",
            original_error=last_error
        )

    def get_metrics(self) -> Dict[str, Any]:
        """获取 LLM 调用指标"""
        return {
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "success_rate": self.metrics.success_rate,
            "total_tokens": self.metrics.total_tokens,
            "total_duration": self.metrics.total_duration,
            "avg_duration": self.metrics.avg_duration,
            "last_error": self.metrics.last_error,
        }

    def reset_metrics(self):
        """重置指标"""
        self.metrics = LLMCallMetrics()