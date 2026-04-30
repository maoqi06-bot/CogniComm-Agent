"""嵌入模型封装模块。"""

import os
import numpy as np
from typing import List, Union, Optional
from abc import ABC, abstractmethod


class BaseEmbeddings(ABC):
    """嵌入模型抽象基类。"""

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI 嵌入模型实现，支持自定义端点。

    Args:
        model (str): 模型名称，默认 text-embedding-ada-002。
        api_key (Optional[str]): API 密钥。若不提供则读取环境变量。
        base_url (Optional[str]): 自定义 API 网关地址。

    Returns:
        OpenAIEmbeddings: 初始化的嵌入模型实例。

    Examples:
        >>> config = {
        ...     "model": "text-embedding-ada-002",
        ...     "base_url": "https://sg.uiuiapi.com/v1/"
        ... }
        >>> embedder = OpenAIEmbeddings(**config)
        >>> vector = embedder.embed_query("SCA for non-convex trajectory optimization")
        >>> print(len(vector))
        1536
    """

    def __init__(
            self,
            model: str = "text-embedding-ada-002",
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 EMBEDDING_API_KEY 或 OPENAI_API_KEY")

        self.base_url = (
            base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "https://sg.uiuiapi.com/v1/")
        )
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
        self._dimension = _infer_embedding_dimension(model)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """批量将文本转换为向量。

        Args:
            texts (Union[str, List[str]]): 单个文本或文本列表。

        Returns:
            np.ndarray: 形状为 (N, dimension) 的向量矩阵。

        Examples:
            >>> texts = ["LSTM state prediction", "AltMin beamforming"]
            >>> vectors = embedder.embed(texts)
            >>> vectors.shape[0] == 2
            True
        """
        if isinstance(texts, str):
            texts = [texts]
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except Exception as e:
            raise RuntimeError(f"OpenAI API 调用失败: {e}")

    def embed_query(self, text: str) -> np.ndarray:
        """为单个查询文本生成向量。"""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


def _infer_embedding_dimension(model_name: str) -> int:
    normalized = (model_name or "").strip().lower()
    if normalized == "text-embedding-3-large":
        return 3072
    if normalized in {"text-embedding-3-small", "text-embedding-ada-002"}:
        return 1536

    env_dimension = os.getenv("EMBEDDING_DIMENSION") or os.getenv("OPENAI_EMBEDDING_DIMENSION")
    if env_dimension:
        try:
            return int(env_dimension)
        except ValueError:
            pass
    return 1536


def resolve_embedding_provider(default: str = "openai") -> str:
    return (os.getenv("EMBEDDING_PROVIDER") or default).strip().lower()


def resolve_embedding_model(default: str = "text-embedding-3-small") -> str:
    return (
        os.getenv("EMBEDDING_MODEL")
        or default
    ).strip()


def resolve_embedding_api_key() -> Optional[str]:
    return os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")


def resolve_embedding_base_url(default: str = "https://sg.uiuiapi.com/v1/") -> str:
    return (
        os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or default
    ).strip()


def create_embeddings(
        provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
) -> BaseEmbeddings:
    """嵌入模型工厂函数。"""
    provider = provider.lower()
    if provider == "openai":
        model_name = model_name or resolve_embedding_model()
        api_key = api_key or resolve_embedding_api_key()
        base_url = base_url or resolve_embedding_base_url()
        return OpenAIEmbeddings(model_name, api_key, base_url)
    else:
        raise ValueError(f"不支持的嵌入提供商: {provider}")
