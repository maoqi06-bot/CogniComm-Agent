"""数据模型定义。"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import hashlib


@dataclass
class Document:
    """文档数据模型。

    用于存储从学术论文（PDF）、代码库（.py）或手册中提取的完整内容及元数据。

    Args:
        id (str): 文档唯一标识，若未提供则根据内容生成 MD5 哈希。
        content (str): 文档的完整文本内容。
        metadata (Dict[str, Any]): 元数据字典，包含文件路径、类型、加载时间等。
        embedding (Optional[List[float]]): 可选，文档的全局向量表示。

    Returns:
        Document: 实例化的文档对象。

    Examples:
        >>> data = {
        ...     "id": "twc_draft_001",
        ...     "content": "This paper investigates ISAC in UAV networks...",
        ...     "metadata": {"file_name": "main.tex", "file_type": "tex"}
        ... }
        >>> doc = Document(**data)
        >>> print(doc.id)
        twc_draft_001
    """
    id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        """初始化后处理，如果没有提供 id，则根据内容生成 MD5 哈希作为 id。"""
        if not self.id:
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class DocumentChunk:
    """文档块数据模型。

    大文档被分割后的子块，是向量检索和混合检索的最小单元。

    Args:
        id (str): 块唯一标识（文档ID_索引）。
        document_id (str): 所属源文档的 ID。
        content (str): 切分后的文本块内容。
        chunk_index (int): 本块在原文档中的顺序索引。
        metadata (Dict[str, Any]): 元数据字典，继承自父文档。
        embedding (Optional[List[float]]): 本文本块的向量表示。

    Returns:
        DocumentChunk: 实例化的文档块对象。

    Examples:
        >>> data = {
        ...     "id": "twc_draft_001_0",
        ...     "document_id": "twc_draft_001",
        ...     "content": "The objective function for trajectory optimization is...",
        ...     "chunk_index": 0
        ... }
        >>> chunk = DocumentChunk(**data)
        >>> print(chunk.full_id)
        twc_draft_001_0
    """
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def full_id(self) -> str:
        """获取完整块标识。"""
        return f"{self.document_id}_{self.chunk_index}"


@dataclass
class SearchResult:
    """搜索结果数据模型。

    封装从检索器（如混合检索器）返回的单一匹配结果及其得分。

    Args:
        chunk (DocumentChunk): 匹配命中的底层文档块实例。
        score (float): 相似度或混合召回的归一化得分。
        content (str): 块内容的快捷访问属性。
        metadata (Dict[str, Any]): 元数据的快捷访问属性。

    Returns:
        SearchResult: 检索结果对象。

    Examples:
        >>> chunk_data = {"id": "cvxpy_01", "document_id": "doc1", "content": "cp.Minimize()", "chunk_index": 1}
        >>> result = SearchResult(chunk=DocumentChunk(**chunk_data), score=0.95, content="cp.Minimize()", metadata={})
        >>> print(result.score)
        0.95
    """
    chunk: DocumentChunk
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)