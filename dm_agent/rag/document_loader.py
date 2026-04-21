"""高级文档加载与分块模块，支持 PDF 解析、语法感知与上下文增强。"""

import os
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import fitz  # PyMuPDF

from .models import Document, DocumentChunk


class AdvancedDocumentLoader:
    """专业级文档加载器，处理科研 PDF 与代码库。

    自动识别文件类型，应用最优分块策略，并执行上下文增强（Context Injection），
    将文档元数据注入每个块的头部，提升孤立文本块的检索准确度。

    Args:
        chunk_size (int): 文本块的最大字符长度。
        chunk_overlap (int): 文本块之间的重叠字符数。

    Returns:
        AdvancedDocumentLoader: 加载器实例。

    Examples:
        >>> config = {"chunk_size": 1500, "chunk_overlap": 200}
        >>> loader = AdvancedDocumentLoader(**config)
        >>> chunks = loader.load_and_chunk("./src/hybrid_beamforming.py")
        >>> len(chunks) > 0
        True
    """

    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.py', '.json', '.csv', '.pdf'}

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.code_splitters = {
            ".py": RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        }
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def load_file(self, file_path: str) -> Document:
        """智能加载单个文件，处理不同格式。"""
        print(f"[LOADER] {time.time():.3f} 开始加载文件: {file_path}")
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        content = ""

        if ext == ".pdf":
            with fitz.open(path) as doc_pdf:
                text_list = []
                for i, page in enumerate(doc_pdf):
                    text_list.append(f"--- Page {i + 1} ---\n{page.get_text('text')}")
                content = "\n".join(text_list)
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        doc_id = hashlib.md5(f"{path.name}{path.stat().st_size}".encode()).hexdigest()[:12]
        doc = Document(
            id=doc_id,
            content=content,
            metadata={
                "file_name": path.name,
                "file_path": str(path),
                "file_type": ext[1:],
                "file_size": path.stat().st_size,
            }
        )
        return doc

    def _inject_context(self, chunk_text: str, doc: Document) -> str:
        """上下文增强：将全局元数据拼接到切片头部。"""
        file_name = doc.metadata.get("file_name", "Unknown")
        file_type = doc.metadata.get("file_type", "Unknown")

        # 针对科研与代码任务构建强语义前缀
        prefix = f"[Metadata | File: {file_name} | Type: {file_type}]\n"
        return prefix + chunk_text

    def chunk_document(self, doc: Document) -> List[DocumentChunk]:
        """对文档进行智能分块及上下文增强。"""
        ext = f".{doc.metadata.get('file_type', 'txt')}"

        if ext in self.code_splitters:
            chunks_text = self.code_splitters[ext].split_text(doc.content)
        else:
            chunks_text = self.default_splitter.split_text(doc.content)

        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            # 执行上下文增强
            enhanced_content = self._inject_context(chunk_text.strip(), doc)

            chunks.append(
                DocumentChunk(
                    id=f"{doc.id}_{i}",
                    document_id=doc.id,
                    content=enhanced_content,
                    chunk_index=i,
                    metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(chunks_text)}
                )
            )
        return chunks

    def load_and_chunk(self, source: str, recursive: bool = True) -> List[DocumentChunk]:
        """加载目录或文件并执行完整分块流程。"""
        path = Path(source)
        docs = []

        if path.is_file():
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                docs = [self.load_file(str(path))]
            else:
                print(f"⚠️ 跳过不支持的文件格式: {path.name}")
        elif path.is_dir():
            print(f"📂 正在扫描目录: {path.absolute()}")
            all_files = list(path.rglob("*")) if recursive else list(path.glob("*"))

            for file_path in all_files:
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    try:
                        rel_path = file_path.relative_to(path)
                        print(f"  📄 发现匹配文件: {rel_path}")
                        docs.append(self.load_file(str(file_path)))
                    except Exception as e:
                        print(f"❌ 加载 {file_path} 失败: {e}")
        else:
            raise ValueError(f"无效路径: {source}")

        all_chunks = []
        for doc in docs:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            print(f"✅ 文件 [{doc.metadata['file_name']}] 已切分为 {len(doc_chunks)} 块")

        print(f"\n✨ 扫描结束：总共加载 {len(docs)} 个文档，生成 {len(all_chunks)} 个知识切片。")
        return all_chunks