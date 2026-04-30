"""
基础 RAG 技能类。
集成工业级检索链路：上下文增强分块、BM25+Dense 混合检索、RRF 融合及 Cross-Encoder 精排。
"""

import os
import sys
import threading
import logging
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from ...tools.base import Tool
from ..base import BaseSkill, SkillMetadata
from ...rag.embeddings import (
    create_embeddings,
    resolve_embedding_api_key,
    resolve_embedding_base_url,
    resolve_embedding_model,
    resolve_embedding_provider,
)
from ...rag.vector_store import FAISSVectorStore
from ...rag.document_loader import AdvancedDocumentLoader
from ...rag.observability import TraceManager
# 导入重构后的高级检索组件
from ...rag.retriever import Retriever, HybridRetriever, get_shared_reranker

logger = logging.getLogger(__name__)


class BaseRAGSkill(BaseSkill):
    """
    RAG 知识库基础抽象类。
    支持通过 config 字典动态初始化元数据，并实现增量文档加载逻辑。
    """

    # 类变量：确保全局所有 RAG 技能共享同一个重排模型

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._tracer = None
        # 1. 初始化元数据
        self._metadata = SkillMetadata(
            name=config.get("skill_id", "rag_skill"),
            display_name=config.get("display_name", "RAG知识库"),
            description=config.get("description", "基于向量检索的专业知识库"),
            keywords=config.get("keywords", []),
            patterns=config.get("patterns", []),
            priority=config.get("priority", 10),
            version=config.get("version", "1.0.0"),
        )

        # 2. RAG 参数配置
        self.skill_id = config.get("skill_id", "rag_skill")
        base_path = Path(__file__).parent.parent.parent.parent
        self.index_path = str(base_path / "dm_agent" / "data" / "indices" / config.get("index_subdir", "default_idx"))
        self.builtin_dir = base_path / "dm_agent" / "data" / "knowledge_base" / config.get("data_subdir",
                                                                                           "default_docs")

        # 指纹库文件路径与统计文件路径
        self.manifest_path = Path(self.index_path) / "manifest.json"
        self.stats_path = Path(self.index_path) / "index_stats.json"

        self.top_k = config.get("top_k", 3)
        self.threshold = config.get("threshold", 0.5)
        self.use_hybrid = config.get("use_hybrid", True)
        self.use_reranker = config.get("use_reranker", True)
        self.reranker_model = config.get("reranker_model", "BAAI/bge-reranker-base")
        self.embedding_provider = resolve_embedding_provider()
        self.embedding_api_key = resolve_embedding_api_key()
        self.embedding_base_url = resolve_embedding_base_url()
        self.embedding_model = resolve_embedding_model()
        if self.use_reranker and os.getenv("RAG_PRELOAD_RERANKER", "true").lower() == "true":
            get_shared_reranker(self.reranker_model, preload=True)

        # 3. 内部状态与组件
        self._embeddings = None
        self._vector_store = None
        self._retriever = None
        self._document_loader = AdvancedDocumentLoader()
        self._initialized = False
        self._init_error = None
        self._lock = threading.Lock()

    def get_metadata(self) -> SkillMetadata:
        return self._metadata

    def get_prompt_addition(self) -> str:
        if not self._ensure_initialized():
            return f"\n⚠️ 注意：{self._metadata.display_name} 暂时不可用 ({self._init_error})。\n"

        return f"""
## 专家领域：{self._metadata.display_name}
你可以通过以下工具检索该领域的专业论文、公式和代码实现：
1. `{self.skill_id}_search`: 检索相关专业资料。
2. 如果只是检索和解释专业知识或者概念，就不需要在通过default tools中的工具在进行搜索，我们只听过{self.skill_id}_search来进行检索和回答。并且如没有在专业知识库中寻找到相关知识，直接说不知道即可。
3. 如果有写代码或者其他代码需求才通过default tools中的工具进行检索。
"""

    def get_tools(self) -> List[Tool]:
        def search_func(arguments: Dict[str, Any]) -> str:
            # --- 1. 初始化 Trace (增加安全保护) ---
            # 如果外部注入了 tracer 就用外部的，否则才新建（兼容性处理）
            if hasattr(self, "_tracer") and self._tracer is not None:
                tracer = self._tracer
                print("🔗 使用注入的全局 Tracer 记录检索过程")
            else:
                tracer = TraceManager()
                tracer.start_trace("Query")
            query = arguments.get("query", "")

            # 预设变量，确保在异常分支中也能被访问
            node_recall = None
            node_rerank = None

            try:
                # 2. 检查初始化
                if not self._ensure_initialized():
                    error_info = f"错误: {self._init_error}"
                    tracer.add_node("Init_Check", "Error", output_val=error_info)
                    return error_info

                # 3. 追踪：召回阶段 (Recall)
                node_recall = tracer.add_node("Hybrid_Recall", "BM25 + FAISS", input_val=query)

                # 一次调用获取足够结果
                k_request = arguments.get("k", self.top_k)
                all_results = self._retriever.retrieve(query, k=k_request * 3)
                raw_docs = all_results[:k_request * 2]  # 前2k作为原始召回
                results = all_results[:k_request]  # 前k作为精排结果

                # 【修复点】：增加对 raw_docs 的判空保护，防止列表推导式报错
                found_count = len(all_results) if all_results else 0

                recall_results = []
                if raw_docs:
                    # 使用 getattr 防御对象属性不存在的情况
                    recall_results = [{"doc": getattr(d, 'content', str(d))[:50], "score": getattr(d, 'score', 0)} for d
                                      in raw_docs]

                tracer.end_node(node_recall, output_val=recall_results)

                # 4. 追踪：重排阶段 (Rerank)
                node_rerank = tracer.add_node("Rerank", self.reranker_model, input_val=found_count)

                # results 已在上面一次获取，无需再次检索
                # --- [检查点：确保 self._tracer 存在且已注入] ---
                if hasattr(self, "_tracer") and self._tracer is not None:
                    # 为 Ragas 准备上下文列表，存入全局 Trace 的 metadata
                    self._tracer.metadata["retrieved_contexts"] = [res.content for res in results]

                if self._tracer:
                    # [核心] 为 Ragas 准备上下文列表
                    self._tracer.metadata["retrieved_contexts"] = [res.content for res in results]

                # 【修复点】：确保 results 不为 None 再进行迭代
                res_list = results if results else []
                res_scores = [{"score": getattr(res, 'score', 0)} for res in res_list]
                tracer.end_node(node_rerank, output_val=res_scores)

                if not res_list:
                    return "未找到相关专业背景资料。"

                # 5. 格式化输出
                output = [f"从《{self._metadata.display_name}》中检索到以下内容："]
                for i, res in enumerate(res_list, 1):
                    file_name = res.metadata.get("file_name", "未知文件")
                    # 限制长度并保留核心逻辑
                    output.append(f"\n[{i}] 来源: {file_name} (相关度: {res.score:.4f})\n{res.content[:1500]}...")

                output_text = "\n".join(output)
                if tracer:
                    tracer.metadata.setdefault("rag_eval_samples", []).append({
                        "question": query,
                        "contexts": [res.content for res in res_list],
                        "answer": output_text,
                        "eval_scope": "rag_query",
                        "source": f"{self.skill_id}_search",
                        "context_scores": [float(getattr(res, "score", 0) or 0) for res in res_list],
                        "context_sources": [
                            res.metadata.get("file_name") or res.metadata.get("source") or "unknown"
                            for res in res_list
                        ],
                    })

                return output_text

            except Exception as e:
                # 捕获运行时的意外错误并记录
                error_msg = str(e)
                print(f"❌ [DEBUG] 检索过程发生异常: {error_msg}")

                if tracer:
                    # 如果在 recall 阶段崩溃，手动关闭 node_recall 防止 null
                    if node_recall and node_recall.end_time == 0:
                        tracer.end_node(node_recall, output_val=f"FAILED: {error_msg}")
                    tracer.add_node("Critical_Error", type(e).__name__, output_val=error_msg)

                import traceback
                traceback.print_exc()
                return f"检索执行异常: {error_msg}"



            finally:

                # 1. 无论成功失败，只要有 output，就塞进 metadata

                if tracer and 'output' in locals():
                    tracer.metadata["retrieved_context"] = "\n".join(output)

                    tracer.metadata["config_top_k"] = self.top_k

                    tracer.metadata["config_threshold"] = self.threshold

                # 2. 【核心逻辑】决定谁来执行最后的“存盘”动作

                if not hasattr(self, "_tracer") or self._tracer is None:

                    # 只有当 tracer 是工具自己临时创建的时，才在这里保存

                    if tracer:
                        save_path = tracer.finish_and_save()

                        print(f"📡 [DEBUG] 临时 Trace 保存成功 -> {save_path}")

                else:

                    # 如果是全局 Tracer，千万不要在这里 finish_and_save！！

                    # 我们只需要静静地把数据填进 tracer.metadata，

                    # 保存动作由 main.py 在拿到 LLM 答案后统一执行。

                    print(f"🔗 检索数据已挂载到全局 Tracer (等待 Agent 回答后保存)")

        return [
            Tool(
                name=f"{self.skill_id}_search",
                description=f"在{self._metadata.display_name}知识库中检索。参数: {{\"query\": \"关键词\"}}",
                runner=search_func
            )
        ]

    def _get_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"计算哈希失败 {file_path}: {e}")
            return ""

    def _ensure_initialized(self) -> bool:
        with self._lock:
            if self._initialized:
                return True

            print(f"📥 [{self._metadata.display_name}] 正在初始化 RAG 组件...")
            try:
                os.makedirs(self.index_path, exist_ok=True)
                self._embeddings = create_embeddings(
                    provider=self.embedding_provider,
                    model_name=self.embedding_model,
                    api_key=self.embedding_api_key,
                    base_url=self.embedding_base_url,
                )
                self._vector_store = FAISSVectorStore(
                    embeddings=self._embeddings,
                    index_path=self.index_path,
                )

                reranker = None
                if self.use_reranker:
                    print(f"🔄 后台预热共享 Reranker 模型: {self.reranker_model}")
                    reranker = get_shared_reranker(self.reranker_model, preload=True)

                if self.use_hybrid:
                    self._retriever = HybridRetriever(
                        self._vector_store,
                        reranker=reranker,
                        top_k=self.top_k,
                        threshold=self.threshold
                    )
                else:
                    self._retriever = Retriever(self._vector_store, top_k=self.top_k, threshold=self.threshold)

                self._sync_knowledge_base()

                if self.use_hybrid and hasattr(self._retriever, "_build_bm25_index"):
                    self._retriever._build_bm25_index()

                self._initialized = True
                print(f"✅ [{self._metadata.display_name}] 初始化成功！")
                return True
            except Exception as e:
                self._init_error = str(e)
                print(f"❌ [{self._metadata.display_name}] 初始化失败: {e}")
                return False

    def _sync_knowledge_base(self):
        """同步本地文件夹与向量数据库（增量更新），并记录完整的 Ingestion Trace。"""
        if not self.builtin_dir.exists():
            print(f"⚠️ 知识库原始目录不存在: {self.builtin_dir}")
            return

        # --- [新增] 初始化 Ingestion 追踪 ---
        tracer = TraceManager()
        tracer.start_trace("Ingestion")
        # 记录同步任务开始
        sync_node = tracer.add_node("Sync_Knowledge_Base", "Local_FS", input_val=str(self.builtin_dir))

        old_manifest = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    old_manifest = json.load(f)
            except Exception as e:
                print(f"⚠️ 读取指纹库失败: {e}")

        current_manifest = {}
        files_to_add = []
        all_files = list(self.builtin_dir.rglob("*"))
        for file_path in all_files:
            if file_path.is_file() and file_path.suffix.lower() in self._document_loader.SUPPORTED_EXTENSIONS:
                rel_path = str(file_path.relative_to(self.builtin_dir))
                file_hash = self._get_file_hash(file_path)
                current_manifest[rel_path] = file_hash
                if rel_path not in old_manifest or old_manifest[rel_path] != file_hash:
                    files_to_add.append(file_path)

        files_to_remove = [p for p in old_manifest if p not in current_manifest]

        # 如果没有变动
        if not files_to_add and not files_to_remove:
            tracer.end_node(sync_node, output_val="Already up-to-date")
            tracer.finish_and_save() # 记录一次“检查”动作
            self._save_index_stats()
            print(f"✨ [{self._metadata.display_name}] 已经是最新。")
            return

        # --- [新增] 追踪：删除过期文档 ---
        if files_to_remove:
            del_node = tracer.add_node("Delete_Stale_Docs", "FAISS", input_val=files_to_remove)
            for rel_path in files_to_remove:
                self._vector_store.delete_by_metadata({"file_path": rel_path})
            tracer.end_node(del_node, output_val=f"Removed {len(files_to_remove)} docs")

        # --- [新增] 追踪：增量加载与切分 ---
        if files_to_add:
            load_node = tracer.add_node("Document_Processing", "AdvancedLoader", input_val=[f.name for f in files_to_add])
            total_new_chunks = 0
            for file_path in files_to_add:
                try:
                    doc = self._document_loader.load_file(str(file_path))
                    chunks = self._document_loader.chunk_document(doc)
                    if chunks:
                        self._vector_store.add_chunks(chunks)
                        total_new_chunks += len(chunks)
                except Exception as e:
                    print(f"  ❌ 处理 {file_path.name} 失败: {e}")
            tracer.end_node(load_node, output_val={"added_files": len(files_to_add), "new_chunks": total_new_chunks})

        # --- [新增] 追踪：索引持久化 ---
        save_node = tracer.add_node("Index_Persistence", "FAISS_Disk", input_val=self.index_path)
        self._vector_store.save()
        tracer.end_node(save_node, output_val="Success")

        # 保存指纹库
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(current_manifest, f, indent=2, ensure_ascii=False)

        # 结束总追踪并保存
        tracer.end_node(sync_node, output_val="Sync Completed")
        tracer.finish_and_save()

        # 更新统计信息
        self._save_index_stats()

    def _save_index_stats(self):
        """将向量库真实分块数保存至磁盘，供 Dashboard 读取。"""
        try:
            # 从 VectorStore 获取真实的向量总数 (ntotal)
            stats = self._vector_store.get_stats()
            total_chunks = stats.get("total_chunks", 0)
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump({"total_chunks": total_chunks, "update_time": time.time()}, f)
            print(f"📚 统计信息已更新。当前总分块数: {total_chunks}")
        except Exception as e:
            logger.error(f"保存统计信息失败: {e}")

    def on_activate(self) -> None:
        self._ensure_initialized()
