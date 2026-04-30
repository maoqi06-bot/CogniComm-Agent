# dm_agent/mcp/rag_mcp_server.py
"""
RAG MCP 服务器 - 多领域支持版本（增强路径灵活性）
支持将检索结果追加到主程序 trace 文件
"""

import os
import sys
import json
import asyncio
import logging
import threading
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, NamedTuple

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dm_agent.rag.embeddings import (
    create_embeddings,
    resolve_embedding_api_key,
    resolve_embedding_base_url,
    resolve_embedding_model,
    resolve_embedding_provider,
)
from dm_agent.rag.vector_store import FAISSVectorStore
from dm_agent.rag.document_loader import AdvancedDocumentLoader
from dm_agent.rag.observability import TraceManager, append_trace_payload
from dm_agent.rag.retriever import Retriever, HybridRetriever, get_shared_reranker

logger = logging.getLogger(__name__)


def _safe_print(message: str) -> None:
    """Print diagnostic text without crashing legacy Windows consoles."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe + "\n")
        sys.stdout.flush()


BASE_PATH = Path(__file__).parent.parent.parent
TRACE_DIR = BASE_PATH / "data" / "traces"   # 新增

# ---------- 默认配置 ----------
DEFAULT_CONFIG = {
    "skill_id": os.getenv("RAG_DEFAULT_SKILL_ID", "default"),
    "display_name": os.getenv("RAG_DEFAULT_DISPLAY_NAME", "通用知识库"),
    "index_subdir": os.getenv("RAG_DEFAULT_INDEX_SUBDIR", "default_idx"),
    "data_subdir": os.getenv("RAG_DEFAULT_DATA_SUBDIR", "default_docs"),
    "top_k": int(os.getenv("RAG_TOP_K", "3")),
    "threshold": float(os.getenv("RAG_THRESHOLD", "0.5")),
    "use_hybrid": os.getenv("RAG_USE_HYBRID", "true").lower() == "true",
    "use_reranker": os.getenv("RAG_USE_RERANKER", "true").lower() == "true",
    "reranker_model": os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base"),
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
}

if DEFAULT_CONFIG["use_reranker"] and os.getenv("RAG_PRELOAD_RERANKER", "true").lower() == "true":
    get_shared_reranker(DEFAULT_CONFIG["reranker_model"], preload=True)

BASE_PATH = Path(__file__).parent.parent.parent
KNOWLEDGE_BASE_ROOT = BASE_PATH / "dm_agent" / "data" / "knowledge_base"
INDEX_ROOT = BASE_PATH / "dm_agent" / "data" / "indices"

# ---------- 全局组件 ----------
class RAGInstance(NamedTuple):
    skill_id: str
    config: Dict[str, Any]
    embeddings: Any
    vector_store: Any
    retriever: Any
    document_loader: AdvancedDocumentLoader
    index_path: Path
    data_path: Path

_rag_instances: Dict[str, RAGInstance] = {}
_instances_lock = threading.Lock()
_document_loader = AdvancedDocumentLoader()



def _get_file_hash(file_path: Path) -> str:
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"计算哈希失败 {file_path}: {e}")
        return ""


def _resolve_data_path(config: Dict[str, Any]) -> Path:
    if "data_path" in config:
        data_path = Path(config["data_path"])
        if data_path.exists():
            return data_path
        else:
            logger.warning(f"指定的数据路径不存在: {data_path}，将尝试使用默认路径")

    data_subdir = config.get("data_subdir", "")
    if data_subdir:
        sub_path = KNOWLEDGE_BASE_ROOT / data_subdir
        if sub_path.exists():
            return sub_path
        else:
            logger.warning(f"数据子目录不存在: {sub_path}，将回退到根目录: {KNOWLEDGE_BASE_ROOT}")

    return KNOWLEDGE_BASE_ROOT


def _resolve_index_path(config: Dict[str, Any]) -> Path:
    if "index_path" in config:
        index_path = Path(config["index_path"])
        index_path.mkdir(parents=True, exist_ok=True)
        return index_path

    index_subdir = config.get("index_subdir", "")
    if index_subdir:
        idx_path = INDEX_ROOT / index_subdir
    else:
        idx_path = INDEX_ROOT / config.get("skill_id", "default")

    idx_path.mkdir(parents=True, exist_ok=True)
    return idx_path


def _sync_knowledge_base(instance: RAGInstance):
    data_path = instance.data_path
    if not data_path.exists():
        logger.warning(f"知识库原始目录不存在: {data_path}，跳过同步")
        return

    index_path = instance.index_path
    manifest_path = index_path / "manifest.json"
    stats_path = index_path / "index_stats.json"

    tracer = TraceManager()
    tracer.start_trace("Ingestion")
    sync_node = tracer.add_node("Sync_Knowledge_Base", "Local_FS", input_val=str(data_path))

    old_manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                old_manifest = json.load(f)
        except Exception as e:
            logger.warning(f"读取指纹库失败: {e}")

    current_manifest = {}
    files_to_add = []
    all_files = list(data_path.rglob("*"))
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in _document_loader.SUPPORTED_EXTENSIONS:
            rel_path = str(file_path.relative_to(data_path))
            file_hash = _get_file_hash(file_path)
            current_manifest[rel_path] = file_hash
            if rel_path not in old_manifest or old_manifest[rel_path] != file_hash:
                files_to_add.append(file_path)

    files_to_remove = [p for p in old_manifest if p not in current_manifest]

    if not files_to_add and not files_to_remove:
        tracer.end_node(sync_node, output_val="Already up-to-date")
        tracer.finish_and_save()
        _save_index_stats(instance, stats_path)
        logger.info(f"领域 {instance.skill_id} 知识库已是最新")
        return

    if files_to_remove:
        del_node = tracer.add_node("Delete_Stale_Docs", "FAISS", input_val=files_to_remove)
        for rel_path in files_to_remove:
            instance.vector_store.delete_by_metadata({"file_path": rel_path})
        tracer.end_node(del_node, output_val=f"Removed {len(files_to_remove)} docs")

    if files_to_add:
        load_node = tracer.add_node("Document_Processing", "AdvancedLoader", input_val=[f.name for f in files_to_add])
        total_new_chunks = 0
        for file_path in files_to_add:
            try:
                doc = _document_loader.load_file(str(file_path))
                chunks = _document_loader.chunk_document(doc)
                if chunks:
                    instance.vector_store.add_chunks(chunks)
                    total_new_chunks += len(chunks)
            except Exception as e:
                logger.error(f"处理 {file_path.name} 失败: {e}")
        tracer.end_node(load_node, output_val={"added_files": len(files_to_add), "new_chunks": total_new_chunks})

    save_node = tracer.add_node("Index_Persistence", "FAISS_Disk", input_val=str(index_path))
    instance.vector_store.save()
    tracer.end_node(save_node, output_val="Success")

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(current_manifest, f, indent=2, ensure_ascii=False)

    tracer.end_node(sync_node, output_val="Sync Completed")
    tracer.finish_and_save()

    _save_index_stats(instance, stats_path)


def _save_index_stats(instance: RAGInstance, stats_path: Path):
    try:
        stats = instance.vector_store.get_stats()
        total_chunks = stats.get("total_chunks", 0)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({"total_chunks": total_chunks, "update_time": time.time()}, f)
        logger.info(f"领域 {instance.skill_id} 统计信息已更新，总分块数: {total_chunks}")
    except Exception as e:
        logger.error(f"保存统计信息失败: {e}")


def _create_rag_instance(skill_id: str, config: Dict[str, Any]) -> Optional[RAGInstance]:
    try:
        data_path = _resolve_data_path(config)
        index_path = _resolve_index_path(config)

        if not data_path.exists():
            logger.error(f"数据目录不存在且无法回退: {data_path}")
            return None

        embeddings = create_embeddings(
            provider=config.get("embedding_provider", resolve_embedding_provider()),
            model_name=config.get("embedding_model", resolve_embedding_model()),
            api_key=config.get("embedding_api_key") or resolve_embedding_api_key(),
            base_url=config.get("embedding_base_url") or resolve_embedding_base_url(),
        )

        vector_store = FAISSVectorStore(
            embeddings=embeddings,
            index_path=str(index_path),
        )

        reranker = None
        if config.get("use_reranker", DEFAULT_CONFIG["use_reranker"]):
            model_name = config.get("reranker_model", DEFAULT_CONFIG["reranker_model"])
            logger.info(f"后台预热共享 Reranker 模型: {model_name}")
            reranker = get_shared_reranker(model_name, preload=True)

        top_k = config.get("top_k", DEFAULT_CONFIG["top_k"])
        threshold = config.get("threshold", DEFAULT_CONFIG["threshold"])
        use_hybrid = config.get("use_hybrid", DEFAULT_CONFIG["use_hybrid"])

        if use_hybrid:
            retriever = HybridRetriever(
                vector_store,
                reranker=reranker,
                top_k=top_k,
                threshold=threshold
            )
        else:
            retriever = Retriever(
                vector_store,
                top_k=top_k,
                threshold=threshold
            )

        instance = RAGInstance(
            skill_id=skill_id,
            config=config,
            embeddings=embeddings,
            vector_store=vector_store,
            retriever=retriever,
            document_loader=_document_loader,
            index_path=index_path,
            data_path=data_path
        )

        _sync_knowledge_base(instance)

        if use_hybrid and hasattr(retriever, "_build_bm25_index"):
            retriever._build_bm25_index()

        return instance
    except Exception as e:
        logger.error(f"创建 RAG 实例失败 {skill_id}: {e}", exc_info=True)
        return None


def _get_or_create_rag(skill_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[RAGInstance]:
    with _instances_lock:
        if skill_id in _rag_instances:
            return _rag_instances[skill_id]

        if config is None:
            config = {
                "skill_id": skill_id,
                "index_subdir": skill_id + "_idx",
                "data_subdir": skill_id + "_docs",
                **DEFAULT_CONFIG
            }

        instance = _create_rag_instance(skill_id, config)
        if instance:
            _rag_instances[skill_id] = instance
            logger.info(f"已创建 RAG 实例: {skill_id} (数据目录: {instance.data_path})")
        return instance


def _append_to_trace(trace_id: str, nodes: List[Dict], metadata: Dict[str, Any]):
    """将检索节点和元数据追加到主程序的 trace 文件中"""
    append_trace_payload(trace_id, nodes=nodes, metadata=metadata, log_dir=str(TRACE_DIR))


# ---------- MCP 服务器 ----------
app = Server("rag-mcp-server")


@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search",
            description="在指定知识库中检索相关专业资料。",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "知识库标识（如 wireless_comm, default）"},
                    "query": {"type": "string", "description": "检索关键词或问题"},
                    "k": {"type": "integer", "description": "返回结果数量（可选）"},
                    "trace_id": {"type": "string", "description": "可选的追踪ID，用于关联到主程序的trace"}
                },
                "required": ["skill_id", "query"]
            }
        ),
        types.Tool(
            name="initialize_expert_context",
            description="初始化一个专家领域的知识库（按需加载）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "知识库唯一标识"},
                    "config_json": {"type": "string", "description": "JSON 格式的配置字符串"}
                },
                "required": ["skill_id", "config_json"]
            }
        )
    ]


@app.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "initialize_expert_context":
        skill_id = arguments.get("skill_id")
        config_json = arguments.get("config_json")
        if not skill_id or not config_json:
            return [types.TextContent(type="text", text="错误：缺少 skill_id 或 config_json 参数")]

        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            return [types.TextContent(type="text", text="错误：config_json 不是有效的 JSON")]

        instance = _get_or_create_rag(skill_id, config)
        if instance:
            return [types.TextContent(type="text", text=f"成功初始化领域知识库: {skill_id}")]
        else:
            return [types.TextContent(type="text", text=f"初始化失败: {skill_id}")]

    elif name == "search":
        skill_id = arguments.get("skill_id")
        query = arguments.get("query", "").strip()
        trace_id = arguments.get("trace_id")
        logger.debug(f"search called: skill_id={skill_id}, trace_id={trace_id}")
        if not skill_id or not query:
            return [types.TextContent(type="text", text="错误：缺少 skill_id 或 query 参数")]

        k = arguments.get("k")
        instance = _get_or_create_rag(skill_id)
        if not instance:
            return [types.TextContent(type="text", text=f"错误：无法加载知识库 {skill_id}，请先初始化或检查配置")]

        if k is None:
            k = instance.config.get("top_k", DEFAULT_CONFIG["top_k"])
        else:
            try:
                k = int(k)
            except ValueError:
                k = DEFAULT_CONFIG["top_k"]

        try:
            use_hybrid = instance.config.get("use_hybrid", DEFAULT_CONFIG["use_hybrid"])
            # 一次调用获取足够结果（k*3 覆盖粗排召回需求）
            all_results = instance.retriever.retrieve(query, k=k * 3)
            results = all_results[:k]
            raw_docs = all_results[:k * 2] if use_hybrid else results

            if not results:
                return [types.TextContent(type="text", text="未找到相关专业背景资料。")]

            recall_results = []
            if raw_docs:
                recall_results = [
                    {"doc": getattr(d, 'content', str(d))[:50], "score": getattr(d, 'score', 0)}
                    for d in raw_docs
                ]

            res_scores = [{"score": getattr(res, 'score', 0)} for res in results]

            # 准备节点数据（用于追加或独立保存）
            recall_node_dict = {
                "name": "Hybrid_Recall",
                "type": "BM25 + FAISS",
                "input_val": query,
                "output_val": recall_results,
                "start_time": time.time(),
                "end_time": time.time(),
            }
            rerank_node_dict = {
                "name": "Rerank",
                "type": instance.config.get("reranker_model", DEFAULT_CONFIG["reranker_model"]),
                "input_val": len(raw_docs),
                "output_val": res_scores,
                "start_time": time.time(),
                "end_time": time.time(),
            }

            contexts = [res.content for res in results]
            context_scores = [float(getattr(res, "score", 0) or 0) for res in results]
            context_sources = [
                res.metadata.get("file_name") or res.metadata.get("source") or "unknown"
                for res in results
            ]
            metadata = {
                "retrieved_contexts": contexts,
                "config_top_k": k,
                "config_threshold": instance.config.get("threshold", DEFAULT_CONFIG["threshold"]),
            }

            output_lines = [f"从《{instance.config.get('display_name', skill_id)}》中检索到以下内容："]
            for i, res in enumerate(results, 1):
                file_name = res.metadata.get("file_name", "未知文件")
                output_lines.append(f"\n[{i}] 来源: {file_name} (相关度: {res.score:.4f})\n{res.content[:1500]}...")
            output_text = "\n".join(output_lines)
            metadata.setdefault("rag_eval_samples", []).append({
                "question": query,
                "contexts": contexts,
                "answer": output_text,
                "eval_scope": "rag_query",
                "source": f"{skill_id}_search",
                "context_scores": context_scores,
                "context_sources": context_sources,
            })

            if trace_id:
                # 追加到主程序 trace
                _safe_print(f"[MCP RAG] append trace: {trace_id}")
                _append_to_trace(trace_id, [recall_node_dict, rerank_node_dict], metadata)
            else:
                # 创建独立 trace（使用 TraceManager 公开 API）
                tracer = TraceManager()
                tracer.start_trace("Query")
                recall_node = tracer.add_node("Hybrid_Recall", "BM25 + FAISS", input_val=query)
                tracer.end_node(recall_node, output_val=recall_results)
                rerank_node = tracer.add_node("Rerank", instance.config.get("reranker_model", DEFAULT_CONFIG["reranker_model"]), input_val=len(raw_docs))
                tracer.end_node(rerank_node, output_val=res_scores)
                tracer.metadata.update(metadata)
                tracer.finish_and_save()

            return [types.TextContent(type="text", text=output_text)]

        except Exception as e:
            logger.exception(f"检索异常: {e}")
            error_msg = str(e)
            if trace_id:
                error_node = {
                    "name": "Critical_Error",
                    "type": type(e).__name__,
                    "output_val": error_msg,
                    "start_time": time.time(),
                    "end_time": time.time(),
                }
                _append_to_trace(trace_id, [error_node], {})
            return [types.TextContent(type="text", text=f"检索执行异常: {error_msg}")]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-mcp-server",
                server_version="2.2.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
