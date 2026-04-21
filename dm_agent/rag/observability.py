import time
import uuid
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class TraceNode:
    """记录 RAG 流程中的单个环节"""
    node_id: str
    method: str      # e.g., Hybrid_Recall, Rerank, Ingest_Split
    provider: str    # e.g., FAISS, BGE-M3, LangChain
    start_time: float
    end_time: float = 0.0
    input_data: Any = None
    output_data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

class TraceManager:
    """追踪管理器：持久化存储全链路 Trace 数据"""

    # 修改 TraceManager 的 __init__ 部分
    def __init__(self, log_dir: str = None, trace_id: str = None):
        self.trace_id = trace_id or f"trace_{int(time.time())}"
        self.current_trace_id = trace_id
        self.metadata = {}  # [新增] 用于存储全局信息，如检索上下文和评估分数
        self.nodes = [] # 新增
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            # 稳健方案：从当前文件向上找，直到发现 'dm_agent' 文件夹
            current_file = Path(__file__).resolve()
            root = current_file.parent
            # 最多向上找 5 层，防止死循环
            for _ in range(5):
                if (root / "dm_agent").exists():
                    break
                root = root.parent

            # 最终路径设定在项目根目录下的 data/traces
            self.log_dir = root / "data" / "traces"

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # 最后的兜底：如果根目录没法写，就写在当前运行目录
            self.log_dir = Path("./data/traces")
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_trace(self, trace_type: str = "Query"):
        """开启新追踪，区分 Query (检索) 和 Ingestion (摄取)"""
        self.current_trace_id = f"{trace_type.lower()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.nodes = []
        return self.current_trace_id

    def add_node(self, method: str, provider: str, input_val: Any = None) -> TraceNode:
        """记录步骤 开始"""
        node = TraceNode(
            node_id=uuid.uuid4().hex[:8],
            method=method,
            provider=provider,
            start_time=time.time(),
            input_data=input_val
        )
        self.nodes.append(node)
        return node

    def end_node(self, node: TraceNode, output_val: Any = None, metadata: Dict = None):
        """记录步骤结束并存储输出结果"""
        node.end_time = time.time()
        node.output_data = output_val
        if metadata:
            node.metadata.update(metadata)

    def finish_and_save(self) -> Path:
        """保存为 JSON；如果同名 trace 已存在，则合并而不是覆盖。

        MCP RAG 服务器和主 Agent 可能位于不同进程。MCP 会先把检索节点写入
        同一个 trace 文件，主进程稍后再写入 question/llm_answer。这里必须做
        merge，否则主进程的空 nodes 会覆盖 MCP 写入的检索链路。
        """
        if not self.current_trace_id: return None
        new_data = {
            "trace_id": self.current_trace_id,
            "trace_type": self.current_trace_id.split('_')[0],
            "total_duration": sum(n.duration for n in self.nodes if n.end_time > 0),
            "timestamp": time.time(),
            "nodes": [asdict(n) for n in self.nodes],
            "metadata": self.metadata  # [关键] 将体检报告存入 JSON 根部
        }
        file_path = self.log_dir / f"{self.current_trace_id}.json"
        existing = _read_trace_file(file_path)
        trace_data = _merge_trace_payload(existing, new_data)
        _atomic_write_json(file_path, trace_data)
        return file_path


def _normalize_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy MCP node fields to the dashboard/evaluator schema."""
    return {
        "node_id": node.get("node_id") or uuid.uuid4().hex[:8],
        "method": node.get("method") or node.get("name") or "Unknown",
        "provider": node.get("provider") or node.get("type") or "Unknown",
        "start_time": node.get("start_time") or time.time(),
        "end_time": node.get("end_time") or time.time(),
        "input_data": node.get("input_data", node.get("input_val")),
        "output_data": node.get("output_data", node.get("output_val")),
        "metadata": node.get("metadata") or {},
    }


def _read_trace_file(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_trace_payload(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    trace_id = incoming.get("trace_id") or existing.get("trace_id")
    merged_nodes = []
    seen = set()
    for node in existing.get("nodes", []) + incoming.get("nodes", []):
        normalized = _normalize_node(node)
        input_key = json.dumps(normalized.get("input_data"), ensure_ascii=False, sort_keys=True, default=str)
        key = (
            normalized.get("node_id"),
            normalized.get("method"),
            normalized.get("provider"),
            input_key,
        )
        if key in seen:
            continue
        seen.add(key)
        merged_nodes.append(normalized)

    metadata = {}
    existing_meta = existing.get("metadata") or {}
    incoming_meta = incoming.get("metadata") or {}
    metadata.update(existing_meta)
    metadata.update(incoming_meta)
    for list_key in ("retrieved_contexts", "rag_eval_samples"):
        combined = []
        for item in existing_meta.get(list_key) or []:
            if item not in combined:
                combined.append(item)
        for item in incoming_meta.get(list_key) or []:
            if item not in combined:
                combined.append(item)
        if combined:
            metadata[list_key] = combined

    return {
        "trace_id": trace_id,
        "trace_type": incoming.get("trace_type") or existing.get("trace_type") or str(trace_id).split("_")[0],
        "total_duration": sum(
            max(0, (node.get("end_time") or 0) - (node.get("start_time") or 0))
            for node in merged_nodes
        ),
        "timestamp": time.time(),
        "nodes": merged_nodes,
        "metadata": metadata,
    }


def _atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_name(f"{file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, file_path)


def append_trace_payload(trace_id: str, nodes: List[Dict[str, Any]] = None, metadata: Dict[str, Any] = None, log_dir: str = None) -> Path:
    manager = TraceManager(log_dir=log_dir, trace_id=trace_id)
    file_path = manager.log_dir / f"{trace_id}.json"
    incoming = {
        "trace_id": trace_id,
        "trace_type": str(trace_id).split("_")[0],
        "timestamp": time.time(),
        "nodes": nodes or [],
        "metadata": metadata or {},
    }
    existing = _read_trace_file(file_path)
    merged = _merge_trace_payload(existing, incoming)
    _atomic_write_json(file_path, merged)
    return file_path
