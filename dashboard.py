import streamlit as st
import pandas as pd
import json
import os
import datetime
import plotly.express as px
import subprocess
import sys
from pathlib import Path

# --- 1. 初始化路径 ---
ROOT = Path(__file__).resolve().parent
# 确保路径与你的数据存储结构一致
INDEX_DIR = ROOT / "dm_agent" / "data" / "indices"
TRACE_DIR = ROOT / "data" / "traces"
RAGAS_REPORT = ROOT / "data" / "ragas_report.json"
RAGAS_STATUS = ROOT / "data" / "ragas_eval_status.json"
RAGAS_LOG = ROOT / "data" / "ragas_eval.log"
MULTI_AGENT_MEMORY_TIMELINE = ROOT / "data" / "multi_agent_memory_timeline.jsonl"
MULTI_AGENT_MEMORY_APPROVALS = ROOT / "data" / "multi_agent_memory_approvals.json"
MEMORY_REPLAY_EXPORT_DIR = ROOT / "task" / "memory_replay_exports"

st.set_page_config(page_title="DM-Agent RAG 控制塔", layout="wide")


# --- 2. 增强版计数逻辑 ---
def get_index_stats():
    total_chunks = 0
    indices = []
    if INDEX_DIR.exists():
        for meta_file in INDEX_DIR.glob("*.meta.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        lengths = [len(v) for v in data.values() if isinstance(v, list)]
                        count = max(lengths) if lengths else 1
                    else:
                        count = 0
                    total_chunks += count
                    indices.append(
                        {"name": meta_file.name.replace(".meta.json", ""), "count": count, "path": meta_file})
            except Exception:
                continue
    return total_chunks, indices


def _read_ragas_status():
    if not RAGAS_STATUS.exists():
        return {}
    try:
        with open(RAGAS_STATUS, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_process_running(pid):
    if not pid:
        return False
    try:
        subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.returncode == 0 and str(pid) in result.stdout
    except Exception:
        return False


def start_ragas_background_eval():
    status = _read_ragas_status()
    if status.get("status") == "running" and _is_process_running(status.get("pid")):
        return status, False

    RAGAS_STATUS.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "running",
        "pid": None,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "report_path": str(RAGAS_REPORT),
        "log_path": str(RAGAS_LOG),
    }
    with open(RAGAS_STATUS, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    with open(RAGAS_LOG, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "dm_agent.rag.evaluator"],
            cwd=str(ROOT),
            stdout=log_file,
            stderr=log_file,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    status["pid"] = proc.pid
    with open(RAGAS_STATUS, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    return status, True


def load_multi_agent_memory_timeline(limit: int = 500):
    if not MULTI_AGENT_MEMORY_TIMELINE.exists():
        return []
    rows = []
    try:
        with open(MULTI_AGENT_MEMORY_TIMELINE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-limit:]
    except Exception:
        return []


def load_memory_approvals():
    if not MULTI_AGENT_MEMORY_APPROVALS.exists():
        return []
    try:
        with open(MULTI_AGENT_MEMORY_APPROVALS, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_memory_approvals(records):
    MULTI_AGENT_MEMORY_APPROVALS.parent.mkdir(parents=True, exist_ok=True)
    with open(MULTI_AGENT_MEMORY_APPROVALS, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def update_memory_approval(approval_id, status, error=None):
    records = load_memory_approvals()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for record in records:
        if record.get("approval_id") == approval_id:
            record["status"] = status
            record["reviewed_at"] = now
            if error:
                record["review_error"] = str(error)
            break
    save_memory_approvals(records)


def approve_memory_record(record):
    from dm_agent.memory.memory_manager import MemoryManager
    from dm_agent.memory.long_term_memory import MemoryCategory, MemoryPriority

    template = record.get("template", {})
    manager = MemoryManager()
    manager.add_memory(
        content=record.get("content", ""),
        category=MemoryCategory(template.get("category", "skill_knowledge")),
        priority=MemoryPriority[str(template.get("priority", "normal")).upper()],
        importance_score=float(template.get("importance_score", 0.55)),
        tags=set(template.get("tags", [])),
        metadata=record.get("metadata", {}),
        source=template.get("source", "multi_agent_human_approved"),
    )


def build_memory_call_graph_dot(df_memory):
    def node_id(value):
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(value))
        return safe[:80] or "unknown"

    lines = [
        "digraph multi_agent_memory {",
        "  rankdir=LR;",
        "  node [shape=box, style=rounded];",
    ]
    for _, row in df_memory.iterrows():
        task = str(row.get("task_id") or "unknown_task")
        agent = str(row.get("agent_name") or "unknown_agent")
        kind = str(row.get("kind") or "event")
        task_node = "task_" + node_id(task)
        agent_node = "agent_" + node_id(agent)
        kind_node = "kind_" + node_id(kind)
        lines.append(f'  {task_node} [label="{task}"];')
        lines.append(f'  {agent_node} [label="{agent}"];')
        lines.append(f'  {kind_node} [label="{kind}"];')
        lines.append(f"  {task_node} -> {agent_node};")
        lines.append(f"  {agent_node} -> {kind_node};")
    lines.append("}")
    return "\n".join(lines)


def export_memory_replay(task_id, rows):
    MEMORY_REPLAY_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_task = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(task_id or "all"))
    json_path = MEMORY_REPLAY_EXPORT_DIR / f"{stamp}_{safe_task}_memory_replay.json"
    md_path = MEMORY_REPLAY_EXPORT_DIR / f"{stamp}_{safe_task}_memory_replay.md"
    payload = {
        "task_id": task_id,
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "event_count": len(rows),
        "events": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Agent Memory Replay\n\n")
        f.write(f"- Task: {task_id}\n")
        f.write(f"- Exported at: {payload['exported_at']}\n")
        f.write(f"- Events: {len(rows)}\n\n")
        for row in rows:
            f.write(
                f"## {row.get('created_at_iso', '')} | "
                f"{row.get('agent_name', '')} | {row.get('kind', '')}\n\n"
            )
            f.write(str(row.get("content", "")) + "\n\n")
    return json_path, md_path


def compute_multi_agent_memory_stats(events, approvals):
    df_events = pd.DataFrame(events)
    lookup_stats = {
        "lookups": 0,
        "hits": 0,
        "misses": 0,
        "hit_rate": None,
        "by_task": pd.DataFrame(),
        "preference_lookups": 0,
        "preference_hits": 0,
        "preference_misses": 0,
        "preference_hit_rate": None,
        "preference_by_task": pd.DataFrame(),
    }
    if not df_events.empty and "kind" in df_events:
        lookup_df = df_events[
            df_events["kind"].isin(["long_term_memory_hit", "long_term_memory_miss"])
        ].copy()
        if not lookup_df.empty:
            lookup_df["is_hit"] = lookup_df["kind"] == "long_term_memory_hit"
            lookup_stats["lookups"] = len(lookup_df)
            lookup_stats["hits"] = int(lookup_df["is_hit"].sum())
            lookup_stats["misses"] = int((~lookup_df["is_hit"]).sum())
            lookup_stats["hit_rate"] = lookup_stats["hits"] / lookup_stats["lookups"]
            group_cols = ["task_id"]
            if "agent_name" in lookup_df:
                group_cols.append("agent_name")
            by_task = (
                lookup_df.groupby(group_cols, dropna=False)
                .agg(
                    lookups=("kind", "count"),
                    hits=("is_hit", "sum"),
                )
                .reset_index()
            )
            by_task["hit_rate"] = by_task["hits"] / by_task["lookups"]
            lookup_stats["by_task"] = by_task

        preference_df = df_events[
            df_events["kind"].isin(["user_preference_memory_hit", "user_preference_memory_miss"])
        ].copy()
        if not preference_df.empty:
            preference_df["is_hit"] = preference_df["kind"] == "user_preference_memory_hit"
            lookup_stats["preference_lookups"] = len(preference_df)
            lookup_stats["preference_hits"] = int(preference_df["is_hit"].sum())
            lookup_stats["preference_misses"] = int((~preference_df["is_hit"]).sum())
            lookup_stats["preference_hit_rate"] = (
                lookup_stats["preference_hits"] / lookup_stats["preference_lookups"]
            )
            group_cols = ["task_id"]
            if "agent_name" in preference_df:
                group_cols.append("agent_name")
            preference_by_task = (
                preference_df.groupby(group_cols, dropna=False)
                .agg(
                    lookups=("kind", "count"),
                    hits=("is_hit", "sum"),
                )
                .reset_index()
            )
            preference_by_task["hit_rate"] = (
                preference_by_task["hits"] / preference_by_task["lookups"]
            )
            lookup_stats["preference_by_task"] = preference_by_task

    df_approvals = pd.DataFrame(approvals)
    approval_stats = {
        "total": len(approvals),
        "pending": 0,
        "approved": 0,
        "rejected": 0,
        "errors": 0,
        "pass_rate": None,
        "by_kind": pd.DataFrame(),
    }
    if not df_approvals.empty and "status" in df_approvals:
        statuses = df_approvals["status"].fillna("unknown").astype(str)
        approval_stats["pending"] = int((statuses == "pending").sum())
        approval_stats["approved"] = int((statuses == "approved").sum())
        approval_stats["rejected"] = int((statuses == "rejected").sum())
        approval_stats["errors"] = int((statuses == "error").sum())
        decided = approval_stats["approved"] + approval_stats["rejected"]
        if decided:
            approval_stats["pass_rate"] = approval_stats["approved"] / decided
        if "memory_kind" in df_approvals:
            by_kind = (
                df_approvals.assign(status=statuses)
                .groupby(["memory_kind", "status"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            approval_stats["by_kind"] = by_kind
    return lookup_stats, approval_stats


# --- 3. UI 布局 ---
st.sidebar.title("🚀 DM-Agent 开发者中心")
page = st.sidebar.radio("功能路由",
                        ["📊 系统总览", "📂 数据浏览器", "📥 Ingestion 追踪", "🔍 Query 链路诊断", "🧪 自动化评估", "Multi-Agent Memory"])

total_c, index_list = get_index_stats()

# --- 4. 页面功能实现 ---

if page == "📊 系统总览":
    st.header("RAG 运行状态概览")
    c1, c2, c3 = st.columns(3)
    c1.metric("总索引分块 (Chunks)", total_c)
    c2.metric("已捕获追踪 (Traces)", len(list(TRACE_DIR.glob("query_*.json"))))
    c3.metric("最后观测时间", datetime.datetime.now().strftime("%H:%M:%S"))
    st.subheader("核心组件状态")
    st.table(pd.DataFrame([
        {"层级": "向量引擎", "组件": "FAISS", "状态": "Running"},
        {"层级": "重排模型", "组件": "BGE-Reranker", "状态": "Online"},
        {"层级": "评估框架", "组件": "Ragas", "状态": "Standby"}
    ]))

elif page == "📂 数据浏览器":
    st.header("知识库内容深度浏览器")
    if index_list:
        sel_idx = st.selectbox("选择索引库", index_list, format_func=lambda x: f"{x['name']} ({x['count']} chunks)")
        try:
            with open(sel_idx['path'], 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            processed_chunks = []
            if isinstance(raw_data, list):
                for item in raw_data:
                    if isinstance(item, dict):
                        processed_chunks.append({"content": item.get("content", item.get("text", str(item))),
                                                 "metadata": item.get("metadata", item.get("meta", {}))})
                    else:
                        processed_chunks.append({"content": str(item), "metadata": {}})
            elif isinstance(raw_data, dict):
                content_keys = ['docs', 'documents', 'chunks', 'content', 'data', 'texts']
                meta_keys = ['metadatas', 'metadata', 'metas', 'info']
                target_c = next((k for k in content_keys if k in raw_data), None)
                target_m = next((k for k in meta_keys if k in raw_data), None)
                if target_c:
                    contents = raw_data[target_c]
                    metas = raw_data.get(target_m, [{} for _ in contents])
                    for c, m in zip(contents, metas):
                        processed_chunks.append({"content": str(c), "metadata": m if isinstance(m, dict) else {}})
                else:
                    processed_chunks.append({"content": json.dumps(raw_data, indent=2), "metadata": {}})

            search_q = st.text_input("🔍 搜索关键词...").strip().lower()
            display_data = [c for c in processed_chunks if search_q in c['content'].lower() or search_q in str(
                c['metadata']).lower()] if search_q else processed_chunks
            st.info(f"当前库总计: {len(processed_chunks)} | 匹配到: {len(display_data)}")

            for i, item in enumerate(display_data[:50]):
                meta = item['metadata']
                source = meta.get('file_name') or meta.get('source') or "未知来源"
                with st.expander(f"Chunk #{i} | 来源: {source}"):
                    c1, c2 = st.columns([2, 1])
                    c1.write(item['content'])
                    c2.json(meta)
        except Exception as e:
            st.error(f"加载失败: {e}")
    else:
        st.warning("未发现索引文件。")

elif page == "📥 Ingestion 追踪":
    st.header("数据同步追踪")
    i_traces = sorted(TRACE_DIR.glob("ingestion_*.json"), key=os.path.getmtime, reverse=True)
    if i_traces:
        sel_i = st.selectbox("选择同步批次", i_traces, format_func=lambda x: x.name)
        with open(sel_i, 'r', encoding='utf-8') as f:
            st.json(json.load(f))
    else:
        st.info("暂无记录。")

elif page == "🔍 Query 链路诊断":
    st.header("🔍 检索全链路白盒化诊断")
    q_traces = sorted(TRACE_DIR.glob("query_*.json"), key=os.path.getmtime, reverse=True)

    if q_traces:
        sel_q = st.selectbox("选择查询记录", q_traces, format_func=lambda x: f"🕒 {x.name}")

        with open(sel_q, 'r', encoding='utf-8') as f:
            q_data = json.load(f)

        # 提取节点数据
        nodes_list = q_data.get('nodes', [])
        if not nodes_list:
            st.warning("⚠️ 该 Trace 记录中没有节点数据。")
        else:
            df_q = pd.DataFrame(nodes_list)

            # --- 健壮的时间计算逻辑 ---
            # 1. 确保时间戳列存在
            if 'start_time' in df_q.columns and 'end_time' in df_q.columns:
                # 过滤掉未正常关闭的节点 (end_time 为 0 的情况)
                df_q = df_q[df_q['end_time'] > 0].copy()

                # 2. 计算耗时
                df_q['duration_sec'] = df_q['end_time'] - df_q['start_time']

                # 3. 绘制耗时甘特图/柱状图
                fig = px.bar(
                    df_q,
                    x="duration_sec",
                    y="method",
                    orientation='h',
                    color="provider",  # 按技术提供方着色 (FAISS/BGE等)
                    title="各环节耗时流水 (秒)",
                    hover_data=["provider", "node_id"],
                    labels={"duration_sec": "耗时 (s)", "method": "处理环节"}
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- 展示元数据 (Metadata) ---
            if q_data.get('metadata'):
                with st.expander("📝 检索上下文 (Contexts)"):
                    contexts = q_data['metadata'].get('retrieved_contexts', [])
                    for i, ctx in enumerate(contexts):
                        st.text_area(f"Context #{i + 1}", ctx[:500] + "...", height=100)

        # --- 核心改进：动态探测 Trace 节点 ---
        nodes = {n.get('method') or n.get('name', 'Unknown'): n for n in q_data['nodes']}
        st.subheader("🎯 检索效果对比")

        # 允许用户选择节点，以防硬编码的名称不匹配
        available_methods = list(nodes.keys())
        c1, c2 = st.columns(2)

        with c1:
            method_1 = st.selectbox("选择原始检索节点 (Recall)", available_methods,
                                    index=min(1, len(available_methods) - 1))
            res_1 = nodes.get(method_1, {}).get("output_data", [])
            st.write(res_1[:3] if res_1 else "该节点无输出数据")

        with c2:
            method_2 = st.selectbox("选择精排结果节点 (Rerank)", available_methods,
                                    index=min(2, len(available_methods) - 1))
            res_2 = nodes.get(method_2, {}).get("output_data", [])
            st.write(res_2[:3] if res_2 else "该节点无输出数据")

        with st.expander("查看原始 Trace JSON"):
            st.json(q_data)
    else:
        st.info("暂无查询记录。")

elif page == "🧪 自动化评估":
    st.header("Ragas 自动化语义评估")
    st.caption("评估默认以 RAG query 为样本边界。点击按钮会启动后台评估进程，不会阻塞 DM-Code-Agent 主任务输入。")

    col_run, col_reload = st.columns(2)
    with col_run:
        if st.button("🔄 后台刷新 Ragas 数据"):
            status, started = start_ragas_background_eval()
            if started:
                st.success(f"已启动后台评估进程 PID={status.get('pid')}。完成后本页会读取新的 ragas_report.json。")
            else:
                st.info(f"已有评估进程正在运行 PID={status.get('pid')}。")
    with col_reload:
        if st.button("📈 仅刷新图表显示"):
            st.rerun()

    status = _read_ragas_status()
    if status:
        st.info(
            f"评估状态: {status.get('status', 'unknown')} | "
            f"PID: {status.get('pid', 'N/A')} | "
            f"更新时间: {status.get('updated_at') or status.get('started_at') or 'N/A'}"
        )
    if RAGAS_REPORT.exists():
        try:
            # [优化] 使用更稳健的读取方式，防止 main.py 写入时冲突
            with open(RAGAS_REPORT, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df_eval = pd.DataFrame(data)
            if df_eval.empty:
                st.warning("评估报告为空。请点击后台刷新 Ragas 数据。")
                st.stop()
            # 兼容性处理列名
            rel_col = 'answer_relevancy' if 'answer_relevancy' in df_eval.columns else 'answer_relevance'
            scope_col = 'eval_scope' if 'eval_scope' in df_eval.columns else None
            source_col = 'source' if 'source' in df_eval.columns else None

            if scope_col:
                scopes = ["全部"] + sorted(str(v) for v in df_eval[scope_col].dropna().unique())
                selected_scope = st.selectbox("评估口径", scopes)
                if selected_scope != "全部":
                    df_eval = df_eval[df_eval[scope_col].astype(str) == selected_scope]

            # --- Generation quality and retrieval quality are different views. ---
            st.subheader("生成质量")
            gen_cols = st.columns(4)
            gen_cols[0].metric("Faithfulness (忠实度)", f"{df_eval['faithfulness'].mean():.2%}")
            gen_cols[1].metric("Answer relevancy (回答相关度)", f"{df_eval[rel_col].mean():.2%}")
            gen_cols[2].metric("评估样本数", len(df_eval))
            if "generation_mode" in df_eval.columns:
                raw_ratio = (df_eval["generation_mode"] == "raw_retrieval_dump").mean()
                gen_cols[3].metric("原始检索返回占比", f"{raw_ratio:.2%}")
            else:
                gen_cols[3].metric("原始检索返回占比", "N/A")

            retrieval_cols_available = [
                col
                for col in [
                    "retrieval_quality",
                    "context_query_overlap_max",
                    "max_context_score",
                    "source_count",
                    "weak_retrieval",
                ]
                if col in df_eval.columns
            ]
            if retrieval_cols_available:
                st.subheader("检索质量")
                ret_cols = st.columns(4)
                ret_cols[0].metric("Retrieval quality (检索质量)", f"{df_eval['retrieval_quality'].mean():.2%}")
                ret_cols[1].metric("Query-context overlap", f"{df_eval['context_query_overlap_max'].mean():.2%}")
                if "max_context_score" in df_eval.columns and df_eval["max_context_score"].notna().any():
                    ret_cols[2].metric("Max rerank score", f"{df_eval['max_context_score'].mean():.2%}")
                else:
                    ret_cols[2].metric("Max rerank score", "N/A")
                weak_count = int(df_eval.get("weak_retrieval", pd.Series(dtype=bool)).fillna(False).sum())
                ret_cols[3].metric("Weak retrieval", weak_count)

            chart_cols = [
                col
                for col in ["faithfulness", rel_col, "retrieval_quality", "context_query_overlap_max"]
                if col in df_eval.columns
            ]
            st.line_chart(df_eval[chart_cols])

            # --- [新增] 样本详情：查看具体的 ISAC 论文片段 ---
            st.subheader("📝 详细样本分析")
            display_cols = ['question', 'faithfulness', rel_col]
            for col in [
                "retrieval_quality",
                "context_query_overlap_max",
                "max_context_score",
                "source_count",
                "generation_mode",
            ]:
                if col in df_eval.columns:
                    display_cols.append(col)
            if scope_col:
                display_cols.append(scope_col)
            if source_col:
                display_cols.append(source_col)
            if 'answer_relevancy_source' in df_eval.columns:
                display_cols.append('answer_relevancy_source')
            st.dataframe(df_eval[display_cols], use_container_width=True)

            # 允许点击查看某条具体的检索内容
            selected_idx = st.number_input("选择样本索引查看详情", min_value=0, max_value=len(df_eval) - 1, step=1)
            if st.button("查看该样本上下文"):
                row = df_eval.iloc[selected_idx]
                st.info(f"**问题:** {row['question']}")
                st.warning(f"**检索到的论文片段 (Contexts):**")
                for i, ctx in enumerate(row['contexts']):
                    st.text_area(f"片段 {i}", ctx, height=150)

        except Exception as e:
            st.error(f"解析报告失败: {e}")
    else:
        st.warning("⚠️ 报告文件不存在。请确保运行了 evaluator.py 并生成了 ragas_report.json。")

if page == "Multi-Agent Memory":
    st.header("Multi-Agent Memory")
    st.caption(
        "This page only observes and controls the multi-agent memory layer. "
        "The single-agent memory system remains unchanged."
    )

    col_refresh, col_note = st.columns([1, 3])
    with col_refresh:
        if st.button("Refresh memory data"):
            st.rerun()
    with col_note:
        st.info("Long-term writes that require human approval are handled here, not in the CLI task path.")

    events = load_multi_agent_memory_timeline()
    approvals = load_memory_approvals()
    lookup_stats, approval_stats = compute_multi_agent_memory_stats(events, approvals)

    st.subheader("Cross-Task Memory Metrics")
    metric_cols = st.columns(8)
    hit_rate = lookup_stats["hit_rate"]
    preference_hit_rate = lookup_stats["preference_hit_rate"]
    pass_rate = approval_stats["pass_rate"]
    metric_cols[0].metric("Long-term lookups", lookup_stats["lookups"])
    metric_cols[1].metric("Long-term hits", lookup_stats["hits"])
    metric_cols[2].metric(
        "Long-term hit rate",
        "N/A" if hit_rate is None else f"{hit_rate:.2%}",
    )
    metric_cols[3].metric("Preference lookups", lookup_stats["preference_lookups"])
    metric_cols[4].metric(
        "Preference hit rate",
        "N/A" if preference_hit_rate is None else f"{preference_hit_rate:.2%}",
    )
    metric_cols[5].metric("Approval total", approval_stats["total"])
    metric_cols[6].metric("Approval pending", approval_stats["pending"])
    metric_cols[7].metric(
        "Approval pass rate",
        "N/A" if pass_rate is None else f"{pass_rate:.2%}",
    )

    detail_col_a, detail_col_b = st.columns(2)
    with detail_col_a:
        st.caption("Long-term memory hit rate by task/agent")
        if lookup_stats["by_task"].empty:
            st.info("No long-term memory lookup events yet.")
        else:
            st.dataframe(lookup_stats["by_task"], use_container_width=True)
    with detail_col_b:
        st.caption("Relevant user preference hit rate by task/agent")
        if lookup_stats["preference_by_task"].empty:
            st.info("No user preference lookup events yet.")
        else:
            st.dataframe(lookup_stats["preference_by_task"], use_container_width=True)

    st.caption("Approval status by memory kind")
    if approval_stats["by_kind"].empty:
        st.info("No approval records yet.")
    else:
        st.dataframe(approval_stats["by_kind"], use_container_width=True)

    st.subheader("Human Approval Queue")
    pending = [record for record in approvals if record.get("status") == "pending"]
    st.metric("Pending long-term memory approvals", len(pending))
    if pending:
        for record in pending:
            label = (
                f"{record.get('created_at_iso', '')} | "
                f"{record.get('agent_name', '')} | "
                f"{record.get('memory_kind', '')} | "
                f"{record.get('approval_id', '')}"
            )
            with st.expander(label):
                st.write(record.get("content", ""))
                st.json(record.get("metadata", {}))
                approve_col, reject_col = st.columns(2)
                with approve_col:
                    if st.button("Approve and write long-term memory", key=f"approve_{record.get('approval_id')}"):
                        try:
                            approve_memory_record(record)
                            update_memory_approval(record.get("approval_id"), "approved")
                            st.success("Approved and written to long-term memory.")
                            st.rerun()
                        except Exception as exc:
                            update_memory_approval(record.get("approval_id"), "error", error=exc)
                            st.error(f"Approval failed: {exc}")
                with reject_col:
                    if st.button("Reject", key=f"reject_{record.get('approval_id')}"):
                        update_memory_approval(record.get("approval_id"), "rejected")
                        st.warning("Rejected.")
                        st.rerun()
    else:
        st.info("No pending multi-agent memory approvals.")

    st.subheader("Timeline And Task Call Graph")
    if not events:
        st.info("No multi-agent memory events yet. Run a P2 task to populate data/multi_agent_memory_timeline.jsonl.")
    else:
        df_memory = pd.DataFrame(events)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Events", len(df_memory))
        metric_cols[1].metric("Tasks", df_memory["task_id"].nunique() if "task_id" in df_memory else 0)
        metric_cols[2].metric("Agents", df_memory["agent_name"].nunique() if "agent_name" in df_memory else 0)
        metric_cols[3].metric("Approvals", len(approvals))

        task_ids = ["ALL"] + sorted([str(x) for x in df_memory.get("task_id", pd.Series()).dropna().unique()])
        selected_task = st.selectbox("Task", task_ids)
        if selected_task != "ALL":
            df_memory = df_memory[df_memory["task_id"].astype(str) == selected_task]

        agent_names = ["ALL"] + sorted([str(x) for x in df_memory.get("agent_name", pd.Series()).dropna().unique()])
        selected_agent = st.selectbox("Agent", agent_names)
        if selected_agent != "ALL":
            df_memory = df_memory[df_memory["agent_name"].astype(str) == selected_agent]

        if not df_memory.empty and "created_at" in df_memory:
            df_memory["start"] = pd.to_datetime(df_memory["created_at"], unit="s")
            df_memory["finish"] = df_memory["start"] + pd.to_timedelta(2, unit="s")
            st.subheader("Event Flow")
            fig = px.timeline(
                df_memory.sort_values("start"),
                x_start="start",
                x_end="finish",
                y="agent_name",
                color="kind",
                hover_data=["task_id", "shared", "content"],
                title="Multi-Agent Memory Event Flow",
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        if not df_memory.empty:
            st.subheader("Task-Level Call Graph")
            try:
                st.graphviz_chart(build_memory_call_graph_dot(df_memory), use_container_width=True)
            except Exception as exc:
                st.warning(f"Graphviz rendering failed: {exc}")

        st.subheader("Replay Export")
        export_rows = df_memory.to_dict("records")
        export_task = selected_task if selected_task != "ALL" else "all_tasks"
        if st.button("Export current memory replay"):
            json_path, md_path = export_memory_replay(export_task, export_rows)
            st.success(f"Exported replay:\n\nJSON: {json_path}\n\nMarkdown: {md_path}")

        st.subheader("Event Table")
        if "created_at_iso" in df_memory:
            df_memory = df_memory.sort_values("created_at_iso", ascending=False)
        display_cols = [
            c
            for c in ["created_at_iso", "task_id", "agent_name", "kind", "shared", "content", "metadata"]
            if c in df_memory.columns
        ]
        st.dataframe(df_memory[display_cols], use_container_width=True)

        st.subheader("Memory Replay")
        for _, row in df_memory.head(30).iterrows():
            title = f"{row.get('created_at_iso', '')} | {row.get('agent_name', '')} | {row.get('kind', '')}"
            with st.expander(title):
                st.write(row.get("content", ""))
                st.json(row.to_dict())
