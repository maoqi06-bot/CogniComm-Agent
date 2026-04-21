# 2026-04-18 RAG Trace 合并写入与 Ragas 评估修复

## 问题背景

项目中通过 `data/traces/query_*.json` 保存 RAG 运行链路，并用 Ragas 评估最终回答的：

- Faithfulness
- Answer Relevancy

Ragas 需要同一个样本中同时具备三类数据：

1. `question`
2. `retrieved_contexts`
3. `llm_answer`

但实际运行中经常出现两种不完整 trace：

- 只有检索节点、召回结果、reranker 结果，没有最终回答。
- 只有 `question` 和 `llm_answer`，没有检索节点和 `retrieved_contexts`。

这会导致 `RagasObserver` 或 `evaluator.py` 找不到完整样本，最终无法评估。

## 根因分析

### MCP RAG 模式

MCP RAG server 和主 Agent 是两个进程。

典型执行顺序是：

1. 主进程创建 `TraceManager`，生成 `trace_id`。
2. 主进程通过 MCP skill 把 `trace_id` 传给 RAG server。
3. RAG server 在独立进程中完成 Hybrid Recall 和 Rerank，并把节点追加到 `data/traces/query_xxx.json`。
4. 主 Agent 生成最终回答。
5. 主进程调用 `tracer.finish_and_save()` 写入 `question` 和 `llm_answer`。

旧实现的问题是第 5 步会用主进程内存里的 trace 覆盖同名 JSON 文件。主进程内存里通常没有 MCP 进程写入的 nodes，因此最终文件只剩 `question/llm_answer`，检索链路被覆盖。

所以 MCP 模式的核心问题是：跨进程共享同一个 trace 文件时使用了覆盖写，而不是合并写。

### 内置 RAG 模式

内置 RAG 与主 Agent 在同一进程，理论上可以共享同一个 `TraceManager` 实例。但如果 tracer 没有被正确注入到具体 skill，RAG skill 会临时创建自己的 trace 并自行保存。

这样会出现：

- 临时 trace：有检索节点和上下文，但没有最终回答。
- 主 trace：有最终回答，但没有检索节点。

所以内置 RAG 的核心问题不是进程隔离，而是同一次任务没有共享同一个 trace 生命周期。

## 本次修复

### 1. TraceManager 保存时合并已有文件

修改文件：`dm_agent/rag/observability.py`

`TraceManager.finish_and_save()` 不再直接覆盖同名 JSON，而是：

1. 读取已有 trace 文件。
2. 合并已有 nodes 和当前内存 nodes。
3. 合并已有 metadata 和当前 metadata。
4. 重新计算 `total_duration`。
5. 使用临时文件 + `os.replace()` 原子写入。

这样 MCP RAG server 先写入的检索节点不会被主进程最终保存覆盖。

### 2. 新增 append_trace_payload 统一追加入口

修改文件：`dm_agent/rag/observability.py`

新增：

```python
append_trace_payload(trace_id, nodes=None, metadata=None, log_dir=None)
```

它是跨进程追加 trace 的统一入口，会自动执行合并写入。

### 3. MCP server 改用统一追加入口

修改文件：`dm_agent/rag/rag_mcp_server.py`

`_append_to_trace()` 不再手动读写 JSON，而是调用：

```python
append_trace_payload(trace_id, nodes=nodes, metadata=metadata, log_dir=str(TRACE_DIR))
```

同时统一兼容旧字段：

- `name` -> `method`
- `type` -> `provider`
- `input_val` -> `input_data`
- `output_val` -> `output_data`

这能保证 dashboard 和 evaluator 按统一字段读取。

### 4. evaluator 优先读取完整上下文

修改文件：`dm_agent/rag/evaluator.py`

旧逻辑优先从 `Hybrid_Recall.output_data[*].doc` 提取 contexts，但这里常常只是 50 字符预览，不适合作为 Ragas 上下文。

现在优先读取：

```python
metadata.retrieved_contexts
```

只有不存在时才回退到 Recall 节点预览。

问题也优先读取：

```python
metadata.question
```

### 5. Dashboard 兼容新旧节点字段

修改文件：`dashboard.py`

Query 链路诊断中构造节点字典时，兼容：

- `method`
- `name`

避免旧 trace 或 MCP trace 字段不一致导致页面报错。

## 为什么不是简单加文件锁

文件锁只能解决“两个进程同时写”导致的文件损坏问题，但不能解决“后写者覆盖先写者业务数据”的问题。

本次更关键的是语义合并：

- MCP 写入 nodes 和 retrieved contexts。
- 主进程写入 question 和 llm_answer。

两者不是互斥关系，而是同一个 trace 的不同部分。因此应该合并，而不是互相覆盖。

## 后续建议

1. 主 Agent 创建 trace 后，应把 `trace_id` 和 `TraceManager` 统一注入所有 RAG skill。
2. 多 Agent 模式中，如果子 Agent 会调用 RAG，也需要把父任务 trace_id 或子任务 trace_id 显式传入。
3. RAG 检索工具返回给 LLM 的文本可以截断，但写入 `metadata.retrieved_contexts` 的内容应保留足够完整的上下文，供 Ragas 评估。
4. Dashboard 可以增加“Trace 完整性检查”，直接标记缺少 Q/C/A 的 trace。

## 验证标准

一个可评估的 query trace 至少应包含：

```json
{
  "nodes": [
    {"method": "Hybrid_Recall", "...": "..."},
    {"method": "Rerank", "...": "..."}
  ],
  "metadata": {
    "question": "...",
    "retrieved_contexts": ["..."],
    "llm_answer": "..."
  }
}
```

满足这个结构后，`RagasObserver.instant_eval()` 和 `dm_agent/rag/evaluator.py` 才能稳定计算 Faithfulness 与 Answer Relevancy。
