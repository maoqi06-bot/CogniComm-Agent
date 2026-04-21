# 2026-04-18 多 Agent Trace 丢失与子任务超时后继续运行修复

## 问题现象

多 Agent 任务中，日志显示任务的代码、结果和报告看起来已经生成，但 Agent 仍继续运行后续步骤，继续检索、读取文件、运行 Python，造成明显资源浪费。

同时，`data/traces/query_*.json` 中仍然只有检索节点，缺少 `question` 和 `llm_answer`，导致 Ragas 无法拿到完整的 Q/C/A 样本。

## 根因

### 1. 多 Agent 没有注入全局 Trace

单 Agent 的 `execute_task()` 中会创建 `TraceManager`，并把 tracer 注入各个 RAG skill。

但 P2 多 Agent 路径以前没有做这件事。于是 CodeAgent 调用 `wireless_comm_search` 时，RAG skill 发现没有外部 tracer，就自己创建临时 trace：

```text
临时 Trace 保存成功 -> data/traces/query_xxx.json
```

这些临时 trace 只有检索节点和上下文，没有最终 LLM 回答。

### 2. ReactAgent 对话历史跨子任务复用

`CodeAgent` 内部复用同一个 `ReactAgent`。如果不重置对话历史，前一个子任务的步骤、工具输出、压缩摘要会影响后一个子任务。

这会导致：

- 子任务边界不清晰。
- step 编号和行为看起来延续。
- 模型反复检查、读取、运行已完成的产物。

### 3. ThreadPool 超时不能强杀正在运行的线程

调度器使用 `future.result(timeout=600)` 判定子任务超时，但 Python 线程池无法强制终止已经运行中的线程。

因此某个子任务被标记为超时后，它内部的 ReactAgent 仍可能继续跑，后续批次又开始执行，于是日志中出现“前一个任务还在输出，后一个任务也开始了”的交错现象。

### 4. 执行工具没有子进程超时

`run_python` 没有 timeout，日志中出现了单次脚本运行 418 秒的情况。即使调度器后续判定超时，子进程仍已经消耗大量时间。

## 本次修改

### 1. OrchestratorAgent 创建并注入全局 Trace

修改文件：`dm_agent/multi_agent/__init__.py`

`OrchestratorAgent.run(..., trace=True)` 现在会：

1. 创建 `TraceManager`。
2. `start_trace("Query")`。
3. 写入 `metadata.question` 和 `metadata.task_id`。
4. 将同一个 `tracer` 和 `trace_id` 注入所有 skill：

```python
skill.current_trace_id = trace_id
skill._tracer = tracer
```

最终聚合答案生成后写入：

```python
tracer.metadata["llm_answer"] = final_answer
tracer.finish_and_save()
```

返回结果中也会携带：

```python
result["rag_trace_id"]
```

### 2. CodeAgent 每个子任务前重置对话

修改文件：`dm_agent/multi_agent/__init__.py`

`CodeAgent.process()` 进入子任务时调用：

```python
self._react_agent.reset_conversation()
```

这样每个子任务都有干净的 ReAct 上下文，减少“上一任务残留导致下一任务继续检查/运行”的情况。

### 3. RAGAgent 直接检索也写入全局 Trace

修改文件：`dm_agent/multi_agent/__init__.py`

如果知识查询任务走的是 `RAGAgent.process()` 中的直接 `_retriever.retrieve()`，现在也会把检索节点和 `retrieved_contexts` 写入注入的全局 tracer。

### 4. 调度器超时后取消 future

修改文件：`dm_agent/multi_agent/__init__.py`

任务超时时增加：

```python
future.cancel()
```

说明：这不能强杀已经开始运行的 Python 线程，但能阻止尚未开始的任务继续执行，并明确把该子任务标记为失败。

真正减少资源浪费还需要工具级超时。

### 5. 执行工具增加默认超时

修改文件：`dm_agent/tools/execution_tools.py`

以下工具现在支持统一超时：

- `run_python`
- `run_shell`
- `run_tests`
- `run_linter`

默认：

```bash
TOOL_EXEC_TIMEOUT_SECONDS=120
```

工具参数也可以传：

```json
{"path": "xxx.py", "timeout": 30}
```

超时后工具会返回：

```text
returncode: timeout after 120s
```

这样可以避免一个仿真脚本长时间占住 Agent。

### 6. P2 多 Agent 完成后触发 RagasObserver

修改文件：`main.py`

P2 路径现在会读取 `result["rag_trace_id"]`，并调用：

```python
RagasObserver.instant_eval(...)
```

如果评估端点、API Key 或网络不可用，会给出 warning，不影响主任务结果保存。

## 预期效果

新的多 Agent query trace 应包含：

```json
{
  "metadata": {
    "question": "...",
    "retrieved_contexts": ["..."],
    "llm_answer": "..."
  },
  "nodes": [
    {"method": "Hybrid_Recall", "...": "..."},
    {"method": "Rerank", "...": "..."}
  ]
}
```

日志中不应再大量出现多 Agent 路径下的：

```text
临时 Trace 保存成功
```

如果仍出现，说明某个 skill 没有 `_tracer` 属性或没有接收到 `current_trace_id`，需要继续把该 skill 纳入统一注入逻辑。

## 注意

Python 线程池不能安全强杀运行中的线程，因此调度器层面的 `future.cancel()` 不是完整中断方案。真正避免资源浪费，需要依靠：

1. 工具级 timeout。
2. 更小的 CodeAgent `max_steps`。
3. 子任务完成条件更明确。
4. 对长仿真脚本使用参数控制规模。
