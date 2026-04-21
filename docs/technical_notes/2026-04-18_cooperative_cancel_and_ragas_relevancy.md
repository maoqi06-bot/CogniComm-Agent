# 多 Agent 超时残留执行与 Ragas Relevancy 为空的修复说明

## 背景

本次日志中出现了两个独立问题：

1. P2 多 Agent 任务已经汇总完成，主菜单也已经返回，但旧任务仍继续输出 `Step 19/20/...` 并继续调用 `edit_file`、`read_file` 等工具。
2. Ragas 评估可以输出 `Faithfulness`，但 `answer_relevancy` 仍为 `None`，并伴随错误：

```text
Unknown model: text-embedding-ada-002
```

## 原因分析

### 1. 旧任务在主菜单后继续执行

这不是主线程或 `main.py` 重新启动了旧任务，而是 `ThreadPoolExecutor` 中的工作线程在任务超时后没有真正停止。

Python 的线程池只能对尚未开始运行的 Future 执行 `future.cancel()`。如果子任务已经进入运行状态，尤其已经阻塞在一次 LLM HTTP 调用中，`future.cancel()` 不会杀死该线程。旧实现中调度器在超时后把任务标记为失败，但工作线程内的 `ReactAgent.run()` 不知道自己已经被取消，因此 LLM 返回后会继续解析动作并调用工具。

这就解释了日志现象：主调度流程已经认为任务超时并进入聚合/菜单，但后台工作线程还在完成自己原来的 ReAct 循环。

### 2. Relevancy 为 None

`answer_relevancy` 依赖 embedding 模型生成向量。当前上游接口拒绝了默认模型 `text-embedding-ada-002`，因此 Ragas 的相关度指标无法计算，最后显示为 `None`。

这个问题和 trace 是否写入同一个 JSON 不是同一类问题。trace 中已经有 `question`、`retrieved_contexts`、`llm_answer` 后，如果 embedding 模型不可用，Ragas 仍然无法计算 relevancy。

## 修复内容

### 1. 协作式取消

为每个 `SubTask` 增加 `cancel_event`，调度器在任务超时时设置该事件：

```python
task.cancel_event.set()
future.cancel()
```

`CodeAgent` 将该事件传入 `ReactAgent.run()`。`ReactAgent` 在以下位置检查取消状态：

- 每轮 ReAct 循环开始前
- LLM 返回后、工具调用前
- 动作解析后
- 工具执行后

这样即使当前 LLM 调用不能被强制中断，调用返回后也会停止后续工具调用，避免继续改文件、读文件或写报告。

### 2. Ragas embedding 可配置与本地兜底

即时评估和自动评估均改为使用可配置模型：

```text
RAGAS_LLM_MODEL
RAGAS_EMBEDDING_MODEL
RAGAS_OPENAI_BASE_URL
```

默认 embedding 模型从 `text-embedding-ada-002` 改为 `text-embedding-3-small`。如果上游 embedding 或完整 Ragas 评估失败，系统会：

1. 尝试仅计算 `faithfulness`。
2. 使用本地词项重叠方法生成 `answer_relevancy` 兜底值。
3. 在报告中写入 `answer_relevancy_source`，标明结果来自 `ragas` 或 `local_fallback`。

## 影响范围

- 超时任务不会立刻杀死正在进行的 HTTP 请求，但请求返回后会停止，不再继续执行工具。
- `answer_relevancy` 不会再因为 embedding 模型不可用而保持 `None`。
- 如果使用第三方 OpenAI 兼容网关，需要确认该网关支持 `RAGAS_EMBEDDING_MODEL` 设置的模型。

## 建议配置

如果当前网关不支持 `text-embedding-3-small`，请显式指定其支持的 embedding 模型：

```powershell
$env:RAGAS_EMBEDDING_MODEL = "你的网关支持的embedding模型名"
$env:RAGAS_OPENAI_BASE_URL = "你的OpenAI兼容接口地址"
```

如果只希望临时避免 Ragas 消耗，可以后续增加一个总开关，例如 `RAGAS_AUTO_EVAL=false`，在主流程中跳过自动评估。
