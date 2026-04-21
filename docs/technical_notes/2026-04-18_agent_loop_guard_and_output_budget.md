# 2026-04-18 Agent 死循环、解析失败与输出预算治理

## 背景

在多 Agent 协作模式中执行“分析 ISAC 研究现状并生成 Python 仿真示例和文档”时，日志暴露出三类问题：

1. Reranker 后台预热线程在 Windows GBK 控制台下打印 emoji，触发 `UnicodeEncodeError`。
2. CodeAgent / ReactAgent 在编写代码和文档时反复 `read_file`、`edit_file`、`run_python`，最终达到 `max steps` 或 600 秒任务超时。
3. LLM 生成超长 JSON，其中包含未正确转义的长代码字符串，导致 `Unterminated string` 解析失败；解析失败后又继续进入下一轮，进一步扩大上下文。

这些问题并不主要是“知识库中缺少代码示例”导致的。知识库示例能提升任务质量，但不能自动保证 Agent 输出合法 JSON，也不能防止控制循环重复执行同一类动作。因此需要在 Agent 执行框架层加入硬约束。

## 本次修改

### 1. Reranker 打印改为编码安全

修改文件：`dm_agent/rag/retriever.py`

新增 `_safe_print()`，在旧版 Windows 控制台编码不支持 emoji 或特殊符号时，使用 `errors="replace"` 降级输出，避免后台线程因为打印日志崩溃。

同时将 Reranker 预热相关日志改为 ASCII 前缀：

- `[INFO] Loading Reranker model`
- `[OK] Reranker preload finished`
- `[WARN] Reranker preload failed`

这样不会再因为 GBK 控制台无法编码 `🔄`、`⚠️` 等字符而中断预热线程。

### 2. 工具输出增加预算截断

修改文件：

- `dm_agent/tools/file_tools.py`
- `dm_agent/tools/execution_tools.py`

新增输出截断逻辑：

- `read_file` 默认最多返回 `TOOL_MAX_READ_CHARS=12000` 字符。
- `run_python`、`run_shell`、`run_tests`、`run_linter` 默认最多返回 `TOOL_MAX_OUTPUT_CHARS=12000` 字符。

被截断的输出会保留头尾内容，并提示使用 `line_start` / `line_end` 做精确读取。

这样可以避免一次读取完整大文件或完整错误栈后，把十几万字符塞回 Agent 上下文，导致后续每一步 LLM 调用越来越慢、越来越容易失控。

### 3. Agent 历史上下文真正替换为压缩版本

修改文件：`dm_agent/core/agent.py`

之前上下文压缩只影响当次发送给 LLM 的 `messages_to_send`，但 `self.conversation_history` 仍然保留完整历史。下一轮又会基于完整历史判断和压缩，导致历史持续膨胀。

现在压缩后会执行：

```python
self.conversation_history = compressed_history
```

这样压缩结果会成为新的运行历史，避免 token 雪崩。

### 4. 解析失败熔断

修改文件：`dm_agent/core/agent.py`

新增环境变量：

- `AGENT_MAX_PARSE_FAILURES`，默认 `3`

当模型连续多次返回无法解析的 JSON 时，Agent 会提前停止，而不是一直把错误观察写回上下文继续尝试。

这直接针对日志中的问题：

```text
Failed to parse response: Unterminated string starting at ...
```

此类错误通常来自模型把长代码直接放进 JSON 字符串，但没有正确转义换行或引号。继续循环往往不会自然恢复，只会导致上下文越来越大。

### 5. 重复工具调用熔断

修改文件：`dm_agent/core/agent.py`

新增环境变量：

- `AGENT_MAX_REPEAT_ACTIONS`，默认 `4`

Agent 会对 `action + action_input` 生成签名。如果同一个工具和同一组参数重复调用超过阈值，会提前终止该任务并返回诊断信息。

这能阻止如下模式：

- 反复读取同一个文件。
- 反复运行同一个脚本。
- 反复对同一区间做 `edit_file`。
- 在代码修复任务中“改一点、跑一下、再改一点”但没有实际收敛。

### 6. AI 原始响应和观察结果截断入历史

修改文件：`dm_agent/core/agent.py`

新增环境变量：

- `AGENT_MAX_RAW_CHARS`，默认 `4000`
- `AGENT_MAX_OBSERVATION_CHARS`，默认 `6000`

Agent 仍会记录步骤，但写入对话历史和步骤结果的原始响应、工具观察会被限制长度。这样能显著降低多轮修复、文档写作、代码生成任务中的上下文膨胀。

### 7. CodeAgent 默认步数降低

修改文件：`dm_agent/multi_agent/__init__.py`

CodeAgent 默认 `max_steps` 从 `50` 降为 `30`，并支持：

```bash
CODE_AGENT_MAX_STEPS=30
```

复杂任务不应该靠单个子 Agent 走 50 步以上来硬磨，而应该通过任务分解、模板化代码生成、清晰的完成条件来收敛。

## 是否必须提供正确代码案例

不必须，但很有帮助。

代码案例能解决的是“模型不知道该怎么写”的问题；本次日志体现的核心问题是“模型知道一些方向，但执行控制没有收敛”。两者不是同一层问题。

建议后续同时做两件事：

1. 在知识库中保留短小、可运行、依赖少的代码模板，例如 OFDM ISAC 仿真最小示例。
2. 在工具层或 Agent 层提供“模板生成工具”，让模型选择模板并填参数，而不是让模型一次性把几百行代码塞进 JSON。

第二点比单纯增加知识库代码更关键。因为知识库内容通常会被当作检索上下文返回，LLM 仍然需要把代码重新组织成工具调用 JSON；只要输出很长，仍可能出现 JSON 转义失败。

## 后续建议

1. 为代码生成增加专用文件写入协议，例如分片写入或基于 patch 的结构化编辑，减少超长 JSON 字符串。
2. 在 Orchestrator 分解任务时限制文档和代码任务的粒度，避免一个子任务同时承担“设计、编码、运行、修复、分析、写报告”。
3. 对 `read_file` 工具提示模型优先按行读取，而不是读取全文。
4. 对多 Agent 聚合阶段继续压缩各子任务结果，只传“产物路径、摘要、错误摘要、关键指标”，不要传完整执行轨迹。
5. 对失败子任务引入降级策略：如果代码子任务超时或达到步数限制，后续文档任务应基于现有产物和错误摘要继续，而不是让失败任务无限重试。

## 验证重点

后续运行时应重点观察：

- Reranker 预热线程是否还出现 `UnicodeEncodeError`。
- 日志中 token 数是否不再持续涨到数万甚至十万级。
- 同一工具同一参数是否会被重复调用超过阈值。
- `Unterminated string` 是否在连续 3 次内被熔断。
- 多 Agent 子任务失败后是否能更快失败并返回诊断，而不是卡满 600 秒。
