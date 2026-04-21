# 2026-04-20 多 Agent 子任务超时、部分完成与依赖恢复

## 背景

在 P2 多 Agent 模式中执行任务：

```text
检索ISAC与RIS波束赋形的相关内容，实现一个最简单的混合波束赋形（HBF）的推导和python实现，结果放入task文件夹中
```

日志显示前三个 RAG 子任务均已完成，后续 CodeAgent 也实际写出了多个文件：

```text
task/isac_ris_hbf_analysis.md
task/hbf_system_model.md
task/hbf_derivation.md
task/hbf_algorithm_steps.md
task/hbf_simple.py
task/test_hbf.py
```

但最终任务报告显示 `3/8` 个子任务完成，后续 `task_5`、`task_6`、`task_7`、`task_8` 均被跳过。

## 问题现象

关键日志如下：

```text
Task task_4 timed out after 600s
Skipping task_5: Skipped because dependencies failed: task_4
Skipping task_6: Skipped because dependencies failed: task_5
Skipping task_7: Skipped because dependencies failed: task_5, task_6
Skipping task_8: Skipped because dependencies failed: task_7
```

可以看到，真正失败点不是 RAG 检索，而是 `task_4` 超时后被调度器标记为 `failed`，后续所有依赖 `task_4` 或其下游任务的子任务都被级联跳过。

## 根因分析

### 1. 子任务粒度过大

`task_4` 原本只是分析任务：

```text
基于 task_1、task_2、task_3 的检索结果，分析 ISAC 与 RIS 波束赋形的结合点，确定最简单的 HBF 推导场景
```

但 CodeAgent 在该子任务中实际做了：

- 场景分析
- 系统模型文档
- 数学推导文档
- 算法步骤文档
- Python 实现
- 测试脚本
- 运行检查

这些工作本应分别属于后续 `task_5`、`task_6`、`task_7`。因此单个子任务承担了过多责任，容易耗尽调度器超时时间。

### 2. CodeAgent 没有及时结束当前子任务

CodeAgent 已经写出了可用产物，但没有在 600 秒内调用 `task_complete`。调度器只能通过 `future.result(timeout=600)` 判断该子任务超时。

这属于完成协议不够强的问题：模型倾向于继续检查、扩写和提前完成后续任务，而不是在当前子任务达到可供下游使用的程度后立即结束。

### 3. 依赖策略过硬

之前的调度逻辑中，任一依赖任务只要状态为 `failed`，下游任务就直接跳过。

这对纯计算流水线是合理的，但对 Agent 任务不够鲁棒。因为 Agent 子任务即使超时，也可能已经写出文件、生成文档或完成大部分有价值工作。将其直接视为硬失败，会浪费已有产物，并造成级联失败。

### 4. 超时时间固定且不可配置

`OrchestratorAgent` 中调度器固定使用：

```python
TaskScheduler(..., task_timeout=600)
```

复杂文档和代码任务中，单次 LLM 调用可能超过 100 秒，多个工具调用叠加后很容易达到 600 秒。固定超时时间缺少按任务环境调节的空间。

## 本次修改

### 1. 强化 CodeAgent 子任务边界

修改文件：

```text
dm_agent/multi_agent/prompts.py
```

在多 Agent CodeAgent prompt 中新增约束：

- 严格只完成当前子任务。
- 不提前执行后续子任务。
- 当前子任务核心产物已经写入或验证完成后，立刻调用 `task_complete`。
- 不要反复 `list/read/check` 同一批文件。
- 分析或推导类子任务优先生成简洁 Markdown 或简洁摘要，并在少量工具调用内结束。

这可以减少 CodeAgent 在一个子任务里“把整条流水线都做完”的倾向。

### 2. 增加 `partial` 子任务状态

修改文件：

```text
dm_agent/multi_agent/runtime.py
```

`SubTask.status` 现在支持：

```text
pending, running, completed, partial, failed
```

当子任务因为调度器超时而未显式完成时，不再直接标记为 `failed`，而是标记为：

```python
task.status = "partial"
task.result = {
    "success": False,
    "partial": True,
    "error": "Task timeout after ...",
    "message": "Subtask timed out before an explicit task_complete call. Any files already written by this subtask may still be usable by downstream tasks."
}
```

这样可以保留“部分完成”的语义，而不是把所有有效中间产物都当作失败丢弃。

### 3. 下游依赖允许使用 `partial` 上游

修改文件：

```text
dm_agent/multi_agent/runtime.py
```

依赖上下文构造逻辑现在会把 `completed` 和 `partial` 都作为可注入的上游结果：

```python
if dep.status in {"completed", "partial"} and dep.result:
    ...
```

调度循环中也会将 `partial` 加入 `completed_tasks`：

```python
if sub_task.status in {"completed", "partial"}:
    completed_tasks[sub_task.id] = sub_task
```

因此后续子任务不会再因为上游 timeout 直接整条链跳过，而是能够基于已有文件、已有摘要和 timeout 说明继续工作。

### 4. 聚合器纳入 `partial` 结果

修改文件：

```text
dm_agent/multi_agent/runtime.py
```

`ResultAggregator` 现在会把 `partial` 任务也纳入最终总结上下文：

```python
if task.status in {"completed", "partial"} and task.result:
    ...
```

这样最终报告可以明确说明哪些任务完整完成、哪些任务部分完成、哪些任务真正失败。

### 5. 子任务超时时间改为环境变量可配置

修改文件：

```text
dm_agent/multi_agent/runtime.py
```

调度器超时时间从固定 `600` 秒改为：

```python
task_timeout = int(os.getenv("MULTI_AGENT_TASK_TIMEOUT", "900"))
self.scheduler = TaskScheduler(self.logger, max_parallel, task_timeout=task_timeout)
```

用户可以在 PowerShell 中设置：

```powershell
$env:MULTI_AGENT_TASK_TIMEOUT="1200"
```

复杂任务可以放宽到 1200 秒，普通任务保持默认 900 秒。

### 6. CLI 和报告展示 `partial_count`

修改文件：

```text
main.py
```

CLI 现在输出：

```text
子任务: X 完成 / Y 部分完成 / Z 失败 / 共 N
```

Markdown 报告也会写入：

```text
完成子任务
部分完成子任务
失败子任务
```

执行追踪中，`partial` 子任务显示为：

```text
[PARTIAL]
```

并附带 timeout 或其他说明。

## 为什么这不是检索问题

本次任务中 RAG 子任务均已成功完成，且知识库中确实存在 RIS-aided ISAC HBF 相关内容，例如：

- RIS-aided ISAC hybrid beamforming
- weighted sum of SCNR and SINR
- alternating optimization
- FP / quadratic transformation / manifold optimization

真正的问题是 CodeAgent 在 `task_4` 中做了过多后续工作，且没有及时提交完成信号。因此主要改进点在 Agent 任务边界、调度鲁棒性和超时恢复，而不是 RAG 检索质量。

## 使用建议

对于复杂的“检索 + 推导 + 代码 + 测试 + 报告”任务，建议运行前设置：

```powershell
$env:MULTI_AGENT_TASK_TIMEOUT="1200"
$env:CODE_AGENT_MAX_STEPS="20"
```

其中：

- `MULTI_AGENT_TASK_TIMEOUT` 控制单个子任务最长等待时间。
- `CODE_AGENT_MAX_STEPS` 控制 CodeAgent 单个子任务最多工具调用轮数。

看起来矛盾，但二者作用不同：

- timeout 稍微放宽，避免长 LLM 调用导致误杀。
- max steps 适当收紧，迫使 CodeAgent 更早收束并调用 `task_complete`。

## 预期效果

后续类似任务中，如果某个 CodeAgent 子任务已经写出部分文件但没有及时完成：

1. 调度器会把它标记为 `partial`，而不是 `failed`。
2. 下游任务仍会继续执行。
3. 下游任务能看到上游 timeout 说明和已有结果。
4. 最终报告会区分完成、部分完成和失败。
5. 用户可以从报告中更准确判断任务状态，而不是看到大量子任务被误判为失败。

## 验证

已执行语法检查：

```text
syntax ok
```

已验证 CodeAgent prompt 包含正确协议和边界约束：

```text
has action protocol True
has boundary True
```

已用模拟超时任务验证调度器行为：

```text
Task task_a timed out after 0.05s
partial True failed? False
```

说明 timeout 任务现在会进入 `partial` 状态，不再作为硬失败处理。

## 后续方向

1. 在 TaskDecomposer 中增加更强的粒度控制，避免一个子任务同时承担分析、推导、代码、测试、报告。
2. 为 CodeAgent 增加“当前子任务产物检测器”，当检测到目标文件已生成时提示其尽快 `task_complete`。
3. 为依赖边增加软/硬依赖类型：核心依赖失败时跳过，辅助依赖失败或 partial 时继续。
4. 在 Dashboard 中展示 partial 子任务及其产物路径。
5. 将 CodeAgent 的完成协议做成独立 profile 参数，让不同领域控制不同收束策略。
