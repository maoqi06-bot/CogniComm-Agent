# Multi-Agent Memory Design

## 目标

单 Agent 模式已经具备完整记忆模块：`ReactAgent` 可以在任务前检索长期记忆，在执行中通过记忆工具写入、更新、删除长期记忆，并在任务结束后自动提取可保存信息。这个模式继续保留，因为它是用户可选择的单智能体工作方式。

多 Agent 模式不能简单复制“每个 ReactAgent 自己管理全部记忆”的设计。P2 模式中存在 Orchestrator、RAGAgent、CodeAgent 等不同角色，如果每个子 Agent 都直接读写长期记忆，容易产生重复写入、低价值写入、上下文冲突和工具越权。因此多 Agent 模式新增 `MultiAgentMemoryHub` 作为统一协调层，而不是改动单 Agent 的记忆系统。

## 当前实现

新增 `dm_agent/multi_agent/memory.py`，提供 `MultiAgentMemoryHub`：

- 任务级共享短期记忆：记录 Orchestrator、RAGAgent、CodeAgent 在当前任务中的关键事件和子任务结果。
- 子 Agent 私有短期记忆：记录某个 Agent 的局部工作状态，只注入给对应 Agent。
- 长期记忆桥接：可选复用现有 `MemoryManager`，在子任务执行前检索长期记忆，在总任务结束后写入任务摘要。
- 子 Agent 记忆策略：通过 `AgentMemoryPolicy` 约束不同角色的长期记忆读写权限。
- 异步长期写入：任务摘要等长期记忆写入默认提交到后台线程，避免阻塞 CLI 下一轮输入。
- Timeline 持久化：短期记忆事件 append 到 `data/multi_agent_memory_timeline.jsonl`，dashboard 可直接读取。
- 长期记忆命中统计：每次多 Agent 构建记忆上下文时记录 `long_term_memory_hit` 或 `long_term_memory_miss`，只用于统计，不注入共享短期记忆，避免污染 Agent 上下文。
- Memory replay：`OrchestratorAgent.run(..., trace=True)` 会把 `memory_replay` 放入结果字典，便于调试某次任务如何使用短期/长期记忆。
- 人工审批队列：`engineering_experience` 默认不直接写入长期记忆，而是写入 `data/multi_agent_memory_approvals.json`，由 dashboard 人工审批后再进入长期记忆。

## 记忆分层

### 1. 短期共享记忆

生命周期为一次多 Agent 任务。适合保存：

- 用户原始任务
- RAGAgent 生成的知识结论
- CodeAgent 的代码实现结果
- 已完成子任务摘要
- 失败子任务的错误信息

这些内容会作为 `Shared task memory` 注入下游 Agent，减少重复检索和重复执行。

### 2. 子 Agent 私有工作记忆

生命周期同样为一次多 Agent 任务，但只给对应子 Agent 使用。适合保存：

- CodeAgent 的局部实现计划
- RAGAgent 的检索查询和生成状态
- 某个 Agent 的中间判断

这样可以让 Agent 保持连续性，同时避免把过细的内部工作状态污染全局上下文。

### 3. 长期记忆

长期记忆仍然使用已有单 Agent 的 `MemoryManager` 和 `LongTermMemoryStore`。多 Agent 只通过 `MultiAgentMemoryHub` 统一访问：

- 任务/子任务开始前：检索相关长期记忆，形成增强上下文。
- 总任务结束后：写入任务摘要，包含原始任务、成功/失败子任务数、最终答案摘要。
- CodeAgent：可沉淀经过质量过滤和人工审批的工程经验。

多 Agent 的长期记忆写入不直接开放给所有子 Agent，避免每个子 Agent 都调用 `add_memory` 造成重复或低质量记忆。

## Memory Policy

默认 memory policy：

| Agent | 长期记忆读取 | 长期记忆写入 | 允许写入类型 |
| --- | --- | --- | --- |
| Orchestrator | 是 | 是 | `task_summary` |
| RAGAgent | 是 | 否 | 无 |
| CodeAgent | 是 | 是 | `engineering_experience`, `debugging_lesson`, `implementation_pattern`, `simulation_result` |

RAGAgent 可以使用长期记忆增强研究上下文，但不能直接把检索结果写进长期记忆。CodeAgent 可以沉淀工程经验，但 `engineering_experience` 默认必须经过质量过滤和人工审批。Orchestrator 负责最终任务摘要。

## 写入模板

`MemoryWriteTemplate` 负责把不同类型的长期记忆映射到已有单 Agent 记忆系统的分类：

- `task_summary` -> `conversation_summary`
- `engineering_experience` -> `skill_knowledge`
- `research_note` -> `skill_knowledge`
- `debugging_lesson` -> `skill_knowledge`
- `implementation_pattern` -> `skill_knowledge`
- `simulation_result` -> `skill_knowledge`

不同领域 profile 可以覆盖模板。例如 wireless profile 会给记忆增加 `wireless`、`isac`、`simulation` 等标签；AI、robotics、education、data_science 等 profile 会配置更细粒度的实验、调试和实现模式标签。

## 工程经验质量过滤与人工审批

`engineering_experience` 默认必须先通过质量门控：

- 输出长度至少达到 `engineering_experience_min_chars`。
- 不能包含明显失败、取消、报错标记。
- 需要包含代码、文件、测试、实现、运行等工程关键词之一。

通过质量门控后，系统不会立刻写入长期记忆，而是进入人工审批队列：

```text
data/multi_agent_memory_approvals.json
```

dashboard 的 `Multi-Agent Memory` 页面提供审批入口：

- Approve：写入长期记忆，并把审批状态改为 `approved`。
- Reject：不写入长期记忆，并把审批状态改为 `rejected`。
- Error：写入失败时保留错误信息，便于排查。

这样既避免低价值代码输出污染长期记忆，也保留用户对工程经验沉淀的控制权。

## 预置领域 Profile

`dm_agent.multi_agent.domain_profiles` 提供 `build_domain_profiles(domain)`：

- `wireless` / `isac` / `通信` / `无线通信`
- `medical` / `medicine` / `医学` / `医疗`
- `legal` / `law` / `法律`
- `finance` / `financial` / `金融`
- `ai` / `machine_learning` / `deep_learning` / `人工智能`
- `robotics` / `control` / `机器人`
- `education` / `teaching` / `教育`
- `data_science` / `analytics` / `数据科学`
- `cybersecurity` / `security` / `网络安全`

这些 profile 会覆盖 RAGAgent 的研究风格、检索生成温度、长期记忆读写策略，以及 CodeAgent 的工程记忆模板。它们是轻量化扩展点，不改变 Orchestrator/RAGAgent/CodeAgent 的核心实现。

## Dashboard

Dashboard 的 `Multi-Agent Memory` 页面读取：

```text
data/multi_agent_memory_timeline.jsonl
data/multi_agent_memory_approvals.json
```

页面展示：

- 事件数、任务数、Agent 数、审批数
- 按任务过滤
- 按 Agent 过滤
- 事件流甘特图
- 任务级调用图
- 事件表格
- 记忆回放展开项
- 人工审批入口
- 记忆回放导出
- 跨任务长期记忆命中率与审批通过率统计

任务级调用图把 `task_id -> agent_name -> event kind` 连成图，帮助定位某次任务中哪些 Agent 产生了哪些记忆事件。

## 跨任务统计

Dashboard 会基于 timeline 和 approval queue 计算跨任务统计：

- Long-term lookups：多 Agent 构建上下文时发生的长期记忆检索次数。
- Long-term hits：长期记忆返回非空增强上下文的次数。
- Long-term hit rate：`hits / lookups`，用于观察长期记忆是否真的帮助了多 Agent 任务。
- Approval total：进入人工审批队列的长期记忆候选数。
- Approval pending：尚未处理的候选数。
- Approval pass rate：`approved / (approved + rejected)`，用于观察工程经验沉淀质量。

统计还会按 `task_id`、`agent_name` 展示长期记忆命中率，并按 `memory_kind`、`status` 展示审批分布。这样可以区分“长期记忆没有命中”与“长期记忆命中了但被下游 Agent 忽略”的不同问题。

## Memory Replay Export

dashboard 支持导出当前筛选后的记忆回放：

```text
task/memory_replay_exports/<timestamp>_<task_id>_memory_replay.json
task/memory_replay_exports/<timestamp>_<task_id>_memory_replay.md
```

JSON 文件用于程序化分析；Markdown 文件用于保存调试报告或人工复盘。

## 与单 Agent 模式的关系

- 单 Agent 模式：保持原有设计，`ReactAgent` 自己管理长期记忆工具和自动提取。
- 多 Agent 模式：由 Orchestrator 持有 `MultiAgentMemoryHub`，RAGAgent 和 CodeAgent 通过 hub 获取上下文、记录事件和提交受控写入。

两个模式不会互相破坏。多 Agent 的 timeline、approval queue 和 replay export 都是新增数据面，不改变单 Agent 的记忆存储协议。

## 后续扩展

- 为 `engineering_experience` 增加可选 LLM 判别器，作为人工审批前的二级打分。
- 允许 profile 配置不同的人工审批策略，例如医疗、法律默认更严格，教学和数据分析可更宽松。
