# 多 Agent 记忆清理与 Dashboard 管理

## 背景

本次排查发现，多 Agent 记忆系统存在几个明显问题：

- `memory_metadata.json` 中多条记忆的优先级和重要性分数高度相似。
- 大量记忆的 `created_at` / `updated_at` 几乎一致，说明恢复或写入时丢失了真实时间信息。
- 部分记忆内容乱码或价值很低，却被恢复到了长期记忆。
- timeline 中出现 `unknown_agent`，说明子任务结果没有稳定记录真实执行 Agent。

这些问题会污染后续检索上下文，使多 Agent 调度拿到低质量记忆，进而影响任务分解、RAG 检索和最终回答质量。

## 已完成修复

### 记忆写入与恢复质量门

- 多 Agent 写长期记忆前增加质量过滤：
  - 拒绝乱码内容；
  - 拒绝低信息量内容；
  - 拒绝无有效完成结果的 task summary。
- 从向量库恢复长期记忆时增加恢复过滤：
  - 不再恢复明显乱码；
  - 不再恢复低价值事件；
  - 不再把已经标记为 `recovered_from_vector_store` 的旧污染数据继续复活。
- 向量 metadata 中补齐：
  - `importance_score`
  - `source`
  - `tags`
  - `created_at`
  - `updated_at`
  - `last_accessed`
  - `is_pinned`
  - `decay_factor`

这样即使 `memory_metadata.json` 过期或缺失，也不会在恢复时把所有记忆退化成默认时间、默认优先级和默认重要性。

### Agent 名称修复

- 调度器在派发子任务时写入真实 `agent.name`。
- 记录子任务结果时，如果历史对象缺少 `agent_name`，会按任务类型兜底：
  - `knowledge_query` -> `rag_agent`
  - `code_execution` / `analysis` -> `code_agent`
  - 其他 -> `orchestrator`

后续 timeline 不应再出现新的 `unknown_agent`。

### Dashboard 管理能力

Dashboard 的 `Multi-Agent Memory` 页面新增长期记忆管理区域：

- 展示长期记忆 metadata；
- 支持按 category、source、tag 和关键词筛选；
- 支持删除单条记忆；
- 支持按类别批量删除；
- 支持输入 `RESET MEMORY` 后将所有历史记忆隔离到 `data/memory_quarantine/<timestamp>/` 并重建空目录。

## 历史数据处理策略

旧记忆不直接永久删除，而是先隔离再重建：

- `dm_agent/data/memory/*`
- `data/multi_agent_memory_timeline.jsonl`
- `data/multi_agent_memory_approvals.json`

隔离后的旧数据保留在 `data/memory_quarantine/<timestamp>/`，便于必要时回看问题来源。新的长期记忆目录会重新创建为空目录，后续任务会从干净状态开始写入。

## 验证

已新增并运行测试：

```powershell
python -m unittest discover -s test -p test_multi_agent_memory_quality.py
python -m unittest discover -s test -p test_dashboard_memory_admin.py
```

覆盖场景包括：

- 子任务结果不会再写成 `unknown_agent`；
- 低价值 task summary 不写入长期记忆；
- 向量恢复时保留真实 metadata；
- 乱码向量内容不会被恢复；
- metadata 文件缺失或损坏时 dashboard helper 返回空结果；
- 全量 reset 只有确认文本匹配时才执行；
- 隔离重建会移动旧文件并保留目录结构。

## 后续建议

- 在下一次多 Agent 任务后检查 timeline，确认不再出现新的 `unknown_agent`。
- 观察新写入的长期记忆是否具备合理的 priority / importance 分布。
- 后续如需更精细治理，可在 dashboard 增加编辑、导入/导出和按 source 批量清理能力。
