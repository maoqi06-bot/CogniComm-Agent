# 2026-05-01 智能增量记忆管理改造

## 背景

长期记忆系统原先已经支持记忆抽取、向量检索、访问计数、重要性衰减和手动更新工具，但在新记忆写入时缺少真正的增量管理。

典型问题是：当系统抽取到一条与历史记忆相似的新记忆时，旧逻辑主要依赖内容 hash 去重或后续简单 consolidate。它无法判断：

- 新记忆是否与旧记忆矛盾。
- 新记忆是否只是对旧记忆的补充。
- 新记忆是否应该替换旧偏好或旧项目事实。
- 新记忆是否虽然相似但属于独立主题。

这会导致记忆库逐渐堆积重复、过期或互相冲突的内容，后续检索时给 Agent 注入不稳定上下文。

## 本次修改

### 1. 新增集中式记忆提示词

新增文件：

```text
dm_agent/prompts/memory_prompts.py
```

集中管理三类提示词：

- `build_memory_extraction_prompt()`：从对话中抽取长期记忆。
- `build_memory_resolution_prompt()`：判断新旧记忆关系并输出结构化决策。
- `build_memory_guidance_prompt()`：注入给 Agent 的长期记忆工具使用指南。

`dm_agent/prompts/__init__.py` 已导出这些构建函数，`ReactAgent` 不再直接维护长期记忆指南文本。

### 2. 新增智能写入流程

`MemoryManager.add_memory()` 和 `extract_and_store()` 现在统一走：

```python
add_or_update_memory(...)
```

写入流程变为：

1. 构造候选记忆并过滤低价值运行/工具错误。
2. 按内容和类别检索相似历史记忆。
3. 精确重复时只更新访问计数，不重复入库。
4. 有相似候选且存在 LLM client 时，调用 `resolve_memory_conflict()`。
5. 根据 LLM JSON 决策执行：
   - `create_new`
   - `update_existing`
   - `replace_existing`
   - `ignore`

新增默认配置：

```python
smart_memory_update_enabled = True
memory_resolution_top_k = 5
memory_resolution_min_similarity = 0.72
memory_resolution_temperature = 0.0
```

### 3. 增强记忆更新元数据

当新记忆合并或替换旧记忆时，会保留决策轨迹：

- `memory_update_history`
- `last_merge_reason`
- `resolved_from_memory_ids`
- `superseded_by`

更新后的记忆会合并标签、继承较高重要性，并保留固定状态等关键信息，便于后续审计和调试。

### 4. 修复向量索引一致性

`LongTermMemoryStore.update(content=...)` 原先只更新 `id_to_chunk`，没有真正刷新 FAISS 向量，因此可能出现“文本已更新，但向量仍按旧内容检索”的问题。

本次改为在内容变更或删除后，从 `_memory_index` 这个权威数据源重建向量索引：

```python
_rebuild_all_vectors()
```

这样后续检索会命中新内容，不再被旧向量污染。

### 5. 降低可选依赖对记忆测试的影响

为了让记忆模块在没有完整 RAG 依赖的环境中也能导入和测试：

- `dm_agent/rag/vector_store.py` 将 `faiss` 改为延迟失败。
- `SkillManager` 和多 Agent runtime 对 `BaseRAGSkill` 做可选导入。

真实创建 FAISS 索引时仍需要安装 `faiss-cpu`，但纯记忆数据模型和智能写入测试不再被 RAG 技能依赖阻塞。

## 测试

扩展 `test/test_long_term_memory.py`，覆盖：

- 重复记忆不重复入库。
- 补充型新记忆更新旧记忆。
- 矛盾偏好由新记忆替换旧记忆。
- 相似但独立的新记忆正常新增。
- LLM 返回非法 JSON 时保守新增。
- 无 LLM client 时精确重复仍能去重。
- 内容更新后向量 chunk 反映新内容。

验证命令：

```bash
python -m pytest test/test_long_term_memory.py
```

结果：

```text
18 passed, 1 skipped
```

## 后续建议

当前改造只影响后续写入，不迁移或清洗已有历史记忆。后续可以基于同一套 `resolve_memory_conflict()` 逻辑做一次离线记忆整理任务，用于清理已有重复、矛盾或过期记忆。
