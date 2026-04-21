# Ragas 评估口径：RAG 查询级而不是总任务级

## 问题

多 Agent 任务通常包含多类行为：

- RAG 检索
- 代码编写与调试
- 文件组织
- 最终报告汇总

如果把“用户最初的总任务”和“最终汇总报告”直接送入 Ragas 的 `faithfulness` 与 `answer_relevancy`，分数会被非 RAG 行为污染。例如用户要求“检索 ISAC 并完善 WMMSE 代码”，最终回答会包含代码执行状态、文件路径、失败子任务、建议等内容，这些内容不一定来自 RAG 检索上下文，因此忠实度和相关度天然偏低。

## 正确口径

如果目标是评估 RAG 系统，默认样本边界应是一次 RAG 查询：

```text
question = 本次 RAG query
contexts = 本次检索得到的文本块
answer   = RAG 工具或 RAG 子任务基于这些上下文返回的回答
```

这才对应 Ragas 的基本语义：

- `faithfulness`: 回答是否被检索上下文支持。
- `answer_relevancy`: 回答是否切中本次 RAG query。

总任务级评估更适合称为 Agent 端到端评估，不应与 RAG 系统评估混为一谈。

## 本次修复

系统现在默认收集 `rag_eval_samples`：

```json
{
  "question": "本次检索 query",
  "contexts": ["检索片段1", "检索片段2"],
  "answer": "RAG 工具返回内容",
  "eval_scope": "rag_query",
  "source": "wireless_comm_search"
}
```

即时评估和自动评估都优先读取 `metadata.rag_eval_samples`，逐条计算 Ragas 分数。只有显式设置：

```powershell
$env:RAGAS_EVAL_SCOPE = "task"
```

才会回退到“总问题 + 最终答案 + 全部检索上下文”的端到端评估。

## 建议

Dashboard 中应把评估口径显示出来，例如读取 `eval_scope`：

- `rag_query`: RAG 系统质量评估。
- `task_end_to_end`: Agent 总任务端到端评估。

论文或报告中如果引用 Ragas 分数，也应明确说明评估对象是“RAG 检索生成样本”还是“完整 Agent 任务”。
