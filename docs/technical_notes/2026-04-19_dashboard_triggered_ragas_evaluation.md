# Ragas 评估从主流程移到 Dashboard 手动触发

## 背景

多 Agent 任务结束后，系统会立即执行 Ragas 评估。一次任务中可能产生十几条 RAG query 样本，而 Ragas 的 `faithfulness` 和 `answer_relevancy` 都需要调用 LLM 或 embedding API，因此会出现如下问题：

- 用户已经看到“任务完成”，但必须继续等待评估结束才能回到下一次输入。
- 评估阶段的网络波动会影响主任务体验，例如 `APIConnectionError`。
- 多条样本逐条评估时会打印多段进度条，日志看起来像任务仍在执行。

## 修复

主流程 `main.py` 不再自动调用 `RagasObserver.instant_eval()`。任务结束后只保存 trace 和报告，并提示：

```text
Ragas 评估已改为 Dashboard 手动触发，不再阻塞主任务流程。
```

Dashboard 的“自动化评估”页面现在提供两个按钮：

- `后台刷新 Ragas 数据`：启动独立后台进程运行 `python -m dm_agent.rag.evaluator`。
- `仅刷新图表显示`：只重新读取已有 `data/ragas_report.json`。

后台评估进程会写入：

- `data/ragas_report.json`：评估结果。
- `data/ragas_eval_status.json`：评估状态、PID、更新时间。
- `data/ragas_eval.log`：评估日志。

## 关于相关度偏低

最近一次 trace 中已经按 `rag_eval_samples` 采集了 RAG query 级样本，默认没有再用“总任务问题 + 最终答案”做 Ragas 评估。

相关度仍偏低的主要原因是：当前 RAG 工具返回的 `answer` 基本是检索结果原文列表，包括 metadata、来源、论文片段，而不是针对 query 生成的自然语言答案。Ragas 的 `answer_relevancy` 更适合评估“生成答案是否切题”，对 raw retrieval dump 往往偏低。

这说明两件事需要区分：

- `faithfulness` 高：返回内容确实被检索上下文支持。
- `answer_relevancy` 低：检索片段或 raw dump 对 query 的语义回答性不足。

如果后续要专门评估“检索质量”，建议在 Dashboard 中增加 context-level 指标，例如 query-context 相似度、context precision、命中文献覆盖率等，而不是只依赖 answer-level relevancy。
