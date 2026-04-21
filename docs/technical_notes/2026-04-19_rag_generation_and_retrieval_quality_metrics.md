# RAG 生成质量与检索质量评估完善

日期：2026-04-19

## 问题

此前 `ragas_report.json` 主要输出 `faithfulness` 和 `answer_relevancy`。这两个指标更适合评估“RAG 生成回答”的质量：

- `faithfulness`：回答是否能被检索上下文支持。
- `answer_relevancy`：回答是否围绕该次 RAG query。

但当前无线通信 RAG 工具很多时候返回的是原始检索片段列表，而不是经过综合生成的答案。因此 `answer_relevancy` 偏低并不一定说明检索结果差，可能只是因为返回文本包含大量来源、元数据和长片段。

## 完善方向

本次将评估拆成两条线：

1. 生成质量：
   - 继续使用 Ragas 的 `faithfulness` 和 `answer_relevancy`。
   - 增加 `generation_mode`，区分 `generated_answer` 和 `raw_retrieval_dump`。

2. 检索质量：
   - 在 trace 的 `rag_eval_samples` 中保存 `context_scores` 和 `context_sources`。
   - evaluator 计算本地、确定性的检索指标：
     - `context_count`
     - `avg_context_score`
     - `max_context_score`
     - `min_context_score`
     - `source_count`
     - `context_query_overlap_max`
     - `context_query_overlap_mean`
     - `retrieval_quality`
     - `weak_retrieval`

## 指标解释

`retrieval_quality` 是一个轻量健康度指标：

- 如果有 reranker 分数，则优先使用 `max_context_score`，并加入 query 与上下文的词面重叠。
- 如果没有 reranker 分数，则退化为 query 与上下文的最大词面重叠。

该指标不是严格学术评测指标，不替代人工标注的 Recall@K、MRR 或 nDCG；它的定位是 dashboard 上的工程诊断信号，用来快速发现“这次检索召回是否明显跑偏”。

## Dashboard 展示

Dashboard 的 Ragas 页面现在分为：

- 生成质量：Faithfulness、Answer relevancy、样本数、原始检索返回占比。
- 检索质量：Retrieval quality、Query-context overlap、Max rerank score、Weak retrieval 数量。

这样可以避免把“RAG 工具返回原始片段导致回答相关度低”和“检索召回本身不相关”混在一起。

## 后续建议

如果希望得到更可靠的检索质量评估，可以继续补充人工标注集：

- 每个 query 标注 1 到多个相关文档或 chunk。
- 计算 Recall@K、MRR、nDCG。
- 用这些指标校准当前的 `retrieval_quality` 工程健康度。

