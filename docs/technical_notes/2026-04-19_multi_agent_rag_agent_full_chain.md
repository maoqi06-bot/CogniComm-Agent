# 多 Agent 模式下 RAG Agent 完整链路修正

日期：2026-04-19

## 背景

项目最初的单 Agent 设计中，RAG 以工具形式挂载到 ReactAgent：

1. RAG 工具负责检索并返回 chunks、来源和相关度。
2. 同一个 ReactAgent 继续基于工具返回内容完成推理、代码编写或文档生成。

在这种模式下，RAG 工具只是“检索增强输入”的提供者，因此返回原始检索结果是可以接受的。

但多 Agent 模式中，RAG 已经变成独立的 RAGAgent。如果它仍然只返回 chunks，就会导致：

- RAGAgent 没有完成标准 RAG 的 generation 阶段。
- Ragas 的 `answer` 字段被错误地写成检索结果 dump。
- CodeAgent 需要知识时可能继续绕过 RAGAgent 调 RAG 工具，破坏职责边界。

## 本次修正

### 1. 保留单 Agent 模式

单 Agent 模式仍保持原有工作方式：

- RAG 通过 skill 工具触发。
- RAG 工具可以通过本地 skill 或 MCP server 挂载。
- 单 Agent 同时具备 RAG、coding、文件操作等能力。

### 2. 多 Agent 模式中 RAGAgent 完成完整 RAG 链路

多 Agent 的 RAGAgent 现在执行：

1. 调用已注册的 RAG skill retriever 检索上下文。
2. 汇总 top contexts。
3. 调用 LLM 基于上下文生成综合答案。
4. 将 generated answer 写入 `rag_eval_samples.answer`。
5. 将检索 chunks、分数和来源写入 `contexts`、`context_scores`、`context_sources`。

这样 Ragas 评估的是：

```text
question + retrieved contexts + generated answer
```

而不是：

```text
question + retrieved contexts + raw retrieval dump
```

### 3. 多 Agent 模式中 CodeAgent 不再挂载 RAG 工具

CodeAgent 新增 `allow_rag_tools` 配置。

- 单 Agent/兼容模式默认仍允许挂载 RAG 工具。
- Orchestrator 创建多 Agent CodeAgent 时设置 `allow_rag_tools=False`。

这保证 CodeAgent 的职责集中在代码、文件、实验和测试，不再绕过 RAGAgent 重复检索。

### 4. 子任务依赖结果注入

调度器原本只保证依赖任务的执行顺序，但不会把上游结果传递给下游任务。

现在 Orchestrator 会在执行依赖任务前，将已完成的上游结果附加到当前子任务描述中：

- 文档/分析/代码任务可以直接使用 RAGAgent 的综合答案。
- CodeAgent 不需要重新检索。
- 多 Agent 协作中的信息流更加清晰。

### 5. Docker 默认开启

多 Agent 模式默认启用 Docker 隔离执行代码。

DockerRunner 现在会把当前项目工作区挂载到容器内的 `/workspace`，并设置 `PYTHONPATH=/workspace`。因此 CodeAgent 在 Docker 中运行 Python 片段时，可以导入和测试当前项目代码，而不是只能执行孤立的临时脚本。

多 Agent 模式下的 `run_python`、`run_shell`、`run_tests` 都会被包装为 Docker 版本，避免 CodeAgent 通过 shell 或测试工具绕过容器执行。

命令行仍可使用：

```bash
--no-docker
```

显式关闭 Docker。

## 当前边界

- 单 Agent：RAG 是工具，最终生成由同一个 Agent 完成。
- 多 Agent：RAGAgent 是完整 RAG Agent，负责检索和生成；CodeAgent 不直接使用 RAG 工具。

这个边界是后续评估 RAG 生成质量和检索质量的基础。
