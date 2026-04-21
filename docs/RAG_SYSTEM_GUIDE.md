# CogniComm-Agent RAG System Guide

> 面向用户与开发者的详细说明文档  
> 适用于单 Agent 模式、P2 多 Agent 模式，以及基于 Skill / MCP 的领域知识库扩展

---

## 1. 为什么需要这份文档

CogniComm-Agent 的 RAG 系统不是一个孤立的小模块，而是同时服务于两类能力：

1. **知识库 RAG**
   - 用于论文、代码、说明文档、工程资料的检索与生成
   - 典型场景：ISAC、RIS、MIMO、UAV、深度学习等领域问答与科研支持

2. **记忆型 RAG**
   - 用于长期记忆、用户偏好、项目上下文、实现经验的存取
   - 典型场景：记住用户习惯、保留任务摘要、跨任务复用工程经验

这两条链共享了“向量化 + 检索 + 排序”的底层思想，但它们的**数据来源、写入策略、调用入口、使用边界**都不完全一样。这个文档就是用来把这些区别和设计讲清楚。

---

## 2. RAG 系统在项目中的定位

在 CogniComm-Agent 中，RAG 主要承担四类责任：

- **知识增强**：给模型补充外部专业知识，降低幻觉
- **证据检索**：从论文、代码、说明文档中找回可复用片段
- **上下文增强**：把检索到的证据喂给 LLM 做综合生成
- **记忆存取**：把跨任务的重要信息存入长期记忆并在后续任务中检索回来

一句话概括：

> RAG 不是把 chunks 原样贴给用户，而是把相关证据检索出来，再作为模型的增强上下文参与生成。

---

## 3. RAG 的总体架构

RAG 相关实现主要分布在以下目录：

```text
dm_agent/
├─ rag/
│  ├─ document_loader.py
│  ├─ embeddings.py
│  ├─ vector_store.py
│  ├─ retriever.py
│  ├─ rag_mcp_server.py
│  ├─ evaluator.py
│  └─ observer.py
├─ skills/
│  └─ builtin/
│     ├─ base_rag_skill.py
│     └─ wirelessCommSkill.py
├─ memory/
│  ├─ long_term_memory.py
│  └─ memory_manager.py
└─ multi_agent/
   ├─ runtime.py
   └─ memory.py
```

对应的逻辑链路可以理解为：

```text
原始数据
  -> 文档加载与切块
  -> Embedding 向量化
  -> FAISS 持久化
  -> Dense / BM25 / Hybrid 检索
  -> RRF 融合
  -> Cross-Encoder 精排
  -> 结果注入 LLM 上下文
  -> 综合生成答案
```

---

## 4. 每个模块是如何设计的

### 4.1 `document_loader.py`：文档加载与切块

核心类：`AdvancedDocumentLoader`

主要职责：

- 识别并加载支持的文件类型
- 按文件类型选择更合适的切块策略
- 对每个 chunk 注入最小上下文元信息

当前支持的文件类型：

- `.txt`
- `.md`
- `.py`
- `.json`
- `.csv`
- `.pdf`

设计要点：

1. **代码和自然语言分开处理**
   - Python 文件使用 `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`
   - 非代码文本使用默认 `RecursiveCharacterTextSplitter`

2. **默认切块参数偏科研文本**
   - `chunk_size = 1500`
   - `chunk_overlap = 200`

3. **Context Injection**
   - 每个 chunk 开头会补一段元信息，例如：
     - 文件名
     - 文件类型
   - 这能提升“孤立 chunk”在检索阶段的可解释性和召回质量

4. **PDF 被转成页级文本**
   - 每页会加 `--- Page x ---`
   - 方便后续定位证据来源

为什么这样设计：

- 科研论文、技术文档和代码的结构差异很大，统一切块往往会损失信息
- 给 chunk 注入轻量元信息，能让 dense retrieval 更容易理解它来自哪里

---

### 4.2 `embeddings.py`：Embedding 封装

核心类与函数：

- `BaseEmbeddings`
- `OpenAIEmbeddings`
- `create_embeddings(...)`
- `resolve_embedding_provider()`
- `resolve_embedding_model()`
- `resolve_embedding_api_key()`
- `resolve_embedding_base_url()`

主要职责：

- 封装统一的 embedding 接口
- 从环境变量解析 embedding 相关配置
- 让知识库 RAG 和记忆系统都能共享一致的 embedding 层

当前默认行为：

- 默认 `EMBEDDING_PROVIDER = openai`
- 默认模型回到兼容旧链路的 `text-embedding-ada-002`
- 支持独立配置：
  - `EMBEDDING_PROVIDER`
  - `EMBEDDING_API_KEY`
  - `EMBEDDING_BASE_URL`
  - `EMBEDDING_MODEL`
  - `EMBEDDING_DIMENSION`

设计意图：

- **主 LLM 和 Embedding 解耦**
  - 你可以用 `deepseek-chat` 做主推理
  - 但 embedding 仍然走 OpenAI-compatible 网关

- **兼容历史项目**
  - 早期系统大量依赖 `text-embedding-ada-002`
  - 为了兼容已有向量库与网关行为，默认值保留旧模型

---

### 4.3 `vector_store.py`：FAISS 向量存储

核心类：`FAISSVectorStore`

主要职责：

- 创建并持久化 FAISS 索引
- 保存 chunk 元数据映射
- 提供 dense retrieval

支持的索引类型：

- `Flat`
- `IVF`
- `HNSW`

支持的距离度量：

- `cosine`
- `l2`

当前默认设计偏向：

- 简洁稳定
- 小中型科研知识库可直接使用
- 通过 `id_to_chunk` 和 `position_to_id` 保留 chunk 映射关系

保存内容包括：

- `*.faiss`
- `*.meta.json`

其中 `meta.json` 里保存：

- 向量维度
- 索引类型
- 距离度量
- chunk 内容
- chunk 元数据
- 向量位置到 chunk ID 的映射

这样设计的好处：

- 不只保存向量，还保存“检索结果如何还原回原始 chunk”
- 方便 HybridRetriever、Dashboard 和长期记忆恢复逻辑使用

---

### 4.4 `retriever.py`：检索器与排序优化

核心类：

- `Retriever`
- `HybridRetriever`
- `CrossEncoderReranker`
- `LazyReranker`
- `SharedRerankerRegistry`

#### 4.4.1 Dense Retriever

`Retriever` 直接调用 `FAISSVectorStore.search(...)`：

- 适合语义检索
- 对近义表达、抽象概念、重述问题更友好

#### 4.4.2 HybridRetriever

`HybridRetriever` 是当前工业级主路径，包含：

1. **Dense Recall**
   - 通过 embedding + FAISS 召回

2. **Sparse Recall**
   - 通过 BM25 召回

3. **RRF 融合**
   - Reciprocal Rank Fusion
   - 合并 dense 与 sparse 的排序结果

4. **Cross-Encoder 精排**
   - 对 coarse recall 的候选结果做更精确的相关性打分

这套流程的价值在于：

- Dense 检索擅长语义相似
- BM25 擅长关键词、公式名、缩写、专有名词
- RRF 可以平衡两者优势
- Reranker 在 Top-K 上进一步提高质量

#### 4.4.3 为什么科研场景需要 Hybrid

科研文本有几个特点：

- 专有名词多
- 数学符号或缩写多
- 同一个概念有多种表达方式

只做 dense，可能漏掉精确术语；只做 BM25，又会忽略语义近似。Hybrid 是目前更稳的折中。

#### 4.4.4 Reranker 的优化设计

项目里对 reranker 做了几个工程优化：

- `LazyReranker`：后台异步加载，避免初次查询阻塞太久
- `SharedRerankerRegistry`：跨 Skill / 实例复用同一个模型
- `RAG_PRELOAD_RERANKER=true` 时提前预热

这样可以避免：

- 多个知识库重复加载同一个 reranker
- 首次查询时体验过差

---

### 4.5 `base_rag_skill.py`：本地 Skill 驱动的 RAG

核心类：`BaseRAGSkill`

这是单 Agent 模式和部分本地 RAG Skill 的主要入口。

它负责：

- 构造 skill 元数据
- 初始化 embedding / vector store / retriever
- 同步知识库目录与索引
- 提供 `xxx_search` 工具给 Agent 调用
- 在检索阶段写入 trace 和 RAGAS 样本

#### 它的默认目录约定

```text
dm_agent/data/knowledge_base/<data_subdir>
dm_agent/data/indices/<index_subdir>
```

例如一个 skill 配置：

```json
{
  "skill_id": "wireless_comm",
  "data_subdir": "wirelessComm",
  "index_subdir": "wireless_idx"
}
```

就会对应：

- 原始数据目录：`dm_agent/data/knowledge_base/wirelessComm`
- 索引目录：`dm_agent/data/indices/wireless_idx`

#### 它的知识库同步逻辑

`_sync_knowledge_base()` 做的是增量同步，而不是每次全量重建。

它会：

1. 扫描数据目录
2. 计算文件哈希
3. 对比旧 `manifest.json`
4. 找出新增 / 修改文件
5. 找出被删除文件
6. 更新 FAISS 索引
7. 保存新的 `manifest.json`
8. 保存 `index_stats.json`

这样设计的好处：

- 知识库增量更新更快
- 避免每次都全量重建索引
- Dashboard 可以显示当前分块数量

---

### 4.6 `rag_mcp_server.py`：MCP 方式的 RAG 中台

这是让 RAG 变成“可被远程调用的知识服务”的关键模块。

主要职责：

- 通过 MCP 暴露 `search` 和 `initialize_expert_context`
- 按领域创建多个 RAG 实例
- 支持把检索 trace 追加到主任务 trace

它的关键设计点：

1. **按领域创建 RAG 实例**
   - `skill_id`
   - `data_subdir`
   - `index_subdir`

2. **路径灵活性**
   - 支持显式 `data_path`
   - 支持 `data_subdir`
   - 支持回退到 `dm_agent/data/knowledge_base`

3. **检索与主任务 trace 对接**
   - MCP 查询可以带 `trace_id`
   - 检索节点会回写到主 trace 文件

4. **适合多 Agent**
   - 多 Agent 下，RAGAgent 可以把它当“知识服务中台”使用

---

### 4.7 `memory/long_term_memory.py` 与 `memory_manager.py`：记忆型 RAG

这部分和知识库 RAG 很像，但目标不一样。

#### 知识库 RAG

- 数据源：论文、代码、文档、资料库
- 写入方式：人工整理后放入数据目录
- 目标：回答专业问题、生成报告、辅助实现

#### 记忆型 RAG

- 数据源：对话、任务摘要、用户偏好、工程经验
- 写入方式：程序自动提取、审批后写入
- 目标：增强连续任务中的上下文记忆和长期偏好

`MemoryManager` 负责：

- 从对话里抽取可长期保存的信息
- 读取长期记忆为当前任务增强上下文
- 根据分类检索偏好、项目事实、技能经验

长期记忆数据目录：

```text
dm_agent/data/memory/
├─ memory_index.faiss
├─ memory_index.meta.json
└─ memory_metadata.json
```

也就是说：

> 知识库 RAG 和记忆型 RAG 底层都用向量检索，但前者面向外部资料，后者面向内部经验与上下文。

---

## 5. RAG 是如何优化的

CogniComm-Agent 当前的 RAG 优化不是单点技巧，而是一整条链路的工程优化。

### 5.1 检索前优化

- 按文件类型切块
- 上下文注入
- 递归目录扫描
- 增量索引同步

### 5.2 召回阶段优化

- Dense + BM25 双路召回
- Top-K recall 扩大为最终返回数量的若干倍

### 5.3 排序阶段优化

- RRF 融合排序
- Cross-Encoder 精排
- Reranker 共享与后台预热

### 5.4 工程体验优化

- `manifest.json` 跟踪知识库变化
- `index_stats.json` 供 Dashboard 读取
- trace 记录 recall / rerank / synthesis 过程
- Dashboard 手动触发 RAGAS 评估，避免阻塞主任务

### 5.5 多 Agent 语义边界优化

在多 Agent 模式下，RAG 不再只是“工具返回一堆 chunks”，而是：

- `RAGAgent` 负责检索
- `RAGAgent` 负责综合生成
- `CodeAgent` 使用上游结果继续工作

这样可以减少：

- 工具耦合混乱
- `unknown tool` 问题
- 原始检索片段直接污染最终答案

---

## 6. 单 Agent 和多 Agent 中 RAG 的区别

| 维度 | 单 Agent | 多 Agent |
|---|---|---|
| 调用方式 | Agent 直接触发 RAG tool / skill | Orchestrator 路由给独立 RAGAgent |
| RAG 职责 | 检索工具 + 主 Agent 自己组织答案 | RAGAgent 负责完整 retrieve + synthesize |
| Code 与 RAG 边界 | 同一个 Agent 内部混合处理 | RAGAgent 与 CodeAgent 分工清晰 |
| 适用场景 | 快速问答、轻量知识增强 | 复杂科研任务、分解任务、跨子任务依赖 |
| 输出形态 | 可能更接近工具式输出 | 更接近专业综合答案 |

### 单 Agent 中的 RAG

典型流程：

1. 用户提问
2. Agent 识别相关 skill
3. 调用 `xxx_search`
4. 得到检索结果
5. 同一个 Agent 继续完成回答或代码任务

优点：

- 轻量
- 简单
- 响应快

缺点：

- 当任务复杂时，RAG 和代码执行容易搅在一起

### 多 Agent 中的 RAG

典型流程：

1. `OrchestratorAgent` 分解任务
2. 知识子任务交给 `RAGAgent`
3. `RAGAgent` 调用领域知识库检索
4. `RAGAgent` 基于证据综合生成答案
5. `CodeAgent` 使用上游知识继续实现

优点：

- 分工明确
- 更适合科研任务
- 更容易做评估和调试

缺点：

- 架构更复杂
- 依赖更完整的调度和记忆系统

---

## 7. 用户最关心的问题：如何把自己的原始数据放进系统

这是这次开源版本最需要补充清楚的一点。

### 7.1 推荐的数据目录

请把你自己的原始知识库数据放在：

```text
dm_agent/data/knowledge_base/<your_domain_name>/
```

例如你要增加一个医疗领域知识库，可以这样放：

```text
dm_agent/data/knowledge_base/medical/
├─ papers/
│  ├─ paper_01.pdf
│  └─ paper_02.pdf
├─ notes/
│  ├─ glossary.md
│  └─ diagnosis_rules.txt
└─ code/
   └─ baseline.py
```

系统会递归扫描这个目录下的支持格式文件。

### 7.2 支持放入哪些原始文件

推荐直接放这些格式：

- `pdf`
- `md`
- `txt`
- `py`
- `json`
- `csv`

如果你的原始资料是：

- Word：建议先转成 `pdf` 或 `md`
- 网页：建议保存成 `md` / `txt`
- 扫描件：建议先 OCR，再进入知识库

### 7.3 索引目录会放在哪里

索引会放在：

```text
dm_agent/data/indices/<your_index_name>/
```

并生成：

- `manifest.json`
- `index_stats.json`

以及同名的：

- `<your_index_name>.faiss`
- `<your_index_name>.meta.json`

### 7.4 如何让系统知道这套数据属于哪个领域

你需要在 skill 配置或 MCP 初始化配置中指定：

- `skill_id`
- `data_subdir`
- `index_subdir`

例如：

```json
{
  "skill_id": "medical",
  "display_name": "Medical Expert KB",
  "data_subdir": "medical",
  "index_subdir": "medical_idx",
  "top_k": 5,
  "threshold": 0.4,
  "use_hybrid": true,
  "use_reranker": true
}
```

这表示：

- 原始数据目录：`dm_agent/data/knowledge_base/medical`
- 索引目录：`dm_agent/data/indices/medical_idx`

### 7.5 第一次加载会发生什么

首次访问该领域时，系统会自动：

1. 扫描目录
2. 加载文件
3. 切块
4. 向量化
5. 建立 FAISS 索引
6. 建立 BM25 索引
7. 保存 `manifest.json`
8. 保存 `index_stats.json`

如果之后你新增或替换文件：

- 系统会根据哈希做增量更新
- 不需要每次手工全量重建

### 7.6 一个实际建议

如果你是做科研领域知识库，建议按下面方式组织：

```text
dm_agent/data/knowledge_base/wirelessComm/
├─ papers/         # 论文 PDF
├─ notes/          # 读书笔记、术语表、方法对比
├─ code/           # 小段示例代码或伪代码
└─ datasets/       # 仅放说明文档，不建议直接塞大二进制原始数据
```

这样会比所有文件平铺在一个目录里更易维护。

---

## 8. Skill 触发的两种 RAG 协议

CogniComm-Agent 当前存在两条 Skill 触发 RAG 的路径，它们的配置协议不一样，不能混用。

### 8.1 本地直连型 RAG Skill

这条路径由 `BaseRAGSkill` 驱动。

特点：

- Skill 自己在本地初始化 embedding、向量库和检索器
- Skill 自己扫描 `dm_agent/data/knowledge_base/<data_subdir>/`
- Skill 自己维护 `dm_agent/data/indices/<index_subdir>/`
- 适合单 Agent 或希望直接挂本地知识库的场景

配置示例：

- `dm_agent/skills/custom/examples/local_rag_skill.json`

关键字段：

- `type = "rag"`
- `skill_id`
- `display_name`
- `index_subdir`
- `data_subdir`
- `top_k`
- `threshold`
- `use_hybrid`
- `use_reranker`

### 8.2 MCP 驱动型 RAG Skill

这条路径由 `GenericMCPSkill` 或类似的 MCP Skill 驱动。

特点：

- Skill 自己不建本地索引
- Skill 只把领域配置发给共享 RAG MCP server
- 真正执行初始化、检索和 trace 追加的是共享 RAG MCP server
- 适合多 Agent 和统一知识中台架构

配置示例：

- `dm_agent/skills/custom/examples/mcp_rag_skill.json`

关键字段：

- `skill_name`
- `display_name`
- `description`
- `rag_server_name`
- `domain_config`
- `prompt_addition`
- `tool_definition`

其中：

- `rag_server_name` 指向同一个共享 RAG MCP server
- `domain_config` 只描述“这是哪个领域、数据在哪、索引在哪、检索参数如何”

### 8.3 重要约定：整个项目共享同一个 RAG MCP server

当前推荐约定是：

> 不同领域的 MCP RAG Skill 不应该各自对应不同 server，而应该都连接到同一个共享 RAG MCP server。

例如：

- `wireless_comm` 领域 skill
- `deep_learning` 领域 skill
- 后续新增的 `medical`、`legal`、`finance` skill

都应当指向同一个 MCP server，例如：

- `wireless-rag`

不同 Skill 真正变化的是：

- Skill 元信息
- Prompt 增强内容
- `domain_config.skill_id`
- `domain_config.data_subdir`
- `domain_config.index_subdir`
- 检索参数

而不是每个领域单独起一台新的 RAG MCP server。

### 8.4 推荐实践

如果你要新增一个领域，优先考虑：

1. 如果只是想增加一个领域知识库并接入多 Agent：
   - 走 MCP 驱动型 RAG Skill
   - 连接共享 `wireless-rag` server

2. 如果你想做一个纯本地、无需 MCP 的轻量知识库：
   - 走本地直连型 `BaseRAGSkill`

这样更符合当前项目架构，也更不容易把协议弄混。

---

## 9. 如何使用 RAG

### 9.1 单 Agent 使用方式

适合：

- 快速专业问答
- 某个领域的局部知识增强
- 在同一个 Agent 中顺手完成代码与解释

示例任务：

```text
检索 ISAC 波束赋形的 WMMSE 思路，并给出一个最简单的 Python 示例
```

Agent 会：

- 激活相关 skill
- 调用 `xxx_search`
- 将检索结果用于回答

### 9.2 多 Agent 使用方式

适合：

- 检索 + 推导 + 编码 + 文档整理
- 有多个子任务依赖的复杂任务

示例任务：

```text
检索 ISAC 与 RIS 波束赋形的相关内容，实现一个最简单的混合波束赋形 HBF 的推导和 Python 实现，结果放入 task 文件夹中
```

系统会：

1. 分解知识子任务和代码子任务
2. `RAGAgent` 完成知识查询和综合生成
3. `CodeAgent` 基于上游结论实现代码和文档

### 9.3 Dashboard 中如何观察 RAG

Dashboard 提供：

- 检索链路诊断
- 上下文查看
- recall / rerank 对比
- RAGAS 指标图表

如果你想看：

- 检索召回质量
- 是否 rerank 起了作用
- 某个 query 的上下文长什么样

就去 Dashboard 的 RAG 页面。

---

## 10. RAG 用来做什么：知识库 vs 记忆

### 10.1 知识库 RAG

面向外部资料：

- 论文
- 文档
- 代码样例
- 技术说明

主要解决：

- “系统知道什么”
- “专业知识从哪来”

### 10.2 记忆型 RAG

面向内部经验：

- 用户偏好
- 项目背景
- 调试经验
- 历史任务摘要

主要解决：

- “系统还记得什么”
- “跨任务如何延续上下文”

### 10.3 两者关系

可以把它们理解成：

- 知识库 RAG：外脑
- 记忆型 RAG：长期记事本

两者都重要，但不要混淆。

---

## 11. 常见问题与排查

### 11.1 为什么 RAG 报 embedding 401？

通常说明 embedding 用的：

- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`

这组三元配置不匹配。

请优先检查：

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=https://sg.uiuiapi.com/v1/
EMBEDDING_MODEL=text-embedding-ada-002
```

### 11.2 为什么 RAGAgent 找不到结果？

常见原因：

- 原始数据没有放到正确目录
- 目录名和 `data_subdir` 不一致
- 文件格式不被支持
- 阈值过高
- embedding 没建成功

### 11.3 为什么 index 里有文件但命中率还是低？

可能原因：

- chunk 切得太碎或太长
- 文档标题、术语和正文不统一
- 用户 query 与文档表达差异过大
- reranker 没加载成功

### 11.4 为什么最终回答里不直接输出所有 chunks？

因为正确的 RAG 是“检索增强生成”，不是“原文拼接返回”。  
chunks 是证据，不是最终用户答案。

---

## 12. 推荐的接入流程

如果你要给 CogniComm-Agent 新增一个自己的领域知识库，建议按下面走：

### 步骤 1：准备原始数据

把论文、笔记、代码、术语表整理好，统一放到：

```text
dm_agent/data/knowledge_base/<domain_name>/
```

### 步骤 2：定义领域配置

指定：

- `skill_id`
- `display_name`
- `data_subdir`
- `index_subdir`

### 步骤 3：接入 Skill 或 MCP

你可以选择：

- 本地 `BaseRAGSkill` 派生方式
- MCP `initialize_expert_context` 方式

### 步骤 4：首次运行自动建索引

系统会自动：

- 切块
- 向量化
- 存索引
- 建 BM25

### 步骤 5：到 Dashboard 检查效果

确认：

- 分块数是否正确
- recall 是否正常
- rerank 是否生效
- RAGAS 指标是否合理

---

## 13. 推荐阅读

如果你还想继续往深处看，建议配合这些文档一起读：

- [`README.md`](../README.md)
- [`ENGINEERING_P2.md`](../ENGINEERING_P2.md)
- [`docs/technical_notes/2026-04-19_dashboard_triggered_ragas_evaluation.md`](./technical_notes/2026-04-19_dashboard_triggered_ragas_evaluation.md)
- [`docs/technical_notes/2026-04-19_rag_generation_and_retrieval_quality_metrics.md`](./technical_notes/2026-04-19_rag_generation_and_retrieval_quality_metrics.md)

---

## 14. 总结

CogniComm-Agent 的 RAG 系统不是一个简单的“向量搜索接口”，而是一整套围绕科研与工程任务设计的知识增强基础设施：

- 它有本地 Skill 路径，也有 MCP 中台路径
- 它既支持知识库，也支持长期记忆
- 它既考虑检索质量，也考虑工程可维护性
- 它既服务单 Agent，也服务多 Agent

如果你打算把它真正用起来，最重要的第一步不是调 prompt，而是：

> 把你的原始领域数据按目录规范整理好，放进 `dm_agent/data/knowledge_base/<your_domain_name>/`，然后让系统围绕这份数据建立自己的知识底座。
