# CogniComm-Agent

CogniComm-Agent 是一个面向科研开发、代码实现与专业知识检索的 LLM Agent 系统。它保留了单 Agent 模式下的轻量交互体验，同时扩展出多 Agent 协作、领域 RAG、长期记忆、MCP 工具接入、Dashboard 观测与人工审批等完整工程能力，适合通信、AI、软件工程等需要“检索 + 推理 + 编码 + 复盘”的复杂工作流。

本项目当前提供两个主要入口：

- [`main.py`](./main.py)：CLI 主入口，负责单 Agent 与多 Agent 模式运行
- [`dashboard.py`](./dashboard.py)：开发者可视化控制台，负责 RAG 评估、记忆审批、时间线和调试观测

> 当前项目名称为 **CogniComm-Agent**。如果你看到旧文档中的 `DM-Agent` 或 `DM-Code-Agent`，请以本仓库当前实现和本文档为准。

## 1. 项目定位

CogniComm-Agent 解决的是下面这一类任务：

- 需要大模型理解需求并分解任务
- 需要结合外部知识库进行 RAG 检索与生成
- 需要真正落地代码、脚本、实验文档或技术报告
- 需要保留长期记忆、用户偏好和工程经验
- 需要对 RAG 质量、多 Agent 轨迹和运行状态进行可视化追踪

它不是单纯的聊天机器人，也不是只会调用代码工具的脚本包装器，而是一个面向研发流程的智能体系统。

## 2. 核心能力概览

### 2.1 单 Agent 模式

单 Agent 模式适合快速任务、连续问答和轻量代码修改。它具备：

- ReAct 推理循环
- 多模型切换：DeepSeek / OpenAI-compatible / Claude / Gemini
- 默认代码工具集
- MCP 工具挂载
- Skill 专家系统
- 上下文压缩
- 单 Agent 长期记忆

### 2.2 多 Agent 模式

多 Agent 模式是当前工程重点，适合复杂任务和科研型工作流。它包含：

- `OrchestratorAgent`：统一分解、调度和汇总任务
- `RAGAgent`：负责完整的“检索 + 综合生成”链路
- `CodeAgent`：负责代码、文件、实验、测试、文档落地
- 批次调度与依赖管理
- 可选 Docker 隔离执行
- 多 Agent 共享短期记忆与可控长期记忆写入
- 面向 Dashboard 的时间线、审批、回放和 RAG 评估

### 2.3 RAG 与知识系统

- 基于向量检索 + BM25 + Reranker 的混合 RAG
- 可通过 Skill 或 MCP 接入特定领域知识库
- 支持领域化 profile 与 RAG Agent 专业化 prompt
- 支持 Dashboard 手动触发 RAGAS 评估，不阻塞主任务流程

### 2.4 工程化能力

结合 [`ENGINEERING_P0.md`](./ENGINEERING_P0.md)、[`ENGINEERING_P1.md`](./ENGINEERING_P1.md)、[`ENGINEERING_P2.md`](./ENGINEERING_P2.md)，项目目前具备：

- P0：重试机制、日志系统、LLM 调用封装
- P1：资源限流、安全执行、健康检查、资源管理器
- P2：多 Agent 架构、Profile 系统、记忆系统、RAG 评估、Dashboard 集成

## 3. 总体架构

### 3.1 运行分层

1. **入口层**
   - [`main.py`](./main.py)
   - [`dashboard.py`](./dashboard.py)

2. **Agent 层**
   - 单 Agent：`ReactAgent`
   - 多 Agent：`OrchestratorAgent` / `RAGAgent` / `CodeAgent`

3. **工具与技能层**
   - 默认工具集 `dm_agent/tools`
   - MCP 工具管理 `dm_agent/mcp`
   - Skill 系统 `dm_agent/skills`

4. **知识与记忆层**
   - RAG：`dm_agent/rag`
   - 单 Agent 记忆：`dm_agent/memory`
   - 多 Agent 记忆：`dm_agent/multi_agent/memory.py`

5. **工程保障层**
   - 日志与重试：`dm_agent/utils`
   - 资源管理与安全执行：`dm_agent/utils`
   - DockerRunner：`dm_agent/multi_agent/runtime.py`

### 3.2 多 Agent 工作流

多 Agent 模式下的主流程如下：

1. 用户在 CLI 中选择 P2 多 Agent 模式
2. `main.py` 完成 LLM / Embedding / Docker 预检
3. `OrchestratorAgent` 使用 `TaskDecomposer` 生成子任务图
4. `TaskScheduler` 按依赖关系分批执行子任务
5. 知识型子任务由 `RAGAgent` 完成“检索 + 生成”
6. 代码型子任务由 `CodeAgent` 完成代码、测试、文档和实验落地
7. `ResultAggregator` 生成最终报告和任务目录产物
8. 任务轨迹、记忆事件、审批队列和 RAGAS 数据进入 Dashboard

## 4. 目录结构

下面是当前仓库最重要的目录与文件：

```text
CogniComm-Agent/
├─ main.py                          # CLI 主入口
├─ dashboard.py                     # Streamlit Dashboard
├─ README.md
├─ README_EN.md
├─ ENGINEERING_P0.md                # P0 工程说明
├─ ENGINEERING_P1.md                # P1 工程说明
├─ ENGINEERING_P2.md                # P2 多 Agent 工程说明
├─ requirements.txt
├─ package.json                     # MCP Node server package
├─ mcp_config.json.example          # MCP 配置样例
├─ config.json.example              # CLI 配置样例
├─ .env.example                     # 环境变量样例
├─ configs/
│  └─ multi_agent/
│     └─ profiles/                  # 多 Agent profile JSON
├─ dm_agent/
│  ├─ __init__.py                   # 对外导出 API
│  ├─ clients/                      # 多 LLM 客户端
│  ├─ core/                         # ReAct Agent 核心循环
│  ├─ tools/                        # 默认工具集
│  ├─ mcp/                          # MCP 加载与管理
│  ├─ prompts/                      # Prompt 构造
│  ├─ skills/                       # Skill 与内置专家技能
│  ├─ rag/                          # RAG、向量库、评估与 MCP server
│  ├─ memory/                       # 单 Agent 记忆系统
│  ├─ multi_agent/                  # 多 Agent runtime / profiles / memory / toolkits
│  └─ utils/                        # 日志、重试、资源管理、安全执行
├─ docs/
│  └─ technical_notes/              # 开发日志与技术说明
├─ task/                            # 任务产物目录
└─ data/                            # RAGAS、记忆时间线、审批记录等运行数据
```

## 5. 关键模块说明

### 5.1 CLI 入口：`main.py`

`main.py` 是项目的主入口，负责：

- 解析命令行参数
- 加载 `.env` 与持久化配置
- 初始化 MCP 服务器
- 加载 SkillManager
- 创建单 Agent 或多 Agent 运行实例
- 运行多 Agent 预检：
  - 主 LLM 鉴权
  - Embedding 配置展示
  - Docker 可见性检查

交互菜单当前包括：

1. 执行新任务
2. 多轮对话模式
3. 查看可用工具列表
4. 配置设置
5. 查看可用技能列表
6. P2: 多 Agent 协作模式
7. 退出程序

### 5.2 Dashboard：`dashboard.py`

Dashboard 是系统的开发者控制台，负责可视化和审批，不替代 CLI 主任务入口。主要页面包括：

- RAG 运行状态概览
- 知识库内容浏览器
- 检索链路白盒化诊断
- RAGAS 评估结果与图表
- Multi-Agent Memory 页面
  - 长期记忆命中率
  - 用户偏好命中率
  - 人工审批队列
  - 时间线 / 调用图 / 记忆回放导出

### 5.3 `dm_agent/clients`

负责对接不同 LLM 提供商，目前支持：

- `DeepSeekClient`
- `OpenAIClient`
- `ClaudeClient`
- `GeminiClient`

统一通过 `create_llm_client(...)` 创建，并由 `PROVIDER_DEFAULTS` 管理默认模型与 URL。

### 5.4 `dm_agent/core`

核心是 `ReactAgent`，它负责：

- 构造系统 prompt 与工具描述
- 进行 ReAct 循环
- 执行工具调用
- 维护步骤状态与上下文压缩

单 Agent 模式和多 Agent 中的 `CodeAgent` 都会复用这套核心循环。

### 5.5 `dm_agent/tools`

默认工具集覆盖常见研发操作，例如：

- 文件读取、写入、编辑
- AST 分析
- 函数签名提取
- 目录扫描
- shell / python / test 执行等

这些工具是单 Agent 和多 Agent 的基础执行能力。

### 5.6 `dm_agent/mcp`

负责加载 `mcp_config.json` 配置并管理 MCP 服务器连接。当前样例配置见 [`mcp_config.json.example`](./mcp_config.json.example)。

已提供的样例 MCP 服务器包括：

- `wireless-rag`
- `playwright`
- `context7`
- `filesystem`
- `sqlite`
- `github`

注意：`wireless-rag` 当前使用 `${PROJECT_ROOT}/bin/index.js`，避免绝对路径污染开源仓库。

### 5.7 `dm_agent/skills`

Skill 系统用于给 Agent 注入更专业的提示词、规则和工具组合。当前仓库包含：

- 内置代码类技能
- 数据库技能
- 前端开发技能
- Skill 创建技能
- RAG 领域技能
- 通信领域相关技能

Skill 会在单 Agent 中直接参与工具和 prompt 构造，在多 Agent 中则主要为 `RAGAgent` 和 `CodeAgent` 提供专业化上下文。

### 5.8 `dm_agent/rag`

RAG 模块是 CogniComm-Agent 的知识系统核心，包含：

- `document_loader.py`：文档加载与切块
- `embeddings.py`：Embedding 封装与独立配置解析
- `vector_store.py`：FAISS 向量库
- `retriever.py`：混合检索、RRF 融合、重排
- `rag_mcp_server.py`：多领域 RAG MCP 服务器
- `evaluator.py` / `observer.py`：RAGAS 评估和观测

当前设计中，**多 Agent 下的 RAGAgent 负责完整 RAG 链路，而不是只返回检索 chunks。**

### 5.9 `dm_agent/memory`

这是单 Agent 模式下的记忆系统，包含：

- `ContextCompressor`
- `LongTermMemoryStore`
- `MemoryManager`

它负责单 Agent 的上下文压缩、长期记忆写入和检索。

### 5.10 `dm_agent/multi_agent`

这是多 Agent 系统的核心目录，主要包含：

- `runtime.py`
  - `OrchestratorAgent`
  - `RAGAgent`
  - `CodeAgent`
  - `DockerRunner`
  - `TaskDecomposer`
  - `TaskScheduler`
  - `ResultAggregator`
- `profiles.py`
  - `AgentProfile`
  - `CodeAgentProfile`
  - `RAGAgentProfile`
- `profile_loader.py`
- `domain_profiles.py`
- `toolkits.py`
- `memory.py`

#### 关键职责

- `OrchestratorAgent`：全局调度、依赖拼接、结果汇总
- `RAGAgent`：调用领域 RAG skill 完成检索与综合回答
- `CodeAgent`：执行代码、文件、实验、测试、文档任务
- `DockerRunner`：提供隔离执行能力，当前若不可见会自动降级

### 5.11 `dm_agent/utils`

该目录承载 P0 和 P1 工程化能力，包括：

- 日志系统
- 重试机制
- 超时控制
- 资源管理器
- 信号量限流
- 安全 Shell 执行
- 健康检查

## 6. 记忆系统说明

### 6.1 单 Agent 记忆

单 Agent 记忆系统保持独立，不因为多 Agent 扩展而退化。它适合：

- 多轮对话连续上下文
- 重要事实保留
- 简单项目偏好记忆

### 6.2 多 Agent 记忆

多 Agent 记忆系统主要由 [`dm_agent/multi_agent/memory.py`](./dm_agent/multi_agent/memory.py) 提供，具备：

- 共享短期记忆时间线
- 角色化 memory policy
- 长期记忆候选写入
- 用户偏好写入与检索
- 人工审批队列
- Dashboard 回放与导出

当前默认策略中：

- `orchestrator` 负责任务摘要和用户偏好类的全局写入
- `rag_agent` 以读取长期记忆为主
- `code_agent` 主要写工程经验、实现模式、调试经验、实验结果

## 7. Profile 与扩展机制

多 Agent Profile 用来定义不同 Agent 的专业能力、prompt、工具过滤和记忆偏好。当前样例目录：

- [`configs/multi_agent/profiles`](./configs/multi_agent/profiles)

项目支持两类扩展：

1. **新增 Skill**
   - 在 Skill 目录中实现新技能
   - 由 SkillManager 自动装载或手动注册

2. **新增多 Agent Profile**
   - 添加 profile JSON
   - 通过 loader 构造领域化 RAGAgent / CodeAgent 行为

这使得用户可以在不重写 runtime 的前提下，为不同研究领域扩展知识和执行能力。

## 8. 配置说明

### 8.1 环境变量

复制 `.env.example`：

```bash
cp .env.example .env
```

常用配置包括：

```env
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
OPENAI_BASE_URL=https://sg.uiuiapi.com/v1/
OPENAI_API_STYLE=auto

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://sg.uiuiapi.com/v1/
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

DOCKER_PATH=
```

注意：

- `.env` 不应提交到仓库
- `mcp_config.json` 中不应硬编码真实 API key
- 开源时请使用 `mcp_config.json.example`

### 8.2 CLI 持久化配置

程序支持将配置写入本地 `config.json`。优先级通常为：

1. 命令行参数
2. `config.json`
3. 代码默认值
4. 环境变量辅助补充

## 9. 安装与运行

### 9.1 安装依赖

Python 依赖：

```bash
pip install -r requirements.txt
```

Node 依赖（用于本地 MCP server）：

```bash
npm install
```

### 9.2 启动 CLI

```bash
python main.py
```

### 9.3 启动 Dashboard

```bash
streamlit run dashboard.py
```

### 9.4 命令行快速执行

```bash
python main.py --task "分析当前项目结构"
python main.py --task "实现一个最简单的 HBF 示例" --provider openai --model gpt-5
python main.py --task "检索 ISAC 文献并生成实验报告" --multi-agent
```

## 10. 使用方式

### 10.1 单 Agent 模式

适合：

- 代码修改
- 文件操作
- 多轮连续问答
- 快速原型实现

步骤：

1. 运行 `python main.py`
2. 选择“执行新任务”或“多轮对话模式”
3. 输入任务描述
4. 查看生成结果与日志

### 10.2 多 Agent 模式

适合：

- 科研型复杂任务
- 检索 + 推导 + 编码 + 文档整合
- 带有明确知识依赖和产物要求的任务

步骤：

1. 运行 `python main.py`
2. 选择 `6. P2: 多 Agent 协作模式`
3. 输入完整任务描述
4. 系统将自动执行：
   - 预检
   - 任务分解
   - RAG 子任务
   - Code 子任务
   - 汇总和报告生成

### 10.3 Dashboard 中的 RAG 评估

RAGAS 评估现在采用 **Dashboard 手动触发** 模式：

- 不再阻塞 CLI 主任务
- 可后台刷新数据
- 可查看生成质量与检索质量指标

### 10.4 Dashboard 中的记忆审批

多 Agent 写入长期记忆时，部分内容需要人工审批。你可以在 Dashboard 中：

- 查看待审批条目
- 批准写入长期记忆
- 拒绝低价值记录
- 观察跨任务命中率与通过率

## 11. 常见问题

### 11.1 为什么多 Agent 下 CodeAgent 默认禁用 RAG 工具？

因为在 P2 设计里，RAG 由独立 `RAGAgent` 负责完整“检索 + 生成”链路。这样可以避免 CodeAgent 调用领域工具失败、职责混乱或出现 `unknown tool`。

### 11.2 为什么最终生成答案不直接包含原始 chunks 和元数据？

正确的多 Agent RAG 链路应当将检索结果作为上下文增强输入给 RAGAgent，再由 RAGAgent 做综合生成。最终用户面向的是整理后的答案，而不是生硬的检索原文拼接。

### 11.3 为什么 Docker 会自动降级？

当前代码会在启动前检查 Docker 可见性。如果当前 Python 进程无法找到 `docker.exe`，系统会自动降级为非 Docker 模式，避免任务在中途才失败。

### 11.4 为什么有些长期记忆不会立刻出现？

多 Agent 长期记忆存在：

- 质量过滤
- 人工审批
- 异步写入

因此不是所有事件都会立即成为长期记忆。

## 12. 开发建议

如果你准备继续扩展本项目，建议优先阅读：

- [`ENGINEERING_P0.md`](./ENGINEERING_P0.md)
- [`ENGINEERING_P1.md`](./ENGINEERING_P1.md)
- [`ENGINEERING_P2.md`](./ENGINEERING_P2.md)
- [`docs/technical_notes`](./docs/technical_notes)

推荐扩展方向：

- 新领域 RAG Skill
- 新的多 Agent Profile
- 代码执行工具增强
- 记忆模板和审批策略
- Dashboard 可视化能力

如果你想专门了解 RAG 的模块设计、数据目录约定和使用方式，请继续阅读：

- [`docs/RAG_SYSTEM_GUIDE.md`](./docs/RAG_SYSTEM_GUIDE.md)

## 13. 致谢与引用

CogniComm-Agent 基于本项目团队的持续工程演进构建，同时也受到了以下开源项目的重要启发：

- [`hwfengcs/DM-Code-Agent`](https://github.com/hwfengcs/DM-Code-Agent)

感谢该项目在智能体代码执行、工程组织与早期设计思路上的启发。若你在论文、项目或二次开发中使用本仓库，也欢迎一并注明对原始项目的参考与致谢。

## 14. License

请根据仓库后续发布的许可证文件执行。如果你准备开源分发，建议在仓库根目录补充明确的 `LICENSE` 文件。
