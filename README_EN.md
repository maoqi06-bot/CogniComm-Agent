# CogniComm-Agent

CogniComm-Agent is an LLM agent system for research workflows, code execution, domain RAG, memory, and engineering automation. It keeps the lightweight experience of a single-agent coding assistant while extending the system into a practical multi-agent platform with orchestration, domain profiles, long-term memory, MCP integration, dashboard-based evaluation, and human approval loops.

The two primary entry points are:

- [`main.py`](./main.py): CLI entry for single-agent and multi-agent execution
- [`dashboard.py`](./dashboard.py): developer dashboard for RAG evaluation, memory approval, replay, and observability

> The current project name is **CogniComm-Agent**. If you still see `DM-Agent` or `DM-Code-Agent` in older files, treat this README and the current codebase as the source of truth.

## 1. What This Project Is For

CogniComm-Agent is built for tasks that require more than plain chat:

- LLM-based task understanding and decomposition
- domain knowledge retrieval through RAG
- actual code, scripts, experiments, and report generation
- persistent memory and user preference handling
- white-box observability for RAG quality and multi-agent execution

This is not just a chatbot, and not just a wrapper around code tools. It is an agent framework oriented toward real research and engineering workflows.

## 2. Core Capabilities

### 2.1 Single-Agent Mode

Single-agent mode is optimized for faster, lighter workflows:

- ReAct reasoning loop
- multiple model backends: DeepSeek / OpenAI-compatible / Claude / Gemini
- built-in code tools
- MCP tool mounting
- skill expert system
- context compression
- single-agent long-term memory

### 2.2 Multi-Agent Mode

Multi-agent mode is the main engineering direction of the project. It supports:

- `OrchestratorAgent` for decomposition, scheduling, and final aggregation
- `RAGAgent` for full retrieve + synthesize knowledge workflows
- `CodeAgent` for code, files, tests, experiments, and documentation
- dependency-aware batch scheduling
- optional Docker-based isolated execution
- shared short-term memory plus controlled long-term writes
- dashboard pages for memory replay, approvals, and RAG evaluation

### 2.3 RAG and Domain Knowledge

- hybrid RAG with vector retrieval + BM25 + reranking
- domain-specific knowledge bases via skills or MCP
- domain profiles and specialized prompts
- dashboard-triggered RAGAS evaluation that does not block the CLI task path

### 2.4 Engineering Hardening

The implementation is backed by the engineering stages documented in:

- [`ENGINEERING_P0.md`](./ENGINEERING_P0.md)
- [`ENGINEERING_P1.md`](./ENGINEERING_P1.md)
- [`ENGINEERING_P2.md`](./ENGINEERING_P2.md)

In practice, that means:

- P0: retries, logging, LLM wrappers
- P1: rate limiting, secure execution, health checks, resource management
- P2: multi-agent orchestration, profiles, memory, RAG evaluation, dashboard integration

## 3. Architecture Overview

### 3.1 Runtime Layers

1. **Entry layer**
   - [`main.py`](./main.py)
   - [`dashboard.py`](./dashboard.py)

2. **Agent layer**
   - single-agent `ReactAgent`
   - multi-agent `OrchestratorAgent`, `RAGAgent`, `CodeAgent`

3. **Tool and skill layer**
   - default tools in `dm_agent/tools`
   - MCP integration in `dm_agent/mcp`
   - skill system in `dm_agent/skills`

4. **Knowledge and memory layer**
   - RAG in `dm_agent/rag`
   - single-agent memory in `dm_agent/memory`
   - multi-agent memory in `dm_agent/multi_agent/memory.py`

5. **Engineering support layer**
   - logging and retry utilities
   - security and resource management
   - Docker execution support

### 3.2 Multi-Agent Workflow

In P2 mode, the high-level flow is:

1. the user selects multi-agent mode in the CLI
2. `main.py` runs LLM, embedding, and Docker preflight checks
3. `OrchestratorAgent` uses `TaskDecomposer` to generate a subtask graph
4. `TaskScheduler` executes subtasks in dependency-respecting batches
5. knowledge subtasks are handled by `RAGAgent`
6. implementation subtasks are handled by `CodeAgent`
7. `ResultAggregator` produces the final answer and report
8. traces, memory events, approval records, and RAG evaluation data become visible in the dashboard

## 4. Repository Structure

```text
CogniComm-Agent/
├─ main.py                          # CLI entry point
├─ dashboard.py                     # Streamlit dashboard
├─ README.md
├─ README_EN.md
├─ ENGINEERING_P0.md
├─ ENGINEERING_P1.md
├─ ENGINEERING_P2.md
├─ requirements.txt
├─ package.json                     # Node package for local MCP server
├─ mcp_config.json.example
├─ config.json.example
├─ .env.example
├─ configs/
│  └─ multi_agent/
│     └─ profiles/
├─ dm_agent/
│  ├─ clients/
│  ├─ core/
│  ├─ tools/
│  ├─ mcp/
│  ├─ prompts/
│  ├─ skills/
│  ├─ rag/
│  ├─ memory/
│  ├─ multi_agent/
│  └─ utils/
├─ docs/
│  └─ technical_notes/
├─ task/
└─ data/
```

## 5. Module Guide

### 5.1 `main.py`

`main.py` is the operational entry point. It handles:

- CLI parsing
- loading `.env` and persisted config
- MCP startup
- skill loading
- single-agent execution
- multi-agent execution
- runtime preflight:
  - LLM authentication
  - embedding configuration display
  - Docker visibility checks

The interactive menu currently includes:

1. Execute a new task
2. Multi-turn conversation mode
3. View available tools
4. Configuration settings
5. View available skills
6. P2: Multi-agent collaboration mode
7. Exit

### 5.2 `dashboard.py`

The dashboard is a developer control plane. It is not the main task entry point. It provides:

- RAG runtime status
- knowledge base browsing
- retrieval-chain diagnostics
- RAGAS charts and reports
- multi-agent memory views:
  - long-term hit rate
  - preference hit rate
  - approval queue
  - event timeline
  - task-level call graph
  - replay export

### 5.3 `dm_agent/clients`

Model client implementations live here:

- `DeepSeekClient`
- `OpenAIClient`
- `ClaudeClient`
- `GeminiClient`

They are created through `create_llm_client(...)` and aligned through `PROVIDER_DEFAULTS`.

### 5.4 `dm_agent/core`

This is where `ReactAgent` lives. It is responsible for:

- prompt assembly
- ReAct loop execution
- tool calling
- step tracking
- context compression hooks

Single-agent mode and the multi-agent `CodeAgent` both rely on this core.

### 5.5 `dm_agent/tools`

The default toolset covers common engineering actions such as:

- file reads and writes
- AST parsing
- function signature extraction
- directory inspection
- shell / Python / test execution

### 5.6 `dm_agent/mcp`

This layer loads and manages MCP server configuration. See [`mcp_config.json.example`](./mcp_config.json.example).

Example MCP servers in the sample config:

- `wireless-rag`
- `playwright`
- `context7`
- `filesystem`
- `sqlite`
- `github`

The `wireless-rag` sample uses `${PROJECT_ROOT}/bin/index.js` so the repository does not depend on machine-specific absolute paths.

### 5.7 `dm_agent/skills`

Skills inject specialized prompts, heuristics, and tools into the system. The repository includes:

- code-oriented skills
- database skill
- frontend skill
- skill creation skill
- RAG/domain-oriented skills
- wireless communication related skills

In single-agent mode, skills directly shape tools and prompts. In multi-agent mode, they mainly support domain RAG and specialized execution behavior.

### 5.8 `dm_agent/rag`

This is the knowledge system backbone:

- `document_loader.py`: document loading and chunking
- `embeddings.py`: embedding wrappers and configuration resolution
- `vector_store.py`: FAISS-backed vector storage
- `retriever.py`: hybrid retrieval, rank fusion, reranking
- `rag_mcp_server.py`: multi-domain RAG MCP server
- `evaluator.py` and `observer.py`: RAGAS evaluation and observation

In the current multi-agent design, **RAGAgent owns the full RAG chain**. It does not just return raw chunks.

### 5.9 `dm_agent/memory`

This directory contains the single-agent memory system:

- `ContextCompressor`
- `LongTermMemoryStore`
- `MemoryManager`

It remains available and unchanged as a selectable mode.

### 5.10 `dm_agent/multi_agent`

This directory contains the multi-agent implementation:

- `runtime.py`
  - `OrchestratorAgent`
  - `RAGAgent`
  - `CodeAgent`
  - `DockerRunner`
  - `TaskDecomposer`
  - `TaskScheduler`
  - `ResultAggregator`
- `profiles.py`
- `profile_loader.py`
- `domain_profiles.py`
- `toolkits.py`
- `memory.py`

Key responsibilities:

- `OrchestratorAgent`: top-level planning, dependency wiring, result merging
- `RAGAgent`: retrieve + synthesize domain knowledge
- `CodeAgent`: code, files, tests, experiments, and technical artifacts
- `DockerRunner`: optional isolated execution with graceful fallback when Docker is unavailable

### 5.11 `dm_agent/utils`

This layer carries the P0 and P1 engineering support:

- structured logging
- retry utilities
- timeout wrappers
- resource manager
- semaphore-based rate limiting
- secure shell execution
- health checks

## 6. Memory System

### 6.1 Single-Agent Memory

The single-agent memory path remains available and useful for:

- multi-turn context retention
- important fact storage
- lightweight project preference recall

### 6.2 Multi-Agent Memory

The multi-agent memory hub in [`dm_agent/multi_agent/memory.py`](./dm_agent/multi_agent/memory.py) supports:

- shared short-term event history
- agent-specific memory policies
- long-term write candidates
- user preference capture and retrieval
- human approval queue
- replay and export in the dashboard

By default:

- `orchestrator` owns global summaries and user preference writes
- `rag_agent` mostly reads long-term memory
- `code_agent` writes implementation patterns, debugging lessons, engineering experience, and simulation results

## 7. Profiles and Extensibility

Profiles let you tailor agent behavior, tools, prompts, and memory policy without rewriting the runtime. The sample directory is:

- [`configs/multi_agent/profiles`](./configs/multi_agent/profiles)

There are two main extension paths:

1. **Add a new skill**
   - implement it under the skill system
   - let `SkillManager` load or register it

2. **Add a new multi-agent profile**
   - add a profile JSON
   - route it through the profile loader for domain-specific behavior

This makes the system flexible for additional research domains.

## 8. Configuration

### 8.1 Environment Variables

Start from `.env.example`:

```bash
cp .env.example .env
```

Common settings include:

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

Important notes:

- do not commit `.env`
- do not hardcode real API keys into `mcp_config.json`
- use `mcp_config.json.example` as the shareable template

### 8.2 Persisted CLI Config

The CLI can persist runtime configuration into a local `config.json`. In practice, the effective precedence is typically:

1. command-line arguments
2. local `config.json`
3. code defaults
4. environment variables as supporting inputs

## 9. Installation and Startup

### 9.1 Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 9.2 Install Node Dependencies

```bash
npm install
```

### 9.3 Start the CLI

```bash
python main.py
```

### 9.4 Start the Dashboard

```bash
streamlit run dashboard.py
```

### 9.5 Quick CLI Examples

```bash
python main.py --task "Analyze the current project structure"
python main.py --task "Implement a minimal HBF example" --provider openai --model gpt-5
python main.py --task "Retrieve ISAC papers and generate an experiment report" --multi-agent
```

## 10. Usage Modes

### 10.1 Single-Agent Mode

Best for:

- code edits
- file operations
- quick technical tasks
- multi-turn interactive development

How to use it:

1. run `python main.py`
2. choose “Execute new task” or “Multi-turn conversation mode”
3. enter your task
4. inspect the result and logs

### 10.2 Multi-Agent Mode

Best for:

- research-style complex tasks
- retrieval + derivation + implementation + documentation
- workflows with explicit knowledge dependencies and artifact outputs

How to use it:

1. run `python main.py`
2. choose `6. P2: Multi-Agent`
3. enter a complete task description
4. let the system perform:
   - preflight
   - decomposition
   - RAG subtasks
   - code subtasks
   - aggregation and reporting

### 10.3 Dashboard-Based RAG Evaluation

RAGAS evaluation is triggered manually from the dashboard:

- it does not block CLI task input
- it runs in the background
- it reports both generation and retrieval quality

### 10.4 Memory Approval in the Dashboard

Some long-term memory writes require human approval. The dashboard lets you:

- inspect pending records
- approve or reject them
- observe cross-task hit rates and approval statistics

## 11. Frequently Asked Questions

### 11.1 Why are RAG tools disabled for CodeAgent in multi-agent mode?

Because RAG is owned by a dedicated `RAGAgent` in P2. This keeps responsibilities clean and avoids tool routing confusion or missing-tool errors.

### 11.2 Why doesn’t the final answer simply dump raw chunks and metadata?

Because the intended RAG behavior is retrieval-augmented generation, not retrieval-only output. Retrieved contexts are used to ground generation, then the answer is synthesized for the user.

### 11.3 Why does Docker sometimes fall back automatically?

The runtime checks Docker visibility before execution. If the current Python process cannot see `docker.exe`, the system downgrades to non-Docker execution instead of failing much later.

### 11.4 Why doesn’t every memory event become long-term memory immediately?

Because multi-agent memory uses:

- quality filtering
- human approval for selected categories
- asynchronous write paths

## 12. Recommended Reading for Developers

If you want to extend the project, start with:

- [`ENGINEERING_P0.md`](./ENGINEERING_P0.md)
- [`ENGINEERING_P1.md`](./ENGINEERING_P1.md)
- [`ENGINEERING_P2.md`](./ENGINEERING_P2.md)
- [`docs/technical_notes`](./docs/technical_notes)

Recommended extension areas:

- new domain RAG skills
- new multi-agent profiles
- richer code execution tools
- additional memory templates and approval policies
- dashboard visualization improvements

## 13. Acknowledgement and Citation

CogniComm-Agent is the result of ongoing engineering work in this repository, and it was also meaningfully inspired by:

- [`hwfengcs/DM-Code-Agent`](https://github.com/hwfengcs/DM-Code-Agent)

We appreciate that project for its early ideas around coding agents, tooling structure, and practical implementation direction. If you build on this repository in a paper, report, or derivative project, we encourage you to acknowledge that upstream inspiration as well.

## 14. License

Please follow the license published for this repository. If you plan to distribute it publicly, adding an explicit `LICENSE` file at the repository root is recommended.
