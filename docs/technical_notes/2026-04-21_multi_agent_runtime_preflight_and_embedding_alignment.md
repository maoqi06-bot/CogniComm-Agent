# Multi-Agent Runtime Preflight and Embedding Alignment

## Background

During recent multi-agent runs, several errors appeared together in the same task execution:

- long-term memory retrieval failed with `Unknown model: text-embedding-ada-002`;
- Docker execution failed during code-agent validation steps;
- some tasks were still reported as `成功` even when downstream subtasks had failed;
- agent JSON parsing occasionally failed with `Extra data`;
- OpenAI-compatible credentials could fail late in the task instead of failing early.

These failures were not caused by one single bug. They came from a combination of runtime-environment drift, outdated embedding defaults, and incomplete success-state handling.

This round focused on improving runtime stability in multi-agent mode without changing the single-agent workflow.

## Problems Identified

### 1. Embedding configuration drift

The main LLM model had already been switched to the OpenAI-compatible `gpt-5` path, but parts of the RAG and memory stack still defaulted to:

- `text-embedding-ada-002`

This created a mismatch between:

- the configured OpenAI-compatible chat model;
- the embedding model used by long-term memory and RAG helpers.

As a result, multi-agent memory context construction could fail before sub-agents even started using long-term memory.

### 2. Docker visibility was not checked before execution

CodeAgent could be created in Docker mode and only discover runtime issues when it actually tried to run:

- `run_shell`
- `run_python`
- `run_tests`

That meant users often learned about Docker issues only in the middle of a task rather than before execution.

### 3. Invalid LLM credentials were discovered too late

In some runs, the task decomposition and early subtasks succeeded, but a later code subtask or final aggregation failed with:

- `401 Invalid token`

The system needed a lightweight LLM preflight so obviously invalid credentials would fail fast.

### 4. Overall task status was too optimistic

`OrchestratorAgent.run()` previously returned:

- `"success": True`

as long as the orchestration flow reached aggregation, even if some subtasks had failed or timed out.

This produced a misleading situation:

- task report showed failed subtasks;
- top-level status still showed `成功`.

### 5. JSON parsing was too brittle

Task decomposition used a permissive parser, but it still relied on extracting from the first `{` to the last `}`. If the model returned a valid JSON object followed by extra explanation text, parsing could still fail.

## Changes Implemented

## 1. Align embedding defaults with the current OpenAI-compatible stack

File:

- `dm_agent/rag/embeddings.py`

Changes:

- Updated the default embedding model from `text-embedding-ada-002` to `text-embedding-3-small`.
- Added support for:
  - `OPENAI_EMBEDDING_MODEL`
  - `OPENAI_EMBEDDING_DIMENSION`
- Added `_infer_embedding_dimension()` so embedding dimensions are derived more safely.

Effect:

- Long-term memory, RAG skill initialization, and MCP-backed retrieval now default to a modern embedding model that is much more likely to work with the current OpenAI-compatible provider setup.

## 2. Propagate provider runtime environment into multi-agent dependencies

File:

- `main.py`

Changes:

- Added `apply_runtime_provider_env(config)`.
- When provider is `openai`, the runtime now synchronizes:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
  - default `OPENAI_EMBEDDING_MODEL`

Effect:

- Components that still read configuration through environment variables now inherit the same runtime provider settings as the main chat client.
- This reduces the chance that the main client uses one gateway while embeddings or memory helpers use another.

## 3. Add multi-agent runtime preflight

File:

- `main.py`

Changes:

- Added `run_multi_agent_preflight(config, client, use_docker)`.
- Preflight now checks:
  - current Python executable path;
  - a cheap LLM auth probe;
  - Docker path visibility via `shutil.which("docker")`;
  - `docker --version` if Docker is requested.

Behavior:

- If LLM preflight fails, multi-agent execution aborts early.
- If Docker preflight fails, the system prints diagnostics and automatically downgrades to non-Docker execution instead of failing later in the task.

Effect:

- Users get earlier, clearer feedback.
- Docker problems are discovered before CodeAgent starts tool execution.

## 4. Improve Docker diagnostics in the runtime layer

File:

- `dm_agent/multi_agent/runtime.py`

Changes:

- Added `DockerRunner.health_check()`.
- Improved Docker execution failure metadata:
  - reports `docker_path`;
  - distinguishes “docker executable not found in PATH” from generic execution errors.

Effect:

- Docker failures are easier to diagnose.
- Later tooling and dashboard integrations can build on a structured health-check path.

## 5. Fix top-level multi-agent status semantics

Files:

- `dm_agent/multi_agent/runtime.py`
- `main.py`

Changes:

- `OrchestratorAgent.run()` now computes:
  - `overall_status = "success"`
  - `overall_status = "partial_success"`
  - `overall_status = "failed"`
- The top-level `success` boolean is now only `True` when all subtasks succeed without partials or failures.
- CLI output and saved Markdown reports now display the localized status text derived from `overall_status`.
- CLI exit code now returns success only for fully successful multi-agent runs.

Effect:

- A run with failed subtasks no longer reports itself as fully successful.
- This matches the actual trace and makes downstream automation safer.

## 6. Make task-decomposition JSON parsing more tolerant

File:

- `dm_agent/multi_agent/runtime.py`

Changes:

- `TaskDecomposer._parse_llm_json_response()` now uses `json.JSONDecoder().raw_decode(...)` when extracting the first JSON object from a mixed response.

Effect:

- If the model emits a valid JSON object followed by explanatory text, the decomposition step can still recover the structured payload.
- This reduces `Extra data` parse failures in the multi-agent orchestration path.

## 7. Document embedding config in `.env.example`

File:

- `.env.example`

Changes:

- Added:
  - `OPENAI_BASE_URL`
  - `OPENAI_EMBEDDING_MODEL`

Effect:

- New environments are easier to configure consistently.
- The embedding model is now explicitly visible as a configurable runtime concern.

## Validation

Static validation completed with AST parsing for:

- `main.py`
- `dm_agent/rag/embeddings.py`
- `dm_agent/multi_agent/runtime.py`

This confirms the modified files are syntactically valid.

## Remaining Limitations

This round improves the runtime foundation, but some follow-up work is still valuable:

1. Extend the same robust JSON extraction logic to the generic `ReactAgent` response parser, not just task decomposition.
2. Add a dashboard panel for:
   - Docker health;
   - runtime interpreter path;
   - effective embedding model.
3. Consider a dedicated startup diagnostic command for:
   - provider auth;
   - embedding availability;
   - Docker availability;
   - long-term memory metadata consistency.
4. Improve token-expiry handling during long runs, since a token can still become invalid after preflight.

## Relationship to Previous Work

This note complements earlier multi-agent notes on:

- runtime stability,
- partial timeout handling,
- dashboard-triggered evaluation,
- multi-agent memory improvements.

The new focus here is runtime alignment:

- environment consistency,
- embedding consistency,
- early failure detection,
- and truthful final task status.
