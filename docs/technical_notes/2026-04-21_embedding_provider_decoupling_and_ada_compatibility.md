# Embedding Provider Decoupling and ADA Compatibility

## Background

After the default provider route was corrected back to the legacy DeepSeek path, the main multi-agent LLM started working again:

- `provider = deepseek`
- `model = deepseek-chat`
- `DEEPSEEK_API_KEY` auth passed

However, multi-agent execution still failed as soon as long-term memory retrieval or RAG retrieval started. The logs showed repeated errors like:

- `Multi-agent memory bundle retrieval failed: OpenAI API 调用失败: 401 Invalid token`
- `RAG search failed for wireless_comm: OpenAI API 调用失败: 401 Invalid token`

This revealed that the earlier provider fix only restored the **main chat path**. The **embedding path** was still independently calling an OpenAI-compatible gateway.

## Root Cause

The regression was not caused by the main provider fallback itself. It came from a second hidden coupling:

1. long-term memory and RAG embeddings still defaulted to `provider="openai"`;
2. the embedding client implicitly read:
   - `OPENAI_API_KEY`
   - `OPENAI_BASE_URL`
3. the embedding default model had been changed from the older compatibility baseline to:
   - `text-embedding-3-small`

As a result, even when the main orchestration path correctly used DeepSeek, the memory and RAG stack still tried to use the OpenAI-compatible embedding route.

In this project, that route was historically closer to:

- OpenAI-compatible gateway for embeddings
- old default embedding model: `text-embedding-ada-002`

So the main issue had two layers:

- embedding configuration was not isolated from chat-provider configuration;
- the default embedding model had drifted away from the previously working compatibility baseline.

## Fix Implemented

### 1. Decouple embedding runtime from the main LLM provider

Files:

- `dm_agent/rag/embeddings.py`
- `dm_agent/memory/long_term_memory.py`
- `dm_agent/skills/builtin/base_rag_skill.py`
- `dm_agent/rag/rag_mcp_server.py`
- `main.py`

New explicit embedding runtime variables:

- `EMBEDDING_PROVIDER`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSION`

This makes the design clearer:

- main chat model can use one provider;
- embedding-backed retrieval can use another;
- the two are no longer implicitly tied together.

### 2. Restore backward-compatible default embedding model

File:

- `dm_agent/rag/embeddings.py`

Changes:

- `OpenAIEmbeddings` default model was restored to:
  - `text-embedding-ada-002`
- `resolve_embedding_model()` now defaults to:
  - `EMBEDDING_MODEL`
  - otherwise `text-embedding-ada-002`

Importantly, embedding runtime no longer silently inherits `OPENAI_EMBEDDING_MODEL`.

That means a previous experiment with `text-embedding-3-small` no longer keeps poisoning later runs unless the user explicitly opts in via:

- `EMBEDDING_MODEL=text-embedding-3-small`

### 3. Make long-term memory provider-aware

File:

- `dm_agent/memory/long_term_memory.py`

Change:

- When no embedding instance is injected, long-term memory now creates embeddings via:
  - `resolve_embedding_provider()`

instead of hardcoding:

- `provider="openai"`

### 4. Make RAG skill initialization provider-aware

File:

- `dm_agent/skills/builtin/base_rag_skill.py`

Change:

- Base RAG skill now resolves and stores:
  - embedding provider
  - embedding key
  - embedding base URL
  - embedding model

and uses those values during `_ensure_initialized()`.

### 5. Make MCP RAG server provider-aware

File:

- `dm_agent/rag/rag_mcp_server.py`

Change:

- RAG instance creation now uses the same explicit embedding runtime resolution instead of assuming `openai` plus one hardcoded key field.

### 6. Make runtime preflight show embedding runtime separately

File:

- `main.py`

Preflight now reports:

- `Embedding Provider`
- `Embedding Model`
- `Embedding Base URL` (when available)

This makes it much easier to spot mismatches such as:

- main chat on DeepSeek
- embeddings on OpenAI-compatible gateway

## `.env.example` Updates

File:

- `.env.example`

Added explicit embedding configuration:

- `EMBEDDING_PROVIDER=openai`
- `EMBEDDING_API_KEY=`
- `EMBEDDING_BASE_URL=https://sg.uiuiapi.com/v1/`
- `EMBEDDING_MODEL=text-embedding-ada-002`
- `EMBEDDING_DIMENSION=1536`

The old `OPENAI_EMBEDDING_MODEL` line was also restored to the older compatibility baseline in the example file.

## Validation

Static AST validation completed successfully for:

- `main.py`
- `dm_agent/rag/embeddings.py`
- `dm_agent/memory/long_term_memory.py`
- `dm_agent/skills/builtin/base_rag_skill.py`
- `dm_agent/rag/rag_mcp_server.py`

An additional runtime smoke check confirmed that, after applying runtime env defaults in DeepSeek mode, the process now resolves:

- `EMBEDDING_PROVIDER = openai`
- `EMBEDDING_MODEL = text-embedding-ada-002`

which matches the backward-compatible baseline.

## Practical Meaning

This fix does **not** claim that the currently configured OpenAI-compatible embedding key is valid. If that key is invalid, embedding-backed memory and RAG retrieval can still fail.

What this fix does ensure is:

1. the failure is no longer caused by hidden provider coupling;
2. the embedding runtime is explicitly configurable;
3. the default embedding model is now aligned with the older, previously working compatibility path.

## Follow-Up Suggestions

1. Add a dedicated startup check for embedding connectivity, separate from main LLM auth.
2. Surface embedding runtime diagnostics in the dashboard.
3. If the OpenAI-compatible gateway still rejects the embedding request, validate whether:
   - the gateway still supports `text-embedding-ada-002`;
   - the embedding key is still valid;
   - a dedicated embedding key should be configured via `EMBEDDING_API_KEY`.
