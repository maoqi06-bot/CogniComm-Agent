# Default Provider Routing Correction and Current Behavior

## Background

During the 2026-04-21 runtime-preflight work, we temporarily changed the startup routing logic to prefer the legacy DeepSeek path in order to verify whether:

- the new preflight had broken API access; or
- the real issue was simply that the process had been silently routed to a different provider.

That temporary rollback was useful for diagnosis, but it was **not** the intended long-term product behavior.

The current product requirement is:

- the default main LLM route should remain:
  - `provider = openai`
  - `model = gpt-5`
  - `base_url = https://sg.uiuiapi.com/v1/`
- embedding-backed RAG and long-term memory should remain independently configurable.

## What Was Learned

The earlier failure pattern showed two separate problems:

1. **main chat route**
   - startup had been routed to OpenAI-compatible `gpt-5`;
   - when the configured key was invalid, preflight failed immediately with `401 Invalid token`.

2. **embedding route**
   - long-term memory and RAG retrieval used a separate OpenAI-compatible embedding path;
   - even after the main chat route changed, embedding-backed retrieval could still fail independently.

This distinction matters because the main LLM provider and the embedding provider are no longer assumed to be the same.

## Current Design

### Main LLM defaults

`main.py` now keeps the product-facing default route as:

- `provider = "openai"`
- `model = "gpt-5"`
- `base_url = "https://sg.uiuiapi.com/v1/"`

`parse_args()` resolves defaults in this order:

1. saved `config.json`
2. `DEFAULT_PROVIDER` environment override
3. `Config` defaults

This means:

- if the user explicitly saved a provider, that wins;
- if the user explicitly sets `DEFAULT_PROVIDER`, that wins;
- otherwise the project starts on the documented OpenAI-compatible default route.

### Embedding defaults

Embeddings are now intentionally separated from the main LLM route and configured through:

- `EMBEDDING_PROVIDER`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSION`

The compatibility baseline for embeddings currently defaults to:

- `text-embedding-ada-002`

This preserves the older RAG behavior more closely while allowing the main chat route to stay on `gpt-5`.

## Why the Earlier DeepSeek Logs Still Appeared

Those logs were generated while the temporary rollback-to-DeepSeek version was active.

They do **not** reflect the current code state anymore.

A direct runtime check now resolves:

- `provider = openai`
- `model = gpt-5`
- `base_url = https://sg.uiuiapi.com/v1/`

So if a new run still shows DeepSeek, the likely sources are:

1. a saved `config.json`
2. a `DEFAULT_PROVIDER=deepseek` environment variable
3. an older process still running an outdated local copy

## Practical Interpretation

If the main LLM route now starts as OpenAI-compatible and still fails, that failure should be interpreted as:

- an OpenAI-compatible key / gateway issue;

not as:

- an embedding-model regression.

If the main LLM route succeeds but RAG / memory fails, that should be interpreted as:

- an embedding-specific key / gateway issue;

not as:

- the main provider routing being wrong.

## Relationship to Other 2026-04-21 Notes

This note supersedes the earlier temporary rollback interpretation and should now be read together with:

- `2026-04-21_multi_agent_runtime_preflight_and_embedding_alignment.md`
- `2026-04-21_openai_compatible_gateway_chat_completions_fallback.md`
- `2026-04-21_embedding_provider_decoupling_and_ada_compatibility.md`

Together, those notes describe the current intended architecture:

- OpenAI-compatible `gpt-5` as the default main model
- embedding runtime decoupled from the main model
- early preflight for clearer diagnostics

## Follow-Up Suggestions

1. Add a menu-level runtime diagnostics command that separately tests:
   - main LLM auth
   - embedding auth
   - Docker availability
2. Surface the source of the selected provider in startup logs:
   - config file
   - env override
   - default config
3. Add a dedicated embedding connectivity probe to preflight, so RAG failures are visible before task decomposition begins.
