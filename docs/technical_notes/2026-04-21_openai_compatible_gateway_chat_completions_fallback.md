# OpenAI-Compatible Gateway Chat-Completions Fallback

## Background

After multi-agent preflight diagnostics were added, the runtime began surfacing an important question from the user:

> previous versions were calling the API in a similar way, so why is it failing now?

That question turned out to be reasonable.

The current `OpenAIClient` had been implemented against the OpenAI **Responses API**, while other earlier or parallel code paths in the project were still closer to traditional **chat completions** behavior. For many OpenAI-compatible gateways, those two API styles are not interchangeable in practice, even if they share the same:

- API key
- base URL domain
- SDK client

## Problem

The runtime showed:

- provider = `openai`
- model = `gpt-5`
- base URL = non-official OpenAI-compatible gateway
- API key loaded correctly from environment

Yet authentication still failed during preflight.

Even when a gateway claims OpenAI compatibility, it may:

- support `chat.completions` better than `responses.create`;
- require different token policies for different endpoints;
- lag behind official OpenAI endpoint behavior;
- reject newer API styles despite accepting familiar model names.

This means:

- the *configuration shape* can look unchanged,
- but the *API style* can still be the actual compatibility boundary.

## Changes Implemented

### 1. Add API-style selection to `OpenAIClient`

File:

- `dm_agent/clients/openai_client.py`

New behavior:

- `OpenAIClient` now resolves an internal `api_style`.
- Supported styles:
  - `responses`
  - `chat_completions`
  - `auto`

Resolution rule:

- if `OPENAI_API_STYLE` is explicitly set to `responses` or `chat_completions`, use that;
- otherwise, in `auto` mode:
  - use `responses` for official OpenAI-style base URLs;
  - use `chat_completions` for non-official OpenAI-compatible gateways.

### 2. Add chat-completions fallback for compatible gateways

File:

- `dm_agent/clients/openai_client.py`

Changes:

- When `api_style == "chat_completions"`, the client now calls:
  - `client.chat.completions.create(...)`
- When `api_style == "responses"`, it still calls:
  - `client.responses.create(...)`

The extraction layer was also updated so both response types can be normalized into plain text for the rest of the agent runtime.

### 3. Surface the active API style in preflight diagnostics

File:

- `main.py`

Preflight output now prints:

- `API Style`

This makes it obvious whether the current run is using:

- the newer Responses API path;
- or the more traditional chat-completions path.

### 4. Add runtime environment defaults for gateway compatibility

Files:

- `main.py`
- `.env.example`

Changes:

- Added/standardized:
  - `OPENAI_API_STYLE=auto`
  - `DOCKER_PATH=`

The `.env.example` file now documents that API-style selection is part of the runtime compatibility surface for OpenAI-like gateways.

## Why This Helps

This change directly addresses the “it used to call the API the same way” confusion:

- the project may have used the same provider, key, and gateway;
- but it was not necessarily using the same endpoint family.

With this fallback:

- official OpenAI can still use the Responses API;
- compatible third-party gateways can fall back to chat completions without requiring code forks.

## Relationship to Previous Runtime Work

This note builds on the runtime preflight and provider-alignment work by adding one more missing compatibility layer:

- not just *which model and gateway* are active,
- but also *which API style* is used to talk to that gateway.

## Next Steps

Useful follow-up work includes:

1. Persist the effective API style into task reports and dashboard diagnostics.
2. Add a small connectivity doctor command that tests:
   - responses API
   - chat completions API
   - embedding API
   against the same gateway and token.
3. Allow per-provider profile defaults for gateway-specific API style selection.
