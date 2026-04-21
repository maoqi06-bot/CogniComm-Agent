# Default Provider Regression and Backward Compatibility

## Background

After the runtime-preflight work landed, multi-agent startup began failing immediately with:

- `401 Invalid token`

At first glance this looked like a regression caused by:

- embedding-model alignment;
- OpenAI-compatible gateway fallback changes;
- or the new preflight itself.

However, the new diagnostics made the actual problem visible:

- the runtime was starting with `provider = openai`;
- the model was `gpt-5`;
- the base URL was `https://sg.uiuiapi.com/v1/`;
- and the key source was `OPENAI_API_KEY`.

This mattered because earlier successful runs in the same workspace were likely using the legacy DeepSeek path instead:

- `provider = deepseek`
- `model = deepseek-chat`
- `base_url = https://api.deepseek.com`
- `DEEPSEEK_API_KEY`

In other words, the preflight did not break a previously healthy API path. It exposed that the default runtime route had already shifted to a different provider configuration.

## Root Cause

The regression came from overly aggressive defaults in `main.py`.

`Config` had been changed to default to:

- `provider = "openai"`
- `model = "gpt-5"`
- `base_url = "https://sg.uiuiapi.com/v1/"`

Then `parse_args()` used those values whenever there was no saved `config.json`.

That meant:

1. no explicit CLI provider was required;
2. no saved config was required;
3. simply having both `DEEPSEEK_API_KEY` and `OPENAI_API_KEY` in `.env` was enough for the app to start preferring the OpenAI-compatible route.

If the configured OpenAI-compatible key was invalid, expired, or mismatched with the gateway, the new preflight failed immediately with `401 Invalid token`.

This created the impression that the preflight change had broken all API access, when the actual issue was a default-provider routing regression.

## Fix Implemented

File:

- `main.py`

### 1. Restore legacy-safe config defaults

`Config` defaults were changed back to the historical DeepSeek-oriented values:

- `provider = "deepseek"`
- `model = "deepseek-chat"`
- `base_url = "https://api.deepseek.com"`

This avoids forcing OpenAI-compatible routing in environments that were previously relying on DeepSeek.

### 2. Add inferred default-provider selection

New helper functions were introduced:

- `has_usable_api_key(provider)`
- `resolve_default_provider(saved_config)`
- `resolve_default_model(saved_config, provider)`
- `resolve_default_base_url(saved_config, provider)`

Behavior:

1. if `config.json` explicitly defines a provider, use it;
2. else if `DEFAULT_PROVIDER` is set and has a usable key, use it;
3. else prefer the first usable provider found in env, in this order:
   - `deepseek`
   - `openai`
   - `claude`
   - `gemini`
4. else fall back to the `Config` defaults.

This keeps the app backward compatible for existing `.env` files while still allowing explicit provider switching.

### 3. Keep explicit configuration authoritative

The new inference only applies when the provider was not explicitly saved or passed in.

That means:

- explicit `config.json` settings still win;
- explicit CLI `--provider` still wins;
- only the implicit default path changed.

## Why This Matters

This change separates two very different concerns:

1. **early diagnostics**
   - preflight should fail fast when credentials are invalid;
2. **default routing**
   - the app should not silently switch users to a new provider path unless they explicitly asked for it.

With this fix, the preflight remains useful, but it no longer amplifies a provider-default regression.

## Relationship to Earlier 2026-04-21 Notes

This note refines the interpretation of:

- `2026-04-21_multi_agent_runtime_preflight_and_embedding_alignment.md`
- `2026-04-21_openai_compatible_gateway_chat_completions_fallback.md`

Those changes improved observability and gateway compatibility, but they also made it clear that the workspace had drifted onto a different default provider path.

The current fix restores backward-compatible startup behavior without removing the earlier diagnostics.

## Follow-Up Suggestions

1. Add a startup line showing whether the selected provider came from:
   - CLI
   - config file
   - inferred env default
2. Add a dedicated menu action for:
   - provider connectivity diagnostics
   - embedding diagnostics
   - Docker diagnostics
3. Consider persisting the last successful provider in `config.json` after a successful interactive run.
