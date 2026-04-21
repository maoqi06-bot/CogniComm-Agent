# Multi-Agent Preflight Diagnostics and Windows Docker Discovery

## Background

After adding multi-agent runtime preflight, the system began failing fast in a much healthier way:

- invalid LLM credentials now fail before orchestration starts;
- Docker visibility problems are detected before CodeAgent tries to run containerized tools.

However, this exposed a new usability issue: users could see a preflight failure but still not know **why** a setup that "used to call the API the same way" now failed immediately.

Two practical diagnosis gaps remained:

1. The preflight did not print enough information about the *effective* runtime LLM configuration.
2. Docker detection relied only on `PATH`, which is fragile on Windows when Docker Desktop is installed but its executable path is not inherited into the current Python process.

## Problem

The user-visible questions behind the failure were:

- Which provider/model/base URL is actually active right now?
- Is the API key coming from `.env`, command line, or saved config?
- Is Docker genuinely missing, or just not visible in the current process environment?

Without those answers, a `401 Invalid token` or “docker not found” message is technically correct but operationally incomplete.

## Changes Implemented

### 1. Add explicit LLM preflight diagnostics

File:

- `main.py`

New helper functions:

- `mask_secret(value)`
- `detect_key_source(config)`

Preflight output now includes:

- current Python executable
- active provider
- active model
- active base URL
- API key source
- masked API key preview
- active embedding model when using the OpenAI-compatible path

This makes it much easier to distinguish:

- wrong token
- wrong gateway
- wrong provider/model pair
- stale environment variables

without leaking sensitive secrets into logs or terminal output.

### 2. Add Windows-friendly Docker path discovery

File:

- `main.py`

New helper:

- `resolve_docker_path()`

Behavior:

- first checks `shutil.which("docker")`
- if that fails, scans common Windows Docker Desktop install directories such as:
  - `C:\Program Files\Docker\Docker\resources\bin`
  - `C:\Program Files\Docker\Docker\resources`
  - `C:\ProgramData\DockerDesktop\version-bin`
- if found, prepends the directory to the current process `PATH`

Effect:

- Docker can become visible even when the current Python process did not inherit the full system/user shell PATH.
- This directly addresses a common multi-agent runtime issue on Windows + conda + PowerShell setups.

### 3. Make preflight output more actionable

File:

- `main.py`

`run_multi_agent_preflight(...)` now prints enough information to answer:

- “What is this process actually trying to use?”
- “Why does this fail now even though previous versions appeared to work?”

In practice, this helps explain an important reality:

- previous versions may have been using the same runtime configuration,
- but they only surfaced the problem later in the task;
- the new preflight surfaces the same root cause earlier.

So the behavior changed in **timing and observability**, not necessarily in the underlying credential validity.

## Why This Matters

This round improves trust in the toolchain:

- failures happen earlier;
- runtime configuration becomes inspectable;
- Docker visibility issues become diagnosable instead of mysterious.

That is especially important in this project because the execution stack includes:

- conda-managed Python
- environment-file-based configuration
- OpenAI-compatible gateways
- multi-agent orchestration
- optional Docker-isolated code execution

Any mismatch in one layer can look like a failure in another unless diagnostics are precise.

## Relationship to Previous Runtime Work

This note extends the previous runtime stabilization work by adding:

- clearer LLM identity diagnostics;
- Docker discovery beyond raw PATH lookup;
- a better explanation path for “worked before / fails now” scenarios.

It does not change the single-agent workflow. The updates only improve multi-agent startup diagnostics and environment visibility.

## Next Steps

The next useful follow-ups are:

1. Persist preflight diagnostics into the saved task report for failed startups.
2. Add the same effective-runtime summary to the dashboard.
3. Add a dedicated “environment doctor” command for:
   - LLM auth
   - embedding model availability
   - Docker visibility
   - current interpreter and PATH summary
