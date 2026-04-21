# MCP Conda Environment And Portability Fix

**Time**: 2026-04-18

**Problem**

When starting DM-Code-Agent, the MCP RAG server failed with:

```text
ModuleNotFoundError: No module named 'mcp'
```

The user confirmed that the active runtime should be the `CodeAgent` conda environment, and that this environment already contains the `mcp` package. The project also had hard-coded local paths and secrets in MCP configuration, which made it unsuitable for open-source use.

**Root Cause**

The RAG MCP startup chain was:

```text
main.py -> MCPClient -> npx/node -> bin/index.js -> python -> rag_mcp_server.py
```

Although `CodeAgent` had `mcp` installed, `bin/index.js` launched plain `python`, which could resolve to a different Python environment. That caused `rag_mcp_server.py` to run outside the intended conda environment.

The MCP config also used a machine-specific absolute path:

```json
"args": ["-y", "D:/mq/shixi/project/DM_CODE_AGENT/DM-Code-Agent"]
```

It also contained plaintext API tokens, which should not be committed or shared.

**Solution**

Updated `bin/index.js` so the RAG launcher chooses Python in this order:

1. `PYTHON` or `PYTHON_EXECUTABLE`
2. `${CONDA_PREFIX}/python.exe` on Windows, or `${CONDA_PREFIX}/bin/python` on Unix
3. System `python` or `python3`

Updated `dm_agent/mcp/client.py` so MCP subprocesses inherit portable runtime values:

```text
PROJECT_ROOT
PYTHON
PYTHON_EXECUTABLE
```

This ensures that if the main process is launched from `conda activate CodeAgent`, MCP subprocesses use the same Python environment.

Updated `mcp_config.json` and `mcp_config.json.example` to remove hard-coded local paths and secrets:

```json
"command": "node",
"args": ["${PROJECT_ROOT}/bin/index.js"],
"env": {
  "OPENAI_API_KEY": "${OPENAI_API_KEY}"
}
```

GitHub and other optional tokens were also converted to environment-variable placeholders such as `${GITHUB_TOKEN}`.

**Files Changed**

- `bin/index.js`
- `dm_agent/mcp/client.py`
- `mcp_config.json`
- `mcp_config.json.example`
- `requirements.txt`

**Verification**

Verified that `CodeAgent` can import the MCP package:

```powershell
conda run -n CodeAgent python -c "import mcp; import dm_agent.rag.rag_mcp_server as s; print('rag server import ok')"
```

Verified that the Node launcher now resolves the conda Python:

```text
[Launcher] Python: D:\anaconda3\envs\CodeAgent\python.exe
```

Verified syntax and config validity:

```powershell
node --check bin\index.js
python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8-sig')) for p in ['dm_agent/mcp/client.py','dm_agent/mcp/config.py','dm_agent/rag/rag_mcp_server.py']]"
python -c "import json; [json.load(open(p, encoding='utf-8')) for p in ['mcp_config.json','mcp_config.json.example']]"
```

**Operational Note**

Run the project from the intended conda environment:

```powershell
conda activate CodeAgent
python main.py
```

Keep secrets in environment variables or `.env`, not in `mcp_config.json`.

