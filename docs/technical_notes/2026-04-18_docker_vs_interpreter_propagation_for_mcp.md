# MCP 子进程环境问题：Docker 方案与解释器继承方案对比

**时间**: 2026-04-18

## 背景

项目启动 MCP RAG 服务时，存在两层子进程：

```text
main.py
  -> MCPClient 子进程
    -> node bin/index.js
      -> python dm_agent/rag/rag_mcp_server.py
```

之前出现过如下报错：

```text
ModuleNotFoundError: No module named 'mcp'
```

进一步确认后发现，`CodeAgent` conda 环境中实际已经安装了 `mcp` 包：

```text
D:\anaconda3\envs\CodeAgent\python.exe
mcp package exists
```

因此问题不是依赖一定缺失，而是 MCP 子进程没有使用到当前激活的 `CodeAgent` Python 环境。

同时，原始 `mcp_config.json` 中还存在本机绝对路径和明文密钥：

```json
"args": ["-y", "D:/mq/shixi/project/DM_CODE_AGENT/DM-Code-Agent"]
```

这会导致项目难以开源和迁移。

## Docker 是否可以解决

Docker 可以解决这个问题，而且解决方式比较彻底：

1. 把 Python、Node、MCP SDK、FAISS、RAG 依赖都封装到镜像中。
2. 所有人都通过同一个镜像运行 RAG MCP server。
3. 不再依赖宿主机是否激活了正确的 conda 环境。
4. 不再担心 `python` 命令解析到 base 环境、系统环境或其他虚拟环境。

理论上的启动方式可以变成：

```text
main.py -> MCPClient -> docker run dm-code-agent-rag
```

或者：

```text
main.py -> MCPClient -> node launcher -> docker run python rag_mcp_server.py
```

这样环境一致性最好，尤其适合部署到服务器、CI 或团队多人环境。

## Docker 方案的代价

虽然 Docker 更彻底，但对当前项目来说会引入额外复杂度：

1. 用户必须安装 Docker Desktop 或 Docker Engine。
2. Windows 下还可能涉及 WSL2、路径挂载、文件权限和端口/stdio 转发问题。
3. MCP server 使用标准输入输出通信，Docker 场景下要确保 `stdin/stdout` 透明转发。
4. RAG 知识库、FAISS 索引、trace 输出目录都需要正确挂载。
5. 镜像构建会增加维护成本，例如 Python 依赖、Node 依赖、模型缓存、系统库版本。
6. 对本地开发来说，修改代码后需要额外处理镜像重建或 bind mount。

也就是说，Docker 更适合“部署稳定性”问题，不是当前这个本地开发阶段的最小修复。

## 当前更合适的方案

当前选择的是更简单、更合适的方案：

```text
解释器继承 + 配置占位符 + 去除 shell=True
```

核心思路是：

1. 如果用户从 `conda activate CodeAgent` 后启动 `main.py`，那么 MCP 子进程也应该使用同一个 Python。
2. `MCPClient` 在启动 MCP server 前，把当前解释器路径注入到环境变量：

```text
PYTHON
PYTHON_EXECUTABLE
PROJECT_ROOT
```

3. `bin/index.js` 启动 RAG server 时优先使用：

```text
PYTHON / PYTHON_EXECUTABLE
CONDA_PREFIX/python.exe
系统 python
```

4. `mcp_config.json` 不再写死本机路径，而是使用：

```json
"args": ["${PROJECT_ROOT}/bin/index.js"]
```

5. API Key、GitHub Token 等敏感信息不再写入配置文件，而是使用环境变量占位符：

```json
"OPENAI_API_KEY": "${OPENAI_API_KEY}"
```

## 为什么这个方案更适合当前阶段

这个方案更适合当前项目，原因如下：

1. 改动小：只需要调整 MCP 启动链路，不需要引入 Dockerfile、镜像构建和挂载策略。
2. 依赖少：用户只要有 conda 环境即可运行，不强制安装 Docker。
3. 调试简单：Python 报错仍然直接显示在当前终端，不需要进入容器排查。
4. 开源友好：去掉了绝对路径和明文密钥，其他用户 clone 后也可以使用自己的环境变量。
5. 符合当前根因：问题来自解释器选择错误，而不是宿主机环境不可控。

因此，当前阶段不建议为了这个问题引入 Docker。更好的策略是：

```text
本地开发：使用解释器继承方案
团队部署 / CI / 演示环境：后续再补 Docker 方案
```

## 已采用的解决方式

已经采用轻量方案解决：

### 1. `bin/index.js`

RAG launcher 现在按如下优先级选择 Python：

```text
PYTHON 或 PYTHON_EXECUTABLE
CONDA_PREFIX/python.exe
python / python3
```

这样在 `conda activate CodeAgent` 后运行项目时，RAG MCP server 会使用：

```text
D:\anaconda3\envs\CodeAgent\python.exe
```

### 2. `dm_agent/mcp/client.py`

MCPClient 启动 MCP server 时会注入：

```text
PROJECT_ROOT
PYTHON
PYTHON_EXECUTABLE
```

并且支持在配置中展开：

```text
${PROJECT_ROOT}
${PYTHON}
${PYTHON_EXECUTABLE}
${OPENAI_API_KEY}
```

同时取消 Windows 下的 `shell=True`，避免 Node 和 Python 子进程参数拼接带来的安全警告。

### 3. `mcp_config.json`

原来的硬编码路径被替换为：

```json
"command": "node",
"args": ["${PROJECT_ROOT}/bin/index.js"]
```

密钥改为：

```json
"OPENAI_API_KEY": "${OPENAI_API_KEY}"
```

## 验证方式

确认 `CodeAgent` 环境中可以导入 RAG MCP server：

```powershell
conda run -n CodeAgent python -c "import mcp; import dm_agent.rag.rag_mcp_server as s; print('rag server import ok'); print(s.BASE_PATH)"
```

确认 Node launcher 使用的是 conda Python：

```powershell
conda run -n CodeAgent node bin\index.js
```

预期输出包含：

```text
[Launcher] Python: D:\anaconda3\envs\CodeAgent\python.exe
```

确认配置不再包含本机路径和明文 token：

```powershell
rg -n "D:/|D:\\|sk-|github_pat_|DM-Code-Agent-main|DM-Code-Agent_cursor" mcp_config.json mcp_config.json.example bin dm_agent\mcp dm_agent\rag package.json check_mcp_env.py -S
```

## 后续可选增强

后续如果要提供“开箱即用”的部署能力，可以再补 Docker 方案：

1. 新增 `Dockerfile.rag`
2. 新增 `docker-compose.yml`
3. 将 `dm_agent/data/knowledge_base` 和 `dm_agent/data/indices` 挂载进容器
4. 将 `.env` 注入容器
5. 在 `mcp_config.json.example` 中提供 Docker 版 server 配置

但这应该作为部署增强，而不是当前本地开发问题的首选修复。

## 结论

Docker 可以解决，但当前不是最优解。

当前最合适、最简单的方案是：

```text
保持本地 conda 开发环境，修复 MCP 子进程解释器继承，并移除硬编码路径和密钥。
```

该方案已经解决 `mcp` 包存在但子进程无法导入的问题，同时让项目配置更适合开源和迁移。

