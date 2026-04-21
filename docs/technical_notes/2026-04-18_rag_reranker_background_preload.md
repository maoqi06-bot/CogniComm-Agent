# RAG Reranker 后台预热与重复加载优化

**时间**: 2026-04-18

## 问题背景

项目当前同时提供了两种 RAG 使用方式：

1. **MCP RAG Server 挂载方式**
   - `main.py` 启动 MCP server。
   - MCP server 内部运行 `dm_agent/rag/rag_mcp_server.py`。
   - Agent 通过 MCP 工具调用远程 RAG 检索。

2. **内置 RAG Skill 挂载方式**
   - 本地 `BaseRAGSkill` 直接加载向量库、BM25、Reranker。
   - 本地 Agent/RAG 子 Agent 可以直接使用内置 RAG 工具。

这两种方式都会用到同一个精排模型：

```text
BAAI/bge-reranker-base
```

启动日志中可以看到，加载 reranker 模型耗时明显：

```text
🔄 正在加载全局共享 Reranker 模型: BAAI/bge-reranker-base
🔄 正在加载 Reranker 模型: BAAI/bge-reranker-base ...
Warning: You are sending unauthenticated requests to the HF Hub...
Loading weights...
```

虽然项目原来已经使用了“单例模式”，但用户体验仍然不好：

- 启动阶段会被模型加载阻塞。
- MCP RAG server 和本地内置 RAG 是不同进程时，单例无法跨进程共享。
- 如果多个 RAG skill 初始化时都启用 reranker，需要确保同一进程内不会重复加载。

## 原方案的问题

原方案分别在两个地方维护 reranker 单例：

```text
dm_agent/skills/builtin/base_rag_skill.py
dm_agent/rag/rag_mcp_server.py
```

两个地方都有类似逻辑：

```python
if SHARED_RERANKER is None:
    SHARED_RERANKER = CrossEncoderReranker(model_name=model_name)
```

这个方案能解决一部分问题：

- 同一进程内，同一类 RAG 组件不会重复加载模型。

但是它有几个不足：

1. **同步加载，阻塞启动**

   只要初始化 RAG，就会同步执行 `CrossEncoder(model_name)`，用户必须等模型加载完。

2. **单例分散在多个模块**

   `BaseRAGSkill` 和 `rag_mcp_server.py` 各自维护一套单例逻辑，容易出现行为不一致。

3. **无法跨进程共享**

   MCP server 是独立 Python 进程，本地 Agent 是另一个 Python 进程时，Python 单例不能共享内存。  
   这不是代码写法问题，而是进程模型决定的。

4. **首次检索体验不可控**

   如果不在启动时加载，首次检索会卡住；如果启动时加载，启动页又会卡住。

## 采用的新方案

本次采用：

```text
共享注册表 + LazyReranker + 后台线程预热 + 首次检索有限等待
```

核心实现放在：

```text
dm_agent/rag/retriever.py
```

新增了三个核心对象：

```python
class LazyReranker
class SharedRerankerRegistry
def get_shared_reranker(...)
```

## 新方案工作方式

### 1. 后台预热

当 MCP server 启动或内置 RAG skill 创建时，不再同步加载模型，而是启动一个 daemon 线程：

```python
threading.Thread(
    target=self._load,
    name=f"reranker-preload-{self.model_name}",
    daemon=True,
).start()
```

这样模型加载会在后台进行，主流程可以继续展示菜单、加载工具或处理其他初始化任务。

### 2. 同进程共享

所有组件都通过统一入口获取 reranker：

```python
get_shared_reranker(model_name, preload=True)
```

内部使用：

```python
SharedRerankerRegistry._instances
```

以 `model_name` 为 key 保存 `LazyReranker`。  
因此在同一个 Python 进程内，无论有多少 RAG skill 使用同一个模型，都只会创建一个后台加载任务。

### 3. 首次检索有限等待

如果用户第一次检索时模型还没有加载完成，`LazyReranker.rerank()` 会等待一小段时间：

```text
RAG_RERANKER_WAIT_SECONDS
```

默认值：

```text
10 秒
```

如果 10 秒内模型加载完成，就正常精排。  
如果还没完成，就跳过本次精排，直接返回粗排结果，避免用户一直卡住：

```text
BM25 + Dense + RRF -> 返回 Top-K
```

这比“启动时强制等待模型加载”更友好。

### 4. 可关闭后台预热

新增环境变量：

```text
RAG_PRELOAD_RERANKER
```

默认：

```text
true
```

如果希望完全按需加载，可以设置：

```powershell
$env:RAG_PRELOAD_RERANKER="false"
```

此时只有第一次真正调用 reranker 时才会启动加载。

## 修改的文件

### `dm_agent/rag/retriever.py`

新增：

- `LazyReranker`
- `SharedRerankerRegistry`
- `get_shared_reranker`

`LazyReranker` 负责：

- 后台加载真实 `CrossEncoderReranker`
- 保存加载状态
- 保存加载异常
- 首次检索等待
- 超时后降级为粗排结果

### `dm_agent/skills/builtin/base_rag_skill.py`

内置 RAG skill 不再自己维护 `_SHARED_RERANKER`，而是改为：

```python
reranker = get_shared_reranker(self.reranker_model, preload=True)
```

同时在 skill 创建时，如果 `RAG_PRELOAD_RERANKER=true`，会提前触发后台预热。

### `dm_agent/rag/rag_mcp_server.py`

MCP RAG server 不再自己维护 `_SHARED_RERANKER`，而是改为：

```python
reranker = get_shared_reranker(model_name, preload=True)
```

同时在 server 启动时，如果 `RAG_PRELOAD_RERANKER=true`，会对默认 reranker 模型提前预热。

## 关于“跨进程重复加载”的边界

需要明确一点：

```text
Python 单例只能在同一个进程内生效，不能跨进程共享模型内存。
```

因此：

- MCP RAG server 进程内部：可以保证同模型只加载一次。
- 本地 Agent 进程内部：可以保证同模型只加载一次。
- MCP RAG server 和本地 Agent 如果是两个不同 Python 进程：仍然会各自加载一份模型。

如果未来要真正做到跨进程只加载一次，需要改成以下架构之一：

1. 统一只使用 MCP RAG server，让本地 Agent 不再加载内置 RAG。
2. 把 reranker 独立成模型服务，例如 HTTP/gRPC 服务。
3. 使用 Docker/服务化部署，让所有 Agent 调用同一个 RAG/Rerank 服务。

当前本次优化解决的是：

```text
同进程重复加载 + 启动阻塞 + 首次检索卡死
```

跨进程模型共享属于后续架构优化。

## 配置项

### `RAG_PRELOAD_RERANKER`

是否在启动/初始化阶段后台预热 reranker。

默认：

```text
true
```

关闭：

```powershell
$env:RAG_PRELOAD_RERANKER="false"
```

### `RAG_RERANKER_WAIT_SECONDS`

首次检索等待 reranker 加载完成的最长时间。

默认：

```text
10
```

例如改为 3 秒：

```powershell
$env:RAG_RERANKER_WAIT_SECONDS="3"
```

如果超时，本次检索会跳过 reranker，直接返回粗排结果。

## 验证

语法检查：

```powershell
python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8-sig')) for p in ['dm_agent/rag/retriever.py','dm_agent/skills/builtin/base_rag_skill.py','dm_agent/rag/rag_mcp_server.py']]; print('syntax ok')"
```

验证同进程共享：

```powershell
conda run -n CodeAgent python -c "from dm_agent.rag.retriever import get_shared_reranker; r1=get_shared_reranker('BAAI/bge-reranker-base', preload=False); r2=get_shared_reranker('BAAI/bge-reranker-base', preload=False); print(r1 is r2); print(type(r1).__name__)"
```

输出：

```text
True
LazyReranker
```

验证 MCP server 可导入：

```powershell
conda run -n CodeAgent python -c "import os; os.environ['RAG_PRELOAD_RERANKER']='false'; import dm_agent.rag.rag_mcp_server as s; print('mcp server import ok')"
```

输出：

```text
mcp server import ok
```

## 结论

这次选择后台预热方案，而不是继续扩大同步单例。

新的效果是：

1. 启动阶段不再被 reranker 强制阻塞。
2. 同一进程内同名 reranker 不会重复加载。
3. MCP RAG 和内置 RAG 使用统一的共享注册表逻辑。
4. 首次检索时模型未完成加载，也能有限等待并降级返回，用户体验更稳。
5. 跨进程共享模型暂不在本次修复范围内，后续可通过统一 MCP RAG server 或模型服务化解决。

