# Multi-Agent Profiles and File Structure

## 背景

早期 `dm_agent.multi_agent` 的实现集中在 `__init__.py` 中，RAG、Code、Orchestrator、Scheduler、DockerRunner 等类都放在同一个包入口文件里。这种结构能快速验证 P2 多 Agent 流程，但不利于后续扩展不同领域的 RAG Agent、不同工程栈的 Code Agent，以及各子 Agent 独立接入 MCP 工具和轻量 skill。

目前已经将运行时实现拆到 `runtime.py`，并新增 `profiles.py`、`toolkits.py`、`memory.py`、`domain_profiles.py`、`profile_loader.py` 等模块。后续扩展的重点不是继续增加硬编码 profile，而是让用户通过配置文件和指定 skill 目录扩展多 Agent 能力。

## 扩展方式

多 Agent 的推荐扩展方式是“全局调度，子 Agent 独立 profile”：

1. Orchestrator 负责分解任务、调度子任务、聚合结果。
2. RAGAgent 拥有自己的 profile，只加载与知识检索/生成相关的 skill、MCP 工具、记忆策略和生成风格。
3. CodeAgent 拥有自己的 profile，只加载代码、测试、文件操作、Docker 执行相关能力；默认不挂载 RAG 工具。
4. MemoryHub 统一协调多 Agent 短期记忆和长期记忆写入审批，避免每个子 Agent 直接污染长期记忆。

因此扩展不是“给所有 Agent 塞一个更大的工具池”，而是“每个子 Agent 按角色单独扩展”。用户添加深度学习领域时，应提供深度学习 RAG skill 或 MCP 检索工具，并在 JSON 中声明这些能力属于 RAGAgent；如果还需要 PyTorch/实验代码能力，则在 CodeAgent profile 中声明对应工程 skill 和记忆模板。

## JSON Profile

用户可以把 profile JSON 放到：

```text
configs/multi_agent/profiles/
```

系统会在 P2 任务启动时调用 `load_profiles_for_task(task)`，根据 JSON 中的 `domain`、`aliases`、`auto_activate_keywords` 与用户任务文本匹配。匹配成功后，profile 会覆盖或补充内置 `rag`、`code` profile。

示例：

```json
{
  "enabled": true,
  "name": "deep_learning",
  "domain": "deep_learning",
  "aliases": ["deep learning", "深度学习", "神经网络"],
  "auto_activate_keywords": ["transformer", "深度学习"],
  "profiles": {
    "rag": {
      "type": "rag",
      "name": "deep_learning_rag_agent",
      "domain_style": "deep_learning_research",
      "top_k": 6,
      "synthesis_temperature": 0.15,
      "skills": ["deep_learning_rag"],
      "mcp_tools": ["deep_learning_search", "paper_search"],
      "memory_policy": {
        "read_long_term": true,
        "write_long_term": false,
        "allowed_long_term_kinds": []
      }
    },
    "code": {
      "type": "code",
      "name": "deep_learning_code_agent",
      "code_style": "ml_engineering",
      "use_docker": true,
      "allow_rag_tools": false,
      "skills": ["python_expert"],
      "memory_policy": {
        "read_long_term": true,
        "write_long_term": true,
        "allowed_long_term_kinds": [
          "engineering_experience",
          "debugging_lesson",
          "implementation_pattern",
          "simulation_result"
        ]
      }
    }
  }
}
```

仓库中已经提供示例：

```text
configs/multi_agent/profiles/deep_learning.json
```

## Skill 与 MCP 绑定

JSON profile 中的 `skills` / `skill_names` 是子 Agent 的 skill allowlist。

- RAGAgent 注册 RAG skill 时会检查 allowlist。如果配置了 `skills`，只注册匹配的 RAG skill；如果未配置，则保持原有行为，注册所有可用 RAG skill。
- CodeAgent 的 `skills` 主要用于描述该 profile 所属工程能力，后续可以扩展为精确加载工程 skill。目前 CodeAgent 默认仍使用代码工具和 Docker 执行边界。

JSON profile 中的 `mcp_tools` / `mcp_tool_names` / `tool_names` 是 MCP 工具 allowlist。

- RAG 类 MCP 工具先由 `split_mcp_tools()` 分流到 RAGAgent。
- 如果 RAGAgent profile 配置了 MCP allowlist，则只保留名单内工具。
- Code 类 MCP 工具同理分流到 CodeAgent，并由 CodeAgent profile 的 allowlist 过滤。
- 如果没有配置 allowlist，则保持兼容模式，沿用自动分流后的全部工具。

## 记忆配置

每个子 Agent 的 profile 可以独立配置：

- `memory_policy`
- `memory_write_templates`
- `memory_enabled`
- `long_term_memory_enabled`

这样不同领域可以拥有不同长期记忆分类和写入模板。例如深度学习 CodeAgent 可以把实验结果写成 `simulation_result`，无线通信 CodeAgent 可以把 ISAC 仿真经验写成 `engineering_experience`，医疗/法律 profile 可以配置更严格的长期记忆写入策略。

`engineering_experience` 默认仍需要通过质量过滤和人工审批，审批入口在 dashboard 的 `Multi-Agent Memory` 页面。

## 文件结构

当前多 Agent 相关模块：

```text
dm_agent/multi_agent/
  __init__.py              # 包入口，只导出公共类和加载函数
  runtime.py               # Orchestrator/RAGAgent/CodeAgent 运行时
  profiles.py              # AgentProfile/RAGAgentProfile/CodeAgentProfile
  profile_loader.py        # JSON profile 自动发现与加载
  domain_profiles.py       # 内置领域 profile
  prompts.py               # 多 Agent prompt
  toolkits.py              # MCP/tool 分流与 profile 过滤
  memory.py                # 多 Agent 记忆协调
```

用户扩展目录：

```text
configs/multi_agent/profiles/
dm_agent/skills/custom/
```

推荐流程：

1. 在 `dm_agent/skills/custom/` 放置领域 skill 配置，或在 `mcp_config.json` 中挂载对应 MCP 检索服务。
2. 在 `configs/multi_agent/profiles/` 放置领域 profile JSON。
3. 在任务中使用 profile 的关键词或别名，例如“检索深度学习中 Transformer 的相关内容，并实现一个简单 PyTorch 示例”。
4. P2 多 Agent 自动匹配 profile，RAGAgent 使用领域检索/生成能力，CodeAgent 使用工程能力和 Docker 执行边界。

## 与单 Agent 模式的关系

单 Agent 模式保持原有逻辑：skill 通过描述和选择器触发，RAG 工具可以作为工具挂载给同一个 ReactAgent，同时该 Agent 也具备 coding 能力。

多 Agent 模式采用角色隔离：RAGAgent 负责完整 RAG 链路，CodeAgent 负责工程落地，Orchestrator 负责调度和聚合。profile JSON 只影响多 Agent 模式，不改变单 Agent 的 skill 加载和记忆系统。

## 后续方向

- 将 `runtime.py` 中的 RAGAgent、CodeAgent、Orchestrator 继续物理拆分到独立文件。
- 为 CodeAgent 增加工程 skill allowlist 的精确加载，而不只是 profile metadata。
- 在 dashboard 中展示当前任务命中的 profile、子 Agent skill allowlist 和 MCP allowlist。
- 支持用户在 profile JSON 中声明自定义评估指标，例如 RAG 检索质量、实验复现率、代码测试通过率。
