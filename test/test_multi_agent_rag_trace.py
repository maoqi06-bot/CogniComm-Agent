import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dm_agent.multi_agent.runtime import OrchestratorAgent, RAGAgent, SubTask, TaskType
from dm_agent.tools.base import Tool


class FakeClient:
    def respond(self, messages, temperature=0.2):
        return "RAGAgent synthesized answer"


class FakeMcpTool:
    name = "wireless_comm_search"
    description = "Search wireless RAG"

    def execute(self, arguments):
        trace_id = arguments["trace_id"]
        contexts = ["retrieved ISAC beamforming context"]
        trace_path = Path("data") / "traces" / f"{trace_id}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "nodes": [
                        {
                            "method": "Hybrid_Recall",
                            "provider": "fake_mcp",
                            "input_data": arguments["query"],
                            "output_data": [{"doc": contexts[0], "score": 0.9}],
                        }
                    ],
                    "metadata": {
                "retrieved_contexts": contexts,
                "rag_eval_samples": [
                    {
                        "question": arguments["query"],
                        "contexts": contexts,
                        "answer": "raw retrieval dump",
                        "source": "wireless_comm_search",
                    }
                ],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return "raw retrieval dump"


class FakeSearchSkill:
    def get_metadata(self):
        return type("Meta", (), {"name": "fake_search_skill"})()

    def get_tools(self):
        return [Tool(name="wireless_comm_search", description="Search wireless RAG", runner=lambda args: "ok")]


class FakeSkillManager:
    skills = {"fake_search_skill": FakeSearchSkill()}


class FakeTracer:
    def __init__(self):
        self.metadata = {}
        self.nodes = []

    def add_node(self, method, provider, input_val=None):
        node = {"method": method, "provider": provider, "input_data": input_val}
        self.nodes.append(node)
        return node

    def end_node(self, node, output_val=None, metadata=None):
        node["output_data"] = output_val
        if metadata:
            node["metadata"] = metadata


def test_rag_agent_mcp_trace_records_synthesis_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = FakeTracer()
    agent = RAGAgent(client=FakeClient(), mcp_tools=[FakeMcpTool()])
    agent.current_trace_id = "query_test"
    agent._tracer = tracer

    result = agent.process(
        SubTask(
            id="rag_task",
            type=TaskType.KNOWLEDGE_QUERY,
            description="检索ISAC波束赋形内容",
        )
    )

    assert result["success"] is True
    assert result["answer"] == "RAGAgent synthesized answer"
    assert tracer.metadata["retrieved_contexts"] == ["retrieved ISAC beamforming context"]
    sample = tracer.metadata["rag_eval_samples"][0]
    assert sample["question"] == "检索ISAC波束赋形内容"
    assert sample["contexts"] == ["retrieved ISAC beamforming context"]
    assert sample["answer"] == "RAGAgent synthesized answer"
    assert sample["source"] == "rag_agent:synthesis"


def test_orchestrator_registers_non_base_rag_skill_tools():
    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.rag_agent = RAGAgent(client=FakeClient(), mcp_tools=[])
    orchestrator.logger = type("Logger", (), {"info": lambda self, msg: None})()

    orchestrator._register_rag_skills(FakeSkillManager())

    assert [tool.name for tool in orchestrator.rag_agent._mcp_tools] == ["wireless_comm_search"]
