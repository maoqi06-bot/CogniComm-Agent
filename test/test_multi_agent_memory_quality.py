import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.modules.setdefault(
    "faiss",
    SimpleNamespace(
        IndexFlatIP=object,
        IndexFlatL2=object,
        IndexIVFFlat=object,
        IndexHNSWFlat=object,
        METRIC_INNER_PRODUCT=0,
        METRIC_L2=1,
        normalize_L2=lambda vectors: None,
    ),
)
sys.modules.setdefault(
    "langchain_text_splitters",
    SimpleNamespace(RecursiveCharacterTextSplitter=object, Language=object),
)
sys.modules.setdefault("fitz", SimpleNamespace(open=lambda *args, **kwargs: None))
sys.modules.setdefault("rank_bm25", SimpleNamespace(BM25Okapi=object))
sys.modules.setdefault(
    "sentence_transformers",
    SimpleNamespace(CrossEncoder=object, SentenceTransformer=object),
)

from dm_agent.memory.long_term_memory import (
    LongTermMemoryStore,
    MemoryCategory,
    MemoryPriority,
)
from dm_agent.multi_agent.runtime import CodeAgent
from dm_agent.multi_agent.memory import MultiAgentMemoryConfig, MultiAgentMemoryHub
from dm_agent.rag.models import DocumentChunk


class TestMultiAgentMemoryQuality(unittest.TestCase):
    def test_record_subtask_result_resolves_agent_name_from_task_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            timeline_path = str(Path(tmp) / "timeline.jsonl")
            hub = MultiAgentMemoryHub(
                config=MultiAgentMemoryConfig(
                    async_long_term_writes=False,
                    timeline_path=timeline_path,
                )
            )
            task = SimpleNamespace(
                id="task_1",
                type=SimpleNamespace(value="knowledge_query"),
                result={"success": True, "result": "useful result"},
                status="completed",
                agent_name="",
            )

            hub.record_subtask_result(task)

            rows = Path(timeline_path).read_text(encoding="utf-8")
            self.assertIn('"agent_name": "rag_agent"', rows)
            self.assertNotIn("unknown_agent", rows)

    def test_low_value_task_summary_is_not_written_long_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_manager = MagicMock()
            hub = MultiAgentMemoryHub(
                memory_manager=memory_manager,
                config=MultiAgentMemoryConfig(
                    async_long_term_writes=False,
                    timeline_path=str(Path(tmp) / "timeline.jsonl"),
                ),
            )

            hub.store_task_summary(
                original_task="tiny task",
                final_answer="No relevant knowledge found",
                completed_count=1,
                failed_count=0,
            )

            memory_manager.add_memory.assert_not_called()

    def test_code_agent_infers_common_python_requirements(self):
        agent = CodeAgent.__new__(CodeAgent)

        requirements = agent._infer_python_requirements(
            "import numpy as np\nfrom matplotlib import pyplot as plt\n",
            [],
        )

        self.assertIn("numpy", requirements)
        self.assertIn("matplotlib", requirements)

    def test_code_agent_infers_shell_python_script_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "demo.py"
            script_path.write_text("import numpy as np\n", encoding="utf-8")
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                agent = CodeAgent.__new__(CodeAgent)
                requirements = agent._infer_shell_requirements("python demo.py", [])
            finally:
                os.chdir(old_cwd)

        self.assertIn("numpy", requirements)


class TestLongTermMemoryRecoveryQuality(unittest.TestCase):
    def _store_without_init(self):
        store = LongTermMemoryStore.__new__(LongTermMemoryStore)
        store.vector_store = SimpleNamespace(id_to_chunk={})
        store._memory_index = {}
        store._category_index = {cat: set() for cat in MemoryCategory}
        store.default_category = MemoryCategory.CONVERSATION_SUMMARY
        store.default_priority = MemoryPriority.NORMAL
        return store

    def test_recovery_preserves_vector_metadata(self):
        store = self._store_without_init()
        store.vector_store.id_to_chunk["mem-1"] = DocumentChunk(
            id="mem-1",
            document_id="mem-1",
            chunk_index=0,
            content="User prefers concise engineering summaries with verification notes.",
            metadata={
                "memory_id": "mem-1",
                "category": "user_preference",
                "priority": 4,
                "importance_score": 0.72,
                "source": "multi_agent",
                "tags": ["multi_agent", "user_preference"],
                "created_at": 100.0,
                "updated_at": 150.0,
            },
        )

        self.assertEqual(store._recover_entries_from_vector_store(), 1)
        entry = store._memory_index["mem-1"]
        self.assertEqual(entry.priority, MemoryPriority.HIGH)
        self.assertEqual(entry.importance_score, 0.72)
        self.assertEqual(entry.source, "multi_agent")
        self.assertEqual(entry.created_at, 100.0)
        self.assertEqual(entry.updated_at, 150.0)
        self.assertIn("user_preference", entry.tags)

    def test_recovery_rejects_mojibake_vector_content(self):
        store = self._store_without_init()
        store.vector_store.id_to_chunk["bad"] = DocumentChunk(
            id="bad",
            document_id="bad",
            chunk_index=0,
            content="鐢ㄦ埛鍊惧悜浜庝娇鐢ㄧ粨鏋勫寲鐨勬祴璇曟墽琛岃鍒掞紝鍖呮嫭鐩綍妫€鏌ャ€?",
            metadata={"memory_id": "bad", "source": "multi_agent"},
        )

        self.assertEqual(store._recover_entries_from_vector_store(), 0)
        self.assertNotIn("bad", store._memory_index)


if __name__ == "__main__":
    unittest.main()
