import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

from dm_agent.memory.memory_admin import (
    RESET_CONFIRMATION_TEXT,
    filter_memory_rows,
    is_reset_confirmed,
    load_memory_metadata,
    quarantine_memory_files,
)


class TestDashboardMemoryAdmin(unittest.TestCase):
    def test_load_memory_metadata_missing_or_invalid_returns_empty_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            self.assertEqual(load_memory_metadata(missing)["memories"], [])

            invalid = Path(tmp) / "invalid.json"
            invalid.write_text("{not-json", encoding="utf-8")
            loaded = load_memory_metadata(invalid)
            self.assertEqual(loaded["total_memories"], 0)
            self.assertEqual(loaded["memories"], [])

    def test_filter_memory_rows_uses_category_source_tag_and_query(self):
        rows = [
            {
                "id": "a",
                "content": "alpha memory",
                "category": "user_preference",
                "source": "multi_agent",
                "tags": ["ui"],
            },
            {
                "id": "b",
                "content": "beta memory",
                "category": "skill_knowledge",
                "source": "manual",
                "tags": ["code"],
            },
        ]

        filtered = filter_memory_rows(
            rows,
            category="user_preference",
            source="multi_agent",
            tag="ui",
            query="alpha",
        )

        self.assertEqual([row["id"] for row in filtered], ["a"])

    def test_reset_confirmation_is_exact(self):
        self.assertTrue(is_reset_confirmed(RESET_CONFIRMATION_TEXT))
        self.assertFalse(is_reset_confirmed("reset memory"))
        self.assertFalse(is_reset_confirmed(""))

    def test_quarantine_memory_files_moves_known_artifacts_and_recreates_memory_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "dm_agent" / "data" / "memory"
            data_dir = root / "data"
            memory_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            (memory_dir / "memory_metadata.json").write_text(
                json.dumps({"memories": [{"id": "old"}]}),
                encoding="utf-8",
            )
            (memory_dir / "memory_index.meta.json").write_text("{}", encoding="utf-8")
            (data_dir / "multi_agent_memory_timeline.jsonl").write_text("{}", encoding="utf-8")
            (data_dir / "multi_agent_memory_approvals.json").write_text("[]", encoding="utf-8")

            quarantine_dir = quarantine_memory_files(root, timestamp="20260430_120000")

            self.assertTrue(memory_dir.exists())
            self.assertEqual(list(memory_dir.iterdir()), [])
            self.assertTrue(
                (quarantine_dir / "dm_agent" / "data" / "memory" / "memory_metadata.json").exists()
            )
            self.assertTrue(
                (quarantine_dir / "data" / "multi_agent_memory_timeline.jsonl").exists()
            )
            self.assertTrue(
                (quarantine_dir / "data" / "multi_agent_memory_approvals.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
