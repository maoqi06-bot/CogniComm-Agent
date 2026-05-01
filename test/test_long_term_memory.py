"""长期记忆模块测试"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
import shutil


class TestMemoryEntry(unittest.TestCase):
    """测试 MemoryEntry 数据类。"""

    def test_01_create_entry(self):
        """测试创建记忆条目。"""
        from dm_agent.memory.long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority

        entry = MemoryEntry(
            id="test-001",
            content="用户喜欢使用 pytest 进行测试",
            category=MemoryCategory.USER_PREFERENCE,
            priority=MemoryPriority.HIGH,
            importance_score=0.8,
            tags={"testing", "pytest"},
        )

        self.assertEqual(entry.id, "test-001")
        self.assertEqual(entry.category, MemoryCategory.USER_PREFERENCE)
        self.assertEqual(entry.priority, MemoryPriority.HIGH)
        self.assertIn("pytest", entry.tags)
        print("[OK] MemoryEntry creation")

    def test_02_to_dict_from_dict(self):
        """测试序列化/反序列化。"""
        from dm_agent.memory.long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority

        entry = MemoryEntry(
            id="test-serial-001",
            content="Test memory",
            category=MemoryCategory.CONVERSATION_SUMMARY,
            priority=MemoryPriority.NORMAL,
        )

        data = entry.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["content"], "Test memory")

        entry2 = MemoryEntry.from_dict(data)
        self.assertEqual(entry2.content, entry.content)
        self.assertEqual(entry2.category, entry.category)
        print("[OK] Serialization/deserialization")

    def test_03_decay_score_calculation(self):
        """测试记忆衰减计算。"""
        from dm_agent.memory.long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority

        entry = MemoryEntry(
            id="test",
            content="Test memory",
            category=MemoryCategory.CONVERSATION_SUMMARY,
            priority=MemoryPriority.NORMAL,
            importance_score=0.8,
            decay_factor=1.0,
        )

        score = entry.calculate_decay_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        print(f"[OK] Decay score: {score:.4f}")

    def test_04_critical_priority_no_decay(self):
        """测试 CRITICAL 优先级记忆不衰减。"""
        from dm_agent.memory.long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority

        entry = MemoryEntry(
            id="test",
            content="Critical memory never decays",
            category=MemoryCategory.IMPORTANT_FACT,
            priority=MemoryPriority.CRITICAL,
            importance_score=0.95,
        )

        score = entry.calculate_decay_score()
        self.assertAlmostEqual(score, 0.95, places=2)
        print("[OK] CRITICAL priority no decay")

    def test_05_pinned_memory_no_decay(self):
        """测试固定记忆不会衰减。"""
        from dm_agent.memory.long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority
        import time

        entry = MemoryEntry(
            id="test",
            content="Pinned important memory",
            category=MemoryCategory.IMPORTANT_FACT,
            priority=MemoryPriority.HIGH,
            importance_score=0.9,
            is_pinned=True,
        )

        for _ in range(3):
            score = entry.calculate_decay_score()
            self.assertAlmostEqual(score, 0.9, places=2)
            time.sleep(0.05)
        print("[OK] Pinned memory no decay")

    def test_06_access_boost(self):
        """测试访问增加时的重要性提升。"""
        from dm_agent.memory.long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority

        entry = MemoryEntry(
            id="test",
            content="Frequently accessed memory",
            category=MemoryCategory.CONVERSATION_SUMMARY,
            priority=MemoryPriority.NORMAL,
            importance_score=0.5,
            access_count=10,
        )

        score = entry.calculate_decay_score()
        self.assertGreater(score, 0.5)
        print(f"[OK] Access boost: {score:.4f} > 0.5")


class TestMemoryCategoryPriority(unittest.TestCase):
    """测试记忆类别和优先级。"""

    def test_01_category_values(self):
        """测试所有记忆类别。"""
        from dm_agent.memory.long_term_memory import MemoryCategory

        categories = list(MemoryCategory)
        self.assertGreater(len(categories), 5)

        expected = [
            "user_preference",
            "project_context",
            "important_fact",
            "working_state",
            "skill_knowledge",
            "conversation_summary",
        ]
        actual = [c.value for c in categories]
        for e in expected:
            self.assertIn(e, actual)
        print(f"[OK] Categories: {actual}")

    def test_02_priority_values(self):
        """测试所有优先级。"""
        from dm_agent.memory.long_term_memory import MemoryPriority

        priorities = list(MemoryPriority)
        self.assertEqual(len(priorities), 5)

        values = [p.value for p in priorities]
        self.assertEqual(values, sorted(values, reverse=True))
        print(f"[OK] Priority values: {values}")


class TestLongTermMemoryStoreRequiresFaiss(unittest.TestCase):
    """测试 LongTermMemoryStore 类（需要 faiss 和 API key）。"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.storage_path = os.path.join(cls.test_dir, "test_memory")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss not installed")

    def test_store_creation_only(self):
        """测试存储创建（不调用 API）。"""
        from unittest.mock import MagicMock, patch
        from dm_agent.memory.long_term_memory import LongTermMemoryStore, FAISSVectorStore

        mock_embeddings = MagicMock()
        mock_embeddings.dimension = 1536

        with patch.object(FAISSVectorStore, '__init__', lambda self, **kw: None):
            with patch.object(FAISSVectorStore, 'save', lambda self: None):
                store = LongTermMemoryStore(
                    storage_path=self.storage_path,
                    embeddings=mock_embeddings,
                    max_memories=100,
                )
                store.vector_store = MagicMock()
                store.vector_store.id_to_chunk = {}
                store._memory_index = {}
                store._category_index = {cat: set() for cat in store._category_index}

        self.assertIsNotNone(store)
        self.assertEqual(len(store._memory_index), 0)
        print("[OK] Store creation (mock)")


class TestMemoryManagerRequiresFaiss(unittest.TestCase):
    """测试 MemoryManager 类（需要 faiss 和 API key）。"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.storage_path = os.path.join(cls.test_dir, "test_manager_memory")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_manager_creation_only(self):
        """测试管理器创建（不调用 API）。"""
        from unittest.mock import MagicMock, patch
        from dm_agent.memory.memory_manager import MemoryManager
        from dm_agent.memory.long_term_memory import LongTermMemoryStore, FAISSVectorStore

        mock_embeddings = MagicMock()
        mock_embeddings.dimension = 1536

        with patch.object(FAISSVectorStore, '__init__', lambda self, **kw: None):
            with patch.object(FAISSVectorStore, 'save', lambda self: None):
                store = LongTermMemoryStore(
                    storage_path=self.storage_path,
                    embeddings=mock_embeddings,
                )
                store.vector_store = MagicMock()
                store.vector_store.id_to_chunk = {}
                store._memory_index = {}
                store._category_index = {cat: set() for cat in store._category_index}

                manager = MemoryManager(
                    memory_store=store,
                    llm_client=None,
                )

        self.assertIsNotNone(manager)
        self.assertIsNotNone(manager.memory_store)
        print("[OK] Manager creation (mock)")


class TestMemoryManagerQualityFilters(unittest.TestCase):
    def test_operational_encoding_failures_are_filtered_from_retrieval(self):
        from unittest.mock import MagicMock
        from dm_agent.memory.memory_manager import MemoryManager
        from dm_agent.memory.long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority, MemorySearchResult

        bad_entry = MemoryEntry(
            id="bad",
            content="RAG \u5de5\u5177 mcp_wireless-rag_search \u56e0 GBK \u7f16\u7801\u4e0e emoji \u8f93\u51fa\u4e0d\u517c\u5bb9\u800c\u5931\u8d25\u3002",
            category=MemoryCategory.SKILL_KNOWLEDGE,
            priority=MemoryPriority.HIGH,
            tags={"error", "GBK", "emoji"},
        )
        good_entry = MemoryEntry(
            id="good",
            content="\u7528\u6237\u6b63\u5728\u5b66\u4e60 ISAC \u4e0e\u65e0\u7ebf\u901a\u4fe1\u57fa\u7840\u6982\u5ff5\u3002",
            category=MemoryCategory.PROJECT_CONTEXT,
            priority=MemoryPriority.NORMAL,
            tags={"wireless", "isac"},
        )

        store = MagicMock()
        store.search.return_value = [
            MemorySearchResult(entry=bad_entry, score=0.99),
            MemorySearchResult(entry=good_entry, score=0.8),
        ]
        store.update.return_value = None

        manager = MemoryManager(memory_store=store, llm_client=None)
        result = manager.retrieve_for_context("\u68c0\u7d22 ISAC", limit=5)

        self.assertEqual([item.entry.id for item in result.memories], ["good"])
        self.assertNotIn("GBK", result.enhanced_context)

    def test_operational_encoding_failures_are_not_stored_after_extraction(self):
        from unittest.mock import MagicMock
        from dm_agent.memory.memory_manager import MemoryManager
        from dm_agent.memory.long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority

        store = MagicMock()
        manager = MemoryManager(memory_store=store, llm_client=None)

        manager._extract_by_rules = MagicMock(return_value=[
            MemoryEntry(
                id="",
                content="RAG \u5de5\u5177 mcp_wireless-rag_search \u5728\u5f53\u524d\u73af\u5883\u56e0 GBK \u7f16\u7801\u9519\u8bef\u65e0\u6cd5\u6267\u884c\u3002",
                category=MemoryCategory.SKILL_KNOWLEDGE,
                priority=MemoryPriority.HIGH,
                tags={"error", "troubleshooting"},
            )
        ])

        stored = manager.extract_and_store(
            conversation_history=[{"role": "user", "content": "\u68c0\u7d22 ISAC"}],
            current_task="\u68c0\u7d22 ISAC",
        )

        self.assertEqual(stored, [])
        store.add.assert_not_called()


class FakeMemoryStore:
    def __init__(self, search_results=None):
        from dm_agent.memory.long_term_memory import MemoryCategory

        self.search_results = search_results or []
        self.added = []
        self.updated = []
        self._memory_index = {result.entry.id: result.entry for result in self.search_results}
        self._category_index = {cat: set() for cat in MemoryCategory}

    def search(self, **kwargs):
        return self.search_results

    def add(self, **kwargs):
        from dm_agent.memory.long_term_memory import MemoryEntry

        entry = MemoryEntry(id=f"new-{len(self.added) + 1}", **kwargs)
        self.added.append(entry)
        self._memory_index[entry.id] = entry
        return entry

    def update(self, memory_id, **kwargs):
        entry = self._memory_index.get(memory_id)
        if not entry:
            return None
        self.updated.append((memory_id, kwargs))
        if kwargs.get("increment_access"):
            entry.access_count += 1
        for key, value in kwargs.items():
            if key == "increment_access":
                continue
            if value is not None:
                setattr(entry, key, value)
        return entry

    def get(self, memory_id, increment_access=True):
        return self._memory_index.get(memory_id)

    def get_statistics(self):
        return {}


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def respond(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.response


class TestSmartMemoryUpdates(unittest.TestCase):
    def _result(self, entry, score=0.95):
        from dm_agent.memory.long_term_memory import MemorySearchResult

        return MemorySearchResult(entry=entry, score=score)

    def _entry(self, content, memory_id="old"):
        from dm_agent.memory.long_term_memory import MemoryEntry, MemoryCategory, MemoryPriority

        return MemoryEntry(
            id=memory_id,
            content=content,
            category=MemoryCategory.USER_PREFERENCE,
            priority=MemoryPriority.HIGH,
            importance_score=0.7,
            tags={"preference"},
        )

    def test_duplicate_memory_is_ignored_without_extra_add(self):
        from dm_agent.memory.memory_manager import MemoryManager
        from dm_agent.memory.long_term_memory import MemoryCategory

        old = self._entry("User prefers pytest for tests.")
        store = FakeMemoryStore([self._result(old)])
        manager = MemoryManager(memory_store=store, llm_client=FakeLLM("{}"))

        result = manager.add_memory("User prefers pytest for tests.", MemoryCategory.USER_PREFERENCE)

        self.assertEqual(result.id, old.id)
        self.assertEqual(store.added, [])
        self.assertEqual(old.access_count, 1)

    def test_update_existing_merges_new_detail(self):
        from dm_agent.memory.memory_manager import MemoryManager
        from dm_agent.memory.long_term_memory import MemoryCategory

        old = self._entry("User prefers pytest.")
        store = FakeMemoryStore([self._result(old)])
        llm = FakeLLM('{"action":"update_existing","target_memory_id":"old","updated_content":"User prefers pytest with focused unit tests.","tags":["pytest","unit"],"reason":"Adds useful detail."}')
        manager = MemoryManager(memory_store=store, llm_client=llm)

        result = manager.add_memory("User prefers focused unit tests in pytest.", MemoryCategory.USER_PREFERENCE)

        self.assertEqual(result.id, old.id)
        self.assertEqual(result.content, "User prefers pytest with focused unit tests.")
        self.assertIn("memory_update_history", result.metadata)
        self.assertEqual(store.added, [])

    def test_replace_existing_when_new_preference_conflicts(self):
        from dm_agent.memory.memory_manager import MemoryManager
        from dm_agent.memory.long_term_memory import MemoryCategory

        old = self._entry("User prefers unittest.")
        store = FakeMemoryStore([self._result(old)])
        llm = FakeLLM('{"action":"replace_existing","target_memory_id":"old","updated_content":"User prefers pytest.","reason":"Newest explicit preference wins."}')
        manager = MemoryManager(memory_store=store, llm_client=llm)

        result = manager.add_memory("User prefers pytest.", MemoryCategory.USER_PREFERENCE)

        self.assertEqual(result.id, old.id)
        self.assertEqual(result.content, "User prefers pytest.")
        self.assertEqual(store.added, [])

    def test_similar_but_independent_memory_is_created(self):
        from dm_agent.memory.memory_manager import MemoryManager
        from dm_agent.memory.long_term_memory import MemoryCategory

        old = self._entry("User prefers pytest.")
        store = FakeMemoryStore([self._result(old)])
        llm = FakeLLM('{"action":"create_new","reason":"Different topic."}')
        manager = MemoryManager(memory_store=store, llm_client=llm)

        result = manager.add_memory("User prefers Ruff for linting.", MemoryCategory.USER_PREFERENCE)

        self.assertNotEqual(result.id, old.id)
        self.assertEqual(len(store.added), 1)

    def test_invalid_llm_resolution_falls_back_to_create_new(self):
        from dm_agent.memory.memory_manager import MemoryManager
        from dm_agent.memory.long_term_memory import MemoryCategory

        old = self._entry("User prefers pytest.")
        store = FakeMemoryStore([self._result(old)])
        manager = MemoryManager(memory_store=store, llm_client=FakeLLM("not json"))

        result = manager.add_memory("User prefers Ruff for linting.", MemoryCategory.USER_PREFERENCE)

        self.assertNotEqual(result.id, old.id)
        self.assertEqual(len(store.added), 1)

    def test_no_llm_exact_duplicate_does_not_create_new(self):
        from dm_agent.memory.memory_manager import MemoryManager
        from dm_agent.memory.long_term_memory import MemoryCategory

        old = self._entry("User prefers pytest.")
        store = FakeMemoryStore([self._result(old)])
        manager = MemoryManager(memory_store=store, llm_client=None)

        result = manager.add_memory("User prefers pytest.", MemoryCategory.USER_PREFERENCE)

        self.assertEqual(result.id, old.id)
        self.assertEqual(store.added, [])


class TestVectorRebuildOnContentUpdate(unittest.TestCase):
    def test_update_rebuilds_vector_chunks_with_new_content(self):
        from unittest.mock import patch
        from dm_agent.memory.long_term_memory import LongTermMemoryStore, MemoryCategory

        class FakeEmbeddings:
            dimension = 3

            def embed(self, texts):
                import numpy as np

                return np.ones((len(texts), 3), dtype="float32")

            def embed_query(self, text):
                import numpy as np

                return np.ones(3, dtype="float32")

        class FakeVectorStore:
            def __init__(self, **kwargs):
                self.id_to_chunk = {}
                self.position_to_id = []
                self.index_type = kwargs.get("index_type", "Flat")
                self.metric = kwargs.get("metric", "cosine")

            def add_chunks(self, chunks, embeddings=None):
                for chunk in chunks:
                    self.id_to_chunk[chunk.id] = chunk
                    self.position_to_id.append(chunk.id)

            def save(self):
                pass

            def search(self, query, k=5):
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("dm_agent.memory.long_term_memory.FAISSVectorStore", FakeVectorStore):
                store = LongTermMemoryStore(storage_path=tmpdir, embeddings=FakeEmbeddings())
                entry = store.add("old vector content", category=MemoryCategory.PROJECT_CONTEXT)
                store.update(entry.id, content="new vector content")

                self.assertEqual(store.vector_store.id_to_chunk[entry.id].content, "new vector content")
                self.assertEqual(store.vector_store.position_to_id, [entry.id])


def run_tests():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("Testing Long-Term Memory Module...")
    print("=" * 60 + "\n")

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryEntry))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryCategoryPriority))
    suite.addTests(loader.loadTestsFromTestCase(TestLongTermMemoryStoreRequiresFaiss))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagerRequiresFaiss))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagerQualityFilters))
    suite.addTests(loader.loadTestsFromTestCase(TestSmartMemoryUpdates))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorRebuildOnContentUpdate))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
