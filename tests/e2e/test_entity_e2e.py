"""实体抽取 e2e 测试 (跨会话持久化)。"""


class TestEntityE2E:
    """跨会话实体持久化 + delete 清理 + 中文实体。"""

    def test_cross_session_persistence(self, tmp_path):
        """add → close → reopen → search_entities 跨会话持久化。"""
        db = str(tmp_path / "e2e.db")

        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        config = MemoryConfig(db_path=db)
        m = ExperimentalMemory(config=config)
        m.add("Alice works at Google in London", user_id="u1")
        m.close()

        m2 = ExperimentalMemory(config=MemoryConfig(db_path=db))
        entities = m2.search_entities("Google", user_id="u1")
        assert any(e["entity_text"] == "Google" for e in entities)

        all_entities = m2.list_entities(user_id="u1")
        assert len(all_entities) > 0
        m2.close()

    def test_delete_cleans_entity_refs(self, tmp_path):
        """add → delete → search_entities 引用清理。"""
        db = str(tmp_path / "e2e.db")

        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=db))
        result = m.add("Alice works at Google", user_id="u1")
        memory_id = result["results"][0]["id"]

        entities = m.list_entities(user_id="u1")
        assert len(entities) > 0

        m.delete(memory_id)
        m.close()

        m2 = ExperimentalMemory(config=MemoryConfig(db_path=db))
        entities_after = m2.list_entities(user_id="u1")
        google = [e for e in entities_after if e["entity_text"] == "Google"]
        assert len(google) == 0
        m2.close()

    def test_chinese_entity_extraction(self, tmp_path):
        """中文实体抽取。"""
        db = str(tmp_path / "e2e.db")

        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=db))
        m.add("张三在北京的百度公司工作", user_id="u1")

        entities = m.list_entities(user_id="u1")
        texts = {e["entity_text"] for e in entities}
        assert len(entities) > 0
        assert "北京" in texts or "百度" in texts or "张三" in texts
        m.close()
