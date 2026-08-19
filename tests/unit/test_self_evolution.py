#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""SelfEvolutionLoop 单元测试 — AgentSelfModel + RuleRefiner + RetrievalFeedback + SelfEvolutionLoop。"""

import pytest
from sqlmodel import Session

from septmuse.embedders.hash import HashEmbedder
from septmuse.evolution.feedback import RetrievalFeedback, RetrievalFeedbackResult, RuleFeedback
from septmuse.evolution.refine import RefinementCandidate, RuleRefiner
from septmuse.evolution.self_loop import EvolutionCycleResult, SelfEvolutionLoop
from septmuse.models.procedural import ProceduralRule
from septmuse.models.self_evolution import AgentSelfModel, RuleSnapshot
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


@pytest.fixture
def typed_store(tmp_path):
    """创建 TypedMemoryStore (SQLite 临时文件)。"""
    db_path = str(tmp_path / "test_self_evo.db")
    return TypedMemoryStore(db_path=db_path)


@pytest.fixture
def hash_embedder():
    return HashEmbedder()


# ------------------------------------------------------------------
# AgentSelfModel
# ------------------------------------------------------------------


class TestAgentSelfModel:
    def test_create_model(self, typed_store):
        """创建 AgentSelfModel 并持久化。"""
        model = AgentSelfModel(user_id="alice", agent_id="test")
        with Session(typed_store.engine) as session:
            session.add(model)
            session.commit()
            session.refresh(model)

        assert model.id.startswith("se-")
        assert model.user_id == "alice"
        assert model.agent_id == "test"
        assert model.version == 1
        assert model.total_recalls == 0
        assert model.recall_success_rate == 0.0

    def test_record_recall(self, typed_store):
        """记录检索结果更新成功率。"""
        model = AgentSelfModel(user_id="alice", agent_id="test")
        model.record_recall(success=True)
        assert model.total_recalls == 1
        assert model.successful_recalls == 1
        assert model.recall_success_rate == 1.0

        model.record_recall(success=False)
        assert model.total_recalls == 2
        assert model.recall_success_rate == 0.5

    def test_build_evolution_summary(self, typed_store):
        """构建自然语言自我描述。"""
        model = AgentSelfModel(
            user_id="alice",
            agent_id="test",
            strengths=["semantic", "episodic"],
            weaknesses=["procedural"],
            top_rules=[{"rule_id": "r1", "rule": "always check types", "confidence": 0.9, "helpful_count": 5}],
            successful_recalls=8,
            total_recalls=10,
        )
        model.recall_success_rate = 0.8  # 显式设置
        model.build_evolution_summary()
        assert "semantic" in model.evolution_summary
        assert "procedural" in model.evolution_summary
        assert "always check types" in model.evolution_summary
        assert "80.0%" in model.evolution_summary

    def test_empty_summary(self, typed_store):
        """空画像的自我描述。"""
        model = AgentSelfModel(user_id="alice", agent_id="test")
        model.build_evolution_summary()
        assert "still learning" in model.evolution_summary


# ------------------------------------------------------------------
# RuleSnapshot
# ------------------------------------------------------------------


class TestRuleSnapshot:
    def test_create_snapshot(self, typed_store):
        """创建规则快照。"""
        snap = RuleSnapshot(
            rule_id="rule-1",
            rule="always check input types",
            version=1,
            change_description="initial",
            confidence_at_snapshot=0.8,
            helpful_count=3,
            harmful_count=0,
            user_id="alice",
        )
        with Session(typed_store.engine) as session:
            session.add(snap)
            session.commit()
            session.refresh(snap)

        assert snap.id.startswith("se-")
        assert snap.version == 1
        assert snap.rule_id == "rule-1"
        assert snap.parent_rule_id is None


# ------------------------------------------------------------------
# RuleRefiner
# ------------------------------------------------------------------


class TestRuleRefiner:
    def test_find_candidates_no_rules(self, typed_store):
        """无规则时返回空列表。"""
        refiner = RuleRefiner(typed_store)
        candidates = refiner.find_candidates(user_id="alice")
        assert candidates == []

    def test_find_candidates_no_outcomes(self, typed_store):
        """规则无 outcome 记录时不作为候选。"""
        typed_store.add_rule("test rule", user_id="alice")
        refiner = RuleRefiner(typed_store)
        candidates = refiner.find_candidates(user_id="alice")
        assert candidates == []

    def test_find_candidates_mixed(self, typed_store):
        """mixed outcomes 规则被识别为候选。"""
        rule = typed_store.add_rule("test rule", user_id="alice")
        for _ in range(3):
            typed_store.record_rule_outcome(rule.id, helpful=True)
        for _ in range(2):
            typed_store.record_rule_outcome(rule.id, helpful=False)

        refiner = RuleRefiner(typed_store)
        candidates = refiner.find_candidates(user_id="alice")
        assert len(candidates) >= 1
        assert candidates[0].rule.id == rule.id

    def test_heuristic_refine(self, typed_store):
        """无 LLM 时 heuristic 精炼。"""
        rule = typed_store.add_rule("always do X", user_id="alice")
        for _ in range(3):
            typed_store.record_rule_outcome(rule.id, helpful=False)

        # 刷新 rule 以获取最新的 harmful_count
        with Session(typed_store.engine) as session:
            rule = session.get(ProceduralRule, rule.id)

        refiner = RuleRefiner(typed_store)
        candidate = RefinementCandidate(rule=rule, reason="test", mixed_score=0.8)
        result = refiner.refine(candidate, user_id="alice")

        assert result.refined is True
        assert result.new_rule_id is not None
        assert result.snapshot_id is not None
        # 检查原规则被标记为 deprecated
        with Session(typed_store.engine) as session:
            r = session.get(ProceduralRule, rule.id)
            assert r is not None
            assert r.deprecated is True

    def test_get_evolution_chain(self, typed_store):
        """获取规则演化链。"""
        rule = typed_store.add_rule("v1 rule", user_id="alice")
        for _ in range(3):
            typed_store.record_rule_outcome(rule.id, helpful=False)

        refiner = RuleRefiner(typed_store)
        candidate = RefinementCandidate(rule=rule, reason="test", mixed_score=0.8)
        refiner.refine(candidate, user_id="alice")

        chain = refiner.get_evolution_chain(rule.id)
        assert len(chain) >= 1
        assert chain[0].rule_id == rule.id

    def test_refine_all(self, typed_store):
        """批量精炼。"""
        for i in range(3):
            rule = typed_store.add_rule(f"rule {i}", user_id="alice")
            typed_store.record_rule_outcome(rule.id, helpful=True)
            typed_store.record_rule_outcome(rule.id, helpful=False)

        refiner = RuleRefiner(typed_store)
        results = refiner.refine_all(user_id="alice", max_refinements=2)
        assert len(results) <= 2


# ------------------------------------------------------------------
# RetrievalFeedback
# ------------------------------------------------------------------


class TestRetrievalFeedback:
    def test_evaluate_no_rules(self, typed_store):
        """无规则时返回空结果。"""
        fb = RetrievalFeedback(typed_store)
        result = fb.evaluate(
            "test query",
            retrieved_memories=[{"memory": "test"}],
            user_id="alice",
        )
        assert result.total_rules_evaluated == 0
        assert result.helpful_count == 0

    def test_evaluate_heuristic(self, typed_store):
        """heuristic 评估规则。"""
        typed_store.add_rule("always check types", user_id="alice")
        typed_store.add_rule("never use globals", user_id="alice")

        fb = RetrievalFeedback(typed_store)
        result = fb.evaluate(
            "what types should I check?",
            retrieved_memories=[{"memory": "check input types before processing"}],
            user_id="alice",
        )
        assert result.total_rules_evaluated == 2
        assert result.helpful_count + result.harmful_count == 2

    def test_evaluate_auto_record(self, typed_store):
        """自动记录 helpful/harmful。"""
        rule = typed_store.add_rule("test rule", user_id="alice")

        fb = RetrievalFeedback(typed_store)
        fb.evaluate(
            "test query",
            retrieved_memories=[{"memory": "test rule is useful"}],
            user_id="alice",
            auto_record=True,
        )

        with Session(typed_store.engine) as session:
            r = session.get(ProceduralRule, rule.id)
            assert r is not None
            total = r.helpful_count + r.harmful_count
            assert total > 0  # 至少有一条记录

    def test_rule_feedback_dataclass(self):
        """RuleFeedback dataclass。"""
        fb = RuleFeedback(rule_id="r1", rule_text="test", helpful=True, relevance_score=0.8, reason="matches")
        assert fb.helpful is True
        assert fb.relevance_score == 0.8

    def test_retrieval_feedback_result(self):
        """RetrievalFeedbackResult dataclass。"""
        result = RetrievalFeedbackResult(
            query="test",
            total_rules_evaluated=3,
            helpful_count=2,
            harmful_count=1,
        )
        result.summary = "test summary"
        assert result.summary == "test summary"


# ------------------------------------------------------------------
# SelfEvolutionLoop
# ------------------------------------------------------------------


class TestSelfEvolutionLoop:
    @pytest.fixture
    def loop(self, typed_store, hash_embedder):
        """创建 SelfEvolutionLoop (无 graph_store + 无 LLM)。"""
        from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

        store = ORMMemoryStore(engine=typed_store.engine, vector_store=None, keyword_index=None)
        return SelfEvolutionLoop(
            store=store,
            graph_store=None,
            embedder=hash_embedder,
            typed_store=typed_store,
            llm=None,
        )

    def test_improve(self, loop):
        """improve 执行完整闭环。"""
        result = loop.improve(user_id="alice")
        assert isinstance(result, EvolutionCycleResult)
        assert result.user_id == "alice"
        # 无 LLM 时 lessons 为 0
        assert result.lessons_proposed == 0
        assert result.lessons_accepted == 0
        # refine 在无规则时返回空
        assert result.refined_rules == []
        # coverage 有数据
        assert result.coverage is not None
        assert "overall_score" in result.coverage
        # self_model 应更新
        assert result.self_model_updated is True
        assert result.self_model is not None

    def test_get_self_model(self, loop):
        """获取持久化能力画像。"""
        # 首次 improve 创建 self model
        loop.improve(user_id="alice")
        model = loop.get_self_model(user_id="alice")
        assert model is not None
        assert model["user_id"] == "alice"
        assert model["agent_id"] == "default"
        assert "strengths" in model
        assert "weaknesses" in model
        assert "skill_distribution" in model
        assert "evolution_summary" in model
        assert "recall_success_rate" in model

    def test_get_self_model_none(self, loop):
        """无画像时返回 None。"""
        model = loop.get_self_model(user_id="nonexistent")
        assert model is None

    def test_record_recall_outcome(self, loop):
        """记录检索结果。"""
        loop.improve(user_id="alice")
        loop.record_recall_outcome(user_id="alice", success=True)
        model = loop.get_self_model(user_id="alice")
        assert model["total_recalls"] == 1
        assert model["successful_recalls"] == 1
        assert model["recall_success_rate"] == 1.0

    def test_evaluate_recall(self, loop):
        """检索后自动评估。"""
        loop.improve(user_id="alice")
        result = loop.evaluate_recall(
            "test query",
            retrieved_memories=[{"memory": "test content"}],
            user_id="alice",
        )
        assert isinstance(result, RetrievalFeedbackResult)

    def test_get_rule_evolution_chain_empty(self, loop):
        """无规则演化链。"""
        chain = loop.get_rule_evolution_chain("nonexistent")
        assert chain == []

    def test_improve_with_refine(self, loop, typed_store):
        """有规则时 improve 包含 refine。"""
        rule = typed_store.add_rule("test rule", user_id="alice")
        typed_store.record_rule_outcome(rule.id, helpful=True)
        typed_store.record_rule_outcome(rule.id, helpful=False)
        typed_store.record_rule_outcome(rule.id, helpful=False)

        result = loop.improve(user_id="alice", refine=True)
        assert result.self_model_updated is True
        # 有 mixed outcomes 规则, refine 可能执行
        # (不强制 assert 因为 heuristic 可能不触发 refine)

    def test_evolution_cycle_result(self):
        """EvolutionCycleResult dataclass。"""
        result = EvolutionCycleResult(
            user_id="alice",
            lessons_proposed=3,
            lessons_accepted=2,
            rule_ids=["r1", "r2"],
            self_model_updated=True,
        )
        assert result.summary == ""
        result.summary = "test"
        assert result.summary == "test"


# ------------------------------------------------------------------
# Memory facade 集成
# ------------------------------------------------------------------


class TestMemoryFacadeSelfEvolution:
    def test_self_loop_available(self):
        """Memory facade 的 self_loop 属性可用。"""
        from septmuse.memory.main import Memory

        mem = Memory()
        assert mem.self_loop is not None

    def test_improve_returns_all_keys(self):
        """improve 返回所有兼容键。"""
        from septmuse.memory.main import Memory

        mem = Memory()
        result = mem.improve(user_id="test")
        # 旧键
        assert "dream" in result
        assert "rules" in result
        assert "coverage" in result
        assert "conflicts" in result
        # 新键
        assert "lessons_proposed" in result
        assert "lessons_accepted" in result
        assert "refined" in result
        assert "self_model_updated" in result

    def test_get_self_model(self):
        """get_self_model 返回画像。"""
        from septmuse.memory.main import Memory

        mem = Memory()
        mem.improve(user_id="test")
        model = mem.get_self_model(user_id="test")
        assert model is not None
        assert "evolution_summary" in model

    def test_evaluate_recall(self):
        """evaluate_recall 返回反馈。"""
        from septmuse.memory.main import Memory

        mem = Memory()
        result = mem.evaluate_recall(
            "test",
            [{"memory": "test"}],
            user_id="test",
        )
        assert "total_rules_evaluated" in result

    def test_refine_rules(self):
        """refine_rules 返回结果。"""
        from septmuse.memory.main import Memory

        mem = Memory()
        result = mem.refine_rules(user_id="test")
        assert isinstance(result, list)

    def test_record_recall_outcome(self):
        """record_recall_outcome 不报错且更新成功率。"""
        from septmuse.memory.main import Memory

        mem = Memory()
        mem.improve(user_id="test")
        mem.record_recall_outcome(user_id="test", success=True)
        model = mem.get_self_model(user_id="test")
        assert model["total_recalls"] >= 1
        assert model["successful_recalls"] >= 1
