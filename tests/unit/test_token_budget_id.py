"""BudgetItem.id 字段测试 (修 recall id 丢失 bug)."""
from septmuse.retrieval.token_budget import BudgetItem


def test_budget_item_id_default():
    b = BudgetItem(text="hello", score=0.5)
    assert b.id is None


def test_budget_item_id_set():
    b = BudgetItem(text="hello", score=0.5, id="mem-1")
    assert b.id == "mem-1"


def test_budget_item_id_in_fit(tmp_path):
    """id 穿过 token_budget.fit 保留."""
    import os
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    from septmuse.retrieval.token_budget import TokenBudget
    tb = TokenBudget(budget=2000)
    items = [
        BudgetItem(text="first memory", score=0.9, id="mem-aaa"),
        BudgetItem(text="second memory", score=0.5, id="mem-bbb"),
    ]
    result = tb.fit(items)
    # 所有返回项保留原始 id
    for item in result.items:
        assert item.id in ("mem-aaa", "mem-bbb")
