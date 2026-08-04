"""TypedMemoryStore(engine=) 共享 engine 验证。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlmodel import Session, create_engine, select

from septmuse.models.episodic import EpisodeType, EpisodicEvent
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


def test_typed_store_with_shared_engine(tmp_path):
    """传入 engine 时使用共享 engine，不自建。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'shared.db'}")
    store = TypedMemoryStore(engine=engine)
    assert store.engine is engine  # 同一对象

    # 验证 CRUD 正常
    with Session(engine) as session:
        episode = EpisodicEvent(
            content="测试事件",
            event_type=EpisodeType.FACT,
            user_id="u1",
        )
        session.add(episode)
        session.commit()
        stmt = select(EpisodicEvent).where(EpisodicEvent.user_id == "u1")
        result = session.exec(stmt).first()
        assert result is not None
        assert result.content == "测试事件"


def test_typed_store_backward_compat_db_path(tmp_path):
    """旧构造（db_path=）仍可用。"""
    store = TypedMemoryStore(db_path=str(tmp_path / "compat.db"))
    assert store.engine is not None

    with Session(store.engine) as session:
        episode = EpisodicEvent(
            content="兼容测试",
            event_type=EpisodeType.FACT,
            user_id="u2",
        )
        session.add(episode)
        session.commit()
        stmt = select(EpisodicEvent).where(EpisodicEvent.user_id == "u2")
        result = session.exec(stmt).first()
        assert result is not None


def test_typed_store_engine_takes_priority_over_db_path(tmp_path):
    """同时传 engine 和 db_path 时 engine 优先。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'priority.db'}")
    store = TypedMemoryStore(db_path=str(tmp_path / "ignored.db"), engine=engine)
    assert store.engine is engine  # engine 赢
