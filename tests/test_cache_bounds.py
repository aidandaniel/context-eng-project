"""Bounded TTL+LRU cache behavior (H1 memory bounds)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from context_eng import server as srv
from context_eng.cache import TtlLruCache
from context_eng.config import Config
from context_eng.engine import ContextEngine


@pytest.fixture(autouse=True)
def _clear_server_caches():
    srv._engines.clear()
    srv._bundle_owners.clear()
    yield
    srv._engines.clear()
    srv._bundle_owners.clear()


def test_ttl_lru_evicts_least_recently_used():
    cache: TtlLruCache[str, int] = TtlLruCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache.get("a") == 1  # touch a
    cache["c"] = 3  # should evict b
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_ttl_lru_expires_idle_entries():
    cache: TtlLruCache[str, str] = TtlLruCache(maxsize=4, ttl_seconds=10.0)
    with patch("context_eng.cache.monotonic", return_value=100.0):
        cache["x"] = "keep"
    with patch("context_eng.cache.monotonic", return_value=111.0):
        assert cache.get("x") is None
        assert len(cache) == 0


def test_ttl_lru_get_refreshes_idle_ttl():
    cache: TtlLruCache[str, str] = TtlLruCache(maxsize=4, ttl_seconds=10.0)
    with patch("context_eng.cache.monotonic", return_value=0.0):
        cache["x"] = "v"
    with patch("context_eng.cache.monotonic", return_value=9.0):
        assert cache.get("x") == "v"  # refresh expiry to 19
    with patch("context_eng.cache.monotonic", return_value=18.0):
        assert cache.get("x") == "v"


def test_engine_bundle_cache_evicts_oldest(tmp_path: Path):
    (tmp_path / "hello.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    cfg = Config(
        workspace_root=tmp_path,
        max_cached_bundles=2,
        bundle_ttl_seconds=3600.0,
    )
    engine = ContextEngine(config=cfg)
    b1 = engine.get_context_bundle("explain hello.py", max_tokens=2000)
    b2 = engine.get_context_bundle("explain hello.py again", max_tokens=2000)
    b3 = engine.get_context_bundle("explain hello.py third", max_tokens=2000)
    assert engine.analysis_for_bundle(b1.bundle_id) is None
    assert engine.analysis_for_bundle(b2.bundle_id) is not None
    assert engine.analysis_for_bundle(b3.bundle_id) is not None
    with pytest.raises(KeyError):
        engine.expand_context(b1.bundle_id)


def test_engine_bundle_cache_ttl_expires(tmp_path: Path):
    (tmp_path / "hello.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    cfg = Config(
        workspace_root=tmp_path,
        max_cached_bundles=8,
        bundle_ttl_seconds=10.0,
    )
    engine = ContextEngine(config=cfg)
    with patch("context_eng.cache.monotonic", return_value=0.0):
        bundle = engine.get_context_bundle("explain hello.py", max_tokens=2000)
        bid = bundle.bundle_id
        assert engine.analysis_for_bundle(bid) is not None
    with patch("context_eng.cache.monotonic", return_value=11.0):
        assert engine.analysis_for_bundle(bid) is None


def test_server_engine_cache_evicts_and_clears_owners(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENG_ALLOWED_ROOTS", str(tmp_path))
    roots = []
    for i in range(3):
        root = tmp_path / f"proj{i}"
        root.mkdir()
        (root / "a.py").write_text("x = 1\n", encoding="utf-8")
        roots.append(root)

    monkeypatch.setattr(
        srv,
        "_engines",
        TtlLruCache(maxsize=2, on_evict=srv._on_engine_evict),
    )
    monkeypatch.setattr(
        srv,
        "_bundle_owners",
        TtlLruCache(maxsize=64, ttl_seconds=1800.0),
    )

    e0 = srv.get_engine(str(roots[0]))
    result = srv.get_context_bundle("explain a.py", workspace_root=str(roots[0]))
    bid = result["bundle_id"]
    assert srv._bundle_owners.get(bid) is e0

    srv.get_engine(str(roots[1]))
    srv.get_engine(str(roots[2]))  # evicts oldest engine (proj0)

    assert srv._engines.get(roots[0].resolve().as_posix()) is None
    assert srv._bundle_owners.get(bid) is None


def test_track_bundle_uses_bounded_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        srv, "_bundle_owners", TtlLruCache(maxsize=2, ttl_seconds=1800.0)
    )
    engine = ContextEngine(config=Config(workspace_root=tmp_path))
    srv._track_bundle(engine, "b1")
    srv._track_bundle(engine, "b2")
    srv._track_bundle(engine, "b3")
    assert srv._bundle_owners.get("b1") is None
    assert srv._bundle_owners.get("b2") is engine
    assert srv._bundle_owners.get("b3") is engine
