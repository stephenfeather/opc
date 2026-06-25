"""Unit tests for the recall-tuning loop: re-anchoring, miner, journal, decisions.

Covers the parts that need no live DB — the pure logic that the autoresearch-style
loop rests on. DB-backed paths (backfill lookup, the feedback join, fetch_full_
results, store_learning_v2) are exercised separately against a real Postgres.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import uuid
from pathlib import Path

from scripts.benchmarks import backfill_golden_hashes, journal, tune_loop
from scripts.benchmarks.mine_feedback_labels import JOIN_SQL, aggregate_candidates
from scripts.benchmarks.run_rerank_benchmark import (
    build_reranker_config,
    compute_mrr,
    compute_ndcg,
    compute_precision_at_k,
    filter_by_split,
    is_relevant,
)
from scripts.core.config.models import RerankerConfig
from scripts.core.content_hash import content_hash

# ---------------------------------------------------------------------------
# content_hash — must match the canonical stored hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_matches_legacy_inline_definition(self):
        for s in ["hello", "  pad  \n", "multi\nline", "", "café ☕"]:
            assert content_hash(s) == hashlib.sha256(s.strip().encode()).hexdigest()

    def test_normalizes_surrounding_whitespace(self):
        assert content_hash("  x  ") == content_hash("x")


# ---------------------------------------------------------------------------
# is_relevant / metrics — content-hash re-anchor + hard negatives
# ---------------------------------------------------------------------------


class TestReAnchoredRelevance:
    def test_hash_match_is_instance_independent(self):
        h = content_hash("relevant body")
        # id differs (fresh DB) but content matches -> relevant
        assert is_relevant("NEW_UUID", "relevant body", [], [], golden_hashes=[h])
        assert not is_relevant("NEW_UUID", "other", [], [], golden_hashes=[h])

    def test_hash_precedence_over_legacy_ids(self):
        h = content_hash("body")
        # content does not match the hash; legacy id would say relevant, hash wins
        assert not is_relevant("id1", "other", ["id1"], [], golden_hashes=[h])

    def test_hard_negative_overrides_positive(self):
        hn = content_hash("bad")
        assert not is_relevant("id1", "bad", ["id1"], [], golden_negatives=[hn])

    def test_legacy_id_and_keyword_paths_still_work(self):
        assert is_relevant("id1", "x", ["id1"], [])
        assert is_relevant("z", "has HOOK", [], ["hook"])
        assert not is_relevant("z", "nope", [], ["hook"])

    def test_metrics_use_hashes(self):
        h = content_hash("relevant body")
        ids = ["a", "b", "c"]
        conts = ["relevant body", "x", "y"]
        assert compute_precision_at_k(ids, conts, [], [], golden_hashes=[h]) == 1 / 3
        assert compute_mrr(ids, conts, [], [], golden_hashes=[h]) == 1.0
        assert round(compute_ndcg(ids, conts, [], [], 3, golden_hashes=[h]), 4) == 1.0

    def test_ndcg_idcg_uses_full_hash_count(self):
        # Two golden positives but only one retrieved at rank 1: NDCG < 1.
        h1, h2 = content_hash("one"), content_hash("two")
        ndcg = compute_ndcg(["a"], ["one"], [], [], 5, golden_hashes=[h1, h2])
        assert 0.0 < ndcg < 1.0


# ---------------------------------------------------------------------------
# held-out split filtering
# ---------------------------------------------------------------------------


class TestSplitFilter:
    queries = [
        {"id": "a", "split": "train"},
        {"id": "b", "split": "holdout"},
        {"id": "c", "split": "train"},
    ]

    def test_all_returns_everything(self):
        assert len(filter_by_split(self.queries, "all")) == 3

    def test_train_and_holdout(self):
        assert [q["id"] for q in filter_by_split(self.queries, "train")] == ["a", "c"]
        assert [q["id"] for q in filter_by_split(self.queries, "holdout")] == ["b"]

    def test_unlabeled_file_is_noop(self):
        unlabeled = [{"id": "x"}, {"id": "y"}]
        assert filter_by_split(unlabeled, "holdout") == unlabeled


# ---------------------------------------------------------------------------
# feedback label miner aggregation
# ---------------------------------------------------------------------------


class TestMinerAggregation:
    def _rows(self):
        return [
            {"query_hash": "A", "query_text": "do X", "helpful": True, "content_hash": "hX"},
            {"query_hash": "A", "query_text": "do X", "helpful": True, "content_hash": "hX"},
            {"query_hash": "A", "query_text": None, "helpful": True, "content_hash": "hX"},
            {"query_hash": "A", "query_text": "do X", "helpful": False, "content_hash": "hY"},
            {"query_hash": "A", "query_text": "do X", "helpful": False, "content_hash": "hY"},
        ]

    def test_majority_vote_positive_and_negative(self):
        cands = aggregate_candidates(self._rows(), min_judgments=3)
        assert len(cands) == 1
        c = cands[0]
        assert c["query"] == "do X"  # first non-null query text recovered
        assert c["golden_hashes"] == ["hX"]
        assert c["golden_negatives"] == ["hY"]
        assert c["num_judgments"] == 5

    def test_below_threshold_dropped(self):
        rows = [{"query_hash": "B", "query_text": "q", "helpful": True, "content_hash": "h"}]
        assert aggregate_candidates(rows, min_judgments=3) == []

    def test_tie_is_dropped(self):
        rows = [
            {"query_hash": "C", "query_text": "q", "helpful": True, "content_hash": "h"},
            {"query_hash": "C", "query_text": "q", "helpful": False, "content_hash": "h"},
        ]
        assert aggregate_candidates(rows, min_judgments=2) == []

    def test_hash_only_candidate_has_null_query(self):
        rows = [
            {"query_hash": "D", "query_text": None, "helpful": True, "content_hash": "h1"},
            {"query_hash": "D", "query_text": None, "helpful": True, "content_hash": "h1"},
            {"query_hash": "D", "query_text": None, "helpful": True, "content_hash": "h1"},
        ]
        cands = aggregate_candidates(rows, min_judgments=3)
        assert len(cands) == 1 and cands[0]["query"] is None


class TestMinerJoinDedup:
    """The feedback↔recall join must attribute each judgment to ONE recall."""

    def test_join_sql_dedupes_per_feedback_row(self):
        # A single memory_feedback row can match many recall_log rows in the
        # window; DISTINCT ON (mf.id) keeps only the nearest preceding recall so
        # one judgment is not counted N times toward min_judgments (Codex/CodeRabbit).
        assert "DISTINCT ON (mf.id)" in JOIN_SQL
        assert "ORDER BY mf.id, rl.created_at DESC" in JOIN_SQL


# ---------------------------------------------------------------------------
# backfill lookup_hashes — defensive UUID parsing
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, captured: dict):
        self._captured = captured

    async def fetch(self, sql, ids):
        self._captured["ids"] = ids
        return []


class _FakePool:
    def __init__(self, captured: dict):
        self._captured = captured

    def acquire(self):
        captured = self._captured

        class _Acquire:
            async def __aenter__(self):
                return _FakeConn(captured)

            async def __aexit__(self, *exc):
                return False

        return _Acquire()


class TestLookupHashes:
    def test_all_invalid_short_circuits_without_db(self, monkeypatch):
        # No valid UUID -> returns {} and never touches the pool (Gemini HIGH:
        # raw strings would otherwise crash asyncpg's uuid[] bind).
        called = {"pool": False}

        async def boom():
            called["pool"] = True
            raise AssertionError("get_pool must not be called")

        monkeypatch.setattr(backfill_golden_hashes, "get_pool", boom)
        assert asyncio.run(backfill_golden_hashes.lookup_hashes(["not-a-uuid", ""])) == {}
        assert called["pool"] is False

    def test_parses_valid_uuids_and_skips_invalid(self, monkeypatch):
        captured: dict = {}

        async def fake_pool():
            return _FakePool(captured)

        monkeypatch.setattr(backfill_golden_hashes, "get_pool", fake_pool)
        good = str(uuid.uuid4())
        asyncio.run(backfill_golden_hashes.lookup_hashes([good, "bogus"]))
        # Only the valid id is forwarded, parsed to a uuid.UUID object.
        assert captured["ids"] == [uuid.UUID(good)]
        assert all(isinstance(u, uuid.UUID) for u in captured["ids"])

    def test_empty_input_returns_empty(self):
        assert asyncio.run(backfill_golden_hashes.lookup_hashes([])) == {}


# ---------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------


class TestJournal:
    def test_config_hash_stable_and_sensitive(self):
        assert journal.config_hash(RerankerConfig()) == journal.config_hash(RerankerConfig())
        assert journal.config_hash(RerankerConfig()) != journal.config_hash(
            RerankerConfig(project_weight=0.30)
        )
        assert len(journal.config_hash(RerankerConfig())) == 12

    def test_append_and_read_round_trip(self):
        path = Path(tempfile.mkdtemp()) / "j.tsv"
        journal.append_entry(
            timestamp="2026-06-25T00:00:00Z", cfg_hash="abc",
            ndcg=0.5, p_at_k=0.4, mrr=0.6, p95_latency_ms=1.234,
            status="keep", description="line\twith\ttabs\nand newline", path=path,
        )
        rows = journal.read_entries(path)
        assert len(rows) == 1
        assert rows[0]["config_hash"] == "abc"
        assert rows[0]["ndcg@5"] == "0.5000"
        assert rows[0]["status"] == "keep"
        assert "\t" not in rows[0]["description"]

    def test_invalid_status_rejected(self):
        path = Path(tempfile.mkdtemp()) / "j.tsv"
        try:
            journal.append_entry(
                timestamp="t", cfg_hash="x", ndcg=0, p_at_k=0, mrr=0,
                p95_latency_ms=0, status="bogus", description="d", path=path,
            )
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_read_missing_file_is_empty(self):
        assert journal.read_entries(Path(tempfile.mkdtemp()) / "absent.tsv") == []


# ---------------------------------------------------------------------------
# decision rule + opc.toml apply
# ---------------------------------------------------------------------------


class TestDecisionRule:
    base = {"ndcg_at_k": 0.50, "p95_latency_ms": 1.0}

    def test_keep_on_strict_improvement_within_budget(self):
        cand = {"ndcg_at_k": 0.55, "p95_latency_ms": 1.0}
        assert tune_loop.decide(cand, self.base, 1.25) == "keep"

    def test_equal_ndcg_is_not_a_win(self):
        cand = {"ndcg_at_k": 0.50, "p95_latency_ms": 1.0}
        assert tune_loop.decide(cand, self.base, 1.25) == "discard"

    def test_over_latency_budget_discards(self):
        cand = {"ndcg_at_k": 0.99, "p95_latency_ms": 9.0}
        assert tune_loop.decide(cand, self.base, 1.25) == "discard"

    def test_latency_budget_default_and_override(self):
        assert tune_loop.latency_budget(4.0, None) == 5.0  # 25% over
        assert tune_loop.latency_budget(0.1, None) == 1.1  # min +1ms floor
        assert tune_loop.latency_budget(4.0, 2.0) == 2.0  # explicit override

    def test_active_signal_count(self):
        zero = RerankerConfig(
            project_weight=0.0, recency_weight=0.0, confidence_weight=0.0,
            recall_weight=0.0, type_affinity_weight=0.0, tag_overlap_weight=0.0,
            pattern_weight=0.0,
        )
        assert tune_loop.active_signal_count(zero) == 0
        assert tune_loop.active_signal_count(RerankerConfig(project_weight=0.1)) >= 1


class TestApplyWeightsToToml:
    def _toml(self) -> Path:
        text = (
            "[dedup]\n"
            "threshold = 0.85\n\n"
            "[reranker]\n"
            "project_weight = 0.09\n"
            "recall_weight = 0.02\n"
            "rrf_scale_factor = 25\n\n"
            "[recall]\n"
            "project_weight = 999  # must NOT be touched (different section)\n"
        )
        path = Path(tempfile.mkdtemp()) / "opc.toml"
        path.write_text(text)
        return path

    def test_updates_only_reranker_weight_lines(self):
        path = self._toml()
        changed = tune_loop.apply_weights_to_toml(
            path, {"project_weight": 0.30, "recall_weight": 0.05}, RerankerConfig()
        )
        after = path.read_text()
        assert set(changed) == {"project_weight", "recall_weight"}
        assert "project_weight = 0.3" in after
        assert "recall_weight = 0.05" in after
        # other section + non-weight key untouched
        assert "project_weight = 999" in after
        assert "rrf_scale_factor = 25" in after

    def test_idempotent_reapply(self):
        path = self._toml()
        tune_loop.apply_weights_to_toml(path, {"project_weight": 0.30}, RerankerConfig())
        assert (
            tune_loop.apply_weights_to_toml(
                path, {"project_weight": 0.30}, RerankerConfig()
            )
            == []
        )

    def test_rejects_over_budget_config_before_writing(self):
        # CodeRabbit: --apply must never emit an unloadable opc.toml. A swept
        # combo that (with the live kg_weight) pushes total_signal_weight past
        # 1.0 is rejected by RerankerConfig.__post_init__ BEFORE the file write.
        path = self._toml()
        before = path.read_text()
        over = {k: 0.5 for k in (
            "project_weight", "recency_weight", "confidence_weight",
            "recall_weight", "type_affinity_weight", "tag_overlap_weight",
            "pattern_weight",
        )}  # 7 * 0.5 + kg 0.05 = 3.55 > 1.0
        try:
            tune_loop.apply_weights_to_toml(path, over, RerankerConfig())
            raise AssertionError("expected ValueError for over-budget config")
        except ValueError:
            pass
        # File is untouched on rejection.
        assert path.read_text() == before

    def test_preserves_crlf_line_endings(self):
        # C4: a CRLF opc.toml must stay CRLF after a weight rewrite, not gain
        # mixed endings.
        text = "[reranker]\r\nproject_weight = 0.09\r\nrecall_weight = 0.02\r\n"
        path = Path(tempfile.mkdtemp()) / "crlf.toml"
        path.write_bytes(text.encode())
        tune_loop.apply_weights_to_toml(
            path, {"project_weight": 0.30}, RerankerConfig()
        )
        raw = path.read_bytes()
        assert b"project_weight = 0.3\r\n" in raw
        assert b"\r\n" in raw and b"\n\n" not in raw  # no LF-only line introduced


# ---------------------------------------------------------------------------
# build_reranker_config — perturb from the live incumbent, not defaults
# ---------------------------------------------------------------------------


class TestBuildRerankerConfig:
    def test_inherits_non_swept_fields_from_base(self):
        # A base with operator-tuned non-swept fields: the swept candidate must
        # keep them, overriding ONLY the seven swept weights (Codex P2).
        base = RerankerConfig(kg_weight=0.0, recency_half_life_days=90.0)
        wc = {
            "name": "heavy-project",
            "project_weight": 0.30, "recency_weight": 0.05,
            "confidence_weight": 0.05, "recall_weight": 0.05,
            "type_affinity_weight": 0.05, "tag_overlap_weight": 0.05,
            "pattern_weight": 0.05,
        }
        cfg = build_reranker_config(wc, base)
        # Non-swept fields inherited from base, NOT reset to dataclass defaults.
        assert cfg.kg_weight == 0.0
        assert cfg.recency_half_life_days == 90.0
        # Swept weights overridden.
        assert cfg.project_weight == 0.30
        assert cfg.recall_weight == 0.05

    def test_partial_weight_dict_inherits_rest(self):
        base = RerankerConfig()
        cfg = build_reranker_config({"project_weight": 0.30}, base)
        assert cfg.project_weight == 0.30
        assert cfg.recall_weight == base.recall_weight  # untouched -> inherited
