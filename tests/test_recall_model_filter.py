"""Tests for issue #151: embedding_model filter in recall SQL.

Pins the SQL shape so the model filter lands inside the vector ranking
subqueries (never the FTS leg) and the existing positional binds from
issue #139 (project at $5/$6) keep their numbers. Zero-match degradation
(vector leg empty -> RRF == text-only) is covered structurally: the filter
is a WHERE predicate on vector_ranked, so an empty match set leaves the
FULL OUTER JOIN populated by fts_ranked alone.
"""

from __future__ import annotations

# ==================== model_filter_clause helper ====================


class TestModelFilterClause:
    def test_returns_empty_when_label_none(self):
        from scripts.core.recall_backends import model_filter_clause

        assert model_filter_clause(None, param_index=4) == ""

    def test_renders_predicate_with_param_index(self):
        from scripts.core.recall_backends import model_filter_clause

        assert (
            model_filter_clause("voyage-code-3", param_index=7)
            == "AND embedding_model = $7"
        )

    def test_param_index_is_positional(self):
        from scripts.core.recall_backends import model_filter_clause

        assert model_filter_clause("bge", param_index=5) == "AND embedding_model = $5"


# ==================== build_rrf_cte model filter ====================


class TestBuildRrfCteModelFilter:
    def test_default_omits_model_filter(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=True)
        assert "embedding_model" not in sql

    def test_filter_present_in_vector_leg(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(
            chain_filter=True, model_filter="AND embedding_model = $7"
        )
        assert "AND embedding_model = $7" in sql

    def test_filter_only_in_vector_leg_not_fts(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(
            chain_filter=True, model_filter="AND embedding_model = $7"
        )
        # The FTS leg ends at its closing paren before vector_ranked starts.
        fts_segment, _, vector_segment = sql.partition("vector_ranked AS")
        assert "embedding_model" not in fts_segment
        assert "embedding_model" in vector_segment

    def test_filter_coexists_with_project_and_chain(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(
            chain_filter=True,
            project_filter="AND LOWER(project) = $6",
            model_filter="AND embedding_model = $7",
        )
        assert "superseded_by IS NULL" in sql
        assert "AND LOWER(project) = $6" in sql
        assert "AND embedding_model = $7" in sql

    def test_plain_variant_renders_filter(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(
            chain_filter=False, model_filter="AND embedding_model = $6"
        )
        assert "AND embedding_model = $6" in sql
        assert "superseded_by IS NULL" not in sql


# ==================== render_recall_sql model filter ====================


class TestRenderRecallSqlModelFilter:
    def test_vector_template_default_no_filter(self):
        from scripts.core.recall_backends import _PG_VECTOR_SQL, render_recall_sql

        sql = render_recall_sql(
            _PG_VECTOR_SQL,
            include_project=False,
            chain_filter="",
            project_filter="",
        )
        assert "embedding_model" not in sql

    def test_vector_template_renders_model_filter(self):
        from scripts.core.recall_backends import _PG_VECTOR_SQL, render_recall_sql

        sql = render_recall_sql(
            _PG_VECTOR_SQL,
            include_project=False,
            chain_filter="",
            project_filter="",
            model_filter="AND embedding_model = $3",
        )
        assert "AND embedding_model = $3" in sql

    def test_recency_template_renders_model_filter(self):
        from scripts.core.recall_backends import _PG_RECENCY_SQL, render_recall_sql

        sql = render_recall_sql(
            _PG_RECENCY_SQL,
            include_project=False,
            chain_filter="",
            project_filter="",
            model_filter="AND embedding_model = $4",
        )
        assert "AND embedding_model = $4" in sql

    def test_text_fallback_template_unaffected(self):
        from scripts.core.recall_backends import (
            _PG_TEXT_FALLBACK_SQL,
            render_recall_sql,
        )

        sql = render_recall_sql(
            _PG_TEXT_FALLBACK_SQL,
            include_project=False,
            chain_filter="",
            project_filter="",
            model_filter="",
        )
        assert "embedding_model" not in sql
