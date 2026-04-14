"""Tests for knowledge graph entity and relationship extraction."""

from scripts.core.kg_extractor import (
    extract_entities,
    extract_relations,
)

# ---------------------------------------------------------------------------
# Entity extraction tests
# ---------------------------------------------------------------------------


class TestExtractFilePaths:
    def test_simple_path(self):
        entities = extract_entities("Modified scripts/core/reranker.py to add weights")
        names = {e.name for e in entities if e.entity_type == "file"}
        assert "scripts/core/reranker.py" in names

    def test_path_with_extension_only(self):
        entities = extract_entities("Edit the config.yaml file")
        names = {e.name for e in entities if e.entity_type == "file"}
        assert "config.yaml" in names

    def test_nested_path(self):
        entities = extract_entities("See hooks/ts/src/shared/db.ts for details")
        names = {e.name for e in entities if e.entity_type == "file"}
        assert "hooks/ts/src/shared/db.ts" in names

    def test_no_false_version_numbers(self):
        entities = extract_entities("Python 3.12 is required")
        file_names = {e.name for e in entities if e.entity_type == "file"}
        assert "3.12" not in file_names


class TestExtractPythonImports:
    def test_from_import(self):
        entities = extract_entities("from scripts.core.reranker import rerank")
        names = {e.name for e in entities if e.entity_type == "module"}
        assert "scripts.core.reranker" in names

    def test_plain_import(self):
        entities = extract_entities("import asyncpg")
        names = {e.name for e in entities if e.entity_type == "module"}
        assert "asyncpg" in names


class TestExtractEnvVars:
    def test_database_url(self):
        entities = extract_entities("Set DATABASE_URL to connect")
        names = {e.name for e in entities if e.entity_type == "config"}
        assert "database_url" in names

    def test_voyage_api_key(self):
        entities = extract_entities("Export VOYAGE_API_KEY before running")
        names = {e.name for e in entities if e.entity_type == "config"}
        assert "voyage_api_key" in names

    def test_no_short_caps(self):
        """Single-word ALL_CAPS without underscores should not match."""
        entities = extract_entities("The URL is important")
        config_names = {e.name for e in entities if e.entity_type == "config"}
        assert "URL" not in config_names


class TestExtractErrors:
    def test_import_error(self):
        entities = extract_entities("Got ImportError when loading the module")
        names = {e.name for e in entities if e.entity_type == "error"}
        assert "importerror" in names

    def test_connection_refused(self):
        entities = extract_entities("ConnectionRefusedError on startup")
        names = {e.name for e in entities if e.entity_type == "error"}
        assert "connectionrefusederror" in names


class TestExtractTools:
    def test_docker(self):
        entities = extract_entities("Run docker compose up to start")
        names = {e.name for e in entities if e.entity_type == "tool"}
        assert "docker" in names

    def test_pytest(self):
        entities = extract_entities("Use pytest to run the test suite")
        names = {e.name for e in entities if e.entity_type == "tool"}
        assert "pytest" in names


class TestExtractLanguages:
    def test_python(self):
        entities = extract_entities("Written in Python with type hints")
        names = {e.name for e in entities if e.entity_type == "language"}
        assert "python" in names

    def test_typescript(self):
        entities = extract_entities("Hooks are TypeScript compiled to JS")
        names = {e.name for e in entities if e.entity_type == "language"}
        assert "typescript" in names


class TestExtractLibraries:
    def test_pgvector(self):
        entities = extract_entities("Using pgvector for similarity search")
        names = {e.name for e in entities if e.entity_type == "library"}
        assert "pgvector" in names

    def test_asyncpg(self):
        entities = extract_entities("asyncpg pool for database connections")
        names = {e.name for e in entities if e.entity_type == "library"}
        assert "asyncpg" in names


class TestExtractConcepts:
    def test_backtick_concept(self):
        entities = extract_entities("The `semantic dedup` feature prevents duplicates")
        names = {e.name for e in entities if e.entity_type == "concept"}
        assert "semantic dedup" in names

    def test_no_duplicate_with_other_type(self):
        """A backtick term already captured as a tool should not also be a concept."""
        entities = extract_entities("Use `docker` to run containers")
        concept_names = {e.name for e in entities if e.entity_type == "concept"}
        assert "docker" not in concept_names


class TestDeduplication:
    def test_same_entity_mentioned_twice(self):
        content = "reranker.py handles scoring. See reranker.py for details."
        entities = extract_entities(content)
        file_entities = [e for e in entities if e.name == "reranker.py"]
        assert len(file_entities) == 1

    def test_different_types_not_deduped(self):
        """Same name but different type should produce two entities."""
        content = "import pytest; Use pytest to run tests"
        entities = extract_entities(content)
        pytest_entities = [e for e in entities if "pytest" in e.name]
        types = {e.entity_type for e in pytest_entities}
        # Should have both module and tool (or library)
        assert len(types) >= 2


class TestEdgeCases:
    def test_empty_content(self):
        assert extract_entities("") == []

    def test_no_entities(self):
        entities = extract_entities("This is a simple sentence with no technical terms.")
        # May still find some noise, but should be minimal
        assert isinstance(entities, list)

    def test_noise_filtering(self):
        entities = extract_entities("The value is true and not false")
        names = {e.name for e in entities}
        assert "true" not in names
        assert "false" not in names


# ---------------------------------------------------------------------------
# Relation extraction tests
# ---------------------------------------------------------------------------


class TestRelationExtraction:
    def test_related_to_in_same_sentence(self):
        content = "pgvector handles vector search in PostgreSQL"
        entities = extract_entities(content)
        relations = extract_relations(content, entities)
        assert len(relations) > 0
        rel_types = {r.relation for r in relations}
        assert "related_to" in rel_types or "uses" in rel_types

    def test_solves_relation(self):
        content = "ruff fixes the ImportError in the linter pipeline"
        entities = extract_entities(content)
        relations = extract_relations(content, entities)
        solves = [r for r in relations if r.relation == "solves"]
        assert len(solves) > 0

    def test_supersedes_relation(self):
        content = "asyncpg replaced psycopg2 as the database driver"
        entities = extract_entities(content)
        relations = extract_relations(content, entities)
        supers = [r for r in relations if r.relation == "supersedes"]
        assert len(supers) > 0

    def test_uses_relation(self):
        content = "recall_learnings.py uses the reranker.py module"
        entities = extract_entities(content)
        relations = extract_relations(content, entities)
        uses = [r for r in relations if r.relation == "uses"]
        assert len(uses) > 0

    def test_no_self_relation(self):
        content = "docker uses docker to build containers"
        entities = extract_entities(content)
        relations = extract_relations(content, entities)
        self_rels = [r for r in relations if r.source == r.target]
        assert len(self_rels) == 0

    def test_fewer_than_two_entities(self):
        content = "Just docker alone"
        entities = extract_entities(content)
        # Even with 1 entity, should not crash
        relations = extract_relations(content, entities)
        assert relations == []

    def test_directory_containment(self):
        content = "Modified scripts/core/reranker.py and scripts/core/store_learning.py"
        entities = extract_entities(content)
        relations = extract_relations(content, entities)
        contains = [r for r in relations if r.relation == "contains"]
        assert len(contains) > 0

    def test_conflicts_relation(self):
        content = "pgvector conflicts with the old sqlite backend"
        entities = extract_entities(content)
        relations = extract_relations(content, entities)
        conflicts = [r for r in relations if r.relation == "conflicts_with"]
        assert len(conflicts) > 0


# ---------------------------------------------------------------------------
# Integration-style tests (extraction on real-ish learnings)
# ---------------------------------------------------------------------------


class TestRealLearnings:
    def test_sample_learning_1(self):
        content = (
            "Using localhost:11434 or a LAN IP for Ollama from inside a Docker "
            "container fails silently. The describe_photo function catches the "
            "ConnectionRefusedError. Fix: use host.docker.internal instead."
        )
        entities = extract_entities(content)
        types = {e.entity_type for e in entities}
        assert "error" in types  # ConnectionRefusedError
        assert "tool" in types   # docker

    def test_sample_learning_2(self):
        content = (
            "pgvector cosine distance ranges from 0 to 2, not 0 to 1. "
            "When converting distance to a similarity score, use: "
            "score = 1 - (distance / 2). This affects scripts/core/recall_backends.py."
        )
        entities = extract_entities(content)
        names = {e.name for e in entities}
        assert "scripts/core/recall_backends.py" in names
        assert "pgvector" in names

    def test_sample_learning_3(self):
        content = (
            "subprocess.run(..., cwd=OPC_DIR) changes directory AFTER command setup. "
            "When using `uv run python scripts/core/script.py` with a relative path, "
            "uv resolves the path from the original cwd, not the target."
        )
        entities = extract_entities(content)
        types_found = {e.entity_type for e in entities}
        assert "config" in types_found  # OPC_DIR
        assert "tool" in types_found    # uv


class TestRelationTypedResolution:
    """Regression tests for typed entity resolution in relations (issue: overlapping names)."""

    def test_pytest_overlapping_types_carry_entity_type(self):
        """pytest appears as tool, module, and library — relations should carry types."""
        content = "Using pytest to test the asyncpg connection pool"
        entities = extract_entities(content)
        pytest_entities = [e for e in entities if e.name == "pytest"]
        # pytest should be extracted as at least tool and library
        pytest_types = {e.entity_type for e in pytest_entities}
        assert len(pytest_types) >= 2, f"Expected multiple types for pytest, got {pytest_types}"

        relations = extract_relations(content, entities)
        # Relations should have source_type and target_type set
        for rel in relations:
            assert rel.source_type is not None, (
                f"Relation {rel.source}->{rel.target} missing source_type"
            )
            assert rel.target_type is not None, (
                f"Relation {rel.source}->{rel.target} missing target_type"
            )

    def test_overlapping_names_produce_typed_dedup(self):
        """Same-name entities with different types get separate relations."""
        content = "Using pytest to test the asyncpg connection pool"
        entities = extract_entities(content)
        relations = extract_relations(content, entities)

        # With type-aware dedup, pytest-as-tool and pytest-as-library
        # should each get their own relation to asyncpg
        pytest_asyncpg_rels = [
            r for r in relations
            if r.source == "pytest" and r.target == "asyncpg"
        ]
        source_types = {r.source_type for r in pytest_asyncpg_rels}
        # Should have at least 2 different source types (tool, library)
        assert len(source_types) >= 2, (
            f"Expected multiple typed relations for pytest->asyncpg, "
            f"got source_types={source_types}"
        )

    def test_deterministic_relations_across_runs(self):
        """Same input must produce identical relations every time."""
        content = "Using pytest to test the asyncpg connection pool"
        # Run extraction 10 times and verify all produce identical output
        baseline_entities = extract_entities(content)
        baseline_rels = extract_relations(content, baseline_entities)
        baseline_keys = sorted(
            (r.source, r.source_type, r.target, r.target_type, r.relation)
            for r in baseline_rels
        )

        for _ in range(9):
            entities = extract_entities(content)
            rels = extract_relations(content, entities)
            keys = sorted(
                (r.source, r.source_type, r.target, r.target_type, r.relation)
                for r in rels
            )
            assert keys == baseline_keys, (
                f"Nondeterministic relation output:\n"
                f"  baseline: {baseline_keys}\n"
                f"  this run: {keys}"
            )
