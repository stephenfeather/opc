"""Tests for config core — pure functions, no I/O."""

import pytest

from scripts.core.config.core import (
    ConfigValidationError,
    build_config,
    build_section,
    merge_raw,
)
from scripts.core.config.models import (
    DaemonConfig,
    DedupConfig,
    OPCConfig,
    RerankerConfig,
)


class TestBuildSection:
    def test_empty_dict_returns_defaults(self):
        result = build_section(DedupConfig, {})
        assert result == DedupConfig()

    def test_overrides_known_keys(self):
        result = build_section(DedupConfig, {"threshold": 0.90})
        assert result.threshold == 0.90

    def test_unknown_keys_warns(self, caplog):
        result = build_section(DedupConfig, {"threshold": 0.90, "bogus": 42}, section_name="dedup")
        assert result.threshold == 0.90
        assert "unknown key" in caplog.text
        assert "bogus" in caplog.text

    def test_partial_override_keeps_other_defaults(self):
        result = build_section(RerankerConfig, {"project_weight": 0.25})
        assert result.project_weight == 0.25
        assert result.recency_weight == 0.05  # default preserved

    def test_wrong_type_raises(self):
        with pytest.raises(ConfigValidationError, match="expected int.*got str"):
            build_section(DaemonConfig, {"poll_interval": "fast"}, section_name="daemon")

    def test_int_accepted_for_float_field(self):
        result = build_section(DedupConfig, {"threshold": 1})
        assert result.threshold == 1.0
        assert isinstance(result.threshold, float)


class TestMergeRaw:
    def test_env_overrides_file(self):
        file_raw = {"dedup": {"threshold": 0.85}}
        env_raw = {"dedup": {"threshold": 0.90}}
        merged = merge_raw(file_raw, env_raw)
        assert merged["dedup"]["threshold"] == 0.90

    def test_env_adds_missing_keys(self):
        file_raw = {"dedup": {"threshold": 0.85}}
        env_raw = {"daemon": {"poll_interval": 30}}
        merged = merge_raw(file_raw, env_raw)
        assert merged["dedup"]["threshold"] == 0.85
        assert merged["daemon"]["poll_interval"] == 30

    def test_file_preserved_when_no_env(self):
        file_raw = {"dedup": {"threshold": 0.85}, "daemon": {"poll_interval": 60}}
        env_raw = {}
        merged = merge_raw(file_raw, env_raw)
        assert merged == file_raw

    def test_both_empty(self):
        merged = merge_raw({}, {})
        assert merged == {}

    def test_does_not_mutate_inputs(self):
        file_raw = {"dedup": {"threshold": 0.85}}
        env_raw = {"dedup": {"threshold": 0.90}}
        merge_raw(file_raw, env_raw)
        assert file_raw["dedup"]["threshold"] == 0.85
        assert env_raw["dedup"]["threshold"] == 0.90


class TestBuildConfig:
    def test_empty_dict_returns_all_defaults(self):
        cfg = build_config({})
        assert cfg == OPCConfig()

    def test_partial_override(self):
        raw = {"dedup": {"threshold": 0.90}}
        cfg = build_config(raw)
        assert cfg.dedup.threshold == 0.90
        assert cfg.daemon.poll_interval == 60  # other sections default

    def test_unknown_section_warns(self, caplog):
        raw = {"nonexistent_section": {"key": "value"}}
        cfg = build_config(raw)
        assert cfg == OPCConfig()
        assert "unknown section" in caplog.text
        assert "nonexistent_section" in caplog.text

    def test_multiple_sections(self):
        raw = {
            "dedup": {"threshold": 0.80},
            "daemon": {"poll_interval": 30, "max_retries": 10},
            "recall": {"default_k": 10},
        }
        cfg = build_config(raw)
        assert cfg.dedup.threshold == 0.80
        assert cfg.daemon.poll_interval == 30
        assert cfg.daemon.max_retries == 10
        assert cfg.daemon.stale_threshold == 900  # default
        assert cfg.recall.default_k == 10

    def test_wrong_type_in_section_raises(self):
        raw = {"daemon": {"poll_interval": "not_a_number"}}
        with pytest.raises(ConfigValidationError):
            build_config(raw)
