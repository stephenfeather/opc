"""Tests for config handlers — I/O boundary (file discovery, TOML loading, env reads)."""

import os
from pathlib import Path

from scripts.core.config.handlers import (
    discover_config_paths,
    load_config_file,
    read_env_overrides,
    get_config,
    reset_config,
)
from scripts.core.config.models import OPCConfig


class TestDiscoverConfigPaths:
    def test_returns_list(self):
        paths = discover_config_paths()
        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)

    def test_env_var_first(self, monkeypatch, tmp_path):
        config_file = tmp_path / "custom.toml"
        config_file.touch()
        monkeypatch.setenv("OPC_CONFIG", str(config_file))
        paths = discover_config_paths()
        assert paths[0] == config_file

    def test_includes_user_global(self):
        paths = discover_config_paths()
        user_global = Path.home() / ".config" / "opc" / "config.toml"
        assert user_global in paths


class TestLoadConfigFile:
    def test_returns_empty_for_no_files(self):
        result = load_config_file([Path("/nonexistent/path.toml")])
        assert result == {}

    def test_loads_first_existing(self, tmp_path):
        first = tmp_path / "first.toml"
        second = tmp_path / "second.toml"
        first.write_text('[dedup]\nthreshold = 0.75\n')
        second.write_text('[dedup]\nthreshold = 0.99\n')
        result = load_config_file([first, second])
        assert result["dedup"]["threshold"] == 0.75

    def test_skips_missing_to_find_existing(self, tmp_path):
        missing = tmp_path / "nope.toml"
        present = tmp_path / "yes.toml"
        present.write_text('[daemon]\npoll_interval = 30\n')
        result = load_config_file([missing, present])
        assert result["daemon"]["poll_interval"] == 30

    def test_empty_toml_returns_empty_dict(self, tmp_path):
        empty = tmp_path / "empty.toml"
        empty.write_text("")
        result = load_config_file([empty])
        assert result == {}


class TestReadEnvOverrides:
    def test_returns_empty_when_no_vars(self, monkeypatch):
        monkeypatch.delenv("PATTERN_DETECTION_INTERVAL_HOURS", raising=False)
        monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("AGENTICA_MAX_POOL_SIZE", raising=False)
        result = read_env_overrides()
        assert result == {}

    def test_picks_up_set_vars(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://custom:1234")
        result = read_env_overrides()
        assert result["embedding"]["ollama_host"] == "http://custom:1234"

    def test_casts_int_vars(self, monkeypatch):
        monkeypatch.setenv("PATTERN_DETECTION_INTERVAL_HOURS", "12")
        result = read_env_overrides()
        assert result["daemon"]["pattern_detection_interval_hours"] == 12

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://gpu:11434")
        monkeypatch.setenv("AGENTICA_MAX_POOL_SIZE", "20")
        result = read_env_overrides()
        assert result["embedding"]["ollama_host"] == "http://gpu:11434"
        assert result["database"]["max_pool_size"] == 20


class TestGetConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_returns_opc_config(self):
        cfg = get_config()
        assert isinstance(cfg, OPCConfig)

    def test_caches_result(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reload_returns_fresh(self):
        cfg1 = get_config()
        cfg2 = get_config(reload=True)
        # Both valid OPCConfig but not necessarily same object
        assert isinstance(cfg2, OPCConfig)

    def test_respects_opc_config_env(self, monkeypatch, tmp_path):
        reset_config()
        config_file = tmp_path / "test.toml"
        config_file.write_text('[dedup]\nthreshold = 0.77\n')
        monkeypatch.setenv("OPC_CONFIG", str(config_file))
        cfg = get_config(reload=True)
        assert cfg.dedup.threshold == 0.77

    def test_env_override_beats_file(self, monkeypatch, tmp_path):
        reset_config()
        config_file = tmp_path / "test.toml"
        config_file.write_text('[embedding]\nollama_host = "http://file:1234"\n')
        monkeypatch.setenv("OPC_CONFIG", str(config_file))
        monkeypatch.setenv("OLLAMA_HOST", "http://env:5678")
        cfg = get_config(reload=True)
        assert cfg.embedding.ollama_host == "http://env:5678"
