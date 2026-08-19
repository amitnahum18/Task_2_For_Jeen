"""Unit tests for document_indexer.config. Environment variables are
controlled per-test with monkeypatch, independent of any real .env file."""

from __future__ import annotations

import pytest

from document_indexer.config import load_config
from document_indexer.exceptions import ConfigError


@pytest.fixture
def clean_env(monkeypatch):
    """Start each test from a blank slate, then let it set what it needs."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)


class TestLoadConfig:
    def test_loads_valid_configuration(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://localhost/db")
        monkeypatch.setenv("EMBEDDING_DIM", "1536")

        config = load_config()

        assert config.gemini_api_key == "test-key"
        assert config.postgres_url == "postgresql://localhost/db"
        assert config.embedding_dim == 1536

    def test_embedding_dim_defaults_when_unset(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://localhost/db")

        assert load_config().embedding_dim == 768

    def test_missing_gemini_api_key_raises_config_error(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("POSTGRES_URL", "postgresql://localhost/db")

        with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
            load_config()

    def test_missing_postgres_url_raises_config_error(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with pytest.raises(ConfigError, match="POSTGRES_URL"):
            load_config()

    def test_missing_both_lists_both_in_the_error(self, clean_env) -> None:
        with pytest.raises(ConfigError, match="GEMINI_API_KEY.*POSTGRES_URL"):
            load_config()

    def test_non_integer_embedding_dim_raises_config_error(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://localhost/db")
        monkeypatch.setenv("EMBEDDING_DIM", "not-a-number")

        with pytest.raises(ConfigError, match="EMBEDDING_DIM"):
            load_config()

    def test_non_positive_embedding_dim_raises_config_error(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://localhost/db")
        monkeypatch.setenv("EMBEDDING_DIM", "0")

        with pytest.raises(ConfigError, match="EMBEDDING_DIM"):
            load_config()


class TestConfigRepr:
    def test_repr_does_not_expose_secrets(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://user:hunter2@host/db")

        config = load_config()

        assert "super-secret-key" not in repr(config)
        assert "hunter2" not in repr(config)
