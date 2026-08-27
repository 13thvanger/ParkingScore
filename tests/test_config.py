import pytest

from parking_score.config import ConfigurationError, Settings


def _required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FTP_HOST", "ftp.example")
    monkeypatch.setenv("FTP_USER", "user")
    monkeypatch.setenv("FTP_PASSWORD", "password")
    monkeypatch.setenv("AI_API_KEY", "test-key")


def test_length_retry_limit_is_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _required_environment(monkeypatch)
    monkeypatch.setenv("AI_MAX_TOKENS", "1200")
    monkeypatch.setenv("AI_LENGTH_RETRY_MAX_TOKENS", "4800")

    settings = Settings.from_env(None)

    assert settings.ai_max_tokens == 1200
    assert settings.ai_length_retry_max_tokens == 4800


def test_length_retry_limit_cannot_be_below_initial_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _required_environment(monkeypatch)
    monkeypatch.setenv("AI_MAX_TOKENS", "2000")
    monkeypatch.setenv("AI_LENGTH_RETRY_MAX_TOKENS", "1000")

    with pytest.raises(ConfigurationError, match="must be >= AI_MAX_TOKENS"):
        Settings.from_env(None)


def test_default_length_retry_limit_preserves_larger_initial_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _required_environment(monkeypatch)
    monkeypatch.setenv("AI_MAX_TOKENS", "8000")
    monkeypatch.delenv("AI_LENGTH_RETRY_MAX_TOKENS", raising=False)

    settings = Settings.from_env(None)

    assert settings.ai_length_retry_max_tokens == 8000
