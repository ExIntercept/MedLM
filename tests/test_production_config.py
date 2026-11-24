import pytest

from src.config import require_production_secret


def test_require_production_secret_accepts_random_secret(monkeypatch):
    monkeypatch.setattr("src.config.PRODUCTION", True)
    secret = "x" * 48
    assert require_production_secret("APP_SECRET", secret) == secret


@pytest.mark.parametrize("secret", ["", "change-me", "x" * 31])
def test_require_production_secret_rejects_weak_secret(monkeypatch, secret):
    monkeypatch.setattr("src.config.PRODUCTION", True)
    with pytest.raises(RuntimeError):
        require_production_secret("APP_SECRET", secret)
