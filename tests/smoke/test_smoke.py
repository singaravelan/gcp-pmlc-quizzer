"""Smoke tests: quick checks that core modules load and basic utilities work."""

from config import settings
from utils.question_generator import _parse_json


def test_settings_smoke() -> None:
    """Core settings should load with expected primitive types."""
    assert isinstance(settings.APP_TITLE, str)
    assert settings.APP_TITLE
    assert settings.AI_BACKEND in {"claude", "ollama"}


def test_parse_json_smoke() -> None:
    """Basic JSON parsing path should work for a valid payload."""
    raw = '{"ok": true, "count": 2}'
    parsed = _parse_json(raw, fallback={})
    assert parsed == {"ok": True, "count": 2}
