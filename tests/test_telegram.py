"""Unit tests for Telegram bot construction and utilities."""
from __future__ import annotations

import asyncio

import pytest

from agent.chat_providers.telegram import (
    _check_access,
    _parse_allowed_users,
    _send_long,
    _split_message,
    build_bot,
)


class TestParseAllowedUsers:
    def test_empty_string_returns_empty_set(self):
        assert _parse_allowed_users("") == set()
        assert _parse_allowed_users(None) == set()
        assert _parse_allowed_users("  ") == set()

    def test_single_user(self):
        assert _parse_allowed_users("123456") == {123456}

    def test_multiple_users(self):
        assert _parse_allowed_users("123,456,789") == {123, 456, 789}

    def test_ignores_non_numeric(self):
        assert _parse_allowed_users("123,abc,456") == {123, 456}

    def test_trims_whitespace(self):
        assert _parse_allowed_users(" 123 , 456 ") == {123, 456}

    def test_empty_allowlist_denies_access(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "")

        class User:
            id = 123

        class FakeUpdate:
            effective_user = User()

        assert _check_access(FakeUpdate()) == (False, 123)


class TestSplitMessage:
    def test_short_message_unchanged(self):
        text = "Hello world"
        assert _split_message(text, limit=4000) == [text]

    def test_splits_on_paragraphs(self):
        text = "Para 1.\n\nPara 2."
        parts = _split_message(text, limit=8)
        assert len(parts) == 2
        assert "Para 1." in parts[0]
        assert "Para 2." in parts[1]

    def test_adds_numbering_for_multi_part(self):
        text = "A" * 2000
        parts = _split_message(text, limit=100)
        assert len(parts) > 1
        assert parts[0].startswith("[1/")

    def test_splits_on_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        parts = _split_message(text, limit=25)
        assert len(parts) >= 2

    def test_small_limit_still_produces_something(self):
        text = "A" * 100
        parts = _split_message(text, limit=5)
        for p in parts:
            assert len(p) > 0


class TestSendLong:
    def test_sends_assistant_output_as_plain_text(self):
        class FakeChat:
            def __init__(self):
                self.sent = []

            async def send_message(self, text, parse_mode=None):
                self.sent.append({"text": text, "parse_mode": parse_mode})

        class FakeUpdate:
            effective_chat = FakeChat()

        text = "Here is *markdown*, underscores_like_this, and [a link](x)."

        asyncio.run(_send_long(FakeUpdate(), text))

        assert FakeUpdate.effective_chat.sent == [
            {"text": text, "parse_mode": None}
        ]


class TestBuildBot:
    def test_rejects_empty_token(self):
        with pytest.raises(Exception):
            build_bot("")

    def test_builds_with_fake_token(self):
        app = build_bot("123:abc")
        assert app is not None

    def test_all_handlers_registered(self):
        app = build_bot("123:abc")
        total = sum(len(g) for g in app.handlers.values())
        assert total == 6

    def test_start_handler_exists(self):
        app = build_bot("123:abc")
        handler_names = {
            h.callback.__name__
            for group in app.handlers.values()
            for h in group
        }
        assert "start_cmd" in handler_names
        assert "help_cmd" in handler_names
        assert "reset_cmd" in handler_names
        assert "desktop_on_cmd" in handler_names
        assert "desktop_off_cmd" in handler_names

        assert "message_handler" in handler_names
