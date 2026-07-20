"""Regression coverage for shared configuration expansion."""
from __future__ import annotations

import logging
from pathlib import Path

import agent.config_loader as config_loader_module
from agent.config_utils import expand_env_vars
from agent.config_loader import load_agent_config
from agent.mcp_manager import MCPManager


def test_expand_env_vars_expands_root_strings(monkeypatch) -> None:
    monkeypatch.setenv("CONFIG_VALUE", "expanded")

    assert expand_env_vars("${CONFIG_VALUE}") == "expanded"


def test_mcp_and_agent_loaders_expand_nested_config_identically(
    monkeypatch,
) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    persona_path = fixtures / "config_utils_persona.md"
    config_path = fixtures / "config_utils.yaml"
    monkeypatch.setenv("PERSONA_PATH", str(persona_path))
    monkeypatch.setenv("DATA_DIR", "test-data")
    monkeypatch.setenv("SERVICE_HOST", "https://example.test")

    class FakeChatDB:
        def __init__(self, path: str) -> None:
            self.path = path

    class FakeVectorStore:
        def __init__(self, path: str) -> None:
            self.path = path

        def count(self) -> int:
            return 1

    monkeypatch.setattr(config_loader_module, "ChatDB", FakeChatDB)
    monkeypatch.setattr(config_loader_module, "VectorStore", FakeVectorStore)

    mcp_config = MCPManager(str(config_path))._load_config()
    loaded = load_agent_config(str(config_path))

    assert loaded.config == mcp_config
    assert loaded.config["mcp"] == {
        "endpoint": "https://example.test/api",
        "nested": [
            "https://example.test",
            {"path": "test-data/nested"},
        ],
        "number": 7,
    }
    assert loaded.persona == "Test persona\n"
    assert loaded.chatdb.path == "test-data/history.db"
    assert loaded.vector.path == "test-data/chroma"


def test_agent_loader_rebuilds_an_empty_vector_store(monkeypatch, caplog) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    monkeypatch.setenv("PERSONA_PATH", str(fixtures / "config_utils_persona.md"))
    monkeypatch.setenv("DATA_DIR", "test-data")
    monkeypatch.setenv("SERVICE_HOST", "https://example.test")
    turns = [
        {"text": "first", "session_id": "session-a"},
        {"text": "second", "session_id": "session-b"},
    ]

    class FakeChatDB:
        def __init__(self, path: str) -> None:
            self.path = path

        def export_all_turns(self):
            return turns

    class FakeVectorStore:
        instance = None

        def __init__(self, path: str) -> None:
            self.path = path
            self.rebuild_args = None
            FakeVectorStore.instance = self

        def count(self) -> int:
            return 0

        def rebuild(self, **kwargs) -> None:
            self.rebuild_args = kwargs

    monkeypatch.setattr(config_loader_module, "ChatDB", FakeChatDB)
    monkeypatch.setattr(config_loader_module, "VectorStore", FakeVectorStore)

    with caplog.at_level(logging.INFO, logger="agent.graph"):
        load_agent_config(str(fixtures / "config_utils.yaml"))

    assert FakeVectorStore.instance.rebuild_args["texts"] == ["first", "second"]
    assert FakeVectorStore.instance.rebuild_args["metas"] == [
        {"session_id": "session-a"},
        {"session_id": "session-b"},
    ]
    assert len(FakeVectorStore.instance.rebuild_args["ids"]) == 2
    assert "Vector store is empty — rebuilding from 2 SQLite turns" in caplog.text
