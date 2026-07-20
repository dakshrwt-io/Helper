"""Configuration and memory-store initialization for the agent."""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

import yaml

from agent.config_utils import expand_env_vars
from agent.memory.db import ChatDB
from agent.memory.vector import VectorStore

# Keep moved log records under the original logger name.
logger = logging.getLogger("agent.graph")


@dataclass
class LoadedAgentConfig:
    """Configuration values and initialized stores needed by AgentGraph."""

    config: dict[str, Any]
    persona: str
    chatdb: ChatDB
    vector: VectorStore
    max_history_tokens: int


def load_agent_config(config_path: str) -> LoadedAgentConfig:
    """Load agent configuration, persona, and persistent memory stores."""
    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    config = expand_env_vars(raw)
    agent_config = config.get("agent", {})
    max_history_tokens = int(agent_config.get("max_history_tokens", 4000))
    persona_path = os.path.expandvars(
        agent_config.get("persona_path", "agent/persona.md")
    )
    try:
        with open(persona_path, "r", encoding="utf-8") as file:
            persona = file.read()
    except OSError:
        persona = "You are a helpful personal AI assistant."

    memory_config = config.get("memory", {})
    chatdb = ChatDB(
        os.path.expandvars(memory_config.get("sqlite", "data/history.db"))
    )
    vector = VectorStore(
        os.path.expandvars(memory_config.get("chroma", "data/chroma"))
    )

    if vector.count() == 0:
        turns = chatdb.export_all_turns()
        if turns:
            logger.info(
                "Vector store is empty — rebuilding from %d SQLite turns",
                len(turns),
            )
            vector.rebuild(
                ids=[str(uuid.uuid4()) for _ in turns],
                texts=[turn["text"] for turn in turns],
                metas=[{"session_id": turn["session_id"]} for turn in turns],
            )

    return LoadedAgentConfig(
        config=config,
        persona=persona,
        chatdb=chatdb,
        vector=vector,
        max_history_tokens=max_history_tokens,
    )
