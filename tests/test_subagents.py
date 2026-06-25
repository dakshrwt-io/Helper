import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.subagents.types import SubAgentConfig, SubAgentResult


def test_subagent_config_from_dict():
    defaults = {"max_iterations": 10, "max_seconds": 60.0, "tools": [], "model": None}
    cfg = SubAgentConfig.from_dict(
        "test",
        {
            "description": "Test agent",
            "system_prompt": "You are a test agent.",
            "tools": ["mcp"],
            "max_iterations": 5,
            "max_seconds": 30.0,
        },
        defaults,
    )
    assert cfg.name == "test"
    assert cfg.description == "Test agent"
    assert cfg.tools == ["mcp"]
    assert cfg.max_iterations == 5
    assert cfg.max_seconds == 30.0
    assert cfg.model is None


def test_subagent_config_uses_defaults():
    defaults = {"max_iterations": 7, "max_seconds": 45.0, "tools": ["computer"], "model": None}
    cfg = SubAgentConfig.from_dict(
        "minimal",
        {"description": "Minimal agent", "system_prompt": "Minimal."},
        defaults,
    )
    assert cfg.max_iterations == 7
    assert cfg.max_seconds == 45.0
    assert cfg.tools == defaults["tools"]


def test_subagent_result():
    r = SubAgentResult("filesystem", "ok", 3, 2, True)
    assert r.subagent_type == "filesystem"
    assert r.success is True
    assert r.error is None

    r2 = SubAgentResult("bogus", "failed", 0, 0, False, error="unknown subagent")
    assert r2.success is False
    assert r2.error == "unknown subagent"


if __name__ == "__main__":
    test_subagent_config_from_dict()
    test_subagent_config_uses_defaults()
    test_subagent_result()
    print("All subagent type tests passed")
