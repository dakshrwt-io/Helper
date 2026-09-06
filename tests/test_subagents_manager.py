import asyncio
import os
import sys

import pytest
from unittest.mock import MagicMock

from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class _EmptyArgs(BaseModel):
    pass


def _mock_func(**kwargs) -> str:
    return "result"

mcp_tool = StructuredTool.from_function(
    func=_mock_func,
    name="read_file",
    description="Read a file",
    args_schema=_EmptyArgs,
)
computer_tool = StructuredTool.from_function(
    func=_mock_func,
    name="computer_click",
    description="Click somewhere",
    args_schema=_EmptyArgs,
)


@pytest.mark.asyncio
async def test_manager_init():
    from agent.subagents import SubAgentManager

    raw_config = {
        "subagents": {
            "enabled": True,
            "default_max_iterations": 10,
            "default_max_seconds": 60.0,
            "agents": {
                "filesystem": {
                    "description": "File ops",
                    "system_prompt": "You are a filesystem specialist.",
                    "tools": ["mcp"],
                    "max_iterations": 5,
                    "max_seconds": 30.0,
                },
                "computer_control": {
                    "description": "Desktop control",
                    "system_prompt": "You are a desktop automation specialist.",
                    "tools": ["computer"],
                    "max_iterations": 8,
                    "max_seconds": 45.0,
                },
            },
        }
    }

    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()

    manager = SubAgentManager(
        raw_config=raw_config,
        mcp_tools=[mcp_tool],
        computer_tools=[computer_tool],
        llm=llm,
        llm_backend="openrouter",
        model_name="test-model",
    )

    assert set(manager.agent_names) == {"filesystem", "computer_control"}
    assert "filesystem" in manager.configs
    assert "computer_control" in manager.configs

    fs_cfg = manager.configs["filesystem"]
    assert fs_cfg.tools == ["mcp"]
    assert fs_cfg.max_iterations == 5
    assert fs_cfg.max_seconds == 30.0

    cc_cfg = manager.configs["computer_control"]
    assert cc_cfg.tools == ["computer"]
    assert cc_cfg.max_iterations == 8

    print("SubAgentManager init: OK")


@pytest.mark.asyncio
async def test_manager_unknown_subagent_returns_error():
    from agent.subagents import SubAgentManager

    raw_config = {
        "subagents": {
            "enabled": True,
            "agents": {
                "test": {
                    "description": "Test",
                    "system_prompt": "Test.",
                    "tools": [],
                }
            },
        }
    }

    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()

    manager = SubAgentManager(
        raw_config=raw_config,
        mcp_tools=[],
        computer_tools=[],
        llm=llm,
        llm_backend="openrouter",
        model_name="test",
    )

    result = await manager.run("nonexistent", "Do something")
    assert result.success is False
    assert result.error is not None
    assert "nonexistent" in result.error
    print("Manager unknown subagent: OK")


@pytest.mark.asyncio
async def test_manager_disabled():
    from agent.subagents import SubAgentManager

    raw_config = {
        "subagents": {
            "enabled": False,
            "agents": {
                "secret": {
                    "description": "Should not load",
                    "system_prompt": "Secret agent.",
                    "tools": [],
                }
            },
        }
    }

    manager = SubAgentManager(
        raw_config=raw_config,
        mcp_tools=[],
        computer_tools=[],
        llm=MagicMock(),
        llm_backend="openrouter",
        model_name="test",
    )

    assert manager.agent_names == []


@pytest.mark.asyncio
async def test_task_tool_builds():
    from agent.subagents import SubAgentManager, build_task_tool

    raw_config = {
        "subagents": {
            "enabled": True,
            "agents": {
                "filesystem": {
                    "description": "File ops",
                    "system_prompt": "You are a filesystem specialist.",
                    "tools": ["mcp"],
                },
                "computer_control": {
                    "description": "Desktop control",
                    "system_prompt": "You are a desktop automation specialist.",
                    "tools": ["computer"],
                },
            },
        }
    }

    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()

    manager = SubAgentManager(
        raw_config=raw_config,
        mcp_tools=[mcp_tool],
        computer_tools=[computer_tool],
        llm=llm,
        llm_backend="openrouter",
        model_name="test-model",
    )

    task_tool = build_task_tool(manager)
    assert task_tool.name == "task"
    assert "filesystem" in task_tool.description
    assert "computer_control" in task_tool.description
