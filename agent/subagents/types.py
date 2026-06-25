from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubAgentConfig:
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    max_iterations: int = 10
    max_seconds: float = 60.0

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any], defaults: dict[str, Any]) -> SubAgentConfig:
        return cls(
            name=name,
            description=data.get("description", name),
            system_prompt=data.get("system_prompt", f"You are a {name} specialist."),
            tools=data.get("tools", defaults.get("tools", [])),
            model=data.get("model") or defaults.get("model"),
            max_iterations=int(data.get("max_iterations", defaults.get("max_iterations", 10))),
            max_seconds=float(data.get("max_seconds", defaults.get("max_seconds", 60.0))),
        )


@dataclass
class SubAgentResult:
    subagent_type: str
    output: str
    iterations: int
    llm_calls: int
    success: bool
    error: str | None = None
