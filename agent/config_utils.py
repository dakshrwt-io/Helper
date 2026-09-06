"""Small helpers shared by configuration loaders."""
from __future__ import annotations

import os
from typing import Any


class UnexpandedVarError(RuntimeError):
    """Raised when a config value still contains an unexpanded ${VAR} reference."""


def expand_env_vars(obj: Any) -> Any:
    """Return a copy of config containers with environment variables expanded.

    Note: on Windows, os.path.expandvars resolves ${VAR} only when VAR is set;
    undefined references are left untouched — use require_expanded for values
    that must never fall through.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {key: expand_env_vars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(value) for value in obj]
    return obj


def require_expanded(value: str, context: str) -> str:
    """Fail fast when a value still references an undefined environment variable."""
    if "${" in value:
        raise UnexpandedVarError(
            f"{context} contains an unexpanded environment variable: {value!r}. "
            "Set the variable in .env (see .env.example) and restart."
        )
    return value
