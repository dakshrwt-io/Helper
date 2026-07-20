"""Small helpers shared by configuration loaders."""
from __future__ import annotations

import os
from typing import Any


def expand_env_vars(obj: Any) -> Any:
    """Return a copy of config containers with environment variables expanded."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {key: expand_env_vars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(value) for value in obj]
    return obj
