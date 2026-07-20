"""Shared LLM builder used by AgentGraph and SubAgentManager."""
from __future__ import annotations

from typing import Any


_OMIT_MODEL_KWARGS = object()


def _build_openai_compatible(
    ocfg: dict[str, Any],
    default_model: str,
    default_base_url: str,
    temperature: float,
    max_retries: int,
    timeout: int,
    extra_model_kwargs: Any = None,
) -> Any:
    """Build a ChatOpenAI client for an already selected compatible model."""
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": default_model,
        "openai_api_key": ocfg.get("api_key", ""),
        "openai_api_base": ocfg.get("base_url", default_base_url),
        "temperature": temperature,
    }
    if extra_model_kwargs is not _OMIT_MODEL_KWARGS:
        kwargs["model_kwargs"] = extra_model_kwargs
    kwargs["timeout"] = timeout
    kwargs["max_retries"] = max_retries
    return ChatOpenAI(**kwargs)


def build_llm(
    llm_cfg: dict[str, Any],
    backend: str,
    model_name: str | None = None,
    temperature: float = 0.3,
    max_retries: int = 1,
    timeout: int = 60,
) -> tuple[Any, str]:
    """Build an LLM client for the given backend, optionally overriding the model name.

    Returns (llm_instance, actual_model_name).
    """
    backend = backend.lower()

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        ocfg = llm_cfg.get("ollama", {})
        if model_name is None:
            model_name = str(ocfg.get("model", "gemma4:e2b"))
        llm = ChatOllama(
            model=model_name,
            base_url=ocfg.get("base_url", "http://127.0.0.1:11434"),
            temperature=temperature,
        )
        return llm, model_name

    if backend == "deepseek":
        # Preserve the original dependency-error timing for this branch.
        from langchain_openai import ChatOpenAI  # noqa: F401

        ocfg = llm_cfg.get("deepseek", {})
        if model_name is None:
            model_name = str(ocfg.get("model", "deepseek-v4-flash"))
        model_kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        if str(ocfg.get("thinking", "")).lower() == "true":
            extra_body["thinking"] = {"type": "enabled"}
        reasoning = ocfg.get("reasoning_effort", "")
        if reasoning:
            extra_body["reasoning_effort"] = reasoning
        if extra_body:
            model_kwargs["extra_body"] = extra_body
        llm = _build_openai_compatible(
            ocfg,
            model_name,
            "https://api.deepseek.com",
            temperature,
            max_retries,
            timeout,
            extra_model_kwargs=model_kwargs if model_kwargs else None,
        )
        return llm, model_name

    if backend == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        ocfg = llm_cfg.get("nvidia", {})
        if model_name is None:
            model_name = str(ocfg.get("model", "minimaxai/minimax-m3"))
        llm = ChatNVIDIA(
            model=model_name,
            api_key=ocfg.get("api_key", ""),
            temperature=float(ocfg.get("temperature", temperature)),
            top_p=float(ocfg.get("top_p", 0.95)),
            max_completion_tokens=int(ocfg.get("max_completion_tokens", 8192)),
            timeout=timeout,
            max_retries=max_retries,
        )
        return llm, model_name

    # default: openrouter
    # Preserve the original dependency-error timing for this branch.
    from langchain_openai import ChatOpenAI  # noqa: F401

    ocfg = llm_cfg.get("openrouter", {})
    if model_name is None:
        model_name = str(ocfg.get("model", "z-ai/glm-4.5"))
    llm = _build_openai_compatible(
        ocfg,
        model_name,
        "https://openrouter.ai/api/v1",
        temperature,
        max_retries,
        timeout,
        extra_model_kwargs=_OMIT_MODEL_KWARGS,
    )
    return llm, model_name
