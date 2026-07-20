"""Regression coverage for backend-specific LLM construction."""
from __future__ import annotations

import sys
from types import ModuleType

from agent.llm_factory import build_llm


def _install_fake_clients(monkeypatch):
    class ChatOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class ChatOllama:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class ChatNVIDIA:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    openai = ModuleType("langchain_openai")
    openai.ChatOpenAI = ChatOpenAI
    ollama = ModuleType("langchain_ollama")
    ollama.ChatOllama = ChatOllama
    nvidia = ModuleType("langchain_nvidia_ai_endpoints")
    nvidia.ChatNVIDIA = ChatNVIDIA
    monkeypatch.setitem(sys.modules, "langchain_openai", openai)
    monkeypatch.setitem(sys.modules, "langchain_ollama", ollama)
    monkeypatch.setitem(sys.modules, "langchain_nvidia_ai_endpoints", nvidia)
    return ChatOpenAI, ChatOllama, ChatNVIDIA


def test_build_llm_keeps_backend_client_types_and_model_names(monkeypatch) -> None:
    chat_openai, chat_ollama, chat_nvidia = _install_fake_clients(monkeypatch)

    ollama, ollama_model = build_llm(
        {"ollama": {"model": "local-model", "base_url": "http://ollama"}},
        "ollama",
        temperature=0.2,
    )
    deepseek, deepseek_model = build_llm(
        {
            "deepseek": {
                "model": "deep-model",
                "api_key": "deep-key",
                "base_url": "https://deep.example",
                "thinking": "true",
                "reasoning_effort": "high",
            }
        },
        "deepseek",
        temperature=0.4,
        max_retries=5,
        timeout=30,
    )
    nvidia, nvidia_model = build_llm(
        {
            "nvidia": {
                "model": "nvidia-model",
                "api_key": "nvidia-key",
                "temperature": "0.6",
                "top_p": "0.8",
                "max_completion_tokens": "1234",
            }
        },
        "nvidia",
        temperature=0.3,
        max_retries=4,
        timeout=45,
    )
    openrouter, openrouter_model = build_llm(
        {
            "openrouter": {
                "model": "router-model",
                "api_key": "router-key",
                "base_url": "https://router.example",
            }
        },
        "openrouter",
        temperature=0.1,
        max_retries=2,
        timeout=15,
    )

    assert isinstance(ollama, chat_ollama)
    assert isinstance(deepseek, chat_openai)
    assert isinstance(nvidia, chat_nvidia)
    assert isinstance(openrouter, chat_openai)
    assert (ollama_model, deepseek_model, nvidia_model, openrouter_model) == (
        "local-model",
        "deep-model",
        "nvidia-model",
        "router-model",
    )
    assert deepseek.kwargs == {
        "model": "deep-model",
        "openai_api_key": "deep-key",
        "openai_api_base": "https://deep.example",
        "temperature": 0.4,
        "model_kwargs": {
            "extra_body": {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            }
        },
        "timeout": 30,
        "max_retries": 5,
    }
    assert "model_kwargs" not in openrouter.kwargs


def test_build_llm_preserves_deepseek_defaults_and_model_override(monkeypatch) -> None:
    chat_openai, _, _ = _install_fake_clients(monkeypatch)

    deepseek, deepseek_model = build_llm(
        {"deepseek": {"api_key": "deep-key"}},
        "deepseek",
        model_name="override-model",
    )
    openrouter, openrouter_model = build_llm(
        {"openrouter": {"api_key": "router-key"}},
        "openrouter",
        model_name="override-model",
    )

    assert isinstance(deepseek, chat_openai)
    assert isinstance(openrouter, chat_openai)
    assert (deepseek_model, openrouter_model) == ("override-model", "override-model")
    assert deepseek.kwargs["model"] == "override-model"
    assert deepseek.kwargs["openai_api_base"] == "https://api.deepseek.com"
    assert deepseek.kwargs["model_kwargs"] is None
    assert openrouter.kwargs["model"] == "override-model"
    assert openrouter.kwargs["openai_api_base"] == "https://openrouter.ai/api/v1"
    assert "model_kwargs" not in openrouter.kwargs
