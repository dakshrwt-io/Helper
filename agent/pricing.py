"""Per-model token pricing for accurate cost tracking.

Prices are in USD per 1K tokens. The ``DEFAULT`` entry is used when a model
is not explicitly listed.  Users can override any entry via ``model_pricing``
in config.yaml — values from config take priority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelPrice:
    input_per_1k: float
    output_per_1k: float


# ── OpenRouter pricing (selected models) ──────────────────────────────
# Full list: https://openrouter.ai/models
_OPENROUTER: dict[str, ModelPrice] = {
    "z-ai/glm-4.5":             ModelPrice(0.00, 0.00),   # free
    "google/gemini-2.5-flash":  ModelPrice(0.15, 0.60),
    "google/gemini-2.5-pro":    ModelPrice(1.25, 10.00),
    "anthropic/claude-sonnet-4": ModelPrice(3.00, 15.00),
    "anthropic/claude-haiku-4": ModelPrice(1.00, 5.00),
    "openai/gpt-4.1":           ModelPrice(2.00, 8.00),
    "openai/gpt-4.1-mini":      ModelPrice(0.40, 1.60),
    "openai/gpt-4o":            ModelPrice(2.50, 10.00),
    "openai/gpt-4o-mini":       ModelPrice(0.15, 0.60),
    "deepseek/deepseek-r1":     ModelPrice(0.55, 2.19),
    "deepseek/deepseek-v3":     ModelPrice(0.27, 1.10),
    "deepseek/deepseek-chat":   ModelPrice(0.27, 1.10),
    "meta-llama/llama-4-maverick": ModelPrice(0.20, 0.90),
    "mistral/mistral-large":    ModelPrice(2.00, 6.00),
    "qwen/qwen3-235b-a22b":     ModelPrice(0.30, 0.90),
    "x-ai/grok-4":              ModelPrice(2.00, 8.00),
}

# ── DeepSeek direct API pricing ───────────────────────────────────────
# Source: https://api-docs.deepseek.com/quick_start/pricing
_DEEPSEEK: dict[str, ModelPrice] = {
    "deepseek-chat":            ModelPrice(0.27, 1.10),
    "deepseek-reasoner":        ModelPrice(0.55, 2.19),
    "deepseek-v4-flash":        ModelPrice(0.20, 0.80),
    "deepseek-v4-pro":          ModelPrice(0.55, 2.19),
    "deepseek-v3":              ModelPrice(0.27, 1.10),
    "deepseek-r1":              ModelPrice(0.55, 2.19),
}

# ── Groq pricing (vision model) ───────────────────────────────────────
_GROQ: dict[str, ModelPrice] = {
    "meta-llama/llama-4-scout-17b-16e-instruct": ModelPrice(0.10, 0.40),
    "llama-4-scout-17b-16e-instruct":            ModelPrice(0.10, 0.40),
}

# ── NVIDIA NIM pricing ────────────────────────────────────────────────
_NVIDIA: dict[str, ModelPrice] = {
    "minimaxai/minimax-m3": ModelPrice(0.50, 1.00),
    "nvidia/llama-3.1-nemotron-70b-instruct": ModelPrice(0.35, 0.70),
}

# ── Ollama — always free ──────────────────────────────────────────────
_OLLAMA: dict[str, ModelPrice] = {}

# ── Default fallback (conservative $2/M) ──────────────────────────────
DEFAULT = ModelPrice(0.002, 0.002)  # $2 per *million* = $0.002 per *1K*

# ── Flat lookup ───────────────────────────────────────────────────────
_ALL = {}
_ALL.update(_OPENROUTER)
_ALL.update(_DEEPSEEK)
_ALL.update(_GROQ)
_ALL.update(_NVIDIA)
_ALL.update(_OLLAMA)


def get_model_price(
    model_name: str,
    overrides: Optional[dict[str, dict[str, float]]] = None,
) -> ModelPrice:
    """Return (input_per_1k, output_per_1k) for *model_name*.

    Resolution order: user overrides → built-in table → DEFAULT.
    """
    if overrides:
        entry = overrides.get(model_name)
        if entry and "input_per_1k" in entry and "output_per_1k" in entry:
            return ModelPrice(
                input_per_1k=float(entry["input_per_1k"]),
                output_per_1k=float(entry["output_per_1k"]),
            )
    return _ALL.get(model_name, DEFAULT)


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    overrides: Optional[dict[str, dict[str, float]]] = None,
) -> float:
    """Compute USD cost for a single LLM call."""
    price = get_model_price(model_name, overrides)
    return (input_tokens * price.input_per_1k + output_tokens * price.output_per_1k) / 1000.0
