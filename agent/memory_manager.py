"""Conversation-history and vector-memory operations for AgentGraph."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.memory.db import ChatDB
from agent.memory.vector import VectorStore
from agent.trace import TraceCollector, display_content, duration_ms

# Keep moved log records under the original logger name.
logger = logging.getLogger("agent.graph")


class MemoryManager:
    """Load, summarize, recall, and persist one agent's conversation memory."""

    def __init__(
        self,
        chatdb: ChatDB | None,
        vector: VectorStore | None,
        llm: Any,
        max_history_tokens: int,
    ) -> None:
        self.chatdb = chatdb
        self.vector = vector
        self.llm = llm
        self.max_history_tokens = max_history_tokens
        self._summaries: dict[str, tuple[str, int]] = {}

    async def _recall_context(
        self,
        user_text: str,
        session_id: str,
        trace: TraceCollector,
        exclude_hashes: set[str] | None = None,
    ) -> list[str]:
        if not self.vector:
            return []

        recall_start = time.perf_counter()
        trace.emit("memory_recall_started")
        hits = await asyncio.to_thread(
            self.vector.query, user_text, 3, session_id
        )
        recalled: list[str] = []
        exclude = exclude_hashes or set()
        for hit in hits:
            if hit.get("distance", 1.0) >= 0.6:
                continue
            meta = hit.get("meta") or {}
            if meta.get("turn_hash", "") in exclude:
                continue
            recalled.append(hit["text"])
        trace.emit(
            "memory_recall_completed",
            duration_ms=duration_ms(recall_start),
            matches=len(recalled),
        )
        return recalled

    async def _load_history_messages(
        self, session_id: str, trace: TraceCollector
    ) -> tuple[list[Any], str]:
        """Load recent history within token budget. Returns (prior_messages, summary_string).

        Oldest messages that exceed the budget are summarized via LLM call (best-effort).
        Summary is prepended as context so truncated history is not lost.
        """
        if not self.chatdb:
            return [], ""

        history_start = time.perf_counter()
        history = await asyncio.to_thread(self.chatdb.get_history, session_id, limit=0)
        if not history:
            trace.emit("memory_history_loaded", duration_ms=duration_ms(history_start), messages=0)
            return [], ""

        # Token-count from most-recent-first, keep only what fits in budget
        budget = self.max_history_tokens
        kept: list[dict] = []
        token_count = 0
        max_kept_id = 0

        for message in reversed(history):  # most recent first
            tokens = self._count_tokens(message["content"])
            if token_count + tokens > budget and kept:
                break
            kept.append(message)
            token_count += tokens
            if message["id"] > max_kept_id:
                max_kept_id = message["id"]

        kept.reverse()  # back to chronological
        dropped = [message for message in history if message["id"] < max_kept_id]

        summary = ""
        if dropped:
            summary = await self._summarize_session(session_id, dropped)

        prior: list[Any] = []
        for message in kept:
            if message["role"] == "user":
                prior.append(HumanMessage(content=message["content"]))
            else:
                prior.append(AIMessage(content=message["content"]))

        trace.emit(
            "memory_history_loaded",
            duration_ms=duration_ms(history_start),
            messages=len(prior),
            dropped=len(dropped),
            has_summary=bool(summary),
        )
        return prior, summary

    async def _summarize_session(
        self, session_id: str, dropped: list[dict]
    ) -> str:
        """Generate a rolling summary of older truncated messages. Best-effort.

        Caches summary keyed by session_id. Only regenerates when new messages
        have appeared since the last summary was made.
        """
        if not dropped or self.llm is None:
            return ""

        max_id = max(message["id"] for message in dropped)
        cached = self._summaries.get(session_id)
        if cached and cached[1] >= max_id:
            return cached[0]  # cached summary covers all dropped messages

        # Build prompt from dropped messages
        lines: list[str] = []
        for message in dropped[:200]:  # cap to avoid overlong prompts
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{role}: {message['content'][:300]}")
        conversation = "\n\n".join(lines)

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke([
                    SystemMessage(
                        content="Summarize this conversation history in 2-3 concise sentences. "
                        "Capture key topics, decisions, and facts. Be brief."
                    ),
                    HumanMessage(content=conversation),
                ]),
                timeout=30,
            )
            summary = display_content(response.content).strip()
            self._summaries[session_id] = (summary, max_id)
            return summary
        except Exception:
            logger.warning("Session summary generation failed for %s", session_id)
            return ""

    def _build_turn_messages(
        self,
        user_text: str,
        recalled: list[str],
        prior: list[Any],
        summary: str = "",
    ) -> list[Any]:
        messages: list[Any] = []
        if summary:
            messages.append(
                SystemMessage(content=f"Earlier conversation summary:\n{summary}")
            )
        if recalled:
            messages.append(
                SystemMessage(
                    content="\n\nRelevant past context:\n" + "\n---\n".join(recalled)
                )
            )
        messages.extend(prior)
        messages.append(HumanMessage(content=user_text))
        return messages

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Approximate token count using char-length heuristic (1 token ≈ 4 chars)."""
        return max(1, len(text) // 4)

    async def _persist_turn(
        self,
        session_id: str,
        user_text: str,
        final: str,
        trace: TraceCollector,
    ) -> None:
        persist_start = time.perf_counter()
        if self.chatdb:
            add_turn = getattr(self.chatdb, "add_turn", None)
            if callable(add_turn):
                await asyncio.to_thread(add_turn, session_id, user_text, final)
            else:
                await asyncio.to_thread(
                    self.chatdb.add_message,
                    session_id,
                    "user",
                    user_text,
                )
                await asyncio.to_thread(
                    self.chatdb.add_message,
                    session_id,
                    "assistant",
                    final,
                )
        if self.vector and final:
            turn_hash = str(hash(user_text + "|" + final))
            await asyncio.to_thread(
                self.vector.add,
                [str(uuid.uuid4())],
                [f"User: {user_text}\nAssistant: {final}"],
                [{"session_id": session_id, "turn_hash": turn_hash}],
            )
        trace.emit("memory_persisted", duration_ms=duration_ms(persist_start))
