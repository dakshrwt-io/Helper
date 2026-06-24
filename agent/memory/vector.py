"""ChromaDB vector store for semantic recall of past turns."""
from __future__ import annotations

import os
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, path: str = "data/chroma", collection: str = "chat") -> None:
        os.makedirs(path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], texts: list[str], metas: list[dict[str, Any]] | None = None) -> None:
        self._col.add(ids=ids, documents=texts, metadatas=metas or [{} for _ in ids])

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._col.count() == 0:
            return []
        res = self._col.query(query_texts=[text], n_results=top_k)
        out: list[dict[str, Any]] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for d, m, dist in zip(docs, metas, dists):
            out.append({"text": d, "meta": m, "distance": dist})
        return out

    def count(self) -> int:
        return self._col.count()
