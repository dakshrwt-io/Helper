"""ChromaDB vector store for semantic recall of past turns."""
from __future__ import annotations

import logging
import os
import shutil
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, path: str = "data/chroma", collection: str = "chat") -> None:
        os.makedirs(path, exist_ok=True)
        self._path = path
        self._collection_name = collection
        self._client: chromadb.PersistentClient | None = None
        self._col: Any = None
        self._has_documents = False
        self._init_store()

    def _init_store(self) -> None:
        try:
            self._client = chromadb.PersistentClient(path=self._path)
            self._col = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._has_documents = self.count() > 0
        except Exception as exc:
            logger.warning(
                "ChromaDB init failed, attempting recovery: %s", exc
            )
            self._recover_store()

    def _recover_store(self) -> None:
        try:
            if os.path.isdir(self._path):
                shutil.rmtree(self._path)
            os.makedirs(self._path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._path)
            self._col = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._has_documents = False
            logger.info("ChromaDB recovered — store recreated from scratch")
        except Exception as exc:
            logger.error("ChromaDB recovery failed: %s", exc)
            raise

    def add(self, ids: list[str], texts: list[str], metas: list[dict[str, Any]] | None = None) -> None:
        self._col.add(ids=ids, documents=texts, metadatas=metas or [{} for _ in ids])
        if ids:
            self._has_documents = True

    def query(
        self, text: str, top_k: int = 5, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        if not self._has_documents:
            return []
        kwargs: dict[str, Any] = {"query_texts": [text], "n_results": top_k}
        if session_id is not None:
            kwargs["where"] = {"session_id": session_id}
        res = self._col.query(**kwargs)
        out: list[dict[str, Any]] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for d, m, dist in zip(docs, metas, dists):
            out.append({"text": d, "meta": m, "distance": dist})
        return out

    def count(self) -> int:
        try:
            return self._col.count()
        except Exception:
            return 0

    def rebuild(
        self,
        ids: list[str],
        texts: list[str],
        metas: list[dict[str, Any]] | None = None,
        batch_size: int = 500,
    ) -> int:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._has_documents = False
        if ids:
            metadata = metas or [{} for _ in ids]
            step = max(1, batch_size)
            for start in range(0, len(ids), step):
                end = start + step
                self.add(
                    ids=ids[start:end],
                    texts=texts[start:end],
                    metas=metadata[start:end],
                )
        added = self.count()
        self._has_documents = added > 0
        logger.info("Vector store rebuilt with %d documents", added)
        return added
