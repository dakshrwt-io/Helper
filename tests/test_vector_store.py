from __future__ import annotations

from agent.memory.vector import VectorStore


def test_query_uses_cached_non_empty_flag_without_counting() -> None:
    class FakeCollection:
        def count(self) -> int:
            raise AssertionError("query should not count before searching")

        def query(self, query_texts, n_results):
            return {
                "documents": [["remembered"]],
                "metadatas": [[{"session_id": "s"}]],
                "distances": [[0.2]],
            }

    store = VectorStore.__new__(VectorStore)
    store._col = FakeCollection()
    store._has_documents = True

    assert store.query("hello", top_k=3) == [
        {"text": "remembered", "meta": {"session_id": "s"}, "distance": 0.2}
    ]


def test_query_skips_chroma_when_store_is_known_empty() -> None:
    class FakeCollection:
        def query(self, query_texts, n_results):
            raise AssertionError("empty store should not be queried")

    store = VectorStore.__new__(VectorStore)
    store._col = FakeCollection()
    store._has_documents = False

    assert store.query("hello") == []


def test_query_scopes_results_to_session() -> None:
    class FakeCollection:
        def query(self, **kwargs):
            assert kwargs["where"] == {"session_id": "session-a"}
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    store = VectorStore.__new__(VectorStore)
    store._col = FakeCollection()
    store._has_documents = True

    assert store.query("secret", session_id="session-a") == []


def test_rebuild_inserts_documents_in_batches() -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.batches = []

        def add(self, ids, documents, metadatas):
            self.batches.append((list(ids), list(documents), list(metadatas)))

        def count(self) -> int:
            return sum(len(batch[0]) for batch in self.batches)

    class FakeClient:
        def __init__(self, collection) -> None:
            self.collection = collection
            self.deleted = False

        def delete_collection(self, name) -> None:
            self.deleted = True

        def get_or_create_collection(self, name, metadata):
            return self.collection

    collection = FakeCollection()
    store = VectorStore.__new__(VectorStore)
    store._client = FakeClient(collection)
    store._collection_name = "chat"
    store._has_documents = True

    added = store.rebuild(
        ids=["1", "2", "3", "4", "5"],
        texts=["a", "b", "c", "d", "e"],
        metas=[{"i": i} for i in range(5)],
        batch_size=2,
    )

    assert added == 5
    assert [batch[0] for batch in collection.batches] == [
        ["1", "2"],
        ["3", "4"],
        ["5"],
    ]
    assert [batch[1] for batch in collection.batches] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]
    assert store._has_documents is True
