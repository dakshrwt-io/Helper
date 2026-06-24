"""Verify memory layer: SQLite history + ChromaDB recall after a real chat."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.memory.db import ChatDB
from agent.memory.vector import VectorStore


def main() -> None:
    d = ChatDB()
    v = VectorStore()

    print("=== SQLite history (test-tools session) ===")
    h = d.get_history("test-tools", limit=5)
    print(f"rows: {len(h)}")
    for m in h:
        print(f"  [{m['role']}] {m['content'][:80]}")

    print(f"\nspent today: ${d.spent_today()}")

    print("\n=== ChromaDB recall ===")
    print(f"vector store count: {v.count()}")
    hits = v.query("list files directory", top_k=2)
    print(f"hits: {len(hits)}")
    for i, hit in enumerate(hits):
        print(f"  [{i}] dist={hit['distance']:.3f} text={hit['text'][:80]}")


if __name__ == "__main__":
    main()
