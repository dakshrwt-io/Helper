"""SQLite chat history store (per-session)."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class MessageRow(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    role = Column(String)  # user | assistant | tool
    content = Column(Text)
    ts = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def _set_wal(dbapi_connection, _connection_record):
    dbapi_connection.execute("PRAGMA journal_mode=WAL")
    dbapi_connection.execute("PRAGMA busy_timeout=5000")


class ChatDB:
    def __init__(self, path: str = "data/history.db") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}")
        event.listens_for(self.engine, "connect")(_set_wal)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        s = self.Session()
        try:
            s.add(MessageRow(session_id=session_id, role=role, content=content))
            s.commit()
        finally:
            s.close()

    def add_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        s = self.Session()
        try:
            s.add_all(
                [
                    MessageRow(session_id=session_id, role="user", content=user_text),
                    MessageRow(
                        session_id=session_id,
                        role="assistant",
                        content=assistant_text,
                    ),
                ]
            )
            s.commit()
        finally:
            s.close()

    def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        s = self.Session()
        try:
            query = (
                select(MessageRow)
                .filter_by(session_id=session_id)
                .order_by(MessageRow.id.desc())
            )
            if limit > 0:
                query = query.limit(limit)
            rows = s.scalars(query).all()
            return [{"role": r.role, "content": r.content, "ts": str(r.ts), "id": r.id} for r in reversed(rows)]
        finally:
            s.close()

    def export_all_turns(self) -> list[dict]:
        s = self.Session()
        try:
            rows = s.scalars(
                select(MessageRow)
                .order_by(MessageRow.id.asc())
            ).all()
        finally:
            s.close()

        turns: list[dict] = []
        pending_users: dict[str, MessageRow] = {}
        for row in rows:
            if row.role == "user":
                pending_users[row.session_id] = row
            elif row.role == "assistant" and row.session_id in pending_users:
                pending_user = pending_users.pop(row.session_id)
                user_text = pending_user.content or ""
                assistant_text = row.content or ""
                turns.append({
                    "session_id": row.session_id,
                    "text": f"User: {user_text}\nAssistant: {assistant_text}",
                })
        return turns
