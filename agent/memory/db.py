"""SQLite chat history store (per-session)."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
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


class CostRow(Base):
    __tablename__ = "cost_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, index=True)  # YYYY-MM-DD UTC
    spent = Column(Integer, default=0)  # micro-USD


class ChatDB:
    def __init__(self, path: str = "data/history.db") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        s = self.Session()
        try:
            s.add(MessageRow(session_id=session_id, role=role, content=content))
            s.commit()
        finally:
            s.close()

    def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        s = self.Session()
        try:
            rows = (
                s.query(MessageRow)
                .filter_by(session_id=session_id)
                .order_by(MessageRow.id.desc())
                .limit(limit)
                .all()
            )
            return [{"role": r.role, "content": r.content, "ts": str(r.ts)} for r in reversed(rows)]
        finally:
            s.close()

    def today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def add_cost(self, micro_usd: int) -> None:
        d = self.today_str()
        s = self.Session()
        try:
            row = s.query(CostRow).filter_by(date=d).first()
            if row:
                row.spent += micro_usd
            else:
                s.add(CostRow(date=d, spent=micro_usd))
            s.commit()
        finally:
            s.close()

    def spent_today(self) -> float:
        d = self.today_str()
        s = self.Session()
        try:
            row = s.query(CostRow).filter_by(date=d).first()
            return (row.spent / 1_000_000.0) if row else 0.0
        finally:
            s.close()
