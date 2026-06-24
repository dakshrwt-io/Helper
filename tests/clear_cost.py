"""Clear today's cost log (undo test injection)."""
import os
import sys

import sqlalchemy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.memory.db import ChatDB


def main() -> None:
    d = ChatDB()
    s = d.Session()
    s.execute(sqlalchemy.text("DELETE FROM cost_log WHERE date = :d"), {"d": d.today_str()})
    s.commit()
    s.close()
    print(f"cleared, spent today: ${d.spent_today()}")


if __name__ == "__main__":
    main()
