"""Base repository abstraction providing safe connection and transaction helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from src.fasttrack.storage.db import DatabaseManager


class BaseRepository:
    """Base class for all typed repositories communicating with SQLite."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        return self.db.execute_query(query, params)

    def execute_non_query(self, query: str, params: tuple = ()) -> int:
        return self.db.execute_non_query(query, params)

    def execute_scalar(self, query: str, params: tuple = ()) -> Any:
        return self.db.execute_scalar(query, params)
