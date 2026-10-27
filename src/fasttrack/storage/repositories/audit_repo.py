"""Cryptographically hash-chained audit repository providing tamper-evident chain of custody."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.fasttrack.storage.repositories.base import BaseRepository


class AuditRepository(BaseRepository):
    """Data access layer for immutable, hash-chained security operations audit logs."""

    def log_action(
        self,
        actor: str,
        action: str,
        entity_type: str,
        entity_id: str,
        details: str,
    ) -> Dict[str, Any]:
        """Append an action to the audit log, chaining its SHA-256 hash to the previous entry."""
        log_id = f"aud-{uuid.uuid4().hex[:12]}"
        now_utc = datetime.now(timezone.utc).isoformat()

        # Fetch previous entry hash
        prev_row = self.execute_query(
            "SELECT entry_hash FROM audit_log ORDER BY rowid DESC LIMIT 1;"
        )
        prev_hash = prev_row[0]["entry_hash"] if prev_row else "0" * 64

        # Compute tamper-evident hash
        payload = f"{prev_hash}|{now_utc}|{actor}|{action}|{entity_type}|{entity_id}|{details}"
        entry_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        self.execute_non_query(
            """
            INSERT INTO audit_log (id, ts_utc, actor, action, entity_type, entity_id, details, prev_hash, entry_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (log_id, now_utc, actor, action, entity_type, entity_id, details, prev_hash, entry_hash),
        )

        return {
            "id": log_id,
            "ts_utc": now_utc,
            "actor": actor,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "details": details,
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
        }

    def verify_audit_chain(self) -> Tuple[bool, Optional[str]]:
        """Verify the cryptographic hash-chain across all audit entries from beginning to end."""
        rows = self.execute_query("SELECT * FROM audit_log ORDER BY rowid ASC;")
        if not rows:
            return True, "Audit log is empty."

        expected_prev = "0" * 64
        for idx, row in enumerate(rows):
            if row["prev_hash"] != expected_prev:
                return False, f"Broken chain at entry {row['id']} (row {idx}): prev_hash mismatch."

            payload = f"{row['prev_hash']}|{row['ts_utc']}|{row['actor']}|{row['action']}|{row['entity_type']}|{row['entity_id']}|{row['details']}"
            computed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if computed != row["entry_hash"]:
                return False, f"Tampered data detected at entry {row['id']} (row {idx}): entry_hash does not match payload."

            expected_prev = row["entry_hash"]

        return True, f"Verified {len(rows)} entries successfully. Cryptographic chain intact."

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.execute_query(
            "SELECT * FROM audit_log ORDER BY ts_utc DESC LIMIT ?;",
            (limit,),
        )
