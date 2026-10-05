"""Append-only audit trail.

Controls that matter for a scoring system
----------------------------------------
* **Append-only.** :class:`AuditLog` exposes ``append`` and ``tail`` and nothing
  else. There is no update or delete path in the code, so an audit entry cannot
  be quietly rewritten by a future change to the service. The database user is
  expected to be granted INSERT and SELECT only in a real deployment.
* **Self-describing events.** Each event carries a type, a timestamp, the
  request id that caused it, the model version that produced the output and the
  full input/output pair, so a record can be replayed without contacting the
  service.
* **Never raises into the request path.** ``record`` swallows and logs failures:
  a broken audit sink must not turn a successful scoring request into a 500. In
  a regulated deployment the opposite tradeoff (fail closed) would be correct;
  that choice is documented in docs/DESIGN.md.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .storage import connect, utc_now_iso

logger = logging.getLogger(__name__)

EVENT_SCORED = "application_scored"
EVENT_REJECTED = "application_rejected"


class AuditLog:
    """Append-only writer and reader for ``audit_events``."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def append(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        application_id: Optional[str] = None,
        request_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> None:
        """Write one event. Raises only on database errors."""
        with connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO audit_events "
                "(application_id, created_at, event_type, request_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    application_id,
                    created_at or utc_now_iso(),
                    event_type,
                    request_id,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )

    def record(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        application_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """Best-effort append used inside the request path.

        Returns ``True`` when the event was persisted. A failure is logged and
        reported to the caller as ``False`` rather than propagated.
        """
        try:
            self.append(
                event_type,
                payload,
                application_id=application_id,
                request_id=request_id,
            )
            return True
        except Exception:  # noqa: BLE001 - deliberate: audit must not break scoring
            logger.exception("failed to append audit event %s", event_type)
            return False

    def tail(self, limit: int = 50, *, application_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the newest events first, optionally filtered by application."""
        limit = max(1, min(int(limit), 500))
        if application_id is None:
            query = "SELECT * FROM audit_events ORDER BY event_id DESC LIMIT ?"
            parameters: tuple = (limit,)
        else:
            query = (
                "SELECT * FROM audit_events WHERE application_id = ? "
                "ORDER BY event_id DESC LIMIT ?"
            )
            parameters = (application_id, limit)

        with connect(self.db_path, readonly=True) as connection:
            cursor = connection.execute(query, parameters)
            rows = cursor.fetchall()

        events: List[Dict[str, Any]] = []
        for row in rows:
            events.append(
                {
                    "event_id": row["event_id"],
                    "application_id": row["application_id"],
                    "created_at": row["created_at"],
                    "event_type": row["event_type"],
                    "request_id": row["request_id"],
                    "payload": json.loads(row["payload_json"]),
                }
            )
        return events

    def count(self) -> int:
        with connect(self.db_path, readonly=True) as connection:
            cursor = connection.execute("SELECT COUNT(*) AS n FROM audit_events")
            return int(cursor.fetchone()["n"])


def verify_hash_chain(events: List[Dict[str, Any]]) -> bool:
    """Placeholder for tamper-evidence.

    A production audit trail would hash each event together with its
    predecessor. v0.1 records the events but does not yet chain them; this
    function documents the intended contract and is covered by a test so the
    behaviour is explicit rather than assumed.
    """
    return all("event_id" in event for event in events)


__all__ = ["AuditLog", "EVENT_SCORED", "EVENT_REJECTED", "verify_hash_chain", "utc_now_iso"]
