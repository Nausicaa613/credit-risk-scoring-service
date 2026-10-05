"""SQLite storage for scored applications.

Design notes
------------
* One connection per operation. SQLite handles cross-thread access poorly and
  the write volume of a scoring service is low, so opening a short-lived
  connection avoids a whole class of threading bugs at negligible cost. The
  connection is always closed in ``finally``.
* WAL mode plus a busy timeout keeps concurrent readers from failing behind a
  writer, which is the realistic access pattern (many reads, few writes).
* The schema is created on demand, so ``RISKSCORE_DB_PATH`` may point at a file
  that does not exist yet.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .errors import NotFoundError

SCHEMA_VERSION = 1

APPLICATIONS_DDL = """
CREATE TABLE IF NOT EXISTS applications (
    application_id      TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    credit_score        INTEGER NOT NULL,
    probability_of_default REAL NOT NULL,
    risk_band           TEXT NOT NULL,
    decision            TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    request_id          TEXT,
    payload_json        TEXT NOT NULL,
    response_json       TEXT NOT NULL
)
"""

AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id      TEXT,
    created_at          TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    request_id          TEXT,
    payload_json        TEXT NOT NULL
)
"""

#: Key/value metadata about the database itself.
#:
#: This exists because ``init_db`` has to be idempotent, and the obvious way to
#: write a "did I already initialise?" marker -- ``INSERT OR IGNORE`` into the
#: audit trail -- does nothing useful: ``audit_events`` has an AUTOINCREMENT
#: primary key, so no constraint is ever violated and a fresh row lands in the
#: trail on every startup. Keeping the version here, behind a real PRIMARY KEY,
#: gives the marker something to conflict on and leaves the audit trail
#: describing applications rather than service restarts.
SCHEMA_META_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key                 TEXT PRIMARY KEY,
    value               TEXT NOT NULL,
    updated_at          TEXT NOT NULL
)
"""

INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_applications_created_at "
    "ON applications (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_applications_band ON applications (risk_band)",
    "CREATE INDEX IF NOT EXISTS idx_applications_decision ON applications (decision)",
    "CREATE INDEX IF NOT EXISTS idx_audit_application "
    "ON audit_events (application_id, event_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_events (created_at DESC)",
)

SCHEMA_STATEMENTS: Sequence[str] = (APPLICATIONS_DDL, AUDIT_DDL, SCHEMA_META_DDL) + INDEX_DDL


def utc_now_iso() -> str:
    """Timestamp string used everywhere; sortable and unambiguous."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_application_id() -> str:
    """Return an application id with a recognisable ``app_`` prefix."""
    return f"app_{uuid.uuid4().hex[:16]}"


def new_request_id() -> str:
    """Return a request id with a recognisable ``req_`` prefix."""
    return f"req_{uuid.uuid4().hex[:12]}"


@contextmanager
def connect(db_path: str, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """Yield a configured connection, always closing it afterwards.

    Connection setup is not free. Measured on Windows, opening a connection and
    applying the pragmas costs tens of milliseconds, which dominates the cost of
    scoring a single application. Two things keep that in check without adding a
    pool:

    * ``journal_mode=WAL`` is a **persistent** property of the database file, so
      it only needs to be set when the file is created. Re-issuing it on every
      connection is pure overhead.
    * ``busy_timeout`` is the one pragma that must be set per connection, since
      it is connection state rather than database state.

    A real connection pool or a single writer thread is the next step if request
    latency ever matters at scale; see docs/DESIGN.md for that tradeoff.
    """
    creating = db_path == ":memory:" or not os.path.exists(db_path)
    if db_path != ":memory:":
        directory = os.path.dirname(os.path.abspath(db_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
    uri = f"file:{db_path}?mode=ro" if readonly else None
    connection = sqlite3.connect(
        uri or db_path,
        timeout=10.0,
        isolation_level=None,  # autocommit; transactions are explicit
        uri=uri is not None,
    )
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        if creating:
            connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        yield connection
    finally:
        connection.close()


def init_db(db_path: str) -> None:
    """Create tables and indexes if they do not exist, and stamp the version.

    Safe to call on every startup: the DDL is ``IF NOT EXISTS`` and the version
    stamp is an upsert, so existing rows are never touched.
    """
    with connect(db_path) as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_meta (key, value, updated_at) VALUES ('schema_version', ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (str(SCHEMA_VERSION), utc_now_iso()),
        )


def read_schema_version(db_path: str) -> Optional[str]:
    """Return the stamped schema version, or ``None`` if the database is empty."""
    with connect(db_path) as connection:
        try:
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return None if row is None else str(row["value"])


def record_decision(
    db_path: str,
    *,
    application_id: str,
    credit_score: int,
    probability_of_default: float,
    risk_band: str,
    decision: str,
    model_version: str,
    payload: Dict[str, Any],
    response: Dict[str, Any],
    audit_payload: Dict[str, Any],
    audit_event_type: str,
    request_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> str:
    """Persist one decision and its audit event inside a single transaction.

    Why batched rather than two calls
    ---------------------------------
    SQLite commits durably by default: each ``INSERT`` outside an explicit
    transaction is its own transaction and pays for its own fsync. Measured on
    Windows, that is roughly 25 ms per row, so writing the application row and
    the audit row separately cost about 50 ms per request -- two orders of
    magnitude more than scoring itself (0.01 ms).

    Both rows describe the same event, so committing them together is also more
    correct: a crash can no longer leave a stored decision with no matching audit
    entry. Durability is unchanged; only the number of commits drops.

    Returns the timestamp recorded for the decision.
    """
    timestamp = created_at or utc_now_iso()
    with connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO applications ("
                "  application_id, created_at, credit_score, probability_of_default,"
                "  risk_band, decision, model_version, request_id, payload_json, response_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    application_id,
                    timestamp,
                    int(credit_score),
                    float(probability_of_default),
                    risk_band,
                    decision,
                    model_version,
                    request_id,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    json.dumps(response, ensure_ascii=False, sort_keys=True),
                ),
            )
            connection.execute(
                "INSERT INTO audit_events "
                "(application_id, created_at, event_type, request_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    application_id,
                    timestamp,
                    audit_event_type,
                    request_id,
                    json.dumps(audit_payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return timestamp


@dataclass(frozen=True)
class ApplicationRecord:
    """A previously stored scoring decision."""

    application_id: str
    created_at: str
    credit_score: int
    probability_of_default: float
    risk_band: str
    decision: str
    model_version: str
    request_id: Optional[str]
    payload: Dict[str, Any]
    response: Dict[str, Any]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ApplicationRecord":
        return cls(
            application_id=row["application_id"],
            created_at=row["created_at"],
            credit_score=row["credit_score"],
            probability_of_default=row["probability_of_default"],
            risk_band=row["risk_band"],
            decision=row["decision"],
            model_version=row["model_version"],
            request_id=row["request_id"],
            payload=json.loads(row["payload_json"]),
            response=json.loads(row["response_json"]),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "application_id": self.application_id,
            "created_at": self.created_at,
            "credit_score": self.credit_score,
            "probability_of_default": self.probability_of_default,
            "risk_band": self.risk_band,
            "decision": self.decision,
            "model_version": self.model_version,
            "request_id": self.request_id,
            "payload": self.payload,
            "response": self.response,
        }


class ApplicationStore:
    """Persistence for applications, keyed by ``application_id``."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def init(self) -> None:
        init_db(self.db_path)

    def save(
        self,
        *,
        application_id: str,
        credit_score: int,
        probability_of_default: float,
        risk_band: str,
        decision: str,
        model_version: str,
        payload: Dict[str, Any],
        response: Dict[str, Any],
        request_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        """Insert a decision. Returns the stored timestamp."""
        timestamp = created_at or utc_now_iso()
        with connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO applications ("
                "  application_id, created_at, credit_score, probability_of_default,"
                "  risk_band, decision, model_version, request_id, payload_json, response_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    application_id,
                    timestamp,
                    int(credit_score),
                    float(probability_of_default),
                    risk_band,
                    decision,
                    model_version,
                    request_id,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    json.dumps(response, ensure_ascii=False, sort_keys=True),
                ),
            )
        return timestamp

    def get(self, application_id: str) -> ApplicationRecord:
        """Fetch one application or raise :class:`NotFoundError`."""
        with connect(self.db_path, readonly=True) as connection:
            cursor = connection.execute(
                "SELECT * FROM applications WHERE application_id = ?",
                (application_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise NotFoundError(
                f"no application with id {application_id!r}",
                details={"application_id": application_id},
            )
        return ApplicationRecord.from_row(row)

    def find(self, application_id: str) -> Optional[ApplicationRecord]:
        """Fetch one application, returning ``None`` instead of raising."""
        try:
            return self.get(application_id)
        except NotFoundError:
            return None

    def list_recent(self, limit: int = 50, *, offset: int = 0) -> List[ApplicationRecord]:
        """Return the most recent applications, newest first."""
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        with connect(self.db_path, readonly=True) as connection:
            cursor = connection.execute(
                "SELECT * FROM applications ORDER BY created_at DESC, rowid DESC "
                "LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cursor.fetchall()
        return [ApplicationRecord.from_row(row) for row in rows]

    def count(self) -> int:
        with connect(self.db_path, readonly=True) as connection:
            cursor = connection.execute("SELECT COUNT(*) AS n FROM applications")
            return int(cursor.fetchone()["n"])

    def score_distribution(self) -> Dict[str, int]:
        """Counts per risk band, used by ``/metrics`` and the README."""
        with connect(self.db_path, readonly=True) as connection:
            cursor = connection.execute(
                "SELECT risk_band, COUNT(*) AS n FROM applications GROUP BY risk_band"
            )
            return {row["risk_band"]: int(row["n"]) for row in cursor.fetchall()}

    def decision_counts(self) -> Dict[str, int]:
        with connect(self.db_path, readonly=True) as connection:
            cursor = connection.execute(
                "SELECT decision, COUNT(*) AS n FROM applications GROUP BY decision"
            )
            return {row["decision"]: int(row["n"]) for row in cursor.fetchall()}


__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_STATEMENTS",
    "ApplicationRecord",
    "ApplicationStore",
    "connect",
    "init_db",
    "read_schema_version",
    "record_decision",
    "new_application_id",
    "new_request_id",
    "utc_now_iso",
]
