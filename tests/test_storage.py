"""Tests for SQLite persistence, the append-only audit trail and the connection helper."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

from riskscore.audit import AuditLog, EVENT_REJECTED, EVENT_SCORED, verify_hash_chain
from riskscore.errors import NotFoundError
from riskscore.storage import (
    SCHEMA_STATEMENTS,
    SCHEMA_VERSION,
    ApplicationStore,
    connect,
    init_db,
    new_application_id,
    new_request_id,
    read_schema_version,
    record_decision,
    utc_now_iso,
)


class TempDatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.directory.name, "test.db")
        init_db(self.db_path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def save_sample(self, store: ApplicationStore, application_id: str = "app_test000001", **overrides):
        kwargs = {
            "application_id": application_id,
            "credit_score": 612,
            "probability_of_default": 0.184,
            "risk_band": "C",
            "decision": "approve",
            "model_version": "0.1.0",
            "payload": {"age": 40, "purpose": "equipment"},
            "response": {"credit_score": 612, "risk_band": "C"},
            "request_id": "req_test",
        }
        kwargs.update(overrides)
        return store.save(**kwargs)


class IdentifierTests(unittest.TestCase):
    def test_application_ids_are_unique_and_prefixed(self) -> None:
        ids = {new_application_id() for _ in range(200)}
        self.assertEqual(len(ids), 200)
        self.assertTrue(all(value.startswith("app_") for value in ids))

    def test_request_ids_are_unique_and_prefixed(self) -> None:
        ids = {new_request_id() for _ in range(200)}
        self.assertEqual(len(ids), 200)
        self.assertTrue(all(value.startswith("req_") for value in ids))

    def test_timestamps_are_iso_utc_and_sortable(self) -> None:
        first = utc_now_iso()
        second = utc_now_iso()
        self.assertLessEqual(first, second)
        self.assertTrue(first.endswith("Z"))
        self.assertIn("T", first)


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.directory.name, "schema.db")
        init_db(self.db_path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_tables_and_indexes_exist(self) -> None:
        with connect(self.db_path) as connection:
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            indexes = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                )
            }
        self.assertIn("applications", tables)
        self.assertIn("audit_events", tables)
        self.assertIn("schema_meta", tables)
        self.assertIn("idx_applications_created_at", indexes)
        self.assertIn("idx_audit_application", indexes)

    def test_init_is_idempotent(self) -> None:
        # A fresh path, initialised repeatedly: the DDL is IF NOT EXISTS and the
        # version stamp is an upsert, so nothing accumulates.
        fresh = os.path.join(self.directory.name, "idempotent.db")
        init_db(fresh)
        init_db(fresh)
        init_db(fresh)
        with connect(fresh) as connection:
            count = connection.execute("SELECT COUNT(*) AS n FROM schema_meta").fetchone()["n"]
        self.assertEqual(count, 1)
        self.assertEqual(read_schema_version(fresh), str(SCHEMA_VERSION))

    def test_init_does_not_write_into_the_audit_trail(self) -> None:
        """A schema marker is not an application event.

        An earlier revision used ``INSERT OR IGNORE`` into ``audit_events`` as the
        idempotency marker. That silently did nothing, because the table's
        AUTOINCREMENT key never conflicts, so the trail gained a bogus row on
        every startup.
        """
        fresh = os.path.join(self.directory.name, "clean.db")
        init_db(fresh)
        init_db(fresh)
        with connect(fresh) as connection:
            total = connection.execute("SELECT COUNT(*) AS n FROM audit_events").fetchone()["n"]
        self.assertEqual(total, 0)

    def test_read_schema_version_on_an_uninitialised_database(self) -> None:
        self.assertIsNone(read_schema_version(os.path.join(self.directory.name, "blank.db")))

    def test_init_preserves_existing_rows(self) -> None:
        AuditLog(self.db_path).append(EVENT_SCORED, {"n": 1}, application_id="app_keepme01")
        init_db(self.db_path)
        self.assertEqual(
            len(AuditLog(self.db_path).tail(limit=10, application_id="app_keepme01")), 1
        )

    def test_init_creates_missing_parent_directories(self) -> None:
        nested = os.path.join(self.directory.name, "a", "b", "c.db")
        init_db(nested)
        self.assertTrue(os.path.exists(nested))

    def test_schema_statements_are_all_create_if_not_exists(self) -> None:
        for statement in SCHEMA_STATEMENTS:
            normalised = " ".join(statement.split()).upper()
            self.assertTrue(normalised.startswith("CREATE "))
            self.assertIn("IF NOT EXISTS", normalised)

    def test_wal_mode_is_enabled(self) -> None:
        with connect(self.db_path) as connection:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

    def test_row_factory_returns_named_columns(self) -> None:
        with connect(self.db_path) as connection:
            row = connection.execute("SELECT 1 AS value").fetchone()
        self.assertEqual(row["value"], 1)


class ApplicationStoreTests(TempDatabaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = ApplicationStore(self.db_path)

    def test_save_then_get_round_trip(self) -> None:
        self.save_sample(self.store)
        record = self.store.get("app_test000001")
        self.assertEqual(record.credit_score, 612)
        self.assertEqual(record.risk_band, "C")
        self.assertEqual(record.decision, "approve")
        self.assertEqual(record.payload["purpose"], "equipment")
        self.assertEqual(record.response["credit_score"], 612)

    def test_unicode_survives_the_round_trip(self) -> None:
        self.save_sample(
            self.store,
            application_id="app_unicode001",
            payload={"age": 40, "note": "设备采购"},
        )
        self.assertEqual(self.store.get("app_unicode001").payload["note"], "设备采购")

    def test_get_raises_not_found_with_details(self) -> None:
        with self.assertRaises(NotFoundError) as context:
            self.store.get("app_absent")
        self.assertEqual(context.exception.status, 404)
        self.assertEqual(context.exception.details["application_id"], "app_absent")

    def test_find_returns_none_instead_of_raising(self) -> None:
        self.assertIsNone(self.store.find("app_absent"))
        self.save_sample(self.store)
        self.assertIsNotNone(self.store.find("app_test000001"))

    def test_duplicate_primary_key_is_rejected(self) -> None:
        self.save_sample(self.store)
        with self.assertRaises(sqlite3.IntegrityError):
            self.save_sample(self.store, credit_score=700)

    def test_count_and_distribution(self) -> None:
        self.save_sample(self.store, application_id="app_a00000001", risk_band="A", decision="approve")
        self.save_sample(self.store, application_id="app_b00000002", risk_band="E", decision="decline")
        self.save_sample(self.store, application_id="app_c00000003", risk_band="E", decision="decline")
        self.assertEqual(self.store.count(), 3)
        self.assertEqual(self.store.score_distribution(), {"A": 1, "E": 2})
        self.assertEqual(self.store.decision_counts(), {"approve": 1, "decline": 2})

    def test_list_recent_is_newest_first(self) -> None:
        self.save_sample(
            self.store, application_id="app_old0000001", created_at="2026-01-01T00:00:00.000Z"
        )
        self.save_sample(
            self.store, application_id="app_new0000001", created_at="2026-06-01T00:00:00.000Z"
        )
        recent = self.store.list_recent(limit=10)
        self.assertEqual(recent[0].application_id, "app_new0000001")

    def test_list_recent_honours_limit_and_offset(self) -> None:
        for index in range(5):
            self.save_sample(
                self.store,
                application_id=f"app_page{index:05d}",
                created_at=f"2026-01-0{index + 1}T00:00:00.000Z",
            )
        page = self.store.list_recent(limit=2, offset=2)
        self.assertEqual(len(page), 2)
        self.assertEqual(page[0].application_id, "app_page00002")

    def test_list_recent_clamps_absurd_limits(self) -> None:
        self.save_sample(self.store)
        self.assertEqual(len(self.store.list_recent(limit=10_000)), 1)
        self.assertEqual(len(self.store.list_recent(limit=0)), 1)

    def test_record_serialises_to_json(self) -> None:
        self.save_sample(self.store)
        json.dumps(self.store.get("app_test000001").as_dict())

    def test_explicit_created_at_is_preserved(self) -> None:
        self.save_sample(self.store, created_at="2026-03-04T05:06:07.000Z")
        self.assertEqual(self.store.get("app_test000001").created_at, "2026-03-04T05:06:07.000Z")


class RecordDecisionTests(TempDatabaseTestCase):
    """The decision row and the audit row must be committed together."""

    def setUp(self) -> None:
        super().setUp()
        self.store = ApplicationStore(self.db_path)
        self.audit = AuditLog(self.db_path)

    def record(self, application_id: str = "app_atomic00001", **overrides):
        kwargs = {
            "application_id": application_id,
            "credit_score": 640,
            "probability_of_default": 0.21,
            "risk_band": "C",
            "decision": "approve",
            "model_version": "0.1.0",
            "payload": {"age": 40},
            "response": {"credit_score": 640},
            "audit_payload": {"decision": "approve"},
            "audit_event_type": EVENT_SCORED,
            "request_id": "req_atomic",
        }
        kwargs.update(overrides)
        return record_decision(self.db_path, **kwargs)

    def test_writes_both_rows(self) -> None:
        self.record()
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.audit.count(), 1)
        record = self.store.get("app_atomic00001")
        self.assertEqual(record.credit_score, 640)
        events = self.audit.tail(limit=10, application_id="app_atomic00001")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], EVENT_SCORED)

    def test_returns_the_recorded_timestamp(self) -> None:
        timestamp = self.record(created_at="2026-05-06T07:08:09.000Z")
        self.assertEqual(timestamp, "2026-05-06T07:08:09.000Z")
        self.assertEqual(self.store.get("app_atomic00001").created_at, timestamp)
        self.assertEqual(
            self.audit.tail(limit=1, application_id="app_atomic00001")[0]["created_at"], timestamp
        )

    def test_a_failed_second_insert_rolls_back_the_first(self) -> None:
        """Atomicity: no decision may be stored without its audit entry."""
        self.record()
        # Re-using the primary key fails on the applications insert, before the
        # audit insert runs. The audit count must not change.
        with self.assertRaises(sqlite3.IntegrityError):
            self.record(credit_score=700)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.audit.count(), 1)
        self.assertEqual(self.store.get("app_atomic00001").credit_score, 640)

    def test_each_call_stores_its_own_audit_event(self) -> None:
        self.record("app_atomic00002")
        self.record("app_atomic00003")
        self.assertEqual(self.audit.count(), 2)

    def test_unicode_payload_survives(self) -> None:
        self.record("app_atomic00004", payload={"note": "设备采购", "age": 40})
        self.assertEqual(self.store.get("app_atomic00004").payload["note"], "设备采购")

    def test_rollback_leaves_the_connection_usable(self) -> None:
        self.record()
        with self.assertRaises(sqlite3.IntegrityError):
            self.record()
        # A subsequent, valid write must still succeed.
        self.record("app_atomic00005")
        self.assertEqual(self.store.count(), 2)


class AuditLogTests(TempDatabaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.audit = AuditLog(self.db_path)

    def test_append_and_tail(self) -> None:
        self.audit.append(EVENT_SCORED, {"credit_score": 700}, application_id="app_a")
        self.audit.append(EVENT_REJECTED, {"status": 400}, application_id="app_a")
        # ``init_db`` writes a ``schema_initialised`` event, so filter to the
        # scoring events rather than assuming the table started empty.
        events = [
            event
            for event in self.audit.tail(limit=10)
            if event["event_type"] in (EVENT_SCORED, EVENT_REJECTED)
        ]
        self.assertEqual(len(events), 2)
        # Newest first.
        self.assertEqual(events[0]["event_type"], EVENT_REJECTED)
        self.assertEqual(events[1]["payload"]["credit_score"], 700)

    def test_tail_filters_by_application(self) -> None:
        self.audit.append(EVENT_SCORED, {"n": 1}, application_id="app_a")
        self.audit.append(EVENT_SCORED, {"n": 2}, application_id="app_b")
        events = self.audit.tail(limit=10, application_id="app_b")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["payload"]["n"], 2)

    def test_tail_respects_the_limit(self) -> None:
        for index in range(10):
            self.audit.append(EVENT_SCORED, {"n": index})
        self.assertEqual(len(self.audit.tail(limit=3)), 3)

    def test_events_are_ordered_by_monotonic_event_id(self) -> None:
        for index in range(5):
            self.audit.append(EVENT_SCORED, {"n": index})
        ids = [event["event_id"] for event in self.audit.tail(limit=10)]
        self.assertEqual(ids, sorted(ids, reverse=True))

    def test_count_starts_at_zero_on_a_fresh_database(self) -> None:
        """``init_db`` must not seed the trail; it only creates tables."""
        self.assertEqual(self.audit.count(), 0)

    def test_count_tracks_appended_events(self) -> None:
        self.audit.append(EVENT_SCORED, {"n": 1})
        self.audit.append(EVENT_SCORED, {"n": 2})
        self.assertEqual(self.audit.count(), 2)

    def test_record_returns_true_on_success(self) -> None:
        self.assertTrue(self.audit.record(EVENT_SCORED, {"n": 1}, application_id="app_a"))

    def test_record_swallows_database_failures(self) -> None:
        broken = AuditLog(os.path.join(self.directory.name, "audit.db"))
        # Drop the table behind the writer's back to force a genuine failure.
        with connect(broken.db_path) as connection:
            connection.execute("DROP TABLE IF EXISTS audit_events")
        self.assertFalse(broken.record(EVENT_SCORED, {"n": 1}))

    def test_no_update_or_delete_helpers_are_exposed(self) -> None:
        """The append-only guarantee is an API property, so assert on the API."""
        for forbidden in ("update", "delete", "remove", "truncate", "clear"):
            self.assertFalse(hasattr(self.audit, forbidden))

    def test_hash_chain_placeholder_is_explicit(self) -> None:
        self.audit.append(EVENT_SCORED, {"n": 1})
        self.assertTrue(verify_hash_chain(self.audit.tail(limit=5)))
        self.assertTrue(verify_hash_chain([]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
