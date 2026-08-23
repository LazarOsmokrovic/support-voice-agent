"""Phase 0 checkpoint: the mock DB creates its schema and seeds Amazon-style fake data."""

from __future__ import annotations

from data import mock_db


def test_reset_and_seed_populates_all_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_mock_data.db")

    mock_db.reset_and_seed()

    with mock_db.get_connection() as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("customers", "orders", "tickets", "appointments")
        }

    assert counts["customers"] == len(mock_db.CUSTOMERS)
    assert counts["orders"] == len(mock_db.ORDERS)
    assert counts["tickets"] == len(mock_db.TICKETS)
    assert counts["appointments"] == len(mock_db.APPOINTMENTS)


def test_seed_db_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_mock_data.db")

    mock_db.reset_and_seed()
    mock_db.seed_db()  # calling again must not duplicate rows

    with mock_db.get_connection() as conn:
        customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]

    assert customers == len(mock_db.CUSTOMERS)


def test_orders_reference_amazon_style_ids_and_known_customers(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_mock_data.db")
    mock_db.reset_and_seed()

    customer_ids = {row[0] for row in mock_db.CUSTOMERS}

    with mock_db.get_connection() as conn:
        rows = conn.execute("SELECT order_id, customer_id FROM orders").fetchall()

    assert rows, "expected seeded orders"
    for order_id, customer_id in rows:
        # Amazon order IDs are NNN-NNNNNNN-NNNNNNN
        parts = order_id.split("-")
        assert len(parts) == 3
        assert [len(p) for p in parts] == [3, 7, 7]
        assert customer_id in customer_ids
