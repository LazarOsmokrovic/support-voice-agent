"""Phase 1 checkpoint: get_order_status for a valid order, an invalid order-ID
format, and a well-formed but nonexistent order."""

from __future__ import annotations

import data.mock_db as mock_db
from agent.tools.orders import get_order_status


def _fresh_seeded_db(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_orders.db")
    mock_db.reset_and_seed()


def test_valid_order_returns_full_details(tmp_path, monkeypatch):
    _fresh_seeded_db(tmp_path, monkeypatch)
    order_id, _customer_id, item, _qty, _price, status, *_ = mock_db.ORDERS[0]

    result = get_order_status(order_id)

    assert result["found"] is True
    assert result["order_id"] == order_id
    assert result["item"] == item
    assert result["status"] == status


def test_invalid_order_id_format_is_rejected(tmp_path, monkeypatch):
    _fresh_seeded_db(tmp_path, monkeypatch)

    result = get_order_status("not-an-order-id")

    assert result["found"] is False
    assert result["error"] == "invalid_order_id"


def test_well_formed_but_nonexistent_order_is_not_found(tmp_path, monkeypatch):
    _fresh_seeded_db(tmp_path, monkeypatch)

    result = get_order_status("999-9999999-9999999")

    assert result["found"] is False
    assert result["error"] == "not_found"


def test_empty_order_id_is_rejected_without_hitting_the_db(tmp_path, monkeypatch):
    _fresh_seeded_db(tmp_path, monkeypatch)

    result = get_order_status("")

    assert result["found"] is False
    assert result["error"] == "invalid_order_id"
