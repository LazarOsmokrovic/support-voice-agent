"""get_order_status tool: a deterministic lookup against the mock DB.

Per PROJECT_PLAN.md Phase 1: this is a predictable, single-step task, so it's
a plain function the model calls — not something handed to open-ended model
judgment. It never raises; every outcome (bad ID format, not found, found)
comes back as a structured dict the model can read and relay to the customer.
"""

from __future__ import annotations

import re
from typing import Any

from data.mock_db import get_connection

# Amazon's real order-ID shape: 3 digits - 7 digits - 7 digits.
_ORDER_ID_PATTERN = re.compile(r"^\d{3}-\d{7}-\d{7}$")

TOOL_SCHEMA: dict[str, Any] = {
    "name": "get_order_status",
    "description": (
        "Look up the current status of an order by its order ID. Order IDs "
        "look like '112-3487561-2938471' (3 digits, 7 digits, 7 digits, "
        "separated by hyphens). Returns the item, quantity, price, status, "
        "order date, estimated delivery, and tracking number if the order "
        "is found."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID to look up, e.g. '112-3487561-2938471'.",
            }
        },
        "required": ["order_id"],
    },
}


def get_order_status(order_id: str) -> dict[str, Any]:
    """Look up an order by ID. Always returns a dict; never raises."""
    order_id = (order_id or "").strip()

    if not _ORDER_ID_PATTERN.match(order_id):
        return {
            "found": False,
            "error": "invalid_order_id",
            "message": (
                f"'{order_id}' isn't a valid order ID format. Order IDs look "
                "like 112-3487561-2938471 (3-7-7 digits, hyphen-separated)."
            ),
        }

    with get_connection() as conn:
        row = conn.execute(
            "SELECT order_id, item, quantity, price, status, order_date, "
            "estimated_delivery, tracking_number FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()

    if row is None:
        return {
            "found": False,
            "error": "not_found",
            "message": f"No order found with ID {order_id}.",
        }

    return {
        "found": True,
        "order_id": row["order_id"],
        "item": row["item"],
        "quantity": row["quantity"],
        "price": row["price"],
        "status": row["status"],
        "order_date": row["order_date"],
        "estimated_delivery": row["estimated_delivery"],
        "tracking_number": row["tracking_number"],
    }
