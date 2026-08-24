"""issue_refund: the returns & refunds workflow.

PROJECT_PLAN.md's decision path for this phase: verify purchase -> check
return window against policy -> assess stated item condition -> calculate
refund amount -> auto-escalate if above a $ threshold -> require explicit
caller confirmation -> issue refund. This module implements exactly that,
in that order, ties together three earlier phases (order lookup, policy
RAG, escalation), and is the first tool that moves money.

One tool, not two — like book_appointment, the first call already doubles
as the eligibility check: it returns the computed amount and eligibility
before anything is written. CLAUDE.md rule 6 names "issuing a refund"
explicitly as needing a real confirmation turn, so committing requires a
second call in a LATER conversation turn, enforced via the same
PendingActionGate Phase 5 built for booking/cancelling (agent/confirmation.py)
rather than a second copy of that mechanism.

High-value refunds skip the confirmation dance entirely and escalate
instead, per the plan's literal ordering (escalate *before* the
confirmation step) — a human approves those, not the AI even with the
caller's own say-so.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from agent.confirmation import PendingActionGate
from agent.tools.orders import ORDER_ID_PATTERN
from agent.tools.policy_rag import search_policy
from data.mock_db import get_connection

# Both windows are counted from the order's estimated_delivery date, the
# closest thing the mock schema has to an actual delivery date. Kept as
# named constants mirroring data/policies/*.md's stated numbers — if those
# docs' numbers ever change, these need to change with them.
STANDARD_RETURN_WINDOW_DAYS = 30  # data/policies/returns_policy.md
DAMAGED_CLAIM_WINDOW_DAYS = 14  # data/policies/damaged_or_defective_items.md

# Above this, issue_refund escalates instead of asking the caller to
# confirm. Splits the seeded delivered orders meaningfully: Echo Dot
# ($34.99) and the Nike shoes ($129.95) stay under it; the Sony headphones
# ($349.99) go over — both branches are exercised by real seed data.
HIGH_VALUE_REFUND_THRESHOLD = 150.0

ItemCondition = Literal["unopened_or_unwanted", "damaged_or_defective", "opened_software_or_digital"]

TOOL_SCHEMA: dict[str, Any] = {
    "name": "issue_refund",
    "description": (
        "Process a refund for a delivered order. The FIRST call checks "
        "eligibility and proposes the refund — it does not issue anything "
        "yet. Only call it a second time, with the exact same order_id and "
        "condition, after the customer has clearly confirmed in a later "
        "message. High-value refunds skip confirmation and are escalated "
        "to a specialist instead — if the response has escalate: true, "
        "tell the customer that rather than asking them to confirm."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order to refund, e.g. '112-3487561-2938471'."},
            "condition": {
                "type": "string",
                "enum": ["unopened_or_unwanted", "damaged_or_defective", "opened_software_or_digital"],
                "description": (
                    "unopened_or_unwanted: a plain return, item unused or just not wanted. "
                    "damaged_or_defective: arrived broken, defective, or wrong. "
                    "opened_software_or_digital: opened software, digital downloads, or similar — "
                    "never eligible, regardless of window."
                ),
            },
            "reason": {"type": "string", "description": "The customer's own stated reason, for the record."},
        },
        "required": ["order_id", "condition", "reason"],
    },
}


def _get_order(order_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT order_id, customer_id, item, quantity, price, status, estimated_delivery "
            "FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    return dict(row) if row else None


def _window_days(condition: ItemCondition) -> int | None:
    """Return the applicable window in days, or None if never eligible."""
    if condition == "unopened_or_unwanted":
        return STANDARD_RETURN_WINDOW_DAYS
    if condition == "damaged_or_defective":
        return DAMAGED_CLAIM_WINDOW_DAYS
    return None  # opened_software_or_digital: never eligible


def issue_refund(
    order_id: str,
    condition: ItemCondition,
    reason: str,
    state: PendingActionGate,
    customer_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run the full decision path and either propose, escalate, or commit a refund."""
    order_id = (order_id or "").strip()
    now = now or datetime.now()  # noqa: DTZ005 — naive on purpose, matches estimated_delivery's convention

    # 1. Verify purchase.
    if not ORDER_ID_PATTERN.match(order_id):
        return {
            "issued": False,
            "error": "invalid_order_id",
            "message": f"'{order_id}' isn't a valid order ID format.",
        }

    order = _get_order(order_id)
    if order is None:
        return {"issued": False, "error": "not_found", "message": f"No order found with ID {order_id}."}
    if order["customer_id"] != customer_id:
        return {
            "issued": False,
            "error": "not_your_order",
            "message": "This order doesn't belong to the customer on this session.",
        }
    if order["status"] == "Refunded":
        return {"issued": False, "error": "already_refunded", "message": f"Order {order_id} was already refunded."}
    if order["status"] != "Delivered":
        return {
            "issued": False,
            "error": "not_delivered",
            "message": (
                f"Order {order_id} hasn't been delivered yet (status: {order['status']}) — "
                "returns apply to delivered items. To cancel an order before it ships, that's a "
                "separate request."
            ),
        }

    # 2. Check the return window against policy.
    policy_lookup = search_policy(
        "return window for a refund" if condition != "damaged_or_defective" else "damaged or defective item refund"
    )
    policy_reference = policy_lookup["results"][0]["text"] if policy_lookup.get("found") else None

    window_days = _window_days(condition)
    if window_days is None:
        return {
            "issued": False,
            "error": "not_eligible",
            "message": "Opened software, digital downloads, and similar items aren't eligible for return.",
            "policy_reference": policy_reference,
        }

    delivered_on = datetime.fromisoformat(order["estimated_delivery"])
    days_since_delivery = (now - delivered_on).days
    if days_since_delivery > window_days:
        return {
            "issued": False,
            "error": "outside_window",
            "message": (
                f"This order was delivered {days_since_delivery} days ago, past the "
                f"{window_days}-day window for this type of return."
            ),
            "policy_reference": policy_reference,
        }

    # 3. Calculate refund amount. Known simplification: no restocking fee
    # (data/policies/restocking_fees.md) — the mock orders table has no
    # product-category/size data to key a fee off. No separate shipping
    # line item either, so the full order price is always the amount.
    amount = round(order["price"] * order["quantity"], 2)

    # 4. Auto-escalate above the threshold — skips confirmation entirely,
    # matching the plan's literal ordering (escalate INSTEAD OF asking the
    # caller to confirm; a human approves this, not the AI).
    if amount > HIGH_VALUE_REFUND_THRESHOLD:
        state.clear()
        return {
            "issued": False,
            "escalate": True,
            "escalation_reason": f"high-value refund (${amount:.2f}) requires specialist approval",
            "amount": amount,
            "message": f"A ${amount:.2f} refund needs approval from a specialist before it can be issued.",
            "policy_reference": policy_reference,
        }

    # 5. Require explicit confirmation — propose, then commit only on a
    # later call. See agent/confirmation.py.
    if not state.check(key=(order_id, condition)):
        return {
            "issued": False,
            "status": "pending_confirmation",
            "amount": amount,
            "message": f"This order is eligible for a ${amount:.2f} refund. Should I go ahead and issue it?",
            "policy_reference": policy_reference,
        }

    # 6. Issue.
    issued_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO refunds (order_id, customer_id, amount, item_condition, reason, issued_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (order_id, customer_id, amount, condition, reason, issued_at),
        )
        refund_id = cursor.lastrowid
        conn.execute("UPDATE orders SET status = 'Refunded' WHERE order_id = ?", (order_id,))

    return {"issued": True, "refund_id": refund_id, "order_id": order_id, "amount": amount}
