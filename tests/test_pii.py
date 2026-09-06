"""Phase 10a: guardrails/pii.py — canonical PII redaction at storage/egress
boundaries. Extracted from agent/tools/notifications.py, which built a
narrower version in Phase 11 for the outbound webhook.
"""

from __future__ import annotations

from data.mock_db import APPOINTMENTS, ORDERS
from guardrails.pii import HANDOFF_TEXT_FIELDS, redact_fields, redact_text


def test_redact_text_masks_email():
    assert "jane.doe@example.com" not in redact_text("Reach me at jane.doe@example.com please.")
    assert "[redacted-email]" in redact_text("Reach me at jane.doe@example.com please.")


def test_redact_text_masks_card_like_number_and_preserves_spacing():
    result = redact_text("Card number is 4111 1111 1111 1111 for the refund.")
    assert result == "Card number is [redacted-number] for the refund."


def test_redact_text_masks_phone_number():
    result = redact_text("Callback at 555-123-4567 tomorrow.")
    assert "555-123-4567" not in result
    assert "[redacted-phone]" in result


def test_redact_text_preserves_a_real_order_id():
    order_id = ORDERS[0][0]
    sentence = f"Order {order_id} never arrived."
    assert redact_text(sentence) == sentence


def test_redact_text_preserves_seeded_order_and_delivery_dates():
    # ORDERS rows are (order_id, customer_id, item, quantity, price, status,
    # order_date, estimated_delivery, tracking_number) — see data/mock_db.py.
    order = ORDERS[0]
    order_date, estimated_delivery = order[6], order[7]
    sentence = f"Delivered {order_date}, so the return window closes {estimated_delivery}."
    assert redact_text(sentence) == sentence


def test_redact_text_preserves_a_seeded_tracking_number():
    order_id, tracking_number = ORDERS[0][0], ORDERS[0][8]
    assert tracking_number, "seeded order must have a tracking number for this test to mean anything"
    sentence = f"Gave the customer tracking number {tracking_number} for order {order_id}."
    assert redact_text(sentence) == sentence


def test_redact_text_preserves_a_seeded_appointment_datetime():
    # APPOINTMENTS rows are (customer_id, scheduled_time, reason, status).
    scheduled_time = APPOINTMENTS[0][1]
    sentence = f"Callback booked for {scheduled_time}."
    assert redact_text(sentence) == sentence


def test_redact_text_leaves_ordinary_text_untouched():
    text = "Looked up the order status, found no issue."
    assert redact_text(text) == text


def test_redact_text_is_idempotent():
    once = redact_text("Mail jane@example.com or call 555-123-4567.")
    assert redact_text(once) == once


def test_redact_fields_only_touches_named_string_fields():
    data = {
        "customer_intent": "Email jane@example.com",
        "escalation_id": 42,
        "sentiment": "negative",
    }
    result = redact_fields(data, ("customer_intent",))
    assert "[redacted-email]" in result["customer_intent"]
    assert result["escalation_id"] == 42
    assert result["sentiment"] == "negative"


def test_redact_fields_ignores_absent_and_non_string_fields():
    data = {"customer_intent": 123}
    assert redact_fields(data, ("customer_intent", "not_present")) == data


def test_redact_fields_returns_a_copy():
    data = {"customer_intent": "Email jane@example.com"}
    result = redact_fields(data, ("customer_intent",))
    assert data["customer_intent"] == "Email jane@example.com"
    assert result is not data


def test_handoff_text_fields_matches_the_handoff_packet_shape():
    assert HANDOFF_TEXT_FIELDS == (
        "customer_intent",
        "conversation_summary",
        "verified_account_info",
        "actions_taken",
    )
