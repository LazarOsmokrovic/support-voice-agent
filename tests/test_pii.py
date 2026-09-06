"""Phase 10a: guardrails/pii.py — canonical PII redaction at storage/egress
boundaries. Extracted from agent/tools/notifications.py, which built a
narrower version in Phase 11 for the outbound webhook.
"""

from __future__ import annotations

from data.mock_db import ORDERS
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
