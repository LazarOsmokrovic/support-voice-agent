"""Phase 11: agent/tools/notifications.py — redaction, signing, and
delivery of escalation handoff packets to an external automation platform
(n8n). Mirrors tests/test_tts.py's pytest-httpx pattern for the delivery
half (added in Task 3); this file starts with the pure-function half.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_module

from agent.tools.notifications import redact_packet, serialize_packet, sign_payload


def test_redact_packet_masks_email():
    packet = {"customer_intent": "Contact me at jane.doe@example.com about this."}
    result = redact_packet(packet)
    assert "jane.doe@example.com" not in result["customer_intent"]
    assert "[redacted-email]" in result["customer_intent"]


def test_redact_packet_masks_card_like_number():
    packet = {"conversation_summary": "Card number is 4111 1111 1111 1111 for the refund."}
    result = redact_packet(packet)
    assert "4111 1111 1111 1111" not in result["conversation_summary"]
    assert "[redacted-number]" in result["conversation_summary"]


def test_redact_packet_masks_phone_number():
    packet = {"verified_account_info": "Customer ID CUST-1001, callback at 555-123-4567."}
    result = redact_packet(packet)
    assert "555-123-4567" not in result["verified_account_info"]
    assert "[redacted-phone]" in result["verified_account_info"]


def test_redact_packet_leaves_ordinary_text_untouched():
    packet = {"actions_taken": "Looked up order status, found no issue."}
    result = redact_packet(packet)
    assert result["actions_taken"] == "Looked up order status, found no issue."


def test_redact_packet_leaves_non_redacted_fields_untouched():
    packet = {"escalation_id": 42, "reason": "explicit request for a human", "sentiment": "negative"}
    result = redact_packet(packet)
    assert result == packet


def test_sign_payload_is_deterministic_hmac_sha256():
    body = b'{"escalation_id": 1}'
    expected = hmac_module.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert sign_payload(body, "test-secret") == expected


def test_serialize_packet_produces_sorted_deterministic_json():
    packet_a = {"b": 2, "a": 1}
    packet_b = {"a": 1, "b": 2}
    assert serialize_packet(packet_a) == serialize_packet(packet_b)
