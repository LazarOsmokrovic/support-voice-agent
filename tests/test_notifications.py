"""Phase 11: agent/tools/notifications.py — redaction, signing, and
delivery of escalation handoff packets to an external automation platform
(n8n). Mirrors tests/test_tts.py's pytest-httpx pattern for the delivery
half (added in Task 3); this file starts with the pure-function half.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as hmac_module
import time

import httpx
import pytest

from agent.tools.notifications import MAX_ATTEMPTS, notify_escalation, redact_packet, serialize_packet, sign_payload
from data.mock_db import ORDERS


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


def test_redact_packet_masks_card_like_number_preserves_surrounding_spacing():
    packet = {"conversation_summary": "Card number is 4111 1111 1111 1111 for the refund."}
    result = redact_packet(packet)
    assert result["conversation_summary"] == "Card number is [redacted-number] for the refund."


def test_redact_packet_preserves_a_real_order_id():
    # Order IDs (3-7-7 digits, hyphen-separated — ORDER_ID_PATTERN in
    # agent/tools/orders.py) are exactly 17 digits with two separators, the
    # same shape _CARDLIKE_RE looks for. A real seeded order ID must survive
    # redaction intact — it's not PII, and it's the single most useful
    # identifier a human taking a handoff can be given.
    order_id = ORDERS[0][0]
    packet = {"verified_account_info": f"Order {order_id} never arrived."}
    result = redact_packet(packet)
    assert result["verified_account_info"] == f"Order {order_id} never arrived."


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


SAMPLE_PACKET = {
    "escalation_id": 42,
    "reason": "explicit request for a human",
    "customer_intent": "Wants a refund",
    "conversation_summary": "Asked about a refund for a late order.",
    "verified_account_info": "Customer ID CUST-1001",
    "actions_taken": "None yet",
    "sentiment": "negative",
}
WEBHOOK_URL = "https://n8n.example.com/webhook/escalation"


@pytest.mark.asyncio
async def test_notify_escalation_is_a_noop_without_a_webhook_url(monkeypatch, httpx_mock):
    monkeypatch.delenv("ESCALATION_WEBHOOK_URL", raising=False)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is False
    assert len(httpx_mock.get_requests()) == 0


@pytest.mark.asyncio
async def test_notify_escalation_succeeds_on_first_attempt(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is True
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_notify_escalation_retries_after_a_transient_failure(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setattr("agent.tools.notifications.RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=503)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is True
    assert len(httpx_mock.get_requests()) == 2


@pytest.mark.asyncio
async def test_notify_escalation_gives_up_after_exhausting_all_attempts(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setattr("agent.tools.notifications.RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    for _ in range(MAX_ATTEMPTS):
        httpx_mock.add_response(url=WEBHOOK_URL, status_code=503)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is False
    assert len(httpx_mock.get_requests()) == MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_notify_escalation_does_not_raise_on_a_malformed_webhook_url(monkeypatch, httpx_mock):
    # A non-printable character makes httpx raise InvalidURL, a sibling of
    # TransportError rather than a subclass of it — a misconfigured webhook
    # (bad URL) must be handled the same as any other misconfiguration:
    # report failure, never raise, and don't waste retries on it.
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", "http://example.com/\x01hook")

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is False
    assert len(httpx_mock.get_requests()) == 0


@pytest.mark.asyncio
async def test_notify_escalation_does_not_raise_on_a_decoding_error(monkeypatch, httpx_mock):
    # DecodingError is a sibling of TransportError under RequestError (raised
    # when a webhook responds with a malformed/unsupported Content-Encoding)
    # — it must be treated as a retryable transport-level failure, not
    # escape uncaught. Deliberately not wrapped in try/except: an uncaught
    # exception here fails the test.
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setattr("agent.tools.notifications.RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    httpx_mock.add_exception(httpx.DecodingError("bad content-encoding"), url=WEBHOOK_URL)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is True
    assert len(httpx_mock.get_requests()) == 2


@pytest.mark.asyncio
async def test_notify_escalation_does_not_close_a_caller_supplied_client(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)
    client = httpx.AsyncClient()

    try:
        delivered = await notify_escalation(SAMPLE_PACKET, client=client)
        assert delivered is True
        assert client.is_closed is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_notify_escalation_does_not_retry_a_4xx_response(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=401)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is False
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_notify_escalation_signs_the_body_when_a_secret_is_configured(monkeypatch, httpx_mock):
    from agent.tools.notifications import redact_packet, serialize_packet, sign_payload

    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setenv("ESCALATION_WEBHOOK_SECRET", "shh-its-a-secret")
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)

    await notify_escalation(SAMPLE_PACKET)

    request = httpx_mock.get_requests()[0]
    expected_body = serialize_packet(redact_packet(SAMPLE_PACKET))
    expected_signature = sign_payload(expected_body, "shh-its-a-secret")
    assert request.headers["x-signature-256"] == expected_signature


@pytest.mark.asyncio
async def test_notify_escalation_omits_signature_header_without_a_secret(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.delenv("ESCALATION_WEBHOOK_SECRET", raising=False)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)

    await notify_escalation(SAMPLE_PACKET)

    request = httpx_mock.get_requests()[0]
    assert "x-signature-256" not in request.headers


@pytest.mark.asyncio
async def test_notify_escalation_does_not_retry_a_scheme_less_url(monkeypatch, httpx_mock):
    # httpx.UnsupportedProtocol is what real httpx raises for a scheme-less
    # ESCALATION_WEBHOOK_URL like "n8n.example.com/webhook" (the likeliest
    # operator typo) — deep in httpcore's transport, the same layer
    # pytest-httpx replaces wholesale, so (mirroring the DecodingError test
    # above) the exception is injected directly on a syntactically valid URL
    # to exercise the fast-fail branch itself, rather than relying on
    # httpx's own scheme check, which httpx_mock bypasses. It's technically
    # a RequestError subclass, so without a dedicated except branch ahead of
    # the generic one it would be retried MAX_ATTEMPTS times as if
    # transient — burning the whole retry budget on a one-time typo.
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    httpx_mock.add_exception(
        httpx.UnsupportedProtocol("Request URL is missing an 'http://' or 'https://' protocol."),
        url=WEBHOOK_URL,
    )

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is False
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_notify_escalation_gives_up_once_the_total_time_budget_is_exceeded(monkeypatch, httpx_mock):
    # Regression test for the up-to-17s dead-air bug: MAX_ATTEMPTS *
    # WEBHOOK_TIMEOUT_SECONDS + backoff could previously run for 17s,
    # awaited synchronously before either caller speaks its reply on a live
    # call. NOTIFY_TOTAL_BUDGET_SECONDS is monkeypatched to a tiny value so
    # this test proves the cap without actually waiting anywhere near 3s
    # (let alone 17s), and asserts no exception escapes notify_escalation.
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setattr("agent.tools.notifications.NOTIFY_TOTAL_BUDGET_SECONDS", 0.05)

    async def _hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)  # far longer than the tiny budget above
        return httpx.Response(200)  # pragma: no cover — never reached, cancelled first

    httpx_mock.add_callback(_hang, url=WEBHOOK_URL)

    started = time.monotonic()
    delivered = await notify_escalation(SAMPLE_PACKET)
    elapsed = time.monotonic() - started

    assert delivered is False
    assert elapsed < 2.0  # well under the old 17s worst case (and the old default 3-attempt budget)
