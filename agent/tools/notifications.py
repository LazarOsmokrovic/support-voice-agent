"""Outbound escalation notifications to an external automation platform
(n8n) — Phase 11 (AI automation), independent of Phase 10.

When EscalationTracker (agent/tools/escalation.py) decides a human needs to
take over, create_handoff_packet persists the packet to SQLite
(log_escalation) and then calls notify_escalation() here to actually tell
someone. A missing or misconfigured webhook is a normal, working state —
ESCALATION_WEBHOOK_URL unset means a silent no-op, the same
optional-by-default convention as TTS_BACKEND/EMBEDDING_BACKEND
(transport/tts.py, agent/tools/policy_rag.py).

Redaction here is deliberately narrow: it only has to protect this one
outbound payload before it leaves the system, not implement Phase 10's real
PII pipeline (guardrails/pii.py, still an untouched stub). Not a substitute
for that work — just enough to not ship a customer's email, phone, or
card-like number to a third-party webhook by default.

Signing (X-Signature-256, HMAC-SHA256) is the outbound mirror of
transport/telephony.py's inbound X-Twilio-Signature verification.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
from typing import Any

import httpx

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Separator only ever appears *between* digits (never trailing), so a match
# can't swallow a space/dash that belongs to the surrounding text. Still
# matches runs of 13-19 digits: 1 leading digit + 12..18 more.
_CARDLIKE_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")

# Free-text fields on a handoff packet (see agent/tools/escalation.py's
# HandoffFields) that can contain customer-supplied text, and so need
# redaction before leaving the system. escalation_id/reason/sentiment are
# short and structured, never PII, and are left untouched.
_REDACTED_FIELDS = ("customer_intent", "conversation_summary", "verified_account_info", "actions_taken")


def _redact(text: str) -> str:
    """Mask emails, card-like digit runs, and phone-like digit runs.

    Card-like sequences (13-19 digits) are masked before the looser phone
    pattern. This is not required for full coverage — _PHONE_RE's {7,}
    quantifier would consume a card-length digit run just as completely if
    it ran first — but running card detection first means a card-length run
    gets labelled [redacted-number] rather than the less accurate
    [redacted-phone].
    """
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _CARDLIKE_RE.sub("[redacted-number]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    return text


def redact_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `packet` with its free-text fields redacted.
    Fields not present are left absent; non-string values pass through
    untouched (defensive — every real caller passes strings here).
    """
    redacted = dict(packet)
    for field in _REDACTED_FIELDS:
        if field in redacted and isinstance(redacted[field], str):
            redacted[field] = _redact(redacted[field])
    return redacted


def sign_payload(body: bytes, secret: str) -> str:
    """HMAC-SHA256 of `body` with `secret`, hex-encoded."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def serialize_packet(packet: dict[str, Any]) -> bytes:
    """Deterministic JSON serialization (sorted keys), so sign_payload() and
    the actual POST body always agree on the exact bytes signed.
    """
    return json.dumps(packet, sort_keys=True).encode()


logger = logging.getLogger("agent.tools.notifications")

WEBHOOK_TIMEOUT_SECONDS = 5.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (0.5, 1.5)  # sleep after attempt 1, then after attempt 2
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


async def notify_escalation(packet: dict[str, Any], *, client: httpx.AsyncClient | None = None) -> bool:
    """POST a redacted, signed escalation packet to ESCALATION_WEBHOOK_URL.

    Returns True if delivered (2xx on any attempt), False otherwise —
    including when no webhook URL is configured, which is a silent no-op by
    design. Never raises: a broken or misconfigured webhook must never
    affect the escalation itself completing (see agent/tools/escalation.py's
    create_handoff_packet, which also wraps this call defensively).
    """
    url = os.getenv("ESCALATION_WEBHOOK_URL")
    if not url:
        return False

    body = serialize_packet(redact_packet(packet))
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("ESCALATION_WEBHOOK_SECRET")
    if secret:
        headers["X-Signature-256"] = sign_payload(body, secret)

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS)
    try:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await client.post(url, content=body, headers=headers)
            except httpx.InvalidURL as exc:
                # A malformed ESCALATION_WEBHOOK_URL is a configuration error,
                # not a transient one — retrying won't make it valid.
                logger.warning("escalation webhook URL is invalid: %s", exc)
                return False
            except httpx.RequestError as exc:
                logger.warning("escalation webhook attempt %d/%d failed: %s", attempt + 1, MAX_ATTEMPTS, exc)
            else:
                if response.status_code < 300:
                    return True
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    logger.warning(
                        "escalation webhook rejected (status=%d), not retrying", response.status_code
                    )
                    return False
                logger.warning(
                    "escalation webhook attempt %d/%d got status=%d, retrying",
                    attempt + 1, MAX_ATTEMPTS, response.status_code,
                )
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS[attempt])
        logger.warning("escalation webhook: all %d attempts failed", MAX_ATTEMPTS)
        return False
    finally:
        if owns_client:
            await client.aclose()
