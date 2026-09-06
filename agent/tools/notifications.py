"""Outbound escalation notifications to an external automation platform
(n8n) — Phase 11 (AI automation), independent of Phase 10.

When EscalationTracker (agent/tools/escalation.py) decides a human needs to
take over, create_handoff_packet persists the packet to SQLite
(log_escalation) and then calls notify_escalation() here to actually tell
someone. A missing or misconfigured webhook is a normal, working state —
ESCALATION_WEBHOOK_URL unset means a silent no-op, the same
optional-by-default convention as TTS_BACKEND/EMBEDDING_BACKEND
(transport/tts.py, agent/tools/policy_rag.py).

Redaction of the packet before it leaves the system uses guardrails.pii's
canonical redactor (Phase 10a) — this module started with a private copy of
it in Phase 11, since promoted there once database writes needed the same
thing too. redact_packet() below stays a thin, named wrapper: which fields a
handoff packet redacts is this module's concern, even though how to redact
text is not.

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
from typing import Any

import httpx

from guardrails.pii import HANDOFF_TEXT_FIELDS, redact_fields


def redact_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `packet` with its free-text fields redacted.

    Thin wrapper over guardrails.pii — kept as a named function because
    notify_escalation and tests/test_notifications.py both call it, and
    because the choice of WHICH fields a handoff packet redacts is this
    module's concern even though HOW to redact is not.
    """
    return redact_fields(packet, HANDOFF_TEXT_FIELDS)


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

# Worst case without this cap is MAX_ATTEMPTS * WEBHOOK_TIMEOUT_SECONDS +
# sum(RETRY_BACKOFF_SECONDS) = up to 17s, awaited synchronously by both
# callers (agent/session.py's run_turn, transport/pipecat_processors.py's
# DTMF handler) *before* they speak their reply — dead air on a live call at
# the exact moment an already-frustrated caller is being handed off. This is
# a hard ceiling on the whole retry loop, independent of how MAX_ATTEMPTS/
# WEBHOOK_TIMEOUT_SECONDS get tuned later.
NOTIFY_TOTAL_BUDGET_SECONDS = 3.0


async def notify_escalation(packet: dict[str, Any], *, client: httpx.AsyncClient | None = None) -> bool:
    """POST a redacted, signed escalation packet to ESCALATION_WEBHOOK_URL.

    Returns True if delivered (2xx on any attempt), False otherwise —
    including when no webhook URL is configured (silent no-op by design) or
    when the retry loop overruns NOTIFY_TOTAL_BUDGET_SECONDS (treated as a
    failed delivery). Never raises: a broken or misconfigured webhook must
    never affect the escalation itself completing (see
    agent/tools/escalation.py's create_handoff_packet, which also wraps this
    call defensively). A genuine outer cancellation (e.g. Pipecat barge-in)
    is not caught here and propagates as normal.
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

    async def _attempt_delivery() -> bool:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await client.post(url, content=body, headers=headers)
            except (httpx.InvalidURL, httpx.UnsupportedProtocol) as exc:
                # A malformed/scheme-less ESCALATION_WEBHOOK_URL is a
                # configuration error, not a transient one — retrying won't
                # make it valid. (UnsupportedProtocol is technically a
                # RequestError subclass, so it must be caught here, ahead of
                # the generic RequestError branch below, or it'd burn all
                # MAX_ATTEMPTS retrying a typo like "n8n.example.com/hook".)
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

    try:
        return await asyncio.wait_for(_attempt_delivery(), timeout=NOTIFY_TOTAL_BUDGET_SECONDS)
    except TimeoutError:
        # asyncio.TimeoutError (an alias of the builtin TimeoutError on
        # Python 3.11+): the retry loop itself is cancelled by wait_for
        # here, NOT a genuine outer cancellation — asyncio.CancelledError
        # from a real barge-in is a different exception and is not caught by
        # this clause, so it still propagates normally, straight through
        # this try/finally, same as before this fix.
        logger.warning(
            "escalation webhook: exceeded total budget of %.1fs, giving up", NOTIFY_TOTAL_BUDGET_SECONDS
        )
        return False
    finally:
        if owns_client:
            await client.aclose()
