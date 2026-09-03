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

import hashlib
import hmac
import json
import re
from typing import Any

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CARDLIKE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")

# Free-text fields on a handoff packet (see agent/tools/escalation.py's
# HandoffFields) that can contain customer-supplied text, and so need
# redaction before leaving the system. escalation_id/reason/sentiment are
# short and structured, never PII, and are left untouched.
_REDACTED_FIELDS = ("customer_intent", "conversation_summary", "verified_account_info", "actions_taken")


def _redact(text: str) -> str:
    """Mask emails, card-like digit runs, and phone-like digit runs.

    Order matters: card-like sequences (13-19 digits) are masked before the
    looser phone pattern, so a card number is never partially caught and
    left visible by the phone regex running first.
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
