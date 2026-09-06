"""Canonical PII redaction for text leaving the system — Phase 10a.

Extracted from agent/tools/notifications.py, which built a narrow version of
this in Phase 11 for the outbound escalation webhook. Database writes now
need the same thing, and that second real use case is what earns the
extraction — the same convention that produced agent/confirmation.py
(Phase 6), agent/session.py (Phase 7), and transport/pipecat_processors.py
(Phase 9). One definition of what counts as PII here, not two copies drifting.

Scope, deliberately: this redacts free text at STORAGE and EGRESS boundaries
(database writes, structured logs, outbound webhooks) — NOT before the model
reads a turn, despite PROJECT_PLAN.md filing it under "Pre-LLM". This is a
support agent: a caller may legitimately give an email or phone to update
their account, and redacting it before the model sees it would break the
product. The live conversation already reaches Claude turn by turn anyway, so
redacting only at the summarize step would be security theatre. The
authoritative copy of a customer's contact details already lives in the
`customers` table keyed by customer_id, so free-text transcripts never need to
carry a second, uncontrolled copy of it.

Redaction is idempotent: the replacement tokens contain no digits and no '@',
so running it twice is a no-op. The escalation packet relies on this — it is
redacted once when assembled (agent/tools/escalation.py) and again defensively
before the webhook (agent/tools/notifications.py).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Callable

from agent.tools.orders import ORDER_ID_PATTERN

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Separator only ever appears *between* digits (never trailing), so a match
# can't swallow a space/dash that belongs to the surrounding text. Still
# matches runs of 13-19 digits: 1 leading digit + 12..18 more.
_CARDLIKE_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
# Boundary-guarded so a match can't start or end mid-token: without
# (?<![\w-])/(?![\w-]), this matched into an alphanumeric tracking number
# like "TBA123456789US" (the digit run between the letters looks exactly
# like a phone number to this pattern). _CARDLIKE_RE doesn't need the same
# guard — \b already refuses to sit between two word characters (a digit and
# a letter are both \w), so it never fires mid-token to begin with.
_PHONE_RE = re.compile(r"(?<![\w-])\+?\d[\d\-\s]{7,}\d(?![\w-])")

# Free-text fields on a handoff packet (agent/tools/escalation.py's
# HandoffFields) that can contain customer-supplied text. escalation_id,
# reason and sentiment are short and structured, never PII, and stay untouched.
HANDOFF_TEXT_FIELDS: tuple[str, ...] = (
    "customer_intent",
    "conversation_summary",
    "verified_account_info",
    "actions_taken",
)

# An ISO-8601 date (order_date, estimated_delivery) or datetime
# (appointments.scheduled_time) — digit runs this project writes constantly
# in free text, and which are long enough (a date-with-time is 14 digits) to
# otherwise be caught by _CARDLIKE_RE, or (a bare date, hyphen-separated) by
# _PHONE_RE. Anchored so it only exempts a match that IS one of these shapes
# start-to-end, not a substring inside something longer.
_ISO_DATE_OR_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?$")

# Digit-run shapes that are never PII in this project's own data, even
# though they're long enough to match _CARDLIKE_RE or _PHONE_RE: this
# project's own order-ID format, and ISO dates/datetimes. Both are checked
# at both regex passes below — see _mask_unless_known_shape.
_NON_PII_SHAPES: tuple[re.Pattern[str], ...] = (ORDER_ID_PATTERN, _ISO_DATE_OR_DATETIME_RE)


def _mask_unless_known_shape(replacement: str) -> Callable[[re.Match[str]], str]:
    """Build a re.sub replacement function that masks a matched digit run
    with `replacement`, except when the match is exactly the shape of one of
    _NON_PII_SHAPES (this project's own order IDs, or an ISO date/datetime).
    Neither is PII — an order ID is the single most useful identifier a
    human taking a handoff can be given, and a date is just a date.

    Used for both _CARDLIKE_RE and _PHONE_RE: an order ID (17 digits, 2
    separators) is exactly card-length, so it's also long enough to match the
    looser phone pattern; a date-with-time is 14 digits, also card-length.
    If only the card-like pass exempted these shapes, the phone-like pass
    running right after would still catch and mask the very same digits —
    this needs to hold at both stages, not just the first.
    """

    def _mask(match: re.Match[str]) -> str:
        text = match.group()
        if any(pattern.match(text) for pattern in _NON_PII_SHAPES):
            return text
        return replacement

    return _mask


_mask_cardlike = _mask_unless_known_shape("[redacted-number]")
_mask_phonelike = _mask_unless_known_shape("[redacted-phone]")


def redact_text(text: str) -> str:
    """Mask emails, card-like digit runs, and phone-like digit runs.

    Card-like sequences (13-19 digits) are masked before the looser phone
    pattern. This is not required for full coverage — _PHONE_RE's {7,}
    quantifier would consume a card-length digit run just as completely if it
    ran first — but running card detection first means a card-length run gets
    labelled [redacted-number] rather than the less accurate [redacted-phone].
    """
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _CARDLIKE_RE.sub(_mask_cardlike, text)
    text = _PHONE_RE.sub(_mask_phonelike, text)
    return text


def redact_fields(data: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Return a copy of `data` with each named field redacted.

    Fields not present are left absent; non-string values pass through
    untouched (defensive — every real caller passes strings here).
    """
    redacted = dict(data)
    for field in fields:
        if field in redacted and isinstance(redacted[field], str):
            redacted[field] = redact_text(redacted[field])
    return redacted
