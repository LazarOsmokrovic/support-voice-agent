"""Post-LLM grounding detection — Phase 10a.

This is a DETECTOR, not a prover. It catches the common shape of a
hallucinated policy claim — an invented return window, an invented fee — by
checking that policy-shaped numbers in the reply actually appear somewhere in
what the tools returned this turn. It does NOT verify entailment, and a reply
it passes is not thereby proven correct. Overstating that would be worse than
the gap itself.

Deliberately conservative: only numbers attached to a policy-ish unit
(days/weeks/months, a percentage, or an amount of money) are checked, so an
incidental "in 2 ways" cannot trip it. Under Phase 10a's hedge-then-escalate
ladder a false positive costs a customer interaction, so the detector starts
narrow; sub-phase 10c's eval suite is the instrument for widening it from
measurement rather than guesswork — the same method that set Phase 3's RAG
threshold from eight hand-labeled questions instead of intuition.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Tools whose output is the ground truth a reply is checked against. A turn
# that called none of these isn't making a lookup-backed claim, so there is
# nothing to check.
GROUNDING_TOOLS = ("search_policy", "get_order_status")

# A "claim" is a number wearing a policy-ish unit: a duration, a percentage,
# or an amount of money. Bare numbers are deliberately ignored.
_CLAIM_RE = re.compile(
    r"\$\s?(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*%"
    r"|\b(\d+(?:\.\d+)?)\s*(?:business\s+)?(?:day|days|week|weeks|month|months|hour|hours)\b",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

HEDGE_PHRASES: tuple[str, ...] = (
    "Let me double-check that — I want to make sure I give you accurate information.",
    "Actually, let me verify that before I tell you something wrong.",
    "I want to confirm that properly rather than guess — give me a moment.",
)


def hedge_for(index: int) -> str:
    """A hedge line to speak instead of an ungrounded reply.

    `index` is how many consecutive ungrounded replies preceded this one, so a
    customer who hits this twice in a row doesn't hear the identical robotic
    sentence. Deterministic (not random) so tests stay reproducible, and
    caller-supplied rather than stateful so this module stays pure.
    """
    return HEDGE_PHRASES[index % len(HEDGE_PHRASES)]


def _claimed_numbers(reply: str) -> list[str]:
    """Every policy-shaped number asserted in `reply`, normalized to a string."""
    claims: list[str] = []
    for match in _CLAIM_RE.finditer(reply):
        value = next(group for group in match.groups() if group is not None)
        claims.append(value.rstrip("0").rstrip(".") if "." in value else value)
    return claims


def _supported_numbers(tool_calls: list[dict[str, Any]]) -> set[str]:
    """Every number appearing anywhere in this turn's tool output.

    Serializes each output rather than walking its shape, because tool
    outputs are heterogeneous dicts (policy chunks, order rows, refund
    amounts) and any number in any of them is legitimate grounding.
    """
    supported: set[str] = set()
    for call in tool_calls:
        output = call.get("output")
        if output is None:
            continue
        try:
            blob = json.dumps(output, default=str)
        except (TypeError, ValueError):
            blob = str(output)
        for number in _NUMBER_RE.findall(blob):
            supported.add(number.rstrip("0").rstrip(".") if "." in number else number)
    return supported


def check_reply_grounding(reply: str, tool_calls: list[dict[str, Any]]) -> list[str]:
    """Return a finding per policy-shaped claim in `reply` unsupported by this
    turn's tool output. Empty list means nothing suspicious was detected.

    Never raises: a malformed tool output yields no findings rather than a
    guess (fail open on detection — a guardrail must never break a call). The
    inner helpers already handle the expected shape of unserializable tool
    output; this outer guard is for the unexpected shape — a call entry that
    isn't a dict, or a value whose __str__ itself throws — since a guardrail
    that can crash the turn it is meant to protect is worse than no guardrail.
    """
    try:
        if not any(call.get("name") in GROUNDING_TOOLS for call in tool_calls):
            return []

        supported = _supported_numbers(tool_calls)
        unsupported = [claim for claim in _claimed_numbers(reply) if claim not in supported]
        if not unsupported:
            return []
        return [
            "reply asserts "
            + ", ".join(sorted(set(unsupported)))
            + " which appears nowhere in this turn's tool output (possible hallucinated policy detail)"
        ]
    except Exception:
        return []
