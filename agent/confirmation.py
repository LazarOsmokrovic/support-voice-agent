"""A generic "propose now, only commit on a later confirmed turn" gate.

CLAUDE.md rule 6: never let a tool execute an irreversible action (issuing
a refund, booking/cancelling) without an explicit confirmation turn from
the user first. Phase 5's book_appointment/cancel_appointment built this
mechanism first, deliberately not generalized ("one use case isn't enough
to know the right shape yet"). Phase 6's issue_refund needs the identical
protection, so on the second real use case this got extracted here instead
of copied a second time.

A tool calls `gate.check(key)` on every attempt, where `key` identifies the
specific action being proposed (e.g. a slot time + reason, or an
order/condition pair). The same key, proposed then confirmed in a strictly
LATER turn, is what allows a commit — never the same turn, and never a
different key (a new proposal simply replaces whatever was pending, which
is what makes mid-conversation corrections like "actually, next week
instead" work for free).

One gate instance per session per action *family* — e.g. one shared by
book_appointment and cancel_appointment (only one pending scheduling action
at a time), and a separate one for issue_refund, so an unrelated pending
refund and a pending booking never clobber each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PendingActionGate:
    turn: int = 0
    pending: dict[str, Any] | None = None

    def check(self, key: tuple[Any, ...]) -> bool:
        """True if `key` matches a pending proposal from a strictly earlier
        turn (this call should commit) — and clears the pending state.
        False otherwise, whether because nothing was pending, a different
        action was pending, or the matching proposal was from this same
        turn — and (re)registers `key` as the new pending proposal.
        """
        is_confirmation = (
            self.pending is not None
            and self.pending.get("key") == key
            and self.pending["proposed_turn"] < self.turn
        )
        if is_confirmation:
            self.pending = None
            return True
        self.pending = {"key": key, "proposed_turn": self.turn}
        return False

    def clear(self) -> None:
        self.pending = None
