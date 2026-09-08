"""Scripted eval scenarios across all six capabilities — Phase 10c.

Declarative data, written in Python rather than YAML for one reason that
matters: a YAML file cannot `import data.mock_db`, so it invites typing
`112-3487561-2938471` by hand. Hand-typed seed literals produced a Critical
defect in Phase 11 and another in 10a. A Python module lets
tests/test_eval_harness.py cross-check every identifier against the live
seed at collection time, so a scenario referencing an order that no longer
exists fails loudly instead of quietly testing nothing.

Two halves, deliberately kept apart:

  expect            authored BEFORE recording. This is the TEST — what the
                    agent is supposed to do.
  grounding_truth   authored AFTER reading the recording. This is the
                    MEASUREMENT — a human's per-turn judgment of whether the
                    reply was actually grounded, which is what gives Phase
                    10a's detector a false-positive denominator.

Recordings live in eval/recordings/<name>.json, never here: expectations are
hand-authored and reviewed in diffs, recordings are machine-generated and
large, and colocating them would make every re-record produce an
unreviewable diff across the assertions too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# The six capabilities PROJECT_PLAN.md promises. Drives the coverage report;
# tests/test_eval_harness.py asserts all six are represented.
CAPABILITIES: tuple[str, ...] = (
    "order_status",
    "refunds",
    "policy_qa",
    "triage",
    "scheduling",
    "summary",
)

# One label per turn, hand-assigned after reading the recording.
#   grounded        every policy-shaped number in the reply traces to
#                   something the agent legitimately had (this turn's tool
#                   output, an earlier turn's, or the user's own words).
#   ungrounded      the reply asserts a genuinely fabricated policy number.
#   not_applicable  no policy-shaped number, or no search_policy this turn,
#                   so guardrails/validators.py cannot fire by construction
#                   (GROUNDING_TRIGGER_TOOLS = ("search_policy",)).
GroundingLabel = Literal["grounded", "ungrounded", "not_applicable"]


@dataclass(frozen=True)
class ToolExpectation:
    """One tool the scenario says must be called.

    `args_subset` is a SUBSET match, never equality: the model may
    legitimately pass an extra optional argument, and demanding exact dict
    equality would make the suite brittle to prompt edits while measuring
    nothing. `turn` pins which turn it must happen on, or None for anywhere.
    """

    name: str
    args_subset: dict[str, Any] | None = None
    turn: int | None = None


@dataclass(frozen=True)
class DbAssertion:
    """Literal SQL against the scenario's temp database.

    Literal SQL, not a DSL: there are four tables in a local mock, and an
    assertion DSL would be pure overhead over the thing it wraps.

    `rows` is the expected row count. `columns` optionally pins column values
    on the FIRST returned row; None checks only the count.
    """

    sql: str
    params: tuple[Any, ...] = ()
    rows: int = 1
    columns: dict[str, Any] | None = None


@dataclass(frozen=True)
class Expectations:
    """What the agent is supposed to do. Authored before recording.

    `escalation_turn: int | None` is one field carrying two assertions.
    Declaring "fires on turn 2" simultaneously asserts it did NOT fire on
    turn 1 — exactly Phase 4's "neither too eager nor too late" checkpoint,
    which is currently split across two live tests and expressed only in
    prose docstrings.
    """

    tools_called: tuple[ToolExpectation, ...] = ()
    tools_not_called: tuple[str, ...] = ()
    escalation_turn: int | None = None  # None = never fires
    escalation_reason: str | None = None  # exact literal from agent/tools/escalation.py
    end_reason: str | None = None  # "model_ended" | "escalated" | "error" | None
    db_assertions: tuple[DbAssertion, ...] = ()
    # Reads the REAL seeded values at scoring time (mock_db.CUSTOMERS email
    # and phone, mock_db.ORDERS order_id and TBA...US tracking number) and
    # asserts the first two are absent from tickets/escalations/turn-log
    # lines while the last two survived intact. This makes Phase 10a's
    # date-and-tracking-number destruction bug a permanent, always-on check.
    no_pii_in_records: bool = True


@dataclass(frozen=True)
class Scenario:
    """One scripted conversation, its contract, and its ground-truth labels."""

    name: str  # stable slug; also the recording filename stem
    capability: str  # one of CAPABILITIES
    customer_id: str  # must exist in mock_db.CUSTOMERS
    turns: tuple[str, ...]  # the user turns, verbatim
    expect: Expectations
    grounding_truth: tuple[str, ...] = ()  # one GroundingLabel per turn
    close_session: bool = False  # drive close_session() at the end
    # Days added to the recording's timestamp to produce this scenario's
    # frozen clock. Added to the spec deliberately (see the plan): spec §3
    # freezes to recorded_at, but spec §7 wants refund_outside_window to
    # exercise the outside-window path ON PURPOSE — and after the 2026-09-08
    # seed refresh every Delivered order is INSIDE the 30-day window at
    # recorded_at, so no single frozen instant can reach it. The offset is
    # applied identically at record and replay time, so determinism and
    # "record once, replay forever" both hold, and it is visible in a diff.
    clock_offset_days: int = 0
    notes: str = ""  # why this scenario exists


# Filled in Task 11 of the Phase 10c plan with the 20-scenario roster.
SCENARIOS: tuple[Scenario, ...] = ()


def scenario_by_name(name: str) -> Scenario | None:
    """Look up one scenario by its slug, or None if there is no such name."""
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    return None
