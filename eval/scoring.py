"""Deterministic scoring — Phase 10c.

No LLM-as-judge, anywhere, by explicit project-owner decision. Everything
here scores an observable fact: which tools were called with which
arguments, whether escalation fired and why, grounding_flagged, hedge_spoken,
end_reason, the database's end state, and PII in stored records.

Pure functions over a HarnessResult, so they are testable without running an
agent and so the runner is not half assertions.

Two things are deliberately NOT scored. llm_latency_seconds, because replay
latency is meaningless and scoring it would be the eval's own hallucination.
And reply wording, because pinning phrasing measures the model's mood, not
the agent's behaviour.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

from data import mock_db
from eval.harness import HarnessResult
from eval.recording import Recording
from eval.scenarios import Scenario

# The tracking-number shape this project's seed uses. A redactor that eats it
# is destroying the store's own identifiers, which is exactly what happened
# in Phase 10a and was only caught on a whole-branch review.
_TRACKING_PREFIX = "TBA"


@dataclass(frozen=True)
class Failure:
    kind: str  # "tools" | "escalation" | "end_reason" | "db" | "pii" | "drift"
    detail: str


def _is_subset(subset: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(key in actual and actual[key] == value for key, value in subset.items())


def score_tools(scenario: Scenario, result: HarnessResult) -> list[Failure]:
    """Subset-match, never equality: the model may legitimately pass an extra
    optional argument, and demanding exact dict equality would make the suite
    brittle to prompt edits while measuring nothing real.
    """
    failures: list[Failure] = []
    for expectation in scenario.expect.tools_called:
        candidates = [
            call
            for row in result.observed
            if expectation.turn is None or row.turn == expectation.turn
            for call in row.tool_calls
            if call.get("name") == expectation.name
        ]
        where = "anywhere" if expectation.turn is None else f"on turn {expectation.turn}"
        if not candidates:
            failures.append(Failure("tools", f"{expectation.name} expected {where}, never called"))
            continue
        if expectation.args_subset is None:
            continue
        if not any(_is_subset(expectation.args_subset, call.get("input") or {}) for call in candidates):
            got = [call.get("input") for call in candidates]
            failures.append(
                Failure(
                    "tools",
                    f"{expectation.name}({expectation.args_subset}) expected {where}, got {got}",
                )
            )

    for name in scenario.expect.tools_not_called:
        hits = [row.turn for row in result.observed for call in row.tool_calls if call.get("name") == name]
        if hits:
            failures.append(Failure("tools", f"{name} was not supposed to be called, ran on turn(s) {hits}"))
    return failures


def score_escalation(scenario: Scenario, result: HarnessResult) -> list[Failure]:
    """escalation_turn carries two assertions at once — see Expectations."""
    fired = [row for row in result.observed if row.escalation_reason]
    expected_turn = scenario.expect.escalation_turn

    if expected_turn is None:
        if fired:
            row = fired[0]
            return [
                Failure(
                    "escalation",
                    f"expected no escalation, but turn {row.turn} escalated: {row.escalation_reason!r}",
                )
            ]
        return []

    if not fired:
        return [Failure("escalation", f"expected an escalation on turn {expected_turn}, none fired")]

    row = fired[0]
    failures: list[Failure] = []
    if row.turn != expected_turn:
        failures.append(
            Failure("escalation", f"expected an escalation on turn {expected_turn}, it fired on turn {row.turn}")
        )
    if scenario.expect.escalation_reason and row.escalation_reason != scenario.expect.escalation_reason:
        failures.append(
            Failure(
                "escalation",
                f"expected reason {scenario.expect.escalation_reason!r}, got {row.escalation_reason!r}",
            )
        )
    return failures


def score_db(scenario: Scenario, result: HarnessResult) -> list[Failure]:
    """Literal SQL against the scenario's own temp database."""
    failures: list[Failure] = []
    if not scenario.expect.db_assertions:
        return failures
    conn = sqlite3.connect(result.db_path)
    conn.row_factory = sqlite3.Row
    try:
        for assertion in scenario.expect.db_assertions:
            rows = conn.execute(assertion.sql, assertion.params).fetchall()
            if len(rows) != assertion.rows:
                failures.append(
                    Failure(
                        "db",
                        f"expected {assertion.rows} row(s) for `{assertion.sql}` {assertion.params}, "
                        f"found {len(rows)}",
                    )
                )
                continue
            if assertion.columns and rows:
                actual = dict(rows[0])
                for column, value in assertion.columns.items():
                    if actual.get(column) != value:
                        failures.append(
                            Failure(
                                "db",
                                f"`{assertion.sql}` {assertion.params}: expected {column}={value!r}, "
                                f"got {actual.get(column)!r}",
                            )
                        )
    finally:
        conn.close()
    return failures


def _stored_texts(result: HarnessResult) -> list[str]:
    """Every string this scenario durably stored: the turn-log file's real
    bytes plus the tickets and escalations rows it wrote.

    Read from the FILE, not the in-process TurnRecords, on purpose. The file
    went through redact_structure and json.dumps' redacting default — the
    exact code that carried Phase 10a's date-destruction bug, and the only
    way to score PII in stored records at all.

    connect() itself is inside the guarded block, not just the queries after
    it: a scenario with no db_path (HarnessResult's default is Path(), a
    directory) must degrade to "no extra rows" rather than raise, since
    log_lines alone is a legitimate thing to score.
    """
    texts = [json.dumps(line, default=str) for line in result.log_lines]
    try:
        conn = sqlite3.connect(result.db_path)
    except sqlite3.Error:
        return texts
    conn.row_factory = sqlite3.Row
    try:
        for table in ("tickets", "escalations"):
            for row in conn.execute(f"SELECT * FROM {table}").fetchall():  # noqa: S608 — fixed literal names
                texts.append(json.dumps(dict(row), default=str))
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return texts


def score_pii(scenario: Scenario, result: HarnessResult) -> list[Failure]:
    """Reads REAL seeded values at runtime; never hard-codes one.

    Two directions, both of which have been real defects in this repo:
    contact details must be GONE, and the store's own identifiers (order IDs,
    TBA...US tracking numbers) must have SURVIVED. A redactor that passes the
    first half by destroying everything fails the second.
    """
    if not scenario.expect.no_pii_in_records:
        return []

    contacts = [
        value
        for customer_id, _name, email, phone in mock_db.CUSTOMERS
        if customer_id == scenario.customer_id
        for value in (email, phone)
        if value
    ]
    texts = _stored_texts(result)
    blob = "\n".join(texts)

    failures = [
        Failure("pii", f"{contact!r} appears verbatim in a stored record") for contact in contacts if contact in blob
    ]

    # Every seeded order_id and tracking number must appear INTACT wherever a
    # recognizable chunk of it shows up alongside a redaction marker — that
    # combination is the signature of a redactor eating part of the project's
    # own identifier while stripping contact details, which is the Phase 10a
    # bug. Requiring a real 6-char chunk match (not just marker co-presence)
    # keeps this from flagging unrelated identifiers that never appear in the
    # record at all.
    for order_id, _cust, _item, _qty, _price, _status, _od, _ed, tracking in mock_db.ORDERS:
        for identifier in (order_id, tracking):
            if not identifier:
                continue
            chunk = identifier[:6]
            for text in texts:
                if identifier in text:
                    break
                if "[redacted-" in text and chunk in text:
                    failures.append(Failure("pii", f"{identifier} was destroyed by redaction"))
                    break
    return failures


def score_expectations(scenario: Scenario, result: HarnessResult) -> list[Failure]:
    failures = score_tools(scenario, result)
    failures += score_escalation(scenario, result)
    if scenario.expect.end_reason is not None:
        actual = result.observed[-1].end_reason if result.observed else None
        if actual != scenario.expect.end_reason:
            failures.append(
                Failure("end_reason", f"expected end_reason {scenario.expect.end_reason!r}, got {actual!r}")
            )
    failures += score_db(scenario, result)
    failures += score_pii(scenario, result)
    return failures


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def score_drift(recording: Recording, result: HarnessResult) -> list[Failure]:
    """Recording captured what live execution produced; replay recomputes and
    diffs. A mismatch means either the tools regressed (caught) or the
    environment shifted (also worth knowing) — the only mechanism that
    actually verifies replay drives the same code paths.
    """
    failures: list[Failure] = []
    recorded = recording.observed
    if len(recorded) != len(result.observed):
        return [
            Failure(
                "drift",
                f"recording has {len(recorded)} turn(s), replay produced {len(result.observed)} — "
                "the code now takes a different path",
            )
        ]
    for then, now in zip(recorded, result.observed):
        for key in (
            "tool_calls",
            "grounding_flagged",
            "hedge_spoken",
            "escalation_reason",
            "end_reason",
            "block_input_runtime_type",
        ):
            before = _canonical(then.get(key))
            after = _canonical(getattr(now, key))
            if before != after:
                failures.append(Failure("drift", f"turn {now.turn} {key} differs from recording"))
    return failures


def stored_record_counts(result: HarnessResult) -> dict[str, int]:
    """Counts degrade to 0 for a scenario with no usable db_path, same as
    `_stored_texts` — connect() is inside the guarded block for the same
    reason (see there)."""
    counts = {"turn_log": len(result.log_lines), "tickets": 0, "escalations": 0}
    try:
        conn = sqlite3.connect(result.db_path)
    except sqlite3.Error:
        return counts
    try:
        for table in ("tickets", "escalations"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return counts


# A number wearing a policy-ish unit — the same shape
# guardrails/validators.py's _CLAIM_RE looks for. Duplicated here rather
# than imported on purpose: this is the MEASURING instrument, and if it
# shared a regex with the thing being measured, a change to the detector
# would silently move the baseline it is being measured against.
_CLAIM_RE = re.compile(
    r"\$\s?\d+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*%"
    r"|\b\d+(?:\.\d+)?\s*(?:business\s+)?(?:day|days|week|weeks|month|months|hour|hours)\b",
    re.IGNORECASE,
)

LADDER_REASON = "repeated ungrounded replies"


@dataclass(frozen=True)
class GroundingCounts:
    """The confusion matrix, plus the three companion numbers that are
    arguably worth more than the headline rate.

    What this CANNOT establish, stated rather than hidden: with 20 scenarios
    and roughly 60-80 turns, only the subset carrying both a search_policy
    call and a numeric claim is labellable — a single-digit to low-teens
    denominator, putting a 95% confidence interval on the rate at roughly
    +/-25 points. It cannot justify changing
    UNGROUNDED_REPLY_ESCALATION_THRESHOLD on statistical grounds, cannot
    estimate real-traffic behaviour (every scenario is authored by the same
    person who wrote the detector), and cannot find failure modes nobody
    scripted. What it CAN do: prove end to end that the detector fires on a
    genuine fabrication and stays quiet on ordinary correct replies, and
    produce a reproducible baseline whose value is the DELTA after a later
    prompt or regex change, not the level.
    """

    grounded: int = 0
    ungrounded: int = 0
    flagged: int = 0
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    hedged: int = 0
    unreachable_claims: int = 0
    ladder_fired: int = 0
    labeled_turns: int = 0


EMPTY_COUNTS = GroundingCounts()


def rate(numerator: int, denominator: int) -> str:
    """Always n/N with raw counts, never a bare percentage.

    A zero denominator reports insufficient data rather than 0%: with a
    denominator this small, a percentage invites exactly the overstatement
    guardrails/validators.py's own docstring warns against.
    """
    if denominator == 0:
        return f"{numerator}/0 — insufficient data"
    return f"{numerator}/{denominator}"


def grounding_counts(scenario: Scenario, result: HarnessResult) -> GroundingCounts:
    """One scenario's contribution to the aggregate."""
    grounded = ungrounded = flagged = 0
    true_positive = false_positive = true_negative = false_negative = 0
    hedged = unreachable = 0
    ladder = 0

    for index, row in enumerate(result.observed):
        label = scenario.grounding_truth[index] if index < len(scenario.grounding_truth) else "not_applicable"
        if row.grounding_flagged:
            flagged += 1
        if row.hedge_spoken:
            hedged += 1
        if row.escalation_reason == LADDER_REASON:
            ladder = 1

        reply = result.replies[index] if index < len(result.replies) else ""
        searched = any(call.get("name") == "search_policy" for call in row.tool_calls)
        if _CLAIM_RE.search(reply) and not searched:
            # A policy-shaped claim the detector could not possibly reach,
            # because GROUNDING_TRIGGER_TOOLS gates on search_policy being in
            # THIS turn's tool_calls.
            unreachable += 1

        if label == "grounded":
            grounded += 1
            if row.grounding_flagged:
                false_positive += 1
            else:
                true_negative += 1
        elif label == "ungrounded":
            ungrounded += 1
            if row.grounding_flagged:
                true_positive += 1
            else:
                false_negative += 1

    return GroundingCounts(
        grounded=grounded,
        ungrounded=ungrounded,
        flagged=flagged,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        hedged=hedged,
        unreachable_claims=unreachable,
        ladder_fired=ladder,
        labeled_turns=grounded + ungrounded,
    )


def combine_counts(counts: Iterable[GroundingCounts]) -> GroundingCounts:
    total = {f.name: 0 for f in fields(GroundingCounts)}
    for item in counts:
        for name in total:
            total[name] += getattr(item, name)
    return GroundingCounts(**total)
