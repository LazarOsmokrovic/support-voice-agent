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
from pathlib import Path
from typing import Any

from data import mock_db
from eval.harness import HarnessResult
from eval.recording import Recording
from eval.scenarios import Scenario
from guardrails.pii import redact_text

# HarnessResult.db_path defaults to this (eval/harness.py) when a scenario
# never touched a real per-scenario database — only ad-hoc test construction
# produces it, since run_scenario always sets a real seeded file. It is the
# ONLY sqlite3 failure this module tolerates; see _stored_texts and
# stored_record_counts.
_NO_DB_PATH = Path()


@dataclass(frozen=True)
class Failure:
    kind: str  # "tools" | "escalation" | "offer" | "end_reason" | "db" | "pii" | "drift"
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


def score_offer(scenario: Scenario, result: HarnessResult) -> list[Failure]:
    """offer_turn carries two assertions at once — see Expectations.

    Mirrors score_escalation exactly, but reads row.escalation_offered
    instead of row.escalation_reason: a SUGGESTED trigger (agent/tools/
    escalation.py's SUGGESTED_REASONS) now produces an offer, not an
    escalation, and this is the only scorer that can see that distinction —
    score_escalation's `fired` list is built from escalation_reason, which a
    mere offer never sets.
    """
    fired = [row for row in result.observed if row.escalation_offered]
    expected_turn = scenario.expect.offer_turn

    if expected_turn is None:
        if fired:
            row = fired[0]
            return [
                Failure(
                    "offer",
                    f"expected no offer, but turn {row.turn} offered: {row.escalation_offered!r}",
                )
            ]
        return []

    if not fired:
        return [Failure("offer", f"expected an offer on turn {expected_turn}, none fired")]

    row = fired[0]
    failures: list[Failure] = []
    if row.turn != expected_turn:
        failures.append(
            Failure("offer", f"expected an offer on turn {expected_turn}, it fired on turn {row.turn}")
        )
    if scenario.expect.offer_reason and row.escalation_offered != scenario.expect.offer_reason:
        failures.append(
            Failure(
                "offer",
                f"expected reason {scenario.expect.offer_reason!r}, got {row.escalation_offered!r}",
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

    Only the missing-db_path case (_NO_DB_PATH, HarnessResult's default) is
    tolerated: that shape only ever comes from ad-hoc test construction,
    since run_scenario always sets a real seeded file. A genuine sqlite3.Error
    against a real per-scenario database — corruption, permissions, a disk
    error — must propagate. Catching it here would silently score a broken
    run as PII-clean, which is the dangerous direction to be wrong in.
    """
    texts = [json.dumps(line, default=str) for line in result.log_lines]
    if result.db_path == _NO_DB_PATH:
        return texts
    conn = sqlite3.connect(result.db_path)
    conn.row_factory = sqlite3.Row
    try:
        for table in ("tickets", "escalations"):
            for row in conn.execute(f"SELECT * FROM {table}").fetchall():  # noqa: S608 — fixed literal names
                texts.append(json.dumps(dict(row), default=str))
    finally:
        conn.close()
    return texts


def _seeded_identifiers() -> list[tuple[str, str]]:
    """Every value this project must NEVER let its own redactor mangle,
    read from the seed at runtime. Returns (kind, value) pairs.

    Phase 10a destroyed exactly these three shapes: 17-digit order IDs
    (eaten by _CARDLIKE_RE), ISO dates (eaten by _PHONE_RE), and TBA...US
    tracking numbers (mangled mid-token by _PHONE_RE). guardrails/pii.py
    now exempts the first two via _NON_PII_SHAPES and guards the third with
    word boundaries — this is the check that those protections stay.
    """
    values: list[tuple[str, str]] = []
    for order_id, _cust, _item, _qty, _price, _status, order_date, delivery, tracking in mock_db.ORDERS:
        values.append(("order id", order_id))
        if tracking:
            values.append(("tracking number", tracking))
        for date in (order_date, delivery):
            if date:
                values.append(("order date", date))
    for _cust, scheduled_time, _reason, _status in mock_db.APPOINTMENTS:
        values.append(("appointment time", scheduled_time))
    return values


def score_redactor_preserves_identifiers() -> list[Failure]:
    """Assert the redactor leaves this project's own identifiers alone.

    This is the direct form of the check, and it replaced an indirect one
    that did not work. The earlier version inferred destruction from stored
    records: if a record mentioned one identifier, omitted another, and
    contained any '[redacted-' marker, it called the absent one destroyed.
    That produced false positives on any record carrying an unrelated
    redaction (a masked email made an unmentioned tracking number look
    eaten), and — worse — it never covered ISO dates at all, so removing
    _ISO_DATE_OR_DATETIME_RE from guardrails/pii.py turned every
    order_date into '[redacted-phone]' and this suite stayed green. That is
    precisely the Phase 10a regression the check advertises catching.

    Testing the redactor itself removes the guesswork. Destruction happens
    in redact_text; a damaged stored record is only the downstream symptom.
    An exact equality on a known seeded value cannot false-positive, needs
    no anchor, and fires even when no scenario happens to store that
    identifier — so the check is real rather than incidental.
    """
    return [
        Failure("pii", f"the redactor mangles a seeded {kind}: {value!r} -> {redact_text(value)!r}")
        for kind, value in _seeded_identifiers()
        if redact_text(value) != value
    ]


def score_pii(scenario: Scenario, result: HarnessResult) -> list[Failure]:
    """Reads REAL seeded values at runtime; never hard-codes one.

    Two directions, both of which have been real defects in this repo:
    contact details must be GONE, and the store's own identifiers must have
    SURVIVED. A redactor that passes the first half by destroying everything
    fails the second (score_redactor_preserves_identifiers).
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

    # `item` anchors every order, and it is the ONLY anchor used. The
    # earlier version paired order_id and tracking as each other's anchor,
    # which broke twice. Pairing assumes the two co-occur; that holds for
    # get_order_status output but not for refund, policy, escalation or
    # ticket records, so a record naming only the order — plus any
    # unrelated '[redacted-' marker, a masked email say — reported the
    # absent tracking number as destroyed. It also left orders whose
    # tracking_number is NULL (two are seeded) with no anchor at all.
    #
    # `item` has neither problem. It is free text the redactor provably
    # cannot touch: it matches none of _EMAIL_RE / _CARDLIKE_RE /
    # _PHONE_RE, since those need an uninterrupted digit run and a string
    # like "Instant Pot Duo 7-in-1 (6 Qt)" breaks on the letters. So its
    # presence intact means a record genuinely concerns THIS order, even
    # when the order_id that would normally identify it is the very thing
    # destroyed — and its absence means the record simply is not about this
    # order, which is the case that must NOT be flagged.
    for order_id, _cust, item, _qty, _price, _status, _od, _ed, tracking in mock_db.ORDERS:
        for text in texts:
            if item not in text or "[redacted-" not in text:
                continue
            for kind, value in (("order id", order_id), ("tracking number", tracking)):
                if value and value not in text:
                    failures.append(Failure("pii", f"{kind} {value} was destroyed by redaction"))
    return failures


def score_expectations(scenario: Scenario, result: HarnessResult) -> list[Failure]:
    failures = score_tools(scenario, result)
    failures += score_escalation(scenario, result)
    failures += score_offer(scenario, result)
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
            "escalation_offered",
        ):
            before = _canonical(then.get(key))
            after = _canonical(getattr(now, key))
            if before != after:
                failures.append(Failure("drift", f"turn {now.turn} {key} differs from recording"))
    return failures


def stored_record_counts(result: HarnessResult) -> dict[str, int]:
    """Counts degrade to 0 only for a scenario with no usable db_path
    (_NO_DB_PATH, ad-hoc test construction only). A genuine sqlite3.Error
    against a real per-scenario database must propagate, not be scored as
    zero — see `_stored_texts` for the full reasoning; this mirrors it."""
    counts = {"turn_log": len(result.log_lines), "tickets": 0, "escalations": 0}
    if result.db_path == _NO_DB_PATH:
        return counts
    conn = sqlite3.connect(result.db_path)
    try:
        for table in ("tickets", "escalations"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
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
        # Phase 12 (D-9): "repeated ungrounded replies" is a SUGGESTED
        # trigger, so it now lands in escalation_offered, never
        # escalation_reason (agent/session.py's run_turn). Checking both
        # keeps this counter meaningful for recordings made either before or
        # after that split — an offer is still the ladder firing, whether or
        # not the customer went on to accept it.
        if LADDER_REASON in (row.escalation_reason, row.escalation_offered):
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
