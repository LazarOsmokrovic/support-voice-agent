"""Rendering and exit codes — Phase 10c.

Kept out of eval/run_eval.py so the runner is not half print statements, and
so the degenerate cases (no scenarios, nothing labelled, a scenario that
blew up) are testable without running an agent.

SIX outcome states for scored scenarios, not two. PASS/FAIL is "behaviour
matched, or did not". STALE means a staleness hash changed, so the recording
no longer describes the current system — scoring it would score a fiction,
and it is not a failure of the code. DRIFT means replay re-executed the
tools and got different output than the recording observed. MISSING means a
scenario has no recording yet. ERROR means the scenario raised unexpectedly.
Six states rather than two so CI can go red on a genuine regression (FAIL)
and go red DIFFERENTLY on "your fixtures need refreshing" (the rest) — the
fixes differ, and conflating them trains people to ignore the signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from eval.scenarios import CAPABILITIES
from eval.scoring import EMPTY_COUNTS, GroundingCounts, rate

OUTCOMES: tuple[str, ...] = ("PASS", "FAIL", "STALE", "DRIFT", "MISSING", "ERROR")


@dataclass
class ScenarioReport:
    """One scenario's outcome, ready to render or serialise. `details` holds
    the human-readable Failure messages (or an error string) — empty for a
    clean PASS."""

    name: str
    capability: str
    turns: int
    outcome: str
    details: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """The whole suite's result: every scenario, plus the aggregate
    grounding measurement and the PII/storage tallies that only make sense
    across the full run."""

    scenarios: list[ScenarioReport] = field(default_factory=list)
    grounding: GroundingCounts = EMPTY_COUNTS
    pii_leaks: int = 0
    stored_records: dict[str, int] = field(default_factory=dict)
    # Whole-run, not per-scenario: the redactor either preserves this
    # project's own identifiers or it does not, and no scenario changes that.
    # Empty is the healthy state; any entry means guardrails/pii.py is eating
    # order IDs, ISO dates or tracking numbers — Phase 10a's shipped bug.
    redactor_failures: list[str] = field(default_factory=list)

    def count(self, outcome: str) -> int:
        return sum(1 for row in self.scenarios if row.outcome == outcome)


def exit_code(report: EvalReport, strict: bool = False) -> int:
    """0 all pass · 1 any FAIL · 2 any STALE/DRIFT/MISSING/ERROR with no FAIL.

    --strict collapses 2 into 1 for a release gate, where "the fixtures are
    stale" is not an acceptable state to ship in either.
    """
    # A redactor destroying the project's own identifiers is a behavioural
    # regression, not a fixtures problem, so it ranks with FAIL rather than
    # with STALE. It is also independent of the scenarios: it must still go
    # red when every scenario is MISSING, which is exactly the state this
    # suite sits in until the recordings exist.
    if report.count("FAIL") or report.redactor_failures:
        return 1
    fixture_trouble = sum(report.count(name) for name in ("STALE", "DRIFT", "MISSING", "ERROR"))
    if fixture_trouble:
        return 1 if strict else 2
    return 0


def _capability_line(report: EvalReport) -> str:
    tally = {name: 0 for name in CAPABILITIES}
    for row in report.scenarios:
        if row.capability in tally:
            tally[row.capability] += 1
    covered = sum(1 for count in tally.values() if count)
    parts = " · ".join(f"{name} {count}" for name, count in tally.items())
    return f"capability coverage: {parts}  ({covered}/{len(CAPABILITIES)})"


def render(report: EvalReport) -> str:
    """Human-facing text report. Must never crash: an empty scenario list, a
    fully not_applicable grounding set, and an errored scenario are all
    exercised by tests and are expected states, not edge cases to special-
    case away.
    """
    counts = report.grounding
    lines = [
        "Support Voice Agent — eval suite (replay)",
        f"recordings: eval/recordings/  ·  {len(report.scenarios)} scenarios  ·  offline",
        "",
    ]
    for row in report.scenarios:
        lines.append(f"{row.outcome:<6} {row.name:<45} {row.capability:<14} {row.turns} turns")
        lines.extend(f"         - {detail}" for detail in row.details)
    lines.append("")
    lines.append(
        f"{report.count('PASS')} passed · {report.count('FAIL')} failed · "
        f"{report.count('STALE')} stale · {report.count('DRIFT')} drifted · "
        f"{report.count('MISSING')} missing · {report.count('ERROR')} errored"
    )
    lines.append(_capability_line(report))
    lines.append("")
    lines.append("grounding detector")
    lines.append(
        f"  labeled turns          {counts.labeled_turns:>4}    "
        f"(grounded {counts.grounded} · ungrounded {counts.ungrounded})"
    )
    lines.append(f"  flagged                {counts.flagged:>4}")
    lines.append(
        f"  false positives       {rate(counts.false_positive, counts.grounded)}"
        "    [see eval/README.md on what this sample size supports]"
    )
    lines.append(f"  false negatives       {rate(counts.false_negative, counts.ungrounded)}")
    lines.append(f"  hedge spoken            {counts.hedged:>4}")
    lines.append(
        f"  unreachable claims   {counts.unreachable_claims:>4}    "
        "turns asserting a policy number with no search_policy call"
    )
    lines.append(
        f"  ladder fired            {counts.ladder_fired:>4}    "
        'scenario(s) reached "repeated ungrounded replies"'
    )
    lines.append("")
    total_records = sum(report.stored_records.values())
    breakdown = ", ".join(f"{count} {name}" for name, count in report.stored_records.items())
    suffix = f" ({breakdown})" if breakdown else ""
    lines.append(f"pii: {report.pii_leaks} leaks across {total_records} stored records{suffix}")
    if report.redactor_failures:
        lines.append("")
        lines.append("REDACTOR IS DESTROYING THIS PROJECT'S OWN IDENTIFIERS:")
        lines.extend(f"  {detail}" for detail in report.redactor_failures)
    else:
        lines.append("redactor: seeded order IDs, dates and tracking numbers all survive intact")
    return "\n".join(lines)


def to_json(report: EvalReport) -> dict[str, Any]:
    """The whole report as one JSON-serialisable object, for `--json` and for
    CI to diff between runs without parsing the text report."""
    return {
        "scenarios": [asdict(row) for row in report.scenarios],
        "summary": {
            "passed": report.count("PASS"),
            "failed": report.count("FAIL"),
            "stale": report.count("STALE"),
            "drifted": report.count("DRIFT"),
            "missing": report.count("MISSING"),
            "errored": report.count("ERROR"),
        },
        "grounding": asdict(report.grounding),
        "pii_leaks": report.pii_leaks,
        "stored_records": report.stored_records,
        "redactor_failures": report.redactor_failures,
    }
