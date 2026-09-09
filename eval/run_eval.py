"""`python -m eval.run_eval` — the offline entry point. Phase 10c.

No arguments needed, no API key, no network. A key present in the
environment is still never used: the patched constructor (eval/replay.py)
raises on any attempt to reach the model, so a hole in the seam is a loud
failure rather than a surprise bill.

Stays the entry point PROJECT_PLAN.md names; rendering lives in
eval/report.py so this file is not half print statements.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.harness import run_scenario
from eval.recording import frozen_now, load_recording, stale_fields
from eval.replay import FakeAnthropicClient
from eval.report import EvalReport, ScenarioReport, exit_code, render, to_json
from eval.scenarios import SCENARIOS, Scenario, scenario_by_name
from eval.scoring import (
    Failure,
    combine_counts,
    grounding_counts,
    score_drift,
    score_expectations,
    score_redactor_preserves_identifiers,
    stored_record_counts,
)

# The prompt fragment each hash field is about, so a STALE line names the
# constant a human recognises rather than a field name they have to look up.
_HASH_LABELS = {
    "system_prompt_sha256": "SYSTEM_PROMPT",
    "classification_prompt_sha256": "CLASSIFICATION_PROMPT",
    "summary_prompt_sha256": "SUMMARY_PROMPT",
    "handoff_prompt_sha256": "HANDOFF_PROMPT",
    "tool_schemas_sha256": "agent.session.TOOLS",
    "seed_sha256": "data/mock_db.py seed data",
}


def strip_side_effect_env() -> None:
    """Make this CLI as safe as tests/conftest.py makes the test suite.

    MUST be called from main(), not at module scope. agent/core.py calls
    load_dotenv() at import, and this module's imports reach it — so calling
    this at module scope would run BEFORE the variables it is stripping have
    been repopulated. The same load_dotenv reach is why `env -u
    ANTHROPIC_API_KEY python -m pytest` does not actually skip the live
    tests, confirmed empirically on 2026-09-08.

    An escalating scenario calls create_handoff_packet, which calls
    notify_escalation, which POSTs to ESCALATION_WEBHOOK_URL for real. A fake
    escalation packet landing in a real Slack channel is exactly the hazard
    conftest was written for, arriving where conftest cannot reach.
    """
    os.environ.pop("ESCALATION_WEBHOOK_URL", None)
    os.environ.pop("ESCALATION_WEBHOOK_SECRET", None)
    os.environ["TURN_LOG_PATH"] = ""


async def evaluate(scenarios: Sequence[Scenario], workdir: Path) -> EvalReport:
    """Replay, score and aggregate. Never records, never re-records."""
    report = EvalReport()
    counts = []
    stored = {"turn_log": 0, "tickets": 0, "escalations": 0}
    leaks = 0

    for scenario in scenarios:
        row = ScenarioReport(
            name=scenario.name,
            capability=scenario.capability,
            turns=len(scenario.turns),
            outcome="PASS",
        )
        report.scenarios.append(row)

        recording = load_recording(scenario.name)
        if recording is None:
            row.outcome = "MISSING"
            row.details.append(
                f"no recording at eval/recordings/{scenario.name}.json — "
                f"record it: python -m eval.record --scenario {scenario.name}"
            )
            continue

        stale = stale_fields(recording)
        if stale:
            row.outcome = "STALE"
            for name, before, after in stale:
                row.details.append(f"{_HASH_LABELS[name]} changed since recording ({before[:4]}… → {after[:4]}…)")
            row.details.append(f"re-record: python -m eval.record --scenario {scenario.name}")
            continue

        client = FakeAnthropicClient(scenario.name, list(recording.creates), list(recording.parses))
        result = await run_scenario(
            scenario, client, frozen_now(recording, scenario), workdir / scenario.name
        )

        # log_turn() never raises (observability/turn_log.py) — a write that
        # silently failed would otherwise be indistinguishable from a
        # disabled log. This turns that never-raise policy into a checked
        # invariant: every in-process captured record must have produced
        # exactly one line in the turn-log file.
        #
        # Reported as this scenario's ERROR rather than raised. It was a bare
        # `assert`, which had two problems: `python -O` strips it, so the
        # invariant would vanish in exactly the environment least likely to be
        # watched, and an AssertionError aborted the whole run, discarding the
        # other 19 scenarios' results over one scenario's bad write.
        if len(result.log_lines) != len(result.records):
            row.outcome = "ERROR"
            row.details.append(
                f"turn log holds {len(result.log_lines)} line(s) for "
                f"{len(result.records)} captured record(s) — a write silently failed"
            )
            continue

        if result.error:
            row.outcome = "ERROR"
            row.details.append(result.error)
            continue

        drift = score_drift(recording, result)
        if client.creates_remaining or client.parses_remaining:
            drift.append(
                Failure(
                    "drift",
                    f"{client.creates_remaining} create(s) and {client.parses_remaining} parse(s) "
                    "left unused — the code now takes a shorter path",
                )
            )
        failures = score_expectations(scenario, result)
        if failures:
            row.outcome = "FAIL"
            row.details.extend(f"{failure.kind}: {failure.detail}" for failure in failures)
        elif drift:
            row.outcome = "DRIFT"
            row.details.extend(failure.detail for failure in drift)

        leaks += sum(1 for failure in failures if failure.kind == "pii")
        counts.append(grounding_counts(scenario, result))
        for key, value in stored_record_counts(result).items():
            stored[key] = stored.get(key, 0) + value

    report.grounding = combine_counts(counts)
    # Runs once per evaluation, not per scenario: it asserts a property of the
    # redactor itself, which no scenario can influence. It is counted with the
    # PII leaks because it IS the other half of that question — a redactor
    # that hides contact details by destroying the store's own order IDs and
    # dates has not protected anything, it has just broken the data. Phase 10a
    # shipped exactly that bug.
    redactor_failures = score_redactor_preserves_identifiers()
    report.redactor_failures = [failure.detail for failure in redactor_failures]
    leaks += len(redactor_failures)
    report.pii_leaks = leaks
    report.stored_records = stored
    return report


def main(argv: Sequence[str] | None = None) -> int:
    strip_side_effect_env()

    parser = argparse.ArgumentParser(prog="python -m eval.run_eval")
    parser.add_argument("--scenario", action="append", default=[], help="run one scenario; repeatable")
    parser.add_argument("--strict", action="store_true", help="treat STALE/DRIFT/MISSING as failures")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit the whole report as JSON")
    args = parser.parse_args(argv)

    if args.scenario:
        targets = []
        for name in args.scenario:
            scenario = scenario_by_name(name)
            if scenario is None:
                print(f"unknown scenario: {name}")
                return 2
            targets.append(scenario)
    else:
        targets = list(SCENARIOS)

    with TemporaryDirectory(prefix="eval-run-") as tmp:
        report = asyncio.run(evaluate(targets, Path(tmp)))

    print(json.dumps(to_json(report), indent=2) if args.as_json else render(report))
    return exit_code(report, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
