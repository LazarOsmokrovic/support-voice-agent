"""`python -m eval.record` — the ONLY entry point that touches the API.

Record once, replay forever. Scenarios run live against the real API here,
capturing genuine Claude responses into versioned fixtures; scoring
thereafter is offline, free and deterministic. Re-recording is always an
explicit, NAMED operation: scenarios are listed by name or with --all, and
eval/run_eval.py never re-records on its own, because doing so would spend
money unasked and erase the very signal a STALE report is trying to give you.

Runs through the SAME eval/harness.py as replay, against the same freshly
seeded temp database — see that module's docstring for why that sharing is
the design's load-bearing choice.

The worksheet printer at the end is not a nicety. Grounding ground truth is
assigned by a HUMAN, once, after reading the recording — not by the runner
and not by a model, because grading one unvalidated detector with another
unvalidated detector measures nothing. Printing each turn's reply, its
grounding_flagged value and the numbers the tools actually returned, then
emitting a paste-ready grounding_truth block, is what decides whether the
labelling actually gets done.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import anthropic

from agent.core import DEFAULT_MODEL
from eval.harness import HarnessResult, observed_as_dicts, run_scenario
from eval.recording import Recording, current_hashes, save_recording
from eval.scenarios import SCENARIOS, Scenario, scenario_by_name

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class _RecordingMessages:
    def __init__(self, owner: RecordingAnthropicClient) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any):
        response = await self._owner.inner.messages.create(**kwargs)
        # model_dump(mode="json") is what makes replay SDK-native: the same
        # shape the SDK's own deserialiser reads back, rather than a
        # hand-rolled mock that freezes today's assumptions.
        self._owner.creates.append(response.model_dump(mode="json"))
        return response

    async def parse(self, **kwargs: Any):
        response = await self._owner.inner.messages.parse(**kwargs)
        output_format = kwargs.get("output_format")
        self._owner.parses.append(
            {
                "output_format": getattr(output_format, "__name__", str(output_format)),
                "parsed_output": response.parsed_output.model_dump(mode="json"),
            }
        )
        return response


class RecordingAnthropicClient:
    """Delegates every call to a real client and captures both queues."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.creates: list[dict[str, Any]] = []
        self.parses: list[dict[str, Any]] = []
        self.messages = _RecordingMessages(self)


async def record_scenario(
    scenario: Scenario,
    workdir: Path,
    client_factory: Callable[[], Any],
) -> tuple[Recording, HarnessResult]:
    """Run one scenario live and assemble its Recording.

    `recorded_at` is captured BEFORE the run and doubles as the frozen clock
    for both this run and every replay, so the scenario evaluates against the
    dates that were true when it was recorded — years later included.
    """
    recorded_at = datetime.now()  # noqa: DTZ005 — naive on purpose, matches the frozen sites
    client = RecordingAnthropicClient(client_factory())
    frozen = recorded_at + timedelta(days=scenario.clock_offset_days)
    result = await run_scenario(scenario, client, frozen, workdir)
    # Same turn-log invariant the offline runner checks, and it matters more
    # here: this is the path that WRITES the fixture. log_turn() never raises,
    # so a silently dropped write during recording would bake a hole into the
    # recording itself and every later replay would inherit it, with nothing
    # left to reveal that anything was lost. Raising is right in the recorder
    # (unlike the runner, which reports it as one scenario's ERROR) because
    # saving a knowingly incomplete recording is worse than recording nothing.
    if len(result.log_lines) != len(result.records):
        raise RuntimeError(
            f"{scenario.name}: turn log holds {len(result.log_lines)} line(s) for "
            f"{len(result.records)} captured record(s) — a write silently failed; "
            "refusing to save an incomplete recording"
        )
    recording = Recording(
        scenario=scenario.name,
        recorded_at=recorded_at.isoformat(),
        model=DEFAULT_MODEL,
        anthropic_sdk_version=anthropic.__version__,
        creates=client.creates,
        parses=client.parses,
        observed=observed_as_dicts(result.observed),
        **current_hashes(),
    )
    return recording, result


def grounding_worksheet(scenario: Scenario, result: HarnessResult) -> str:
    """Everything a human needs to label this scenario's turns, plus a
    paste-ready grounding_truth block pre-filled with "not_applicable".
    """
    lines = [f"--- grounding worksheet: {scenario.name} ---"]
    for index, row in enumerate(result.observed):
        reply = result.replies[index] if index < len(result.replies) else ""
        numbers = sorted(
            {
                number
                for call in row.tool_calls
                for number in _NUMBER_RE.findall(json.dumps(call.get("output"), default=str))
            }
        )
        lines.append(f"turn {row.turn}: grounding_flagged={row.grounding_flagged} hedge_spoken={row.hedge_spoken}")
        lines.append(f"  reply: {reply}")
        lines.append(f"  tools: {[call.get('name') for call in row.tool_calls]}")
        lines.append(f"  numbers in tool output: {numbers}")
    labels = ", ".join(['"not_applicable"'] * len(result.observed))
    trailing = "," if len(result.observed) == 1 else ""
    lines.append(f"grounding_truth=({labels}{trailing}),   # <- edit each label, then paste into eval/scenarios.py")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.record")
    parser.add_argument("--scenario", action="append", default=[], help="scenario name; repeatable")
    parser.add_argument("--all", action="store_true", help="record every scenario in SCENARIOS")
    args = parser.parse_args(argv)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Recording needs a real key; run_eval never does.")
        return 2
    if not args.all and not args.scenario:
        print("Name scenarios with --scenario NAME (repeatable) or pass --all. There is no implicit re-record.")
        return 2

    if args.all:
        targets = list(SCENARIOS)
    else:
        targets = []
        for name in args.scenario:
            scenario = scenario_by_name(name)
            if scenario is None:
                print(f"unknown scenario: {name}")
                return 2
            targets.append(scenario)

    async def _run() -> int:
        with TemporaryDirectory(prefix="eval-record-") as tmp:
            for scenario in targets:
                recording, result = await record_scenario(
                    scenario, Path(tmp) / scenario.name, anthropic.AsyncAnthropic
                )
                if result.error:
                    print(f"ERROR {scenario.name}: {result.error}")
                path = save_recording(recording)
                print(f"recorded {scenario.name} -> {path}")
                print(grounding_worksheet(scenario, result))
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
