"""Replay: the one seam between this project and the Anthropic API — Phase 10c.

Recorded responses are rebuilt through the SDK'S OWN DESERIALISER, never
hand-rolled with MagicMock the way tests/test_session.py does. That is a
deliberate difference in kind, not in effort. A MagicMock freezes today's
assumption about what `block.input` is forever — fine for a unit test of
control flow, wrong for the suite whose entire job is fidelity. Delegating
to construct_type means an SDK upgrade that changes block.input changes
replay exactly as it changes production, and the dedicated test in
tests/test_eval_harness.py fails by name instead of degrading silently.

`parses` need no equivalent fidelity work: the only attribute any caller
reads is `.parsed_output` (escalation.py:99, escalation.py:231,
summary.py:123) and its type is a Pydantic model this repo owns, so dump
and model_validate back is byte-for-byte what production receives.

Both queues raise descriptively rather than ever returning a stale item. A
green scenario for the wrong reason is worse than a red one.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types import Message

from agent.tools import refunds as refunds_module
from agent.tools import scheduling as scheduling_module

try:  # The SDK's own deserialiser. Private, so guarded and tested by name.
    from anthropic._models import construct_type as _construct_type
except ImportError:  # pragma: no cover - exercised only on an SDK that moves it
    _construct_type = None


class RecordingExhausted(RuntimeError):
    """Replay wanted more model calls than were recorded.

    The signature of a prompt change that added a tool round-trip. Must name
    the scenario and the index, because a bare StopIteration from deep inside
    the tool loop is unreadable.
    """


class RecordingMismatch(RuntimeError):
    """Replay asked for a different structured output than was recorded next."""


def rebuild_message(payload: dict[str, Any]) -> Message:
    """Rebuild one recorded `messages.create` response as a real SDK Message."""
    if _construct_type is not None:
        return _construct_type(value=payload, type_=Message)  # type: ignore[return-value]
    return Message.construct(**payload)


class ParsedResponse:
    """The only attribute production reads off a `messages.parse` response."""

    __slots__ = ("parsed_output",)

    def __init__(self, parsed_output: Any) -> None:
        self.parsed_output = parsed_output


class _FakeMessages:
    def __init__(self, owner: FakeAnthropicClient) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> Message:
        owner = self._owner
        if owner.creates_consumed >= len(owner.creates):
            raise RecordingExhausted(
                f"scenario {owner.scenario!r} asked for create #{owner.creates_consumed + 1} "
                f"but only {len(owner.creates)} recorded — re-record with "
                f"`python -m eval.record --scenario {owner.scenario}`"
            )
        payload = owner.creates[owner.creates_consumed]
        owner.creates_consumed += 1
        return rebuild_message(payload)

    async def parse(self, **kwargs: Any) -> ParsedResponse:
        owner = self._owner
        output_format = kwargs.get("output_format")
        requested = getattr(output_format, "__name__", str(output_format))
        if owner.parses_consumed >= len(owner.parses):
            raise RecordingExhausted(
                f"scenario {owner.scenario!r} asked for parse #{owner.parses_consumed + 1} "
                f"({requested}) but only {len(owner.parses)} recorded — re-record with "
                f"`python -m eval.record --scenario {owner.scenario}`"
            )
        entry = owner.parses[owner.parses_consumed]
        expected = entry["output_format"]
        if expected != requested:
            raise RecordingMismatch(
                f"scenario {owner.scenario!r} parse #{owner.parses_consumed + 1}: "
                f"recording holds {expected}, code requested {requested} — the call order diverged"
            )
        owner.parses_consumed += 1
        return ParsedResponse(output_format.model_validate(entry["parsed_output"]))


class FakeAnthropicClient:
    """Stands in for anthropic.AsyncAnthropic for one scenario's duration.

    Never reaches the network. A create/parse beyond what was recorded is a
    loud, named failure, so a hole in the seam is a crash rather than a
    surprise bill.
    """

    def __init__(
        self,
        scenario: str,
        creates: list[dict[str, Any]],
        parses: list[dict[str, Any]],
    ) -> None:
        self.scenario = scenario
        self.creates = creates
        self.parses = parses
        self.creates_consumed = 0
        self.parses_consumed = 0
        self.messages = _FakeMessages(self)

    @property
    def creates_remaining(self) -> int:
        return len(self.creates) - self.creates_consumed

    @property
    def parses_remaining(self) -> int:
        return len(self.parses) - self.parses_consumed


# Verified by grep, and kept honest by a test rather than by a comment:
# tests/test_eval_harness.py asserts these sets never drift. Five entries,
# not four: agent/session.py:218 is prose inside create_session's docstring
# ("...creates its own anthropic.AsyncAnthropic() by default...") describing
# the real construction that happens in agent/core.py:101 — it never
# executes as code. The grep-guard test matches source text, not AST, so it
# can't tell a docstring mention from a call; the seam (patching the module
# attribute) is unaffected either way, since a docstring never calls
# anything.
MODEL_CONSTRUCTION_SITES: tuple[tuple[str, int], ...] = (
    ("agent/core.py", 101),
    ("agent/session.py", 240),
    ("agent/tools/summary.py", 142),
    ("agent/tools/escalation.py", 214),
    ("agent/tools/escalation.py", 358),
)

# The ONLY two clock reads that change a decision: the refund return-window
# check and which appointment slots exist. Everything else agent/ reads the
# clock for writes a stored string, and those stay real.
FROZEN_CLOCK_SITES: tuple[tuple[str, int], ...] = (
    ("agent/tools/refunds.py", 111),
    ("agent/tools/scheduling.py", 108),
)


def frozen_datetime_class(frozen: datetime) -> type[datetime]:
    """A datetime subclass whose bare now() is pinned to `frozen`.

    Freezing the clock is the ONE place replay is not literally production,
    and it belongs here in the docstring rather than in a footnote discovered
    later. It is unavoidable: the alternative is scenarios that expire
    against the calendar, which is precisely the defect this phase fixes.

    now(tz) with a tzinfo is deliberately NOT frozen. agent/tools/refunds.py
    uses this same module-level name for both the window check
    (`datetime.now()`, line 111, decision-affecting) and `issued_at`
    (`datetime.now(timezone.utc)`, line 205, a stored string). Splitting on
    the argument lands the freeze on exactly the decision.
    """

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is None:
                return frozen
            return datetime.now(tz)

    return _FrozenDateTime


def _blocked_sync_client(*args: Any, **kwargs: Any):
    raise RuntimeError(
        "eval replay blocked a synchronous anthropic.Anthropic() construction — "
        "this project is async throughout, so a sync client means an unpatched code path"
    )


@contextlib.contextmanager
def scenario_patch(client: Any, frozen: datetime, turn_log_path: Path) -> Iterator[None]:
    """Everything one scenario needs held still, for its duration.

    Three separate hazards, one context manager:

    1. THE MODEL SEAM. Patch the anthropic.AsyncAnthropic CONSTRUCTOR, not a
       `client` parameter. create_session(client=...) reaches one of four
       model call sites; the other three build their own client. Patching the
       constructor reaches all four, requires zero changes under agent/
       (CLAUDE.md rule 5), and cannot be defeated by a fifth site added
       later. anthropic.Anthropic is patched to RAISE, so an accidental sync
       path is loud rather than a surprise network call.

    2. THE CLOCK. See frozen_datetime_class.

    3. THE ENVIRONMENT. agent/core.py calls load_dotenv() at import, so a
       developer's real ESCALATION_WEBHOOK_URL is live and an escalating
       scenario would fire a REAL webhook POST. tests/conftest.py's autouse
       fixture protects the test suite but cannot reach a CLI. Stripping here
       means the protection travels with the scenario, whoever runs it.
    """
    saved_async = anthropic.AsyncAnthropic
    saved_sync = anthropic.Anthropic
    saved_refunds_datetime = refunds_module.datetime
    saved_scheduling_datetime = scheduling_module.datetime
    saved_env = {
        key: os.environ.get(key)
        for key in ("ESCALATION_WEBHOOK_URL", "ESCALATION_WEBHOOK_SECRET", "TURN_LOG_PATH")
    }

    frozen_cls = frozen_datetime_class(frozen)
    try:
        anthropic.AsyncAnthropic = lambda *args, **kwargs: client  # type: ignore[assignment]
        anthropic.Anthropic = _blocked_sync_client  # type: ignore[assignment]
        refunds_module.datetime = frozen_cls  # type: ignore[assignment]
        scheduling_module.datetime = frozen_cls  # type: ignore[assignment]
        os.environ.pop("ESCALATION_WEBHOOK_URL", None)
        os.environ.pop("ESCALATION_WEBHOOK_SECRET", None)
        os.environ["TURN_LOG_PATH"] = str(turn_log_path)
        yield
    finally:
        anthropic.AsyncAnthropic = saved_async  # type: ignore[assignment]
        anthropic.Anthropic = saved_sync  # type: ignore[assignment]
        refunds_module.datetime = saved_refunds_datetime  # type: ignore[assignment]
        scheduling_module.datetime = saved_scheduling_datetime  # type: ignore[assignment]
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
