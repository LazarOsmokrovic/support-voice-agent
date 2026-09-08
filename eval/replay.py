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

from typing import Any

from anthropic.types import Message

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
