"""On-disk fixture format for recorded scenarios — Phase 10c.

One committed JSON file per scenario, at eval/recordings/<name>.json. One
file per scenario rather than one combined file, because a combined file
makes every re-record a whole-file diff and a merge-conflict magnet.

Pretty-printed JSON, not JSONL: a recording is one document, not a stream
(unlike logs/turns.jsonl, which genuinely is one), and a pretty-printed
object diffs legibly where a 40 KB single line does not.

THE HASHES ANSWER "WHEN MUST THIS BE RE-RECORDED." A recording becomes a lie
the moment SYSTEM_PROMPT changes, or a tool schema changes, or the seed data
moves. The runner compares hashes and reports STALE with the exact
re-record command. It NEVER re-records itself: that would spend money unasked
and erase the very signal you wanted.

`recorded_at` does double duty — it is provenance AND the frozen clock that
replay hands to agent/tools/refunds.py and agent/tools/scheduling.py, which
is what stops a recorded scenario from silently expiring against the
calendar (see eval/replay.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agent.prompts import CLASSIFICATION_PROMPT, HANDOFF_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT
from agent.session import TOOLS
from data import mock_db
from eval.scenarios import Scenario

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"

HASH_FIELDS: tuple[str, ...] = (
    "system_prompt_sha256",
    "classification_prompt_sha256",
    "summary_prompt_sha256",
    "handoff_prompt_sha256",
    "tool_schemas_sha256",
    "seed_sha256",
)


@dataclass(frozen=True)
class Recording:
    """One scenario's captured live run.

    Two queues, not one. `creates` and `parses` are different SDK methods
    with different consumers, interleaving in a fixed per-turn order
    (create x N for the tool loop, then one parse for classify_turn,
    optionally one for _infer_handoff_fields, optionally one for
    summarize_session). Separate ordered queues mean an extra `create`
    cannot silently shift a classification into a summary slot, and
    `parses` additionally dispatch on output_format class name, so a
    diverging call order becomes a reported error rather than a corrupted
    replay.

    `observed` is the drift check: what live execution actually produced.
    Replay recomputes it and diffs. A mismatch is DRIFT — either the tools
    regressed (caught) or the environment shifted (also worth knowing). It
    is the only mechanism that verifies replay drives the same code paths.
    """

    scenario: str
    recorded_at: str  # naive ISO-8601; doubles as the frozen clock for replay
    model: str
    anthropic_sdk_version: str
    system_prompt_sha256: str
    classification_prompt_sha256: str
    summary_prompt_sha256: str
    handoff_prompt_sha256: str
    tool_schemas_sha256: str
    seed_sha256: str
    creates: list[dict[str, Any]] = field(default_factory=list)
    parses: list[dict[str, Any]] = field(default_factory=list)
    observed: list[dict[str, Any]] = field(default_factory=list)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    """Canonical JSON: sorted keys, no incidental whitespace. A tool schema
    dict reordered by an unrelated edit must not read as a change.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def current_hashes() -> dict[str, str]:
    """Hash everything a recording depends on, as of right now."""
    return {
        "system_prompt_sha256": _sha256(SYSTEM_PROMPT),
        "classification_prompt_sha256": _sha256(CLASSIFICATION_PROMPT),
        "summary_prompt_sha256": _sha256(SUMMARY_PROMPT),
        "handoff_prompt_sha256": _sha256(HANDOFF_PROMPT),
        "tool_schemas_sha256": _sha256(_canonical(TOOLS)),
        "seed_sha256": _sha256(
            _canonical(
                {
                    "customers": mock_db.CUSTOMERS,
                    "orders": mock_db.ORDERS,
                    "tickets": mock_db.TICKETS,
                    "appointments": mock_db.APPOINTMENTS,
                }
            )
        ),
    }


def stale_fields(recording: Recording) -> list[tuple[str, str, str]]:
    """Every hash that has moved since this recording, as
    (field_name, recorded_hash, current_hash). Empty means still valid.
    """
    current = current_hashes()
    return [
        (name, getattr(recording, name), current[name])
        for name in HASH_FIELDS
        if getattr(recording, name) != current[name]
    ]


def recording_path(name: str) -> Path:
    return RECORDINGS_DIR / f"{name}.json"


def save_recording(recording: Recording) -> Path:
    path = recording_path(recording.scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(recording), indent=2) + "\n", encoding="utf-8")
    return path


def load_recording(name: str) -> Recording | None:
    """None means "not recorded yet" — the runner turns that into MISSING
    with the exact record command, never a silent skip.
    """
    path = recording_path(name)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Recording)}
    return Recording(**{key: value for key, value in payload.items() if key in known})


def frozen_now(recording: Recording, scenario: Scenario) -> datetime:
    """The instant this scenario's clock is frozen to.

    Naive, matching the `# noqa: DTZ005 - naive on purpose` convention at
    both frozen sites (agent/tools/refunds.py:111,
    agent/tools/scheduling.py:108) and the seed's naive
    estimated_delivery / scheduled_time strings.
    """
    return datetime.fromisoformat(recording.recorded_at) + timedelta(days=scenario.clock_offset_days)
