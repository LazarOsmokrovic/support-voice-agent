"""Repo-wide pytest fixtures.

This is the project's first conftest.py — keep it minimal and don't grow it
casually; add to it only for something that genuinely needs to apply to
every test.

Why this exists: agent/core.py calls load_dotenv() at import time, so real
ESCALATION_WEBHOOK_URL / ESCALATION_WEBHOOK_SECRET values sitting in a
developer's .env — put there for the Phase 11 manual n8n checkpoint that
README.md's setup steps walk through — leak into os.environ for the whole
pytest process. Any test that exercises agent/tools/escalation.py's
create_handoff_packet without explicitly stubbing notify_escalation (e.g.
tests/test_escalation.py's test_create_handoff_packet_infers_fields_and_logs)
would then fire a REAL outbound webhook POST — a fake escalation packet
landing in a real Slack channel.

Phase 10b added a second hazard of the same shape: turn logging
(observability/turn_log.py) defaults ON to a *relative* path
(logs/turns.jsonl). Any test that reaches agent/session.py's run_turn — or,
from Phase 10b Task 4, transport/pipecat_processors.py's DTMF escalation
path — would otherwise append real records to the repo's own log file on
every test run.

This autouse fixture strips the webhook vars and blanks TURN_LOG_PATH before
every test runs, so no test can ever hit a real endpoint or write to the
repo's log by accident.

Tests that need one of these vars set already do so explicitly via
monkeypatch.setenv inside the test body (see tests/test_notifications.py and
tests/test_session.py's logging tests). That still works correctly: this
fixture and the test share the same function-scoped `monkeypatch` fixture
instance, and the test's own monkeypatch.setenv call runs after this
fixture's monkeypatch.delenv/setenv, so it wins.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ESCALATION_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ESCALATION_WEBHOOK_SECRET", raising=False)
    # Phase 10b: turn logging defaults ON to a relative path, so without this
    # every test touching run_turn would append to the repo's own
    # logs/turns.jsonl. Tests that want logging set the path explicitly with
    # monkeypatch.setenv, which runs after this fixture.
    monkeypatch.setenv("TURN_LOG_PATH", "")
