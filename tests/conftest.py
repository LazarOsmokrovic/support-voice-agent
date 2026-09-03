"""Repo-wide pytest fixtures.

This is the project's first conftest.py — keep it minimal and don't grow it
casually; add to it only for something that genuinely needs to apply to
every test.

Why this exists: agent/core.py calls load_dotenv() at import time, so a real
ESCALATION_WEBHOOK_URL (and ESCALATION_WEBHOOK_SECRET) sitting in a
developer's .env — put there for the Phase 11 manual n8n checkpoint that
README.md's setup steps walk through — leaks into os.environ for the whole
pytest process. Any test that exercises agent/tools/escalation.py's
create_handoff_packet without explicitly stubbing notify_escalation (e.g.
tests/test_escalation.py's test_create_handoff_packet_infers_fields_and_logs)
would then fire a REAL outbound webhook POST — a fake escalation packet
landing in a real Slack channel. This autouse fixture strips both vars from
the environment before every test runs, so no test can ever hit a real
endpoint by accident.

Tests that need one of these vars set already do so explicitly via
monkeypatch.setenv inside the test body (see tests/test_notifications.py).
That still works correctly: this fixture and the test share the same
function-scoped `monkeypatch` fixture instance, and the test's own
monkeypatch.setenv call runs after this fixture's monkeypatch.delenv, so it
wins.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_escalation_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ESCALATION_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ESCALATION_WEBHOOK_SECRET", raising=False)
