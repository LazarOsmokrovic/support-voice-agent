# Phase 11 — Escalation Notifications (n8n) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `EscalationTracker` decides a human needs to take over, actually notify one — POST the handoff packet to a configurable n8n webhook, instead of only writing it to SQLite as today.

**Architecture:** A new `agent/tools/notifications.py` provides redaction, HMAC signing, and a retrying `notify_escalation()` delivery function. `agent/tools/escalation.py::create_handoff_packet` calls it once, right after `log_escalation` persists the row, then records the delivery outcome via a new `mark_notified()`. No other file changes — `agent/core.py`, `agent/session.py`, and everything under `transport/` are untouched.

**Tech Stack:** Python 3.12+, `httpx` (async POST, already a dependency via `transport/tts.py`), `pytest` + `pytest-asyncio` + `pytest-httpx` (already dependencies), stdlib `hmac`/`hashlib`/`re`/`json`.

**Spec:** `docs/superpowers/specs/2026-09-03-phase-11-escalation-notifications-design.md`

## Global Constraints

- **Decoupling (CLAUDE.md rule 5):** `agent/core.py`, `agent/session.py`, and everything under `transport/` must not change. `create_handoff_packet`'s signature must not change — only its internal behavior.
- **Optional-by-default:** `ESCALATION_WEBHOOK_URL` unset → `notify_escalation` is a silent no-op returning `False`, no HTTP call. Same convention as `TTS_BACKEND`/`EMBEDDING_BACKEND` (`transport/tts.py`, `agent/tools/policy_rag.py`).
- **Never raise into the caller:** `notify_escalation` must never raise. `create_handoff_packet` must still return the packet and must still have logged it via `log_escalation` even if notification fails entirely.
- **No migration system:** schema changes go directly into `data/mock_db.py`'s `SCHEMA` string (`CREATE TABLE IF NOT EXISTS`, same as every prior phase). The local `.db` file is gitignored and regenerated via `python -m data.mock_db`.
- **Async throughout:** `notify_escalation` is `async def`; `create_handoff_packet` already is too.
- **Test isolation convention:** every DB test uses `monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "...")` then `mock_db.reset_and_seed()`, exactly as every existing test file does.
- **CLAUDE.md rule 1 exception:** this phase is explicitly independent of Phase 10 (project owner's explicit direction, recorded in the spec) — do not block on or wait for Phase 10.

---

### Task 1: `escalations` table gets `notified`/`notified_at` columns

**Files:**
- Modify: `data/mock_db.py:67-78` (the `escalations` table definition inside `SCHEMA`)
- Test: `tests/test_mock_db.py`

**Interfaces:**
- Produces: two new columns on the `escalations` table — `notified INTEGER NOT NULL DEFAULT 0`, `notified_at TEXT` — consumed by Task 4's `mark_notified`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mock_db.py`:

```python
def test_escalations_have_notified_columns_defaulting_to_unnotified(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_mock_data.db")
    mock_db.reset_and_seed()

    with mock_db.get_connection() as conn:
        row = conn.execute("SELECT notified, notified_at FROM escalations LIMIT 1").fetchone()

    assert row is not None, "expected the seeded ESCALATIONS row"
    assert row["notified"] == 0
    assert row["notified_at"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mock_db.py::test_escalations_have_notified_columns_defaulting_to_unnotified -v`
Expected: FAIL with `sqlite3.OperationalError: no such column: notified`

- [ ] **Step 3: Add the columns**

In `data/mock_db.py`, change the `escalations` table definition inside the `SCHEMA` string from:

```python
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id            TEXT NOT NULL,
    reason                 TEXT NOT NULL,
    customer_intent        TEXT NOT NULL,
    conversation_summary   TEXT NOT NULL,
    verified_account_info  TEXT,
    actions_taken          TEXT,
    sentiment              TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);
```

to:

```python
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id            TEXT NOT NULL,
    reason                 TEXT NOT NULL,
    customer_intent        TEXT NOT NULL,
    conversation_summary   TEXT NOT NULL,
    verified_account_info  TEXT,
    actions_taken          TEXT,
    sentiment              TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    -- Phase 11: whether create_handoff_packet's call to notify_escalation()
    -- (agent/tools/notifications.py) actually delivered this packet to the
    -- configured automation platform. Defaults to unnotified; a session
    -- with no ESCALATION_WEBHOOK_URL configured leaves every row this way.
    notified                INTEGER NOT NULL DEFAULT 0,
    notified_at             TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);
```

(Existing `seed_db()` inserts into `escalations` with an explicit column list that doesn't mention `notified`/`notified_at`, so seeded rows get the defaults automatically — no other change needed in that function.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mock_db.py -v`
Expected: all pass, including the new test.

- [ ] **Step 5: Commit**

```bash
git add data/mock_db.py tests/test_mock_db.py
git commit -m "Phase 11: add notified/notified_at columns to escalations table

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Redaction, signing, and serialization helpers

**Files:**
- Create: `agent/tools/notifications.py`
- Test: `tests/test_notifications.py` (new file)

**Interfaces:**
- Consumes: nothing (pure functions, no I/O).
- Produces: `redact_packet(packet: dict[str, Any]) -> dict[str, Any]`, `sign_payload(body: bytes, secret: str) -> str`, `serialize_packet(packet: dict[str, Any]) -> bytes` — all consumed by Task 3's `notify_escalation`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_notifications.py`:

```python
"""Phase 11: agent/tools/notifications.py — redaction, signing, and
delivery of escalation handoff packets to an external automation platform
(n8n). Mirrors tests/test_tts.py's pytest-httpx pattern for the delivery
half (added in Task 3); this file starts with the pure-function half.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_module

from agent.tools.notifications import redact_packet, serialize_packet, sign_payload


def test_redact_packet_masks_email():
    packet = {"customer_intent": "Contact me at jane.doe@example.com about this."}
    result = redact_packet(packet)
    assert "jane.doe@example.com" not in result["customer_intent"]
    assert "[redacted-email]" in result["customer_intent"]


def test_redact_packet_masks_card_like_number():
    packet = {"conversation_summary": "Card number is 4111 1111 1111 1111 for the refund."}
    result = redact_packet(packet)
    assert "4111 1111 1111 1111" not in result["conversation_summary"]
    assert "[redacted-number]" in result["conversation_summary"]


def test_redact_packet_masks_phone_number():
    packet = {"verified_account_info": "Customer ID CUST-1001, callback at 555-123-4567."}
    result = redact_packet(packet)
    assert "555-123-4567" not in result["verified_account_info"]
    assert "[redacted-phone]" in result["verified_account_info"]


def test_redact_packet_leaves_ordinary_text_untouched():
    packet = {"actions_taken": "Looked up order status, found no issue."}
    result = redact_packet(packet)
    assert result["actions_taken"] == "Looked up order status, found no issue."


def test_redact_packet_leaves_non_redacted_fields_untouched():
    packet = {"escalation_id": 42, "reason": "explicit request for a human", "sentiment": "negative"}
    result = redact_packet(packet)
    assert result == packet


def test_sign_payload_is_deterministic_hmac_sha256():
    body = b'{"escalation_id": 1}'
    expected = hmac_module.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert sign_payload(body, "test-secret") == expected


def test_serialize_packet_produces_sorted_deterministic_json():
    packet_a = {"b": 2, "a": 1}
    packet_b = {"a": 1, "b": 2}
    assert serialize_packet(packet_a) == serialize_packet(packet_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_notifications.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tools.notifications'`

- [ ] **Step 3: Create `agent/tools/notifications.py`**

```python
"""Outbound escalation notifications to an external automation platform
(n8n) — Phase 11 (AI automation), independent of Phase 10.

When EscalationTracker (agent/tools/escalation.py) decides a human needs to
take over, create_handoff_packet persists the packet to SQLite
(log_escalation) and then calls notify_escalation() here to actually tell
someone. A missing or misconfigured webhook is a normal, working state —
ESCALATION_WEBHOOK_URL unset means a silent no-op, the same
optional-by-default convention as TTS_BACKEND/EMBEDDING_BACKEND
(transport/tts.py, agent/tools/policy_rag.py).

Redaction here is deliberately narrow: it only has to protect this one
outbound payload before it leaves the system, not implement Phase 10's real
PII pipeline (guardrails/pii.py, still an untouched stub). Not a substitute
for that work — just enough to not ship a customer's email, phone, or
card-like number to a third-party webhook by default.

Signing (X-Signature-256, HMAC-SHA256) is the outbound mirror of
transport/telephony.py's inbound X-Twilio-Signature verification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CARDLIKE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")

# Free-text fields on a handoff packet (see agent/tools/escalation.py's
# HandoffFields) that can contain customer-supplied text, and so need
# redaction before leaving the system. escalation_id/reason/sentiment are
# short and structured, never PII, and are left untouched.
_REDACTED_FIELDS = ("customer_intent", "conversation_summary", "verified_account_info", "actions_taken")


def _redact(text: str) -> str:
    """Mask emails, card-like digit runs, and phone-like digit runs.

    Order matters: card-like sequences (13-19 digits) are masked before the
    looser phone pattern, so a card number is never partially caught and
    left visible by the phone regex running first.
    """
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _CARDLIKE_RE.sub("[redacted-number]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    return text


def redact_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `packet` with its free-text fields redacted.
    Fields not present are left absent; non-string values pass through
    untouched (defensive — every real caller passes strings here).
    """
    redacted = dict(packet)
    for field in _REDACTED_FIELDS:
        if field in redacted and isinstance(redacted[field], str):
            redacted[field] = _redact(redacted[field])
    return redacted


def sign_payload(body: bytes, secret: str) -> str:
    """HMAC-SHA256 of `body` with `secret`, hex-encoded."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def serialize_packet(packet: dict[str, Any]) -> bytes:
    """Deterministic JSON serialization (sorted keys), so sign_payload() and
    the actual POST body always agree on the exact bytes signed.
    """
    return json.dumps(packet, sort_keys=True).encode()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_notifications.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent/tools/notifications.py tests/test_notifications.py
git commit -m "Phase 11: add redaction/signing helpers for escalation notifications

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `notify_escalation` delivery with retry/backoff

**Files:**
- Modify: `agent/tools/notifications.py` (append to the file created in Task 2)
- Test: `tests/test_notifications.py` (append)

**Interfaces:**
- Consumes: `redact_packet`, `sign_payload`, `serialize_packet` (Task 2, same file).
- Produces: `async def notify_escalation(packet: dict[str, Any], *, client: httpx.AsyncClient | None = None) -> bool` — consumed by Task 4's `create_handoff_packet`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_notifications.py`:

```python
import httpx
import pytest

from agent.tools.notifications import MAX_ATTEMPTS, notify_escalation

SAMPLE_PACKET = {
    "escalation_id": 42,
    "reason": "explicit request for a human",
    "customer_intent": "Wants a refund",
    "conversation_summary": "Asked about a refund for a late order.",
    "verified_account_info": "Customer ID CUST-1001",
    "actions_taken": "None yet",
    "sentiment": "negative",
}
WEBHOOK_URL = "https://n8n.example.com/webhook/escalation"


@pytest.mark.asyncio
async def test_notify_escalation_is_a_noop_without_a_webhook_url(monkeypatch, httpx_mock):
    monkeypatch.delenv("ESCALATION_WEBHOOK_URL", raising=False)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is False
    assert len(httpx_mock.get_requests()) == 0


@pytest.mark.asyncio
async def test_notify_escalation_succeeds_on_first_attempt(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is True
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_notify_escalation_retries_after_a_transient_failure(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setattr("agent.tools.notifications.RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=503)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is True
    assert len(httpx_mock.get_requests()) == 2


@pytest.mark.asyncio
async def test_notify_escalation_gives_up_after_exhausting_all_attempts(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setattr("agent.tools.notifications.RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    for _ in range(MAX_ATTEMPTS):
        httpx_mock.add_response(url=WEBHOOK_URL, status_code=503)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is False
    assert len(httpx_mock.get_requests()) == MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_notify_escalation_does_not_retry_a_4xx_response(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=401)

    delivered = await notify_escalation(SAMPLE_PACKET)

    assert delivered is False
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_notify_escalation_signs_the_body_when_a_secret_is_configured(monkeypatch, httpx_mock):
    from agent.tools.notifications import redact_packet, serialize_packet, sign_payload

    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setenv("ESCALATION_WEBHOOK_SECRET", "shh-its-a-secret")
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)

    await notify_escalation(SAMPLE_PACKET)

    request = httpx_mock.get_requests()[0]
    expected_body = serialize_packet(redact_packet(SAMPLE_PACKET))
    expected_signature = sign_payload(expected_body, "shh-its-a-secret")
    assert request.headers["x-signature-256"] == expected_signature


@pytest.mark.asyncio
async def test_notify_escalation_omits_signature_header_without_a_secret(monkeypatch, httpx_mock):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.delenv("ESCALATION_WEBHOOK_SECRET", raising=False)
    httpx_mock.add_response(url=WEBHOOK_URL, status_code=200)

    await notify_escalation(SAMPLE_PACKET)

    request = httpx_mock.get_requests()[0]
    assert "x-signature-256" not in request.headers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_notifications.py -v -k notify_escalation`
Expected: FAIL with `ImportError: cannot import name 'notify_escalation'`

- [ ] **Step 3: Append the delivery function to `agent/tools/notifications.py`**

Add these imports to the top of the file (alongside the existing ones from Task 2):

```python
import asyncio
import logging
import os

import httpx
```

Then append below `serialize_packet`:

```python
logger = logging.getLogger("agent.tools.notifications")

WEBHOOK_TIMEOUT_SECONDS = 5.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (0.5, 1.5)  # sleep after attempt 1, then after attempt 2
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


async def notify_escalation(packet: dict[str, Any], *, client: httpx.AsyncClient | None = None) -> bool:
    """POST a redacted, signed escalation packet to ESCALATION_WEBHOOK_URL.

    Returns True if delivered (2xx on any attempt), False otherwise —
    including when no webhook URL is configured, which is a silent no-op by
    design. Never raises: a broken or misconfigured webhook must never
    affect the escalation itself completing (see agent/tools/escalation.py's
    create_handoff_packet, which also wraps this call defensively).
    """
    url = os.getenv("ESCALATION_WEBHOOK_URL")
    if not url:
        return False

    body = serialize_packet(redact_packet(packet))
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("ESCALATION_WEBHOOK_SECRET")
    if secret:
        headers["X-Signature-256"] = sign_payload(body, secret)

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS)
    try:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await client.post(url, content=body, headers=headers)
            except httpx.TransportError as exc:
                logger.warning("escalation webhook attempt %d/%d failed: %s", attempt + 1, MAX_ATTEMPTS, exc)
            else:
                if response.status_code < 300:
                    return True
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    logger.warning(
                        "escalation webhook rejected (status=%d), not retrying", response.status_code
                    )
                    return False
                logger.warning(
                    "escalation webhook attempt %d/%d got status=%d, retrying",
                    attempt + 1, MAX_ATTEMPTS, response.status_code,
                )
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS[attempt])
        logger.warning("escalation webhook: all %d attempts failed", MAX_ATTEMPTS)
        return False
    finally:
        if owns_client:
            await client.aclose()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_notifications.py -v`
Expected: all 14 tests PASS (7 from Task 2 + 7 new)

- [ ] **Step 5: Commit**

```bash
git add agent/tools/notifications.py tests/test_notifications.py
git commit -m "Phase 11: add notify_escalation delivery with retry/backoff

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Wire `notify_escalation` into `create_handoff_packet`

**Files:**
- Modify: `agent/tools/escalation.py` (imports at top; new `mark_notified` after `log_escalation` at line 226; `create_handoff_packet` at lines 228-243)
- Test: `tests/test_escalation.py`

**Interfaces:**
- Consumes: `notify_escalation` (Task 3, `agent/tools/notifications.py`).
- Produces: `mark_notified(escalation_id: int, delivered: bool, notified_at: str | None = None) -> None`. `create_handoff_packet`'s signature and return shape are unchanged — only its internal behavior gains a notify+record step.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_escalation.py` (near the existing `create_handoff_packet` test):

```python
@pytest.mark.asyncio
async def test_create_handoff_packet_marks_notified_true_on_successful_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_handoff_notify.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]

    fake_fields = HandoffFields(
        customer_intent="Wanted a refund",
        conversation_summary="Asked about a refund.",
        verified_account_info=f"Customer ID {customer_id}",
        actions_taken="None yet",
        sentiment="negative",
    )
    fake_response = MagicMock()
    fake_response.parsed_output = fake_fields
    fake_client = MagicMock()
    fake_client.messages.parse = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(escalation, "notify_escalation", AsyncMock(return_value=True))

    packet = await create_handoff_packet(
        customer_id,
        [{"role": "user", "content": "I want a refund"}],
        "explicit request for a human",
        client=fake_client,
    )

    with mock_db.get_connection() as conn:
        row = conn.execute(
            "SELECT notified, notified_at FROM escalations WHERE escalation_id = ?", (packet["escalation_id"],)
        ).fetchone()
    assert row["notified"] == 1
    assert row["notified_at"] is not None


@pytest.mark.asyncio
async def test_create_handoff_packet_still_returns_and_logs_when_notify_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_handoff_notify_fail.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]

    fake_fields = HandoffFields(
        customer_intent="Wanted a refund",
        conversation_summary="Asked about a refund.",
        verified_account_info=f"Customer ID {customer_id}",
        actions_taken="None yet",
        sentiment="negative",
    )
    fake_response = MagicMock()
    fake_response.parsed_output = fake_fields
    fake_client = MagicMock()
    fake_client.messages.parse = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(
        escalation, "notify_escalation", AsyncMock(side_effect=RuntimeError("webhook host unreachable"))
    )

    packet = await create_handoff_packet(
        customer_id,
        [{"role": "user", "content": "I want a refund"}],
        "explicit request for a human",
        client=fake_client,
    )

    assert packet["reason"] == "explicit request for a human"
    with mock_db.get_connection() as conn:
        row = conn.execute(
            "SELECT notified, notified_at FROM escalations WHERE escalation_id = ?", (packet["escalation_id"],)
        ).fetchone()
    assert row["notified"] == 0
    assert row["notified_at"] is not None  # mark_notified still ran, just with delivered=False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_escalation.py -v -k notify`
Expected: FAIL — `AttributeError: <module 'agent.tools.escalation'> does not have the attribute 'notify_escalation'` (monkeypatch target doesn't exist yet)

- [ ] **Step 3: Wire it up in `agent/tools/escalation.py`**

Add to the imports at the top of the file (after `from typing import Any, Literal`):

```python
import logging
```

And add this import alongside the existing `agent.tools.summary`/`data.mock_db` imports:

```python
from agent.tools.notifications import notify_escalation
```

Add a module logger, right after the imports (matching `agent/core.py`'s convention):

```python
logger = logging.getLogger("agent.tools.escalation")
```

Add `mark_notified` immediately after `log_escalation` (which ends at line 226):

```python
def mark_notified(escalation_id: int, delivered: bool, notified_at: str | None = None) -> None:
    """Record whether notify_escalation actually delivered this handoff to
    the automation platform. Always called after log_escalation, whether or
    not delivery succeeded — the escalations table is the durable record of
    both what happened and whether a human was actually told.
    """
    notified_at = notified_at or datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE escalations SET notified = ?, notified_at = ? WHERE escalation_id = ?",
            (int(delivered), notified_at, escalation_id),
        )
```

Replace `create_handoff_packet`'s body:

```python
async def create_handoff_packet(
    customer_id: str,
    messages: list[dict[str, Any]],
    reason: str,
    client: anthropic.AsyncAnthropic | None = None,
) -> dict[str, Any]:
    """Assemble a structured handoff packet, persist it, and notify an
    external automation platform (Phase 11) — see agent/tools/notifications.py.

    Not a tool the model calls itself — see the module docstring. Returns
    the full packet, including its escalation_id, for the transport layer
    to relay (e.g. print a transfer notice). Notification delivery never
    affects this return value — persisting the packet must not depend on
    whether anyone was actually told about it.
    """
    fields = await _infer_handoff_fields(customer_id, messages, client=client)
    escalation_id = log_escalation(customer_id, reason, fields)
    packet = {"escalation_id": escalation_id, "reason": reason, **fields.model_dump()}

    try:
        delivered = await notify_escalation(packet)
    except Exception:  # noqa: BLE001 — a broken webhook must never break escalation
        logger.exception("notify_escalation raised unexpectedly")
        delivered = False
    mark_notified(escalation_id, delivered)

    return packet
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_escalation.py -v`
Expected: all tests pass, including the 2 new ones (existing `test_create_handoff_packet_infers_fields_and_logs` still passes too — it doesn't monkeypatch `notify_escalation`, so it hits the real function, which is a no-op returning `False` since `ESCALATION_WEBHOOK_URL` isn't set in the test environment).

- [ ] **Step 5: Commit**

```bash
git add agent/tools/escalation.py tests/test_escalation.py
git commit -m "Phase 11: notify_escalation wired into create_handoff_packet

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Docs — PROJECT_PLAN.md, PROGRESS.md, .env.example, README.md, and final verification

**Files:**
- Modify: `PROJECT_PLAN.md` (insert new Phase 11 section between the end of Phase 10 and `## Rough effort sizing`; add a row to the effort-sizing table; add `notifications.py` to the repo-structure tree)
- Modify: `PROGRESS.md` (new row)
- Modify: `.env.example` (new env vars)
- Modify: `README.md` (new Phase 11 write-up section, appended after Phase 9's)

**Interfaces:**
- Consumes: nothing (docs only).
- Produces: nothing consumed by other tasks — this is the last task.

- [ ] **Step 1: Add `notifications.py` to `PROJECT_PLAN.md`'s repo tree**

In the `## Repo structure` code block, change:

```
    tools/
      orders.py
      policy_rag.py
      escalation.py
      scheduling.py
      refunds.py
      summary.py
```

to:

```
    tools/
      orders.py
      policy_rag.py
      escalation.py
      scheduling.py
      refunds.py
      summary.py
      notifications.py
```

- [ ] **Step 2: Insert the Phase 11 section into `PROJECT_PLAN.md`**

Between the blank line after Phase 10's last bullet (`- **Deploy:** Dockerize, document env vars, write the README.`) and the `---` that precedes `## Rough effort sizing`, insert:

```markdown
## Phase 11 — AI automation: escalation notifications (n8n)

**Independent of Phase 10** — this phase depends only on Phase 4 (ticket triage &
escalation) being done, not on Phase 10's guardrails work. This is a deliberate,
explicit exception to this plan's normal "one phase at a time, in order" rule
(CLAUDE.md rule 1), made at the project owner's direction so Phase 10 stays untouched
while this phase is built. See `docs/superpowers/specs/2026-09-03-phase-11-escalation-notifications-design.md`
for the full design rationale.

- Today, `create_handoff_packet` (Phase 4) only writes the handoff packet to SQLite —
  no human is ever actually told an escalation happened. Close that gap with a real
  outbound notification.
- New `agent/tools/notifications.py`: `notify_escalation(packet)` POSTs a redacted,
  optionally HMAC-signed copy of the handoff packet to `ESCALATION_WEBHOOK_URL` (an n8n
  webhook trigger), with a bounded retry+backoff on transient failures. Unconfigured →
  silent no-op, same optional-by-default convention as `TTS_BACKEND`/`EMBEDDING_BACKEND`.
- Redaction here is narrow and self-contained (emails, phone-like and card-like digit
  runs, in the packet's free-text fields only) — explicitly not Phase 10's eventual real
  PII pipeline (`guardrails/pii.py` stays an untouched stub), just enough to not ship
  unredacted customer text to a third-party webhook by default.
- `escalations` table gains `notified`/`notified_at` columns — a durable record of
  whether a human was actually told, not just that an escalation happened.
- No changes to `agent/core.py`, `agent/session.py`, or anything under `transport/` —
  the entire integration lives inside `agent/tools/`.

**Checkpoint:** automated — all new tests pass offline (redaction, signing, retry/backoff,
and the `create_handoff_packet` integration), no real n8n instance needed. Manual — run
n8n locally via Docker, wire a Webhook-trigger node to a Slack (or console) output, set
`ESCALATION_WEBHOOK_URL`, run `transport/text_cli.py`, trigger a real escalation, and
confirm the notification arrives with a legible, redacted payload.

---

```

- [ ] **Step 3: Add a row to `PROJECT_PLAN.md`'s effort-sizing table**

Change:

```
| 10 — Hardening | L |
```

to:

```
| 10 — Hardening | L |
| 11 — AI automation (n8n) | M |
```

- [ ] **Step 4: Add the Phase 11 row to `PROGRESS.md`**

After the Phase 10 row, add:

```
| 11 | AI automation: escalation notifications (n8n) | Done | 2026-09-03 | `agent/tools/notifications.py` (redact → sign → POST with retry/backoff to `ESCALATION_WEBHOOK_URL`) wired into `create_handoff_packet`; `escalations` gains `notified`/`notified_at`. Independent of Phase 10 by explicit project-owner direction (CLAUDE.md rule 1 exception, documented in the phase's own section). All automated tests pass; the manual "confirm it actually reaches a real n8n instance" checkpoint is outstanding, same honest-limitation convention as Phases 7-9's real-mic/real-call checks. |
```

- [ ] **Step 5: Add env vars to `.env.example`**

Append:

```
# Phase 11 — escalation notifications (n8n). Optional: unset means a silent
# no-op, same as every other optional integration in this project.
ESCALATION_WEBHOOK_URL=
ESCALATION_WEBHOOK_SECRET=
```

- [ ] **Step 6: Append the Phase 11 write-up to `README.md`**

After Phase 9's section (ending `"...a real call to the Twilio number went through and worked."`), append:

```markdown

---

## Phase 11 — AI automation: escalation notifications, n8n (Done)

`create_handoff_packet` (Phase 4) used to only write a handoff packet to SQLite — no
human was ever actually told an escalation happened. This phase closes that gap with a
real outbound notification to an automation platform, and is deliberately **independent
of Phase 10** — an explicit, documented exception to this project's usual "one phase at
a time, in order" rule, made at the project owner's direction; see
`docs/superpowers/specs/2026-09-03-phase-11-escalation-notifications-design.md` for the
full design rationale (approaches considered, why n8n, why not a durable queue).

### `agent/tools/notifications.py` (new)

`notify_escalation(packet)` — a plain `async` function, no new architectural layer:
redacts the packet's free-text fields (emails, phone-like and card-like digit runs —
deliberately narrow, not Phase 10's eventual real PII pipeline in `guardrails/pii.py`,
which stays an untouched stub), signs the serialized body with HMAC-SHA256 if
`ESCALATION_WEBHOOK_SECRET` is set (the outbound mirror of `transport/telephony.py`'s
inbound `X-Twilio-Signature` verification), and POSTs it to `ESCALATION_WEBHOOK_URL`
with up to 3 attempts (5s timeout each, 0.5s/1.5s backoff, retrying only on a
connection/timeout error or a 5xx — a 4xx is treated as permanent and not retried).
`ESCALATION_WEBHOOK_URL` unset is a normal working state: silent no-op, no HTTP call at
all — the same optional-by-default convention `TTS_BACKEND`/`EMBEDDING_BACKEND` already
use. Never raises, by design — a broken webhook must never affect the escalation itself.

### `agent/tools/escalation.py` — wired at the handoff, not the transport

`create_handoff_packet` gets one new step, right after `log_escalation` persists the
row: call `notify_escalation(packet)`, then record the outcome via the new
`mark_notified(escalation_id, delivered)`. The packet is still returned and still logged
even if notification fails or raises — persistence never depends on delivery succeeding.
No signature change, so its callers (`agent/session.py::run_turn`,
`transport/pipecat_processors.py`'s DTMF handler) needed zero changes — CLAUDE.md rule
5's decoupling holds exactly: nothing in `transport/`, `agent/core.py`, or
`agent/session.py` changed for this phase.

### `data/mock_db.py` — `notified`/`notified_at` columns

Two new columns on the existing `escalations` table, added the same way every prior
phase has extended the schema (a `CREATE TABLE IF NOT EXISTS` edit — no migration system
in this project). Run `python -m data.mock_db` to pick them up in a local dev DB.

### Setting up n8n locally (for the manual checkpoint)

1. `docker run -it --rm -p 5678:5678 n8nio/n8n` (or `docker compose`, if you already run
   one elsewhere).
2. In the n8n editor, add a **Webhook** node (POST, e.g. path `/escalation`), and copy
   its "Test URL" or "Production URL" into `ESCALATION_WEBHOOK_URL` in `.env`.
3. (Optional but recommended) Add a **Code** node right after the Webhook node that
   recomputes the HMAC-SHA256 of the raw body using the same secret you put in
   `ESCALATION_WEBHOOK_SECRET`, and compares it to the incoming `X-Signature-256`
   header — reject the workflow (or route to an error branch) on a mismatch.
4. Wire the Webhook node to whatever should actually notify a human — a **Slack** node
   posting the `reason`/`customer_intent`/`conversation_summary` fields into a channel
   is the natural choice; a plain **NoOp**/console output node is enough to just verify
   delivery.
5. Activate the workflow.

### Tests

`tests/test_notifications.py` (new, 14 tests, all offline via `pytest-httpx` — mirrors
`tests/test_tts.py`'s pattern exactly): redaction (email/phone/card-like masking,
ordinary text untouched), HMAC signing, deterministic serialization, no-op with no
webhook URL configured, success on the first attempt, a successful retry after one
transient failure, exhausting all 3 attempts on persistent failure, a 4xx not being
retried, and the signature header present/absent correctly.

`tests/test_escalation.py`: 2 new cases confirming `create_handoff_packet` records
`notified=1` on a successful delivery and `notified=0` (while still returning and
logging the packet) when `notify_escalation` raises.

`tests/test_mock_db.py`: 1 new case confirming the seeded `escalations` row defaults to
`notified=0`, `notified_at=NULL`.

### Checkpoint result

All new tests pass (17 new: 14 in `test_notifications.py`, 2 in `test_escalation.py`, 1
in `test_mock_db.py`), and the full existing suite is unaffected. The manual checkpoint
— actually running n8n locally and confirming a real escalation notification arrives —
is outstanding, same honest-limitation convention as Phase 7-9's real-mic/real-call
checks: the automated half is fully verified here; the hands-on half needs a real n8n
instance running, which is yours to do.
```

- [ ] **Step 7: Run the full test suite**

Run: `pytest -v`
Expected: every pre-existing test still passes (network/API-key-gated ones skip as usual without real credentials), plus all new Phase 11 tests pass.

- [ ] **Step 8: Commit**

```bash
git add PROJECT_PLAN.md PROGRESS.md .env.example README.md
git commit -m "Phase 11: docs — PROJECT_PLAN.md, PROGRESS.md, README.md, .env.example

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** every component in the spec (redaction, signing, delivery/retry,
`mark_notified`, `create_handoff_packet` integration, schema columns, docs, checkpoint)
maps to a task above. The spec's "Next step" (write plan via `writing-plans`) is this
document.

**Placeholder scan:** no TBD/TODO; every step has literal, complete code or exact prose
to insert.

**Type consistency:** `notify_escalation(packet: dict[str, Any], *, client: httpx.AsyncClient | None = None) -> bool` is defined once in Task 3 and consumed with that exact signature (positional `packet` only, no `client` kwarg needed) in Task 4. `redact_packet`/`sign_payload`/`serialize_packet` (Task 2) are consumed with matching names/signatures in Task 3's implementation and tests. `mark_notified(escalation_id: int, delivered: bool, notified_at: str | None = None) -> None` (Task 4) matches `log_escalation`'s existing parameter style exactly.
