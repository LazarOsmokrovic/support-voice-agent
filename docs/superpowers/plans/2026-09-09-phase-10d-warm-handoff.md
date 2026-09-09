# Phase 10d — Warm Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transfer a live Twilio call to a human agent, whispering the Phase 4 handoff packet to that human before bridging, so they answer with full context instead of a cold transfer.

**Architecture:** The call is inside a bidirectional Media Stream, so Pipecat cannot emit TwiML. The only lever is the Twilio REST API redirecting the live call off the stream into `<Dial><Number url="…">`, where `url` is Twilio's built-in whisper. Twilio stays confined to `transport/telephony.py`; the shared processor learns nothing about it beyond an injected `on_escalation` callback.

**Tech Stack:** Python 3.12+, FastAPI, `twilio` (REST client + `RequestValidator` + `twilio.twiml.voice_response`), Pipecat, pytest + pytest-asyncio, `httpx` for endpoint tests.

**Spec:** `docs/superpowers/specs/2026-09-09-phase-10d-warm-handoff-design.md`

## Global Constraints

- **Exactly ONE change under `agent/`, and no more:** an optional `escalation_packet: dict[str, Any] | None = None` field on `TurnOutcome`, populated where the packet already exists at `agent/session.py:337-346`. This is required, not cosmetic — without it the model-driven escalation path has no packet to transfer and the human hears the generic fallback, so the common path would carry less context than the rare DTMF one. It does not violate CLAUDE.md rule 5: the field is the same shape as the existing `notice`, and `TurnOutcome`'s docstring states its purpose is data for a transport to render. Nothing in `agent/` learns Twilio exists. **Any other `agent/` edit means the seam is wrong — stop and report.**
- **Twilio imports live ONLY in `transport/telephony.py`.** `transport/pipecat_processors.py` is shared with the local-mic pipeline (`transport/pipeline.py`) and must never import Twilio. This is why `on_escalation` is injected rather than called directly.
- **Both escalation paths must fire the transfer:** the model-driven path (`_handle_final_transcript`, via `run_turn`) and the DTMF path (`_handle_dtmf_escalation`, which bypasses `run_turn` entirely). A transfer wired into only one leaves "press 0 for a human" doing nothing — the worst failure, since that button exists for when the AI is already failing.
- **`/whisper` and `/transfer-status` MUST validate `X-Twilio-Signature`**, exactly as `/voice` does at `transport/telephony.py:82-86`. Without it anyone can read a customer's handoff briefing by guessing a URL. The signature check is the only access control on these endpoints.
- **`HUMAN_AGENT_NUMBER` unset ⇒ no transfer, no error.** Optional-by-default, matching `ESCALATION_WEBHOOK_URL`. The agent falls back to today's behaviour (speak the notice, end the call).
- **A failed transfer must never drop the call.** Every failure path returns `False` and lets the existing notice-and-end behaviour run.
- **The transfer is idempotent per call.** A model escalation immediately followed by a DTMF press must issue exactly one redirect.
- **`timeout="20"`** on `<Dial>` (Twilio allows 5–600, defaults 30).
- **No live API calls in any test step.** No Twilio account, no network. The REST client is faked.
- **Never hard-code a seeded literal** (order ID, customer ID, email, phone, tracking number) in a test assertion — read it from `data/mock_db.py`. Two Criticals in this project came from exactly that.
- **House style:** `from __future__ import annotations`, docstrings explaining WHY. `observability/turn_log.py` is the reference.
- Run tests with `python -m pytest` from the repo root. Baseline: **292 passed, 3 skipped**.

## File structure map

| File | Responsibility |
|---|---|
| `transport/telephony.py` (modify) | Everything Twilio-specific: the `TRANSFERS` and `SESSIONS` registries, `render_whisper`, `remember_transfer`, `build_transfer_twiml`, `transfer_to_human`, the `/whisper` and `/transfer-status` endpoints, `resolve_session`/`forget_session`, and wiring `on_escalation` into `build_pipeline`. |
| `transport/pipecat_processors.py` (modify) | `build_pipeline` gains `on_escalation`; `ClaudeTurnProcessor` awaits it at both trigger points. No Twilio import. |
| `agent/session.py` (modify — Task 3 Step 0 ONLY) | One optional `escalation_packet` field on `TurnOutcome`, populated at the existing escalation site. The single permitted `agent/` change; see Global Constraints for why it is required. |
| `tests/test_telephony.py` (modify) | Tasks 1, 2, 4, 5, 6. Extends the existing hand-rolled `_sign()` helper rather than importing `RequestValidator`, so a bug in this project's *use* of it is not masked. |
| `tests/test_pipecat_processors.py` (modify) | Task 3 — both escalation paths, the no-callback case, and a raising callback. |
| `tests/test_session.py` (modify) | Task 3 Step 0 — the escalated turn exposes its packet. |
| `.env.example`, `README.md`, `PROGRESS.md` (modify) | `HUMAN_AGENT_NUMBER`, `TWILIO_CALLER_ID`, `TWILIO_ACCOUNT_SID`, the Phase 10d section, the progress row. |

---

### Task 1: The transfer registry and whisper text

**Files:**
- Modify: `transport/telephony.py` (add below `_public_hostname`, around line 73)
- Test: `tests/test_telephony.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `TRANSFERS: dict[str, PendingTransfer]`, `PendingTransfer` (frozen dataclass: `escalation_id: int | None`, `whisper: str`, `session_id: str`), `render_whisper(packet: dict) -> str`, `remember_transfer(call_sid, packet, session_id) -> PendingTransfer`, `TRANSFER_TIMEOUT_SECONDS = 20`.

- [ ] **Step 1: Write the failing tests**

```python
def test_render_whisper_briefs_the_human_from_the_packet():
    """The whisper is the entire point of a warm handoff — the human must
    hear who is waiting and why before the line opens."""
    packet = {
        "escalation_id": 42,
        "reason": "explicit request for a human",
        "customer_intent": "wants a refund outside the return window",
        "conversation_summary": "Asked about order status, then became frustrated.",
        "verified_account_info": "Maria Gonzalez, order 112-3487561-2938471",
        "actions_taken": "Looked up the order; explained the 30-day policy.",
        "sentiment": "negative",
    }
    whisper = telephony.render_whisper(packet)
    assert "42" in whisper
    assert "explicit request for a human" in whisper
    assert "refund outside the return window" in whisper
    assert "negative" in whisper


def test_render_whisper_survives_a_packet_missing_fields():
    """create_handoff_packet infers its fields from a model call, so a
    degraded packet is possible. A thin briefing beats a 500 that leaves
    the human hearing silence."""
    whisper = telephony.render_whisper({"escalation_id": 7})
    assert "7" in whisper
    assert whisper.strip()


def test_remember_transfer_stores_the_whisper_for_the_endpoint_to_read():
    """The /whisper endpoint runs in a SEPARATE HTTP request from the
    redirect, so the text has to outlive the call that built it. Stashing
    it here avoids re-reading the packet from SQLite, which would have
    meant adding a query to agent/ — forbidden this phase."""
    telephony.TRANSFERS.clear()
    packet = {"escalation_id": 9, "customer_intent": "billing question"}
    stored = telephony.remember_transfer("CA-test-sid", packet, "sess-1")
    assert telephony.TRANSFERS["CA-test-sid"] is stored
    assert stored.escalation_id == 9
    assert stored.session_id == "sess-1"
    assert "billing question" in stored.whisper
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_telephony.py -q -k "whisper or remember_transfer"`
Expected: FAIL — `AttributeError: module 'transport.telephony' has no attribute 'render_whisper'`

- [ ] **Step 3: Implement**

```python
# How long <Dial> rings the human before giving up. Twilio allows 5-600 and
# defaults to 30; 20 is long enough for a real pickup and short enough that a
# customer already waiting on hold is not abandoned to silence.
TRANSFER_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class PendingTransfer:
    """One in-flight transfer, keyed by the customer's call SID.

    Exists because a transfer spans three separate HTTP interactions — the
    REST redirect, Twilio's request to /whisper, and its request to
    /transfer-status — and they need to share state the packet already has.
    """

    escalation_id: int | None
    whisper: str
    session_id: str


# Keyed by call_sid. In-process on purpose: this project runs one uvicorn
# worker (see __main__ at the foot of this file), and a distributed store
# would be machinery a mock project cannot justify. Entries are removed when
# the transfer resolves, so it cannot grow without bound.
TRANSFERS: dict[str, PendingTransfer] = {}


def render_whisper(packet: dict[str, Any]) -> str:
    """Turn a handoff packet into ~15 seconds of spoken briefing.

    Short on purpose: the human is holding a ringing phone and the customer
    is waiting on the other leg. Reason and intent come first because they
    are what decides how the human opens the conversation.

    The packet's free text was already redacted by guardrails/pii.py at
    create_handoff_packet (Phase 10a), so contact details arrive masked while
    the order ID survives — the right split, since the order ID is the thing
    that lets the human actually act.
    """
    parts = [f"Handoff {packet.get('escalation_id', 'unknown')}."]
    for label, key in (
        ("Reason", "reason"),
        ("Customer wants", "customer_intent"),
        ("Account", "verified_account_info"),
        ("Already done", "actions_taken"),
        ("Sentiment", "sentiment"),
    ):
        value = packet.get(key)
        if value:
            parts.append(f"{label}: {value}.")
    return " ".join(parts)


def remember_transfer(call_sid: str, packet: dict[str, Any], session_id: str) -> PendingTransfer:
    """Stash what /whisper and /transfer-status will need, before the
    redirect is issued."""
    pending = PendingTransfer(
        escalation_id=packet.get("escalation_id"),
        whisper=render_whisper(packet),
        session_id=session_id,
    )
    TRANSFERS[call_sid] = pending
    return pending
```

Add `from dataclasses import dataclass` and `from typing import Any` to the imports.

- [ ] **Step 4: Run and watch them pass**

Run: `python -m pytest tests/test_telephony.py -q -k "whisper or remember_transfer"`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add transport/telephony.py tests/test_telephony.py
git commit -m "Phase 10d Task 1: transfer registry and whisper briefing text"
```

---

### Task 2: `transfer_to_human` — the REST redirect

**Files:**
- Modify: `transport/telephony.py` (add below `remember_transfer`)
- Test: `tests/test_telephony.py` (append)

**Interfaces:**
- Consumes: `TRANSFERS`, `PendingTransfer`, `remember_transfer`, `TRANSFER_TIMEOUT_SECONDS` (Task 1).
- Produces: `async def transfer_to_human(call_sid: str, packet: dict, session_id: str, *, client=None) -> bool`, and `build_transfer_twiml(escalation_id, human_number, caller_id) -> str`.

**Why a `client` parameter here and nowhere else:** the Twilio REST client is the one dependency these tests cannot fake by patching a constructor (unlike Anthropic, whose four call sites Phase 10c intercepts that way). Injecting it keeps the test honest without a global patch.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_transfer_twiml_dials_the_human_with_a_whisper_url():
    """Asserted as parsed XML, not string matching — a test that greps for
    a substring passes on malformed TwiML that Twilio would reject."""
    import xml.etree.ElementTree as ET

    xml = telephony.build_transfer_twiml(42, "+15551234567", "+15559876543")
    root = ET.fromstring(xml)
    dial = root.find("Dial")
    assert dial is not None
    assert dial.get("timeout") == "20"
    assert "/transfer-status" in dial.get("action")
    number = dial.find("Number")
    assert number.text == "+15551234567"
    assert "/whisper" in number.get("url")
    assert "escalation_id=42" in number.get("url")


@pytest.mark.asyncio
async def test_transfer_to_human_issues_the_redirect(monkeypatch):
    monkeypatch.setenv("HUMAN_AGENT_NUMBER", "+15551234567")
    monkeypatch.setenv("TWILIO_CALLER_ID", "+15559876543")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()

    updated = {}

    class _FakeCalls:
        def __init__(self, sid):
            self.sid = sid

        def update(self, **kwargs):
            updated.update({"sid": self.sid, **kwargs})

    fake_client = type("C", (), {"calls": staticmethod(lambda sid: _FakeCalls(sid))})()

    ok = await telephony.transfer_to_human("CA-1", {"escalation_id": 42}, "sess-1", client=fake_client)

    assert ok is True
    assert updated["sid"] == "CA-1"
    assert "+15551234567" in updated["twiml"]
    assert telephony.TRANSFERS["CA-1"].escalation_id == 42


@pytest.mark.asyncio
async def test_transfer_to_human_is_a_no_op_without_a_configured_number(monkeypatch):
    """Optional-by-default, exactly like ESCALATION_WEBHOOK_URL. A developer
    with no human agent configured must still get a working agent."""
    monkeypatch.delenv("HUMAN_AGENT_NUMBER", raising=False)
    called = False

    def _boom(sid):
        nonlocal called
        called = True
        raise AssertionError("must not touch Twilio without a number configured")

    fake_client = type("C", (), {"calls": staticmethod(_boom)})()
    assert await telephony.transfer_to_human("CA-1", {}, "s", client=fake_client) is False
    assert called is False


@pytest.mark.asyncio
async def test_transfer_to_human_returns_false_when_twilio_rejects(monkeypatch):
    """A failed transfer must degrade to today's behaviour, never drop the
    call. The caller keeps the agent; it does not raise."""
    monkeypatch.setenv("HUMAN_AGENT_NUMBER", "+15551234567")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")

    class _FailingCalls:
        def update(self, **kwargs):
            raise RuntimeError("call is no longer in-progress")

    fake_client = type("C", (), {"calls": staticmethod(lambda sid: _FailingCalls())})()
    assert await telephony.transfer_to_human("CA-1", {"escalation_id": 1}, "s", client=fake_client) is False


@pytest.mark.asyncio
async def test_transfer_to_human_fires_only_once_per_call(monkeypatch):
    """A model escalation immediately followed by a DTMF press must not
    redirect twice — the second would land on a call already dialling."""
    monkeypatch.setenv("HUMAN_AGENT_NUMBER", "+15551234567")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    calls = []

    class _Calls:
        def update(self, **kwargs):
            calls.append(kwargs)

    fake_client = type("C", (), {"calls": staticmethod(lambda sid: _Calls())})()

    first = await telephony.transfer_to_human("CA-1", {"escalation_id": 1}, "s", client=fake_client)
    second = await telephony.transfer_to_human("CA-1", {"escalation_id": 2}, "s", client=fake_client)

    assert first is True
    assert second is False
    assert len(calls) == 1
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_telephony.py -q -k "transfer_to_human or transfer_twiml"`
Expected: FAIL — `AttributeError: module 'transport.telephony' has no attribute 'build_transfer_twiml'`

- [ ] **Step 3: Implement**

```python
def build_transfer_twiml(escalation_id: int | None, human_number: str, caller_id: str | None) -> str:
    """The TwiML that replaces the Media Stream.

    <Number url=...> is Twilio's whisper: that TwiML runs on the CALLED
    party's end after they answer but before the two legs are bridged, so the
    human hears the briefing and the customer does not. It may not contain
    <Dial>.

    <Dial action=...> hands the parent call to /transfer-status when the dial
    ends, which is what makes the no-answer path possible — without it, the
    customer would simply be hung up on.

    The leading <Say> matters more than it looks: issuing the redirect cuts
    the Media Stream, which can truncate the agent's own spoken notice
    mid-word. This guarantees the customer hears something before ringing.
    """
    response = VoiceResponse()
    response.say("Connecting you now. Please hold.")
    dial = Dial(
        action=f"https://{_public_hostname()}/transfer-status",
        timeout=TRANSFER_TIMEOUT_SECONDS,
        caller_id=caller_id,
    )
    dial.number(human_number, url=f"https://{_public_hostname()}/whisper?escalation_id={escalation_id}")
    response.append(dial)
    return str(response)


async def transfer_to_human(
    call_sid: str,
    packet: dict[str, Any],
    session_id: str,
    *,
    client: Any | None = None,
) -> bool:
    """Redirect the customer's live call into a whispered <Dial>.

    Returns True if the redirect was issued, False otherwise. NEVER raises:
    every caller treats False as "carry on as before", so a broken transfer
    costs the customer a handoff, not the call.
    """
    human_number = os.getenv("HUMAN_AGENT_NUMBER")
    if not human_number:
        print("(no HUMAN_AGENT_NUMBER configured — skipping transfer)")
        return False

    if call_sid in TRANSFERS:
        print(f"(transfer already in flight for {call_sid} — ignoring duplicate)")
        return False

    remember_transfer(call_sid, packet, session_id)
    try:
        rest = client or Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        twiml = build_transfer_twiml(packet.get("escalation_id"), human_number, os.getenv("TWILIO_CALLER_ID"))
        rest.calls(call_sid).update(twiml=twiml)
    except Exception as exc:  # noqa: BLE001 — a failed transfer must never drop the call
        TRANSFERS.pop(call_sid, None)
        print(f"(transfer to {human_number} failed: {exc})")
        return False
    print(f"(transferring {call_sid} to {human_number})")
    return True
```

Add to imports: `from twilio.rest import Client` and `from twilio.twiml.voice_response import Dial`.

- [ ] **Step 4: Run and watch them pass**

Run: `python -m pytest tests/test_telephony.py -q -k "transfer_to_human or transfer_twiml"`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add transport/telephony.py tests/test_telephony.py
git commit -m "Phase 10d Task 2: transfer_to_human REST redirect with whisper TwiML"
```

---

### Task 3: The `on_escalation` seam — both trigger points

**Files:**
- Modify: `transport/pipecat_processors.py` — `ClaudeTurnProcessor.__init__`, `_handle_dtmf_escalation`, `_handle_final_transcript`, `build_pipeline`
- Test: `tests/test_pipecat_processors.py` (append)

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (deliberately — this file must not import Twilio).
- Produces: `build_pipeline(transport, session, *, mute_mic_during_tts=False, on_escalation=None)`; `ClaudeTurnProcessor(session=..., on_escalation=None, **kwargs)`. `on_escalation` has type `Callable[[dict[str, Any] | None], Awaitable[bool]] | None` and receives the handoff packet (or `None` when the model-driven path has no packet to hand over).

**This is the task most likely to be got wrong.** There are TWO paths and both must call the callback. `_handle_dtmf_escalation` bypasses `run_turn` entirely — it is Phase 9's deterministic safety net — so wiring only `_handle_final_transcript` would leave "press 0 for a human" printing a notice and doing nothing.

**It also carries this phase's single permitted `agent/` change** (Step 0 below), without which the model-driven path transfers with no context.

- [ ] **Step 0: Expose the handoff packet on `TurnOutcome`**

Write this test first, in `tests/test_session.py`:

```python
@pytest.mark.asyncio
async def test_an_escalated_turn_exposes_the_handoff_packet(monkeypatch):
    """Phase 10d needs it: the transport transfers the call and whispers the
    packet to the human. run_turn built the packet and then dropped it,
    leaving the transport nothing to hand over — so the ordinary escalation
    path would have briefed the human with nothing while the rarer DTMF path
    briefed them fully."""
    session = create_session("CUST-1001", client=_fake_client("Let me get someone."))
    monkeypatch.setattr(
        escalation, "check_escalation", AsyncMock(return_value="explicit request for a human")
    )
    monkeypatch.setattr(
        escalation,
        "create_handoff_packet",
        AsyncMock(return_value={"escalation_id": 3, "customer_intent": "wants a human"}),
    )

    outcome = await run_turn(session, "get me a person")

    assert outcome.end_reason == "escalated"
    assert outcome.escalation_packet is not None
    assert outcome.escalation_packet["escalation_id"] == 3
```

Use whatever fake-client helper `tests/test_session.py` already defines rather than inventing one; match the file's existing style.

Run it: `python -m pytest tests/test_session.py -q -k handoff_packet` → FAIL with `AttributeError: 'TurnOutcome' object has no attribute 'escalation_packet'`.

Then add the field to `TurnOutcome` (`agent/session.py:212-223`), directly below `notice` so related fields sit together:

```python
    # Phase 10d: the transport needs the packet itself, not just the notice —
    # transport/telephony.py whispers it to the human agent before bridging
    # the call. Same purpose as `notice` above: data for a transport to
    # render. run_turn built this and discarded it before 10d.
    escalation_packet: dict[str, Any] | None = None
```

and populate it at the escalation site (`agent/session.py:337-346`), where `packet` is already in scope, by adding `escalation_packet=packet` to the `TurnOutcome(...)` construction. Note the `except` branch at line 342 leaves `packet` unbound — initialise `packet = None` before the `try` so a failed handoff still produces a valid outcome.

Re-run: PASS. Commit separately, since it is the one `agent/` change and should be reviewable on its own:

```bash
git add agent/session.py tests/test_session.py
git commit -m "Phase 10d: expose the handoff packet on TurnOutcome for the transport to whisper"
```

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_dtmf_escalation_fires_the_transfer_callback(monkeypatch):
    """Press 0 is the safety net for when the AI is already failing — it MUST
    transfer, not just announce a transfer."""
    fired = []

    async def _on_escalation(packet):
        fired.append(packet)
        return True

    session = create_session("CUST-1001")
    monkeypatch.setattr(
        escalation, "create_handoff_packet", AsyncMock(return_value={"escalation_id": 5})
    )
    processor = ClaudeTurnProcessor(session=session, on_escalation=_on_escalation, enable_direct_mode=True)
    await _started(processor)

    await processor.process_frame(InputDTMFFrame(button=KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    assert len(fired) == 1
    assert fired[0]["escalation_id"] == 5


@pytest.mark.asyncio
async def test_model_driven_escalation_fires_the_transfer_callback(monkeypatch):
    """The other path: run_turn decided to escalate."""
    fired = []

    async def _on_escalation(packet):
        fired.append(packet)
        return True

    session = create_session("CUST-1001")
    outcome = SimpleNamespace(
        reply="Connecting you with a human agent.",
        notice=None,
        warnings=[],
        llm_latency_seconds=0.0,
        ended=True,
        end_reason="escalated",
        escalation_packet={"escalation_id": 8},
    )
    monkeypatch.setattr(
        "transport.pipecat_processors.run_turn", AsyncMock(return_value=outcome)
    )
    processor = ClaudeTurnProcessor(session=session, on_escalation=_on_escalation, enable_direct_mode=True)
    await _started(processor)

    await processor.process_frame(_transcript("get me a human"), FrameDirection.DOWNSTREAM)

    assert len(fired) == 1


@pytest.mark.asyncio
async def test_no_callback_means_todays_behaviour_is_unchanged(monkeypatch):
    """transport/pipeline.py (local mic) passes no callback. It must behave
    exactly as it did before this phase."""
    session = create_session("CUST-1001")
    monkeypatch.setattr(
        escalation, "create_handoff_packet", AsyncMock(return_value={"escalation_id": 5})
    )
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(InputDTMFFrame(button=KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    assert any(isinstance(f, TextFrame) and "human agent" in f.text for f in sink.frames)


@pytest.mark.asyncio
async def test_a_raising_callback_never_breaks_the_call(monkeypatch):
    """Telephony failures must not crash the pipeline — the customer still
    hears the notice."""
    async def _on_escalation(packet):
        raise RuntimeError("twilio exploded")

    session = create_session("CUST-1001")
    monkeypatch.setattr(
        escalation, "create_handoff_packet", AsyncMock(return_value={"escalation_id": 5})
    )
    processor = ClaudeTurnProcessor(session=session, on_escalation=_on_escalation, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(InputDTMFFrame(button=KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    assert any(isinstance(f, TextFrame) for f in sink.frames)
```

Add `from types import SimpleNamespace` to that test file's imports.

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_pipecat_processors.py -q -k "callback or transfer"`
Expected: FAIL — `TypeError: ClaudeTurnProcessor.__init__() got an unexpected keyword argument 'on_escalation'`

- [ ] **Step 3: Implement**

In `ClaudeTurnProcessor.__init__`:

```python
    def __init__(self, *, session: Session, on_escalation: EscalationHook | None = None, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        # Injected by transport/telephony.py, absent for transport/pipeline.py.
        # This is how a real call transfer happens without this shared module
        # importing Twilio — the local-mic pipeline has no phone call to
        # transfer, and CLAUDE.md rule 5 keeps provider code out of here.
        self._on_escalation = on_escalation
```

Above the class:

```python
EscalationHook = Callable[[dict[str, Any] | None], Awaitable[bool]]
```

Add a helper on the class:

```python
    async def _fire_escalation_hook(self, packet: dict[str, Any] | None) -> bool:
        """Call the transport's transfer hook, swallowing anything it raises.

        A telephony failure must never crash the pipeline: the customer is
        mid-call and the notice still has to reach them.
        """
        if self._on_escalation is None:
            return False
        try:
            return await self._on_escalation(packet)
        except Exception as exc:  # noqa: BLE001 — a broken transfer must not drop the call
            print(f"(escalation hook failed: {exc})")
            return False
```

In `_handle_dtmf_escalation`, immediately before `print(notice)`:

```python
        await self._fire_escalation_hook(packet if escalation_id is not None else None)
```

(`packet` is already in scope from the `try` block; guard on `escalation_id` because the `except` path leaves it unbound.)

In `_handle_final_transcript`, inside the existing `if outcome.ended:` block, before pushing `EndFrame`:

```python
            if outcome.end_reason == "escalated":
                await self._fire_escalation_hook(outcome.escalation_packet)
```

In `build_pipeline`, add the parameter and pass it through:

```python
def build_pipeline(
    transport: BaseTransport,
    session: Session,
    *,
    mute_mic_during_tts: bool = False,
    on_escalation: EscalationHook | None = None,
) -> Pipeline:
```

and at the `ClaudeTurnProcessor(...)` construction inside it, add `on_escalation=on_escalation`.

Add `from collections.abc import Awaitable, Callable` and `from typing import Any` to imports.

- [ ] **Step 4: Run and watch them pass**

Run: `python -m pytest tests/test_pipecat_processors.py -q`
Expected: PASS — all existing tests plus the 4 new ones

- [ ] **Step 5: Commit**

```bash
git add transport/pipecat_processors.py tests/test_pipecat_processors.py
git commit -m "Phase 10d Task 3: on_escalation hook fired from both escalation paths"
```

---

### Task 4: The `/whisper` endpoint

**Files:**
- Modify: `transport/telephony.py` (add after the `/voice` endpoint)
- Test: `tests/test_telephony.py` (append)

**Interfaces:**
- Consumes: `TRANSFERS`, `PendingTransfer` (Task 1).
- Produces: `POST /whisper` returning `<Response><Say>…</Say></Response>`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_whisper_speaks_the_briefing_to_the_human(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42, "customer_intent": "refund dispute"}, "s1")

    url = "https://example.ngrok.app/whisper"
    params = {"CallSid": "CA-whisper-leg", "ParentCallSid": "CA-1"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/whisper", data=params, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 200
    assert "refund dispute" in response.text
    assert "<Say>" in response.text


@pytest.mark.asyncio
async def test_whisper_rejects_an_unsigned_request_and_leaks_nothing(monkeypatch):
    """This endpoint speaks a customer's handoff briefing aloud. The
    signature check is the ONLY access control on it — without it, anyone
    who guesses the URL can read intent, summary and account info."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42, "customer_intent": "refund dispute"}, "s1")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post(
            "/whisper", data={"ParentCallSid": "CA-1"}, headers={"X-Twilio-Signature": "wrong"}
        )

    assert response.status_code == 403
    assert "refund dispute" not in response.text


@pytest.mark.asyncio
async def test_whisper_falls_back_when_the_transfer_is_unknown(monkeypatch):
    """A process restart between redirect and whisper loses the registry.
    The human should still get a usable call, not silence."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()

    url = "https://example.ngrok.app/whisper"
    params = {"ParentCallSid": "CA-unknown"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/whisper", data=params, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 200
    assert "<Say>" in response.text
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_telephony.py -q -k whisper`
Expected: FAIL — 404, the route does not exist

- [ ] **Step 3: Implement**

```python
async def _validate_twilio_signature(request: Request, path: str) -> dict[str, str]:
    """Shared by every Twilio-facing endpoint. Extracted rather than repeated
    because /whisper and /transfer-status must not drift from /voice's
    checking — a weaker check on the endpoint that SPEAKS a customer's
    briefing would be the worst place to have one.
    """
    form = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(os.getenv("TWILIO_AUTH_TOKEN", ""))
    if not validator.validate(f"https://{_public_hostname()}{path}", dict(form), signature):
        raise HTTPException(status_code=403, detail="invalid Twilio request signature")
    return dict(form)


@app.post("/whisper")
async def whisper(request: Request) -> Response:
    """Spoken to the human agent only, after they answer and before the two
    legs are bridged. Twilio requests this via the `url` attribute on
    <Number>; the customer never hears it.
    """
    form = await _validate_twilio_signature(request, "/whisper")
    pending = TRANSFERS.get(form.get("ParentCallSid", ""))
    text = pending.whisper if pending else "A customer is waiting. No context is available for this transfer."
    response = VoiceResponse()
    response.say(text)
    return Response(content=str(response), media_type="application/xml")
```

Refactor `/voice` to use `_validate_twilio_signature(request, "/voice")` too, so there is one implementation.

- [ ] **Step 4: Run and watch them pass**

Run: `python -m pytest tests/test_telephony.py -q`
Expected: PASS — including the pre-existing `/voice` signature tests, which now exercise the shared helper

- [ ] **Step 5: Commit**

```bash
git add transport/telephony.py tests/test_telephony.py
git commit -m "Phase 10d Task 4: /whisper endpoint with shared signature validation"
```

---

### Task 5: The `/transfer-status` endpoint

**Files:**
- Modify: `transport/telephony.py` (add after `/whisper`)
- Test: `tests/test_telephony.py` (append)

**Interfaces:**
- Consumes: `TRANSFERS`, `_validate_twilio_signature` (Tasks 1, 4).
- Produces: `POST /transfer-status` returning `<Hangup/>` on success or `<Connect><Stream url="…?session=…"/></Connect>` on failure.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["busy", "no-answer", "failed", "canceled"])
async def test_transfer_status_returns_the_customer_to_the_agent(monkeypatch, status):
    """The human did not pick up. The customer has been holding — bringing
    them back to an agent that REMEMBERS the conversation is the whole point
    of passing the session id through."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42}, "sess-abc")

    url = "https://example.ngrok.app/transfer-status"
    params = {"CallSid": "CA-1", "DialCallStatus": status}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/transfer-status", data=params, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 200
    assert "<Stream" in response.text
    assert "session=sess-abc" in response.text
    assert "CA-1" not in telephony.TRANSFERS


@pytest.mark.asyncio
async def test_transfer_status_hangs_up_after_a_completed_transfer(monkeypatch):
    """The human answered and the call is over. Reconnecting the AI here
    would drop a finished conversation back onto a bot."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42}, "sess-abc")

    url = "https://example.ngrok.app/transfer-status"
    params = {"CallSid": "CA-1", "DialCallStatus": "completed"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/transfer-status", data=params, headers={"X-Twilio-Signature": signature})

    assert "<Hangup" in response.text
    assert "<Stream" not in response.text


@pytest.mark.asyncio
async def test_transfer_status_rejects_an_unsigned_request(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post(
            "/transfer-status",
            data={"CallSid": "CA-1", "DialCallStatus": "no-answer"},
            headers={"X-Twilio-Signature": "wrong"},
        )

    assert response.status_code == 403
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_telephony.py -q -k transfer_status`
Expected: FAIL — 404, the route does not exist

- [ ] **Step 3: Implement**

```python
# Every DialCallStatus that means the human did NOT take the call. Twilio's
# full set is completed/answered/busy/no-answer/failed/canceled; the first two
# mean the bridge happened and the conversation is over.
_DIAL_FAILED = frozenset({"busy", "no-answer", "failed", "canceled"})


@app.post("/transfer-status")
async def transfer_status(request: Request) -> Response:
    """Twilio requests this when <Dial> ends, and from here the action URL —
    not the original TwiML — controls the parent call.

    On a failed dial the customer is still on the line, having waited through
    the ringing. Reconnecting the Media Stream with the ORIGINAL session id
    means the agent resumes with full history and can apologise and offer a
    callback, rather than greeting them from scratch as a stranger.
    """
    form = await _validate_twilio_signature(request, "/transfer-status")
    call_sid = form.get("CallSid", "")
    pending = TRANSFERS.pop(call_sid, None)
    status = form.get("DialCallStatus", "")

    response = VoiceResponse()
    if status in _DIAL_FAILED and pending is not None:
        print(f"(transfer for {call_sid} ended as {status!r} — returning the caller to the agent)")
        connect = Connect()
        connect.stream(url=f"wss://{_public_hostname()}/media-stream?session={pending.session_id}")
        response.append(connect)
    else:
        response.hangup()
    return Response(content=str(response), media_type="application/xml")
```

- [ ] **Step 4: Run and watch them pass**

Run: `python -m pytest tests/test_telephony.py -q`
Expected: PASS (6 new tests here, plus everything prior)

- [ ] **Step 5: Commit**

```bash
git add transport/telephony.py tests/test_telephony.py
git commit -m "Phase 10d Task 5: /transfer-status returns the caller to the agent on no-answer"
```

---

### Task 6: Session resume, and wiring it together

**Files:**
- Modify: `transport/telephony.py` — `media_stream`
- Test: `tests/test_telephony.py` (append)

**Interfaces:**
- Consumes: `TRANSFERS`, `transfer_to_human` (Tasks 1, 2); `build_pipeline(..., on_escalation=...)` (Task 3).
- Produces: `SESSIONS: dict[str, Session]`, and `media_stream` accepting `?session=<id>`.

- [ ] **Step 1: Write the failing tests**

```python
def test_session_registry_resumes_a_known_session():
    """After a failed transfer the customer comes back on a NEW Media Stream.
    Without this they would meet a brand-new session that has forgotten the
    entire conversation — worse than never attempting the transfer."""
    telephony.SESSIONS.clear()
    session = create_session("CUST-1001", transport="telephony")
    telephony.SESSIONS[session.session_id] = session

    assert telephony.resolve_session(session.session_id) is session


def test_session_registry_falls_back_to_a_new_session_for_an_unknown_id():
    """A cold restart is worse than resuming, but far better than a 500 and
    a dropped call."""
    telephony.SESSIONS.clear()
    resumed = telephony.resolve_session("no-such-session")
    assert resumed is not None
    assert resumed.session_id != "no-such-session"


def test_session_registry_forgets_a_session_when_it_closes():
    """The registry is in-process and unbounded otherwise."""
    telephony.SESSIONS.clear()
    session = create_session("CUST-1001", transport="telephony")
    telephony.SESSIONS[session.session_id] = session
    telephony.forget_session(session.session_id)
    assert session.session_id not in telephony.SESSIONS
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_telephony.py -q -k session_registry`
Expected: FAIL — `AttributeError: module 'transport.telephony' has no attribute 'SESSIONS'`

- [ ] **Step 3: Implement**

```python
# Live sessions keyed by session_id, so a call that comes back from a failed
# transfer can resume the SAME conversation. In-process on purpose (one
# uvicorn worker); entries are removed by forget_session when the call ends.
SESSIONS: dict[str, Session] = {}


def resolve_session(session_id: str | None) -> Session:
    """Resume a session by id, or start a fresh one.

    An unknown id is not an error worth failing a live call over — the caller
    is on the phone right now. A cold start loses history; a 500 loses the
    customer.
    """
    if session_id and session_id in SESSIONS:
        print(f"(resuming session {session_id} after a failed transfer)")
        return SESSIONS[session_id]
    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    SESSIONS[session.session_id] = session
    return session


def forget_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)
```

Then in `media_stream`, replace the `create_session(...)` line and add the hook:

```python
    session = resolve_session(websocket.query_params.get("session"))

    async def _on_escalation(packet: dict[str, Any] | None) -> bool:
        if packet is None:
            return False
        return await transfer_to_human(call_sid, packet, session.session_id)

    pipeline = build_pipeline(transport, session, on_escalation=_on_escalation)
```

and after `close_session(session)` completes, add `forget_session(session.session_id)`.

Add `from agent.session import Session` to the imports.

- [ ] **Step 4: Run and watch them pass**

Run: `python -m pytest tests/test_telephony.py tests/test_pipecat_processors.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add transport/telephony.py tests/test_telephony.py
git commit -m "Phase 10d Task 6: session resume after a failed transfer, wired end to end"
```

---

### Task 7: Documentation and the honest end state

**Files:**
- Modify: `.env.example`, `README.md`, `PROGRESS.md`
- Test: none new — verification is the full suite

**Interfaces:**
- Consumes: everything. Produces no importable name.

- [ ] **Step 1: Add the new environment variables**

In `.env.example`, following the style of the existing entries:

```
# Phase 10d — warm handoff. The human agent's phone, E.164 format. Unset means
# no transfer is attempted and the agent behaves as it did before Phase 10d.
# Any E.164 number works: a US Twilio number (~$0.014/min) or a Voice SDK
# browser client tests the identical code path far more cheaply than
# international mobile termination (Serbian mobile is $0.8211/min).
HUMAN_AGENT_NUMBER=
# Caller ID presented on the outbound leg. MUST be a Twilio-owned number —
# the customer's own number cannot legally be presented.
TWILIO_CALLER_ID=
# Needed for the REST client that issues the transfer redirect.
TWILIO_ACCOUNT_SID=
```

- [ ] **Step 2: Write the README section**

Add a Phase 10d section covering: the mechanism (REST redirect off the Media Stream, `<Number url>` whisper, `<Dial action>` for the no-answer path); that both the model-driven and DTMF paths transfer; that `/whisper` and `/transfer-status` validate `X-Twilio-Signature` and that this is their only access control; the single-worker session-registry limitation; and the cost note above. State plainly that **the live checkpoint has not been run** and needs a funded Twilio key.

- [ ] **Step 3: Update PROGRESS.md**

Set the 10d row to Done with the date, matching the format and tone of the 10a/10b/10c rows. State the automated result and that the live transfer checkpoint has NOT been performed — this project has corrected two phases for claiming an unrun checkpoint.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: 292 passed + the new tests, 3 skipped, no regressions.

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md PROGRESS.md
git commit -m "Phase 10d Task 7: document the warm handoff and its limitations"
```

---

## Self-review

**Spec coverage.** Mechanism → Tasks 2, 4, 5. Two trigger points → Task 3. Rule-5 decoupling → Task 3. No-answer path and session continuity → Tasks 5, 6. Signature validation on both new endpoints → Tasks 4, 5. Idempotency → Task 2. `HUMAN_AGENT_NUMBER` optional → Task 2. Error handling (unset number, REST failure, unknown escalation_id, unknown session, double fire) → Tasks 2, 4, 5, 6. Docs and cost note → Task 7. No gaps.

**Placeholders.** None: every code step carries real code, every test step real assertions.

**Type consistency.** `on_escalation` is `Callable[[dict[str, Any] | None], Awaitable[bool]] | None` in Tasks 3 and 6. `transfer_to_human(call_sid, packet, session_id, *, client=None) -> bool` matches its Task 6 call site. `PendingTransfer` fields (`escalation_id`, `whisper`, `session_id`) are consistent across Tasks 1, 4, 5.

**One risk flagged for the live checkpoint, not solvable offline.** Issuing the redirect tears down the Media Stream, which can truncate the agent's own spoken notice mid-word. The `<Say>` at the top of the transfer TwiML guarantees the customer hears something, but exactly how the two overlap can only be judged on a real call. If it sounds abrupt, the fix is to await `BotStoppedSpeakingFrame` before firing the hook — deliberately not built now, because guessing at the timing without hearing it would be speculation.

**A defect in this plan's own spec, found and fixed during self-review.** The spec claimed zero `agent/` changes. Checking `agent/session.py` showed `TurnOutcome` does not carry the handoff packet — `run_turn` builds it at line 337 and discards it. Left alone, the model-driven escalation path (the ordinary one) would have transferred with `None` and whispered "no context available", while the rarer DTMF path briefed the human fully. Exactly backwards, and it would have passed every offline test, since the fallback is legitimate behaviour. Task 3 Step 0 now adds the one required field, and the spec has been corrected to say so rather than quietly contradicting itself.

Because of that, `outcome.escalation_packet` is read directly in Task 3, not via `getattr`.
