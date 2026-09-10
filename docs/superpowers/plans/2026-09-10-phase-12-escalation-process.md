# Phase 12 — Escalation as a Resolvable Process Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn escalation from an event that ends the call into a state the agent must resolve — a callback the customer agrees to, or a record that they will reach out themselves — before the conversation is allowed to end.

**Architecture:** Escalation state becomes a mutable per-session object living on `SessionGates`, mutated by tool handlers and read by `run_turn`. Triggers split into *mandatory* (open an escalation) and *suggested* (offer one, and only escalate if the customer accepts). Two new sync tools record a resolution; all async work — building the packet, persisting, notifying — stays in `run_turn` and `close_session`, exactly where it lives today.

**Tech Stack:** Python 3.12, SQLite, Pydantic, `anthropic` SDK, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-10-phase-12-escalation-process-design.md`

## Global Constraints

- **CLAUDE.md rule 5:** nothing under `transport/` may change. If a task finds it must, stop and flag it.
- **CLAUDE.md rule 6:** the callback booking goes through `PendingActionGate` propose-then-confirm. No irreversible action without a confirmation turn.
- **CLAUDE.md rule 7:** resolution is a deterministic tool call. The model judges only genuinely ambiguous prose (did the customer accept the offer?), never whether an escalation is resolved.
- **Never raise on an exit path.** `run_turn` and `close_session` already wrap every fallible call; new code follows the same shape.
- **Values come from `data/mock_db.py` at runtime, never hard-coded** — no literal order IDs, slot times, or customer IDs in tests.
- **Tool handlers are synchronous.** `build_dispatch_tool`'s `dispatch_tool` calls `handler(**tool_input)` with no `await` (`agent/session.py:98`). A tool cannot do async I/O. This constraint drives the whole design below.
- Run the suite with keys blanked so no task spends API credit:
  `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
  (`env -u` does **not** work here — `load_dotenv()` repopulates a *missing* variable; an empty string survives.)

---

## Rulings made before execution

The spec is the authority; these are places where implementing it against the real code forced a decision. Each is recorded here rather than silently absorbed.

**R1 — Escalation state lives on `SessionGates`, not as `Session.escalation`.**
The spec says "`Session` gains `escalation: OpenEscalation | None`". That cannot work: `build_dispatch_tool` (`agent/session.py:72`) builds the tool handlers *before* the `Session` exists, so a handler cannot close over a field that is later rebound on `Session`. `SessionGates` is already the "mutable per-session state the handlers close over" container, it is already threaded in, and escalation state is literally a gate — it gates `end_conversation`. So `SessionGates` gains `escalation: EscalationState`, created once and only ever mutated. `Session.gates.escalation` is the accessor. Zero signature changes anywhere; the existing 3-tuple unpacking in `agent/session.py:199`, `tests/test_text_cli.py:43` and `tests/test_text_cli.py:126` keeps working untouched.
*Cost if wrong:* the state is reachable via a slightly longer path than the spec imagined. No behavioural difference.

**R2 — A suggested trigger never opens an escalation; it only speaks an offer.**
The spec describes suggested triggers as opening an escalation once the customer accepts. Implemented literally that needs run_turn to classify "did they decline?" — another judgement call, and the exact class of inference that caused this phase to exist. Instead: a suggested trigger resets its own counter immediately, retires itself for the session, and appends an offer to the turn's `notice`. If the customer accepts, the model calls `schedule_human_callback` and *that* opens the escalation. If they do not, nothing happened and nothing needs detecting.
This satisfies every stated requirement — the offer is made, declining resets the counter, one offer per trigger — with strictly less machinery and no second classifier.
*Cost if wrong:* a customer who accepts an offer and then the call drops leaves no `escalations` row. Judged acceptable: nothing was agreed, and the mandatory triggers (the ones representing a customer's explicit request) still open immediately.

**R3 — All async work stays in `run_turn`; the resolution tools are sync recorders.**
Because handlers are synchronous, `schedule_human_callback` cannot `await create_handoff_packet`. It records the resolution into `gates.escalation` and returns. `run_turn`, after the tool loop, sees an unpersisted resolution and does the async work. This mirrors how escalation already works today — the tool loop finishes, then `run_turn` awaits the packet.
*Cost if wrong:* the packet is built one step later than the tool call. Invisible to the customer, and it is where the existing code already does this.

**R4 — `create_handoff_packet` keeps its name, signature, and behaviour.**
`transport/pipecat_processors.py`'s DTMF-zero handler calls it and must not change (rule 5). It is redefined as "open and immediately resolve as a transfer" — which is exactly what pressing zero means — implemented as `open_escalation` + `resolve_escalation` back to back.
*Cost if wrong:* none identified; the transport's observable behaviour is unchanged.

---

## File Structure

| File | Responsibility |
|---|---|
| `agent/tools/escalation.py` | **Modify.** Gains `EscalationState` (session state, sibling to the existing `EscalationTracker`), `EscalationSignal`, `open_escalation`, `resolve_escalation`, `mark_resolved`. `create_handoff_packet` becomes a wrapper over the first two. |
| `agent/tools/handoff.py` | **Create.** The two resolution tools and their schemas. Sync, no I/O beyond `book_appointment`. |
| `agent/session.py` | **Modify.** `SessionGates.escalation`; `TOOLS` registration; handler wiring; `end_conversation` refusal; `run_turn`'s escalation branch; `close_session`'s unresolved notification. |
| `agent/prompts.py` | **Modify.** Escalation-mode guidance. |
| `data/mock_db.py` | **Modify.** Four new `escalations` columns. |
| `tests/test_escalation.py` | **Modify.** State machine, triggers, packet split. |
| `tests/test_handoff.py` | **Create.** The two resolution tools. |
| `tests/test_session.py` | **Modify.** `run_turn`/`close_session` integration, `end_conversation` refusal. |

---

### Task 1: Escalation state and the mandatory/suggested trigger split

**Files:**
- Modify: `agent/tools/escalation.py` (add near `EscalationTracker`, ~line 135)
- Modify: `data/mock_db.py:67-84` (the `escalations` CREATE TABLE)
- Test: `tests/test_escalation.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `EscalationState` dataclass: fields `status: str = "none"` (one of `"none"`, `"open"`, `"resolved"`), `escalation_id: int | None = None`, `packet: dict[str, Any] | None = None`, `items: list[str] = field(default_factory=list)`, `resolution: str | None = None`, `callback_time: str | None = None`, `offered: set[str] = field(default_factory=set)`, `pending_persist: bool = False`. Methods `open(reason)`, `amend(reason)`, `record_resolution(resolution, callback_time=None)`, `is_open` property.
  - `EscalationSignal` frozen dataclass: `reason: str`, `mandatory: bool`.
  - `MANDATORY_REASONS: frozenset[str]`.
  - `EscalationTracker.record_turn` now returns `EscalationSignal | None` (was `str | None`).
  - `check_escalation` now returns `EscalationSignal | None` (was `str | None`).
  - `EscalationTracker.reset_streak(reason: str) -> None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_escalation.py`:

```python
def test_a_mandatory_trigger_is_marked_mandatory():
    """An explicit request for a human is the customer's decision, not the
    agent's inference, so it opens an escalation without asking permission.
    """
    tracker = escalation.EscalationTracker()
    signal = tracker.record_turn(
        escalation.TurnClassification(
            intent="request_human", sentiment="neutral", policy_restricted=False
        ),
        tool_calls=[],
    )
    assert signal is not None
    assert signal.mandatory is True
    assert signal.reason == "explicit request for a human"


def test_repeated_failed_lookups_are_only_a_suggestion():
    """The live failure this phase exists to fix: a customer mis-dictated an
    order number, said so immediately, and was hung up on. Failed lookups are
    the AGENT's inference that it is failing the customer — sometimes true,
    sometimes just a transposed digit. So it offers rather than imposes.
    """
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    failed_lookup = [{"name": "get_order_status", "output": {"found": False}}]

    for _ in range(escalation.FAILED_LOOKUP_ESCALATION_THRESHOLD):
        signal = tracker.record_turn(neutral, failed_lookup)

    assert signal is not None
    assert signal.reason == "repeated failed lookups"
    assert signal.mandatory is False, "an inference must be offered, never imposed"


def test_resetting_a_streak_gives_the_customer_a_clean_run():
    """Declining an offer has to reset the counter that produced it, or the
    agent simply asks again one turn later — which is what being asked
    repeatedly whether you want a human feels like.
    """
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    failed_lookup = [{"name": "get_order_status", "output": {"found": False}}]

    for _ in range(escalation.FAILED_LOOKUP_ESCALATION_THRESHOLD):
        tracker.record_turn(neutral, failed_lookup)

    tracker.reset_streak("repeated failed lookups")

    assert tracker.consecutive_failed_lookups == 0
    assert tracker.record_turn(neutral, failed_lookup) is None, (
        "one failure after a reset must not immediately re-trigger"
    )


def test_a_turn_with_no_tool_calls_does_not_advance_the_lookup_counter():
    """Turn 10 of the live call escalated while the agent was merely being
    asked to repeat a number back — no lookup happened at all. A counter of
    failed lookups must only move when a lookup was actually attempted.
    """
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    tracker.record_turn(neutral, [{"name": "get_order_status", "output": {"found": False}}])
    before = tracker.consecutive_failed_lookups

    tracker.record_turn(neutral, [])

    assert tracker.consecutive_failed_lookups == before, (
        "a turn with no lookups must neither advance nor reset the streak"
    )


def test_the_state_machine_opens_amends_and_resolves():
    state = escalation.EscalationState()
    assert state.status == "none"
    assert state.is_open is False

    state.open("explicit request for a human")
    assert state.status == "open"
    assert state.is_open is True
    assert state.items == ["explicit request for a human"]

    state.amend("policy-restricted topic")
    assert state.status == "open", "an amendment must never reopen or duplicate"
    assert state.items == ["explicit request for a human", "policy-restricted topic"]

    state.record_resolution("callback", callback_time="2026-09-11T09:00:00")
    assert state.status == "resolved"
    assert state.is_open is False
    assert state.callback_time == "2026-09-11T09:00:00"
    assert state.pending_persist is True


def test_amending_a_resolved_escalation_keeps_it_resolved():
    """A second issue reaches the same human on the same callback. It must
    not drag the call back into an unresolvable state.
    """
    state = escalation.EscalationState()
    state.open("explicit request for a human")
    state.record_resolution("customer_will_reach_out")
    state.pending_persist = False

    state.amend("high-value refund requires approval")

    assert state.status == "resolved"
    assert len(state.items) == 2
    assert state.pending_persist is True, "an amendment after resolution sends an update"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_escalation.py -q`
Expected: FAIL — `AttributeError: module 'agent.tools.escalation' has no attribute 'EscalationState'`, and the mandatory/suggested assertions fail because `record_turn` still returns a bare string.

- [ ] **Step 3: Add the state and signal types**

In `agent/tools/escalation.py`, immediately above `class EscalationTracker`:

```python
# Whose decision each trigger represents. A mandatory trigger is the
# customer's or a rule's — the agent has no standing to second-guess it, so
# it opens an escalation directly. A suggested trigger is the agent's own
# inference that it is failing, which is exactly the kind of judgement that
# hung up on a customer who had merely mis-dictated one digit. Inferences get
# offered; they do not get imposed. This is CLAUDE.md rule 6's propose-then-
# confirm principle applied to handoffs.
MANDATORY_REASONS = frozenset(
    {
        "explicit request for a human",
        "policy-restricted topic",
    }
)


@dataclass(frozen=True)
class EscalationSignal:
    """A fired trigger and whether the agent must act on it or merely offer."""

    reason: str
    mandatory: bool


@dataclass
class EscalationState:
    """One escalation per session, for the whole life of the session.

    That is what actually happens on a support line: a human ringing a
    customer back deals with everything that customer has, rather than
    booking three separate calls for three questions. So a second
    escalation-worthy issue becomes another ITEM on the same handoff, not a
    second handoff — which is also why `items` is a list and not the single
    `reason` string this replaced.

    Mutated in place, never rebound, because the tool handlers in
    build_dispatch_tool close over it before the Session that owns it exists.
    """

    status: str = "none"  # "none" | "open" | "resolved"
    escalation_id: int | None = None
    packet: dict[str, Any] | None = None
    items: list[str] = field(default_factory=list)
    resolution: str | None = None  # "callback" | "customer_will_reach_out" | "transfer"
    callback_time: str | None = None
    # Suggested triggers that have already made their offer. Being asked over
    # and over whether you would like a human is its own kind of failure.
    offered: set[str] = field(default_factory=set)
    # Set whenever there is something run_turn still needs to persist or
    # notify. run_turn clears it once the async work is done, so a dropped
    # connection between the tool call and the write is visible rather than
    # silently lost.
    pending_persist: bool = False

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def open(self, reason: str) -> None:
        if self.status == "none":
            self.status = "open"
        self.items.append(reason)
        self.pending_persist = True

    def amend(self, reason: str) -> None:
        """A second trigger during an existing escalation. Never reopens a
        resolved one and never schedules a second callback — it adds an item
        the human taking the handoff can prepare for.
        """
        if reason in self.items:
            return
        self.items.append(reason)
        self.pending_persist = True

    def record_resolution(self, resolution: str, callback_time: str | None = None) -> None:
        self.status = "resolved"
        self.resolution = resolution
        self.callback_time = callback_time
        self.pending_persist = True
```

Add `field` to the existing `from dataclasses import ...` line at the top of the file if it is not already imported.

- [ ] **Step 4: Change `record_turn` to return a signal, and fix the counter**

Replace the body of `EscalationTracker.record_turn` (`agent/tools/escalation.py:142`) so every `return "<reason>"` becomes an `EscalationSignal`, and the failed-lookup threshold check moves *inside* the `if outcomes:` block:

```python
        tool_escalation = _tool_signaled_escalation(tool_calls)
        if tool_escalation:
            # A tool asking for a specialist (a high-value refund needing
            # approval) is a rule, not an inference.
            return EscalationSignal(tool_escalation, mandatory=True)
        if classification.intent == "request_human":
            return EscalationSignal("explicit request for a human", mandatory=True)
        if classification.policy_restricted:
            return EscalationSignal("policy-restricted topic", mandatory=True)

        if classification.sentiment == "negative":
            self.consecutive_negative_turns += 1
        else:
            self.consecutive_negative_turns = 0
        if self.consecutive_negative_turns >= NEGATIVE_SENTIMENT_ESCALATION_THRESHOLD:
            return EscalationSignal("sustained negative sentiment across multiple turns", mandatory=False)

        outcomes = _turn_tool_outcomes(tool_calls)
        if outcomes:
            if any(outcomes):  # at least one lookup succeeded this turn — clean slate
                self.consecutive_failed_lookups = 0
            else:
                self.consecutive_failed_lookups += 1
            # Checked INSIDE this block on purpose. It used to sit outside,
            # re-reading the counter every single turn — so once the streak
            # was tripped, a turn containing no lookup at all still escalated.
            # One live call produced escalation rows 7, 8 and 9 for one
            # problem that way, the last from a turn that merely asked the
            # agent to repeat a number back.
            if self.consecutive_failed_lookups >= FAILED_LOOKUP_ESCALATION_THRESHOLD:
                return EscalationSignal("repeated failed lookups", mandatory=False)

        if ungrounded:
            self.consecutive_ungrounded_replies += 1
        else:
            self.consecutive_ungrounded_replies = 0
        if self.consecutive_ungrounded_replies >= UNGROUNDED_REPLY_ESCALATION_THRESHOLD:
            return EscalationSignal("repeated ungrounded replies", mandatory=False)

        return None
```

Update the docstring's return description and the type annotation to `EscalationSignal | None`.

Then add, as a method on `EscalationTracker`:

```python
    def reset_streak(self, reason: str) -> None:
        """Clear the counter behind a suggested trigger, so a declined offer
        gives the customer a clean run rather than tripping the same
        threshold on their next breath.
        """
        if reason == "repeated failed lookups":
            self.consecutive_failed_lookups = 0
        elif reason == "sustained negative sentiment across multiple turns":
            self.consecutive_negative_turns = 0
        elif reason == "repeated ungrounded replies":
            self.consecutive_ungrounded_replies = 0
```

Update `check_escalation`'s return annotation to `EscalationSignal | None`; its body needs no change.

- [ ] **Step 5: Add the schema columns**

In `data/mock_db.py`, inside the `escalations` CREATE TABLE (after the Phase 11 `notified_at` column, before the FOREIGN KEY line):

```sql
    -- Phase 12: an escalation is a process, not an event. `items` is a
    -- newline-separated list of every reason attached to this one handoff —
    -- one customer gets one callback, so a second escalation-worthy issue
    -- appends here rather than opening a second row. `resolution` is null
    -- while the escalation is still open, which is also how an abandoned
    -- call is recognised at close_session.
    items                   TEXT,
    resolution              TEXT,
    callback_time           TEXT,
    resolved_at             TEXT,
```

The `.db` file is gitignored and regenerated: `python -m data.mock_db` picks the new columns up via `reset_and_seed()`, the same convention every prior phase used.

- [ ] **Step 6: Run the tests**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_escalation.py -q`
Expected: PASS.

Then run the full suite: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
Expected: failures in `tests/test_session.py` and any existing test asserting `check_escalation`/`record_turn` returns a string. **Fix those call sites now** — they should read `signal.reason`. Do not skip them; a test that still passes while comparing a dataclass to a string is a test that stopped checking anything.

- [ ] **Step 7: Commit**

```bash
git add agent/tools/escalation.py data/mock_db.py tests/test_escalation.py tests/test_session.py
git commit -m "Split escalation triggers into mandatory and suggested"
```

---

### Task 2: Split the packet into open, resolve, and amend

**Files:**
- Modify: `agent/tools/escalation.py:294-360` (`create_handoff_packet`)
- Test: `tests/test_escalation.py`

**Interfaces:**
- Consumes: `EscalationState` from Task 1.
- Produces:
  - `async def open_escalation(customer_id, messages, reason, client=None) -> dict` — infers fields, writes the row, returns the packet. **Sends nothing.**
  - `def mark_resolved(escalation_id, items, resolution, callback_time=None, resolved_at=None) -> None`
  - `async def resolve_escalation(packet, items, resolution, callback_time=None) -> bool` — updates the row, then notifies. Returns delivery success. Never raises.
  - `create_handoff_packet` keeps its existing signature and behaviour.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_opening_an_escalation_writes_a_row_but_tells_nobody(monkeypatch):
    """There is nothing useful to tell a human until the outcome is known.
    A message saying "a customer needs help, we don't know what they want or
    when to ring" is noise the human agent has to chase.

    The row is still written immediately, so a dropped call leaves a record.
    """
    sent = []
    monkeypatch.setattr(
        escalation, "notify_escalation", _recording_notifier(sent), raising=True
    )
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)

    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "get me a person"}], "explicit request for a human"
    )

    assert packet["escalation_id"] is not None
    assert sent == [], "opening must not notify"
    with get_connection() as conn:
        row = conn.execute(
            "SELECT resolution FROM escalations WHERE escalation_id = ?", (packet["escalation_id"],)
        ).fetchone()
    assert row["resolution"] is None, "an open escalation has no resolution yet"


@pytest.mark.asyncio
async def test_resolving_notifies_exactly_once_with_the_agreed_time(monkeypatch):
    """"Notifies twice" is the failure mode a reader cannot see, so this
    counts calls rather than checking the last one.
    """
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent), raising=True)
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)

    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "get me a person"}], "explicit request for a human"
    )
    slot = "2026-09-11T09:00:00"

    delivered = await escalation.resolve_escalation(
        packet, items=["explicit request for a human"], resolution="callback", callback_time=slot
    )

    assert delivered is True
    assert len(sent) == 1, f"exactly one notification per escalation, got {len(sent)}"
    assert sent[0]["resolution"] == "callback"
    assert sent[0]["callback_time"] == slot
    assert sent[0]["items"] == ["explicit request for a human"]


@pytest.mark.asyncio
async def test_a_notification_failure_still_leaves_the_escalation_resolved(monkeypatch):
    """Delivery has never been allowed to affect persistence (Phase 11) and
    that does not change here.
    """
    async def _explode(packet, **kwargs):
        raise RuntimeError("webhook down")

    monkeypatch.setattr(escalation, "notify_escalation", _explode, raising=True)
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)

    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "get me a person"}], "explicit request for a human"
    )

    delivered = await escalation.resolve_escalation(
        packet, items=["explicit request for a human"], resolution="customer_will_reach_out"
    )

    assert delivered is False
    with get_connection() as conn:
        row = conn.execute(
            "SELECT resolution, resolved_at FROM escalations WHERE escalation_id = ?",
            (packet["escalation_id"],),
        ).fetchone()
    assert row["resolution"] == "customer_will_reach_out"
    assert row["resolved_at"] is not None


@pytest.mark.asyncio
async def test_create_handoff_packet_still_opens_and_resolves_in_one_call(monkeypatch):
    """transport/pipecat_processors.py's DTMF-zero handler calls this and must
    not change (CLAUDE.md rule 5). Pressing zero IS a resolution — the caller
    is being transferred right now — so it opens and resolves together.
    """
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent), raising=True)
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)

    packet = await escalation.create_handoff_packet(
        CUSTOMER_ID, [{"role": "user", "content": "0"}], "caller pressed 0"
    )

    assert packet["escalation_id"] is not None
    assert "callback_time" in packet, "the existing contract includes a callback time"
    assert len(sent) == 1
```

Add these helpers near the top of `tests/test_escalation.py` if equivalents do not already exist:

```python
def _recording_notifier(sink: list[dict]):
    async def _notify(packet, **kwargs):
        sink.append(packet)
        return True

    return _notify


async def _stub_infer_fields(customer_id, messages, client=None):
    """Keeps these tests offline. The inference itself is covered separately."""
    return escalation.HandoffFields(
        customer_intent="wants a person",
        conversation_summary="asked for a human",
        verified_account_info=None,
        actions_taken=None,
        sentiment="neutral",
    )
```

- [ ] **Step 2: Run to verify they fail**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_escalation.py -q -k "open_escalation or resolve or handoff_packet_still"`
Expected: FAIL — `open_escalation` and `resolve_escalation` do not exist.

- [ ] **Step 3: Implement the split**

Replace `create_handoff_packet` (`agent/tools/escalation.py:294`) with:

```python
async def open_escalation(
    customer_id: str,
    messages: list[dict[str, Any]],
    reason: str,
    client: anthropic.AsyncAnthropic | None = None,
) -> dict[str, Any]:
    """Assemble a handoff packet and persist it, WITHOUT telling anyone yet.

    Not a tool the model calls itself — see the module docstring.

    The row is written the moment the escalation opens, so a call that drops
    before anything is agreed still leaves a record (close_session finds it
    and notifies as unresolved). Nothing goes out to the automation platform
    here: until an outcome is known there is nothing useful to say to a
    human, and two messages per escalation is noise.
    """
    inferred = await _infer_handoff_fields(customer_id, messages, client=client)
    # Redact ONCE, here, so the DB row and the outbound webhook carry
    # identical text. notify_escalation redacts again defensively for any
    # future caller; redaction is idempotent, so that second pass is a no-op.
    fields = HandoffFields(**redact_fields(inferred.model_dump(), HANDOFF_TEXT_FIELDS))
    escalation_id = log_escalation(customer_id, reason, fields)
    return {
        "escalation_id": escalation_id,
        "reason": reason,
        "items": [reason],
        "resolution": None,
        # The earliest slot a human COULD ring back on, so the agent has a
        # concrete time to offer. Read-only and reserves nothing — the
        # booking happens in agent/tools/handoff.py once the customer picks
        # one, behind rule 6's confirmation turn.
        "callback_time": _next_callback_slot(),
        **fields.model_dump(),
    }


def mark_resolved(
    escalation_id: int,
    items: list[str],
    resolution: str,
    callback_time: str | None = None,
    resolved_at: str | None = None,
) -> None:
    """Record how an escalation ended. `items` is stored newline-separated:
    one customer gets one callback, so every reason attached to this handoff
    lives in one row rather than spawning new ones.
    """
    resolved_at = resolved_at or datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE escalations SET items = ?, resolution = ?, callback_time = ?, "
            "resolved_at = ? WHERE escalation_id = ?",
            ("\n".join(items), resolution, callback_time, resolved_at, escalation_id),
        )


async def resolve_escalation(
    packet: dict[str, Any],
    items: list[str],
    resolution: str,
    callback_time: str | None = None,
) -> bool:
    """Close out an escalation and tell the automation platform about it.

    This is the ONLY place a notification goes out, which is what guarantees
    exactly one per escalation, always carrying the truth: resolved with an
    agreed time, resolved as customer-initiated, or (from close_session)
    unresolved because the call ended first.

    Never raises. Persistence must never depend on delivery succeeding, and
    delivery must never break a call that is already ending.
    """
    escalation_id = packet["escalation_id"]
    packet = {**packet, "items": items, "resolution": resolution, "callback_time": callback_time}

    # Persist BEFORE notifying: the durable record is what matters, and a
    # webhook that hangs for its full retry budget must not leave the row
    # claiming the escalation is still open.
    try:
        mark_resolved(escalation_id, items, resolution, callback_time)
    except Exception:  # noqa: BLE001 — recording an outcome must never break a call
        logger.exception("mark_resolved raised unexpectedly")

    try:
        delivered = await notify_escalation(packet)
    except Exception:  # noqa: BLE001 — a broken webhook must never break escalation
        logger.exception("notify_escalation raised unexpectedly")
        delivered = False

    # Separate try/except from the notify call above: mark_notified is a
    # second, independent thing that can fail (write-lock contention under
    # simultaneous escalations) and it must not discard a packet that
    # log_escalation already durably persisted.
    try:
        mark_notified(escalation_id, delivered)
    except Exception:  # noqa: BLE001 — recording delivery status must never break escalation
        logger.exception("mark_notified raised unexpectedly")

    return delivered


async def create_handoff_packet(
    customer_id: str,
    messages: list[dict[str, Any]],
    reason: str,
    client: anthropic.AsyncAnthropic | None = None,
) -> dict[str, Any]:
    """Open an escalation and immediately resolve it as a live transfer.

    Kept with its original name and signature for
    transport/pipecat_processors.py's DTMF-zero handler (CLAUDE.md rule 5):
    a caller pressing zero is being put through to a person right now, so
    there is no process to work through — the outcome is already known and
    the human agent should hear about it immediately.

    Every other caller uses open_escalation + resolve_escalation, which is
    what makes escalation a process the agent has to resolve.
    """
    packet = await open_escalation(customer_id, messages, reason, client=client)
    await resolve_escalation(
        packet, items=[reason], resolution="transfer", callback_time=packet.get("callback_time")
    )
    return packet
```

- [ ] **Step 4: Run the tests**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_escalation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/tools/escalation.py tests/test_escalation.py
git commit -m "Notify once, at resolution, not at detection"
```

---

### Task 3: The two resolution tools

**Files:**
- Create: `agent/tools/handoff.py`
- Modify: `agent/session.py:38-47` (`TOOLS`), `agent/session.py:57-70` (`SessionGates`), `agent/session.py:82-95` (handlers)
- Test: `tests/test_handoff.py`

**Interfaces:**
- Consumes: `EscalationState` (Task 1), `PendingActionGate` (`agent/confirmation.py`), `scheduling.book_appointment` and `scheduling.find_available_slots`.
- Produces:
  - `SCHEDULE_CALLBACK_SCHEMA`, `RECORD_CALLBACK_DECLINED_SCHEMA` (dicts).
  - `def schedule_human_callback(slot_time, state, escalation, gate, customer_id) -> dict`
  - `def record_customer_will_reach_out(escalation) -> dict`
  - `SessionGates.escalation: EscalationState`

Both tools are **synchronous** — `dispatch_tool` does not await (`agent/session.py:98`). They record an outcome into the shared `EscalationState`; `run_turn` (Task 5) does the persisting and notifying.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_handoff.py`:

```python
"""Phase 12's two resolution tools.

Both are deliberately synchronous and do no network I/O: build_dispatch_tool
calls handlers without awaiting them, so a tool physically cannot do async
work. They record an outcome; run_turn persists and notifies it.
"""

from __future__ import annotations

import pytest

from agent.confirmation import PendingActionGate
from agent.tools import handoff, scheduling
from agent.tools.escalation import EscalationState
from data.mock_db import get_connection, reset_and_seed

CUSTOMER_ID = "CUST-1001"


@pytest.fixture(autouse=True)
def _fresh_db():
    reset_and_seed()


def _open_state() -> EscalationState:
    state = EscalationState()
    state.open("explicit request for a human")
    state.escalation_id = 1
    state.pending_persist = False
    return state


def _free_slot() -> str:
    """From the real calendar, never hard-coded — a literal date here would
    quietly start failing once it fell outside the booking window.
    """
    return scheduling.find_available_slots()["slots"][0]


def test_scheduling_a_callback_proposes_before_it_books():
    """CLAUDE.md rule 6: booking is irreversible, so the first call proposes
    and only a later confirmed turn commits. The extra turn is the point.
    """
    escalation_state = _open_state()
    gate = PendingActionGate()
    slot = _free_slot()

    first = handoff.schedule_human_callback(
        slot_time=slot, state=escalation_state, gate=gate, customer_id=CUSTOMER_ID
    )

    assert first["status"] == "pending_confirmation"
    assert escalation_state.status == "open", "a proposal resolves nothing yet"
    with get_connection() as conn:
        booked = conn.execute("SELECT COUNT(*) AS n FROM appointments").fetchone()["n"]
    assert booked == 0, "nothing may be reserved before the customer confirms"


def test_a_confirmed_callback_books_a_real_appointment_and_resolves():
    """The time Slack shows has to be one that is genuinely held, or two
    customers get promised the same slot.
    """
    escalation_state = _open_state()
    gate = PendingActionGate()
    slot = _free_slot()

    handoff.schedule_human_callback(
        slot_time=slot, state=escalation_state, gate=gate, customer_id=CUSTOMER_ID
    )
    gate.turn += 1  # the customer's "yes" arrives on a later turn
    result = handoff.schedule_human_callback(
        slot_time=slot, state=escalation_state, gate=gate, customer_id=CUSTOMER_ID
    )

    assert result["scheduled"] is True
    assert escalation_state.status == "resolved"
    assert escalation_state.resolution == "callback"
    assert escalation_state.callback_time == slot
    assert escalation_state.pending_persist is True, "run_turn still has to notify"
    with get_connection() as conn:
        row = conn.execute("SELECT scheduled_time FROM appointments").fetchone()
    assert row["scheduled_time"] == slot, "a real appointment, not just a promise"


def test_an_unavailable_slot_leaves_the_escalation_open():
    """The agent has to offer another time, not carry on as though something
    was arranged.
    """
    escalation_state = _open_state()
    gate = PendingActionGate()
    slot = _free_slot()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO appointments (customer_id, scheduled_time, reason, status) "
            "VALUES (?, ?, 'taken', 'scheduled')",
            (CUSTOMER_ID, slot),
        )

    result = handoff.schedule_human_callback(
        slot_time=slot, state=escalation_state, gate=gate, customer_id=CUSTOMER_ID
    )

    assert result["scheduled"] is False
    assert result["error"] == "slot_unavailable"
    assert escalation_state.status == "open"


def test_a_declined_callback_is_a_real_resolution_and_books_nothing():
    """"I'll call back when I know my schedule" is a legitimate ending, not a
    failure — and holding a slot the customer never agreed to is wrong.
    """
    escalation_state = _open_state()

    result = handoff.record_customer_will_reach_out(escalation=escalation_state)

    assert result["recorded"] is True
    assert escalation_state.status == "resolved"
    assert escalation_state.resolution == "customer_will_reach_out"
    assert escalation_state.callback_time is None
    with get_connection() as conn:
        booked = conn.execute("SELECT COUNT(*) AS n FROM appointments").fetchone()["n"]
    assert booked == 0


def test_accepting_an_offer_opens_an_escalation_that_was_never_opened():
    """A suggested trigger only makes an offer — no escalation exists until
    the customer says yes. So the resolution tool has to open one.
    """
    escalation_state = EscalationState()  # nothing open
    assert escalation_state.status == "none"

    handoff.record_customer_will_reach_out(escalation=escalation_state)

    assert escalation_state.status == "resolved"
    assert escalation_state.items, "an escalation opened by acceptance still needs a reason"


def test_resolving_twice_does_not_schedule_a_second_callback():
    """One customer, one callback, by design."""
    escalation_state = _open_state()
    gate = PendingActionGate()
    slot = _free_slot()

    handoff.schedule_human_callback(
        slot_time=slot, state=escalation_state, gate=gate, customer_id=CUSTOMER_ID
    )
    gate.turn += 1
    handoff.schedule_human_callback(
        slot_time=slot, state=escalation_state, gate=gate, customer_id=CUSTOMER_ID
    )
    second = handoff.record_customer_will_reach_out(escalation=escalation_state)

    assert second["recorded"] is False
    assert escalation_state.resolution == "callback", "the first resolution stands"
    with get_connection() as conn:
        booked = conn.execute("SELECT COUNT(*) AS n FROM appointments").fetchone()["n"]
    assert booked == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_handoff.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tools.handoff'`.

- [ ] **Step 3: Write `agent/tools/handoff.py`**

```python
"""Phase 12: the two ways an escalation can end.

Escalation used to be an event that ended the call — a trigger fired, a row
was written, and the conversation stopped. There was no process, no
resolution, and no way back. These two tools are the ways out: the customer
agrees a time for a human to ring them, or says they will make contact
themselves. Until one of them has been called, end_conversation is refused.

Both are SYNCHRONOUS and do no network I/O, because agent/session.py's
dispatch_tool calls handlers without awaiting them. They record an outcome
into the session's shared EscalationState; run_turn does the persisting and
the single outbound notification.
"""

from __future__ import annotations

from typing import Any

from agent.confirmation import PendingActionGate
from agent.tools import scheduling
from agent.tools.escalation import EscalationState

# What an escalation opened by a customer accepting an offer is recorded as.
# A suggested trigger never opens one itself (it only asks), so by the time a
# resolution tool runs there may be nothing open — and a handoff with no
# reason on it tells the human agent nothing.
ACCEPTED_OFFER_REASON = "customer accepted an offer of a callback"

SCHEDULE_CALLBACK_SCHEMA: dict[str, Any] = {
    "name": "schedule_human_callback",
    "description": (
        "Arrange for a human colleague to call the customer back about an "
        "issue you are handing over. Call find_available_slots first and "
        "offer the customer a time. The FIRST call proposes the booking and "
        "asks for confirmation — it does not book yet. Only call it a "
        "second time, with the same time, after the customer has clearly "
        "confirmed in a later message. Use this whenever a human needs to "
        "take over, including when the customer accepts your offer of a "
        "callback."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slot_time": {
                "type": "string",
                "description": "The slot the customer agreed to, exactly as find_available_slots returned it.",
            },
        },
        "required": ["slot_time"],
    },
}

RECORD_CALLBACK_DECLINED_SCHEMA: dict[str, Any] = {
    "name": "record_customer_will_reach_out",
    "description": (
        "Record that the customer would rather contact us themselves than "
        "have a colleague call them back — for example if they do not know "
        "their schedule yet. This is a perfectly good outcome, not a "
        "failure: it closes out the handover so the conversation can end "
        "normally. Do not use it if the customer has agreed to a callback "
        "time; use schedule_human_callback for that."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


def schedule_human_callback(
    slot_time: str,
    state: EscalationState,
    gate: PendingActionGate,
    customer_id: str,
) -> dict[str, Any]:
    """Book a real appointment for a human to ring the customer back.

    A real booking, through Phase 5's book_appointment, rather than a time
    read off the calendar and promised: the slot has to actually be held or
    two customers get told the same one, and the time the human agent is
    given has to be a time that exists. That means CLAUDE.md rule 6 applies —
    the first call proposes, a later confirmed turn commits. The extra turn
    is the correct cost of an irreversible action.
    """
    if state.status == "resolved":
        # One customer, one callback. A second issue reaches the same human
        # on the same call — see EscalationState.amend.
        return {
            "scheduled": False,
            "error": "already_resolved",
            "message": (
                "A callback is already arranged for this customer at "
                f"{state.callback_time}. Anything else they raise will be passed "
                "to the same colleague — there is no need to book a second call."
            ),
        }

    booking = scheduling.book_appointment(
        slot_time=slot_time,
        reason="callback from a human agent",
        state=gate,
        customer_id=customer_id,
    )

    if booking.get("status") == "pending_confirmation":
        return {"scheduled": False, "status": "pending_confirmation", "message": booking["message"]}
    if not booking.get("booked"):
        # slot_unavailable — the escalation stays open and the agent offers
        # another time. Passed straight through so the model sees the real
        # reason rather than a generic failure.
        return {"scheduled": False, "error": booking.get("error"), "message": booking.get("message")}

    _ensure_open(state)
    state.record_resolution("callback", callback_time=slot_time)
    return {
        "scheduled": True,
        "appointment_id": booking["appointment_id"],
        "callback_time": slot_time,
        "message": f"A colleague will call the customer back at {slot_time}.",
    }


def record_customer_will_reach_out(escalation: EscalationState) -> dict[str, Any]:
    """The customer would rather make contact themselves.

    A first-class outcome, not a failure. The human agent is told the
    customer will initiate, and no slot is held — reserving one the customer
    never agreed to would block a time someone else could use.
    """
    if escalation.status == "resolved":
        return {
            "recorded": False,
            "error": "already_resolved",
            "message": f"This handover is already resolved ({escalation.resolution}).",
        }

    _ensure_open(escalation)
    escalation.record_resolution("customer_will_reach_out")
    return {
        "recorded": True,
        "message": "Noted that the customer will get in touch themselves. A colleague has the details.",
    }


def _ensure_open(state: EscalationState) -> None:
    """A suggested trigger only offers; it never opens an escalation. So a
    customer accepting one arrives here with nothing open, and the handover
    has to be opened before it can be resolved.
    """
    if state.status == "none":
        state.open(ACCEPTED_OFFER_REASON)
```

- [ ] **Step 4: Register the tools**

In `agent/session.py`, add `handoff` to the `from agent.tools import ...` line, then:

`TOOLS` (line 38) gains two entries after `scheduling.CANCEL_APPOINTMENT_SCHEMA`:

```python
    handoff.SCHEDULE_CALLBACK_SCHEMA,
    handoff.RECORD_CALLBACK_DECLINED_SCHEMA,
```

`SessionGates` (line 57) gains a field:

```python
    # Phase 12. Not a PendingActionGate, but a gate in the most literal
    # sense — it is what refuses end_conversation while a handover is still
    # unresolved. It lives here rather than on Session because the tool
    # handlers below close over it, and they are built before the Session
    # that owns them exists.
    escalation: escalation.EscalationState = field(default_factory=escalation.EscalationState)
```

`handlers` (line 83) gains two entries:

```python
        "schedule_human_callback": lambda **kw: handoff.schedule_human_callback(
            **kw, state=gates.escalation, gate=gates.scheduling, customer_id=customer_id
        ),
        "record_customer_will_reach_out": lambda **kw: handoff.record_customer_will_reach_out(
            **kw, escalation=gates.escalation
        ),
```

Note the callback booking shares `gates.scheduling` — the existing scheduling gate — because it *is* a booking, and a customer cannot have a pending appointment and a pending callback at once without one clobbering the other.

- [ ] **Step 5: Run the tests**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_handoff.py -q && ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/tools/handoff.py agent/session.py tests/test_handoff.py
git commit -m "Add the two ways an escalation can be resolved"
```

---

### Task 4: Refuse to end the call while a handover is unresolved

**Files:**
- Modify: `agent/session.py` (the `end_conversation` handler, line 94; `should_end_session`, line 107)
- Modify: `agent/tools/summary.py:52` (`end_conversation`)
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `SessionGates.escalation` (Task 3).
- Produces: `end_conversation(escalation: EscalationState | None = None) -> str` — returns a refusal string while open. `should_end_session` now requires the call to have *succeeded*.

- [ ] **Step 1: Write the failing tests**

```python
def test_end_conversation_is_refused_while_a_handover_is_open():
    """The bug that prompted this phase: the agent asked "would you like me
    to find some callback slots?" and hung up on the same turn. Refusing the
    tool is what makes the model work the problem instead.
    """
    from agent.tools import summary
    from agent.tools.escalation import EscalationState

    state = EscalationState()
    state.open("explicit request for a human")

    result = summary.end_conversation(escalation=state)

    assert "callback" in result.lower() or "colleague" in result.lower()
    assert not should_end_session(
        [{"name": "end_conversation", "output": result}]
    ), "a refused end_conversation must not end the session"


def test_end_conversation_is_allowed_once_the_handover_is_resolved():
    from agent.tools import summary
    from agent.tools.escalation import EscalationState

    state = EscalationState()
    state.open("explicit request for a human")
    state.record_resolution("customer_will_reach_out")

    result = summary.end_conversation(escalation=state)

    assert should_end_session([{"name": "end_conversation", "output": result}])


def test_end_conversation_still_works_with_no_escalation_at_all():
    """The overwhelming majority of calls. Nothing about this phase may make
    an ordinary goodbye harder."""
    from agent.tools import summary

    assert should_end_session([{"name": "end_conversation", "output": summary.end_conversation()}])
```

- [ ] **Step 2: Run to verify they fail**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_session.py -q -k end_conversation`
Expected: FAIL — `end_conversation()` takes no arguments.

- [ ] **Step 3: Implement the refusal**

Replace `agent/tools/summary.py:52-58`:

```python
# What a refused end_conversation returns. Recognised by
# agent/session.py's should_end_session, so the two cannot drift: a refusal
# must never be mistaken for the model signing off.
END_REFUSED_PREFIX = "Not yet."


def end_conversation(escalation: Any | None = None) -> str:
    """Normally there is no real work to do — the point is Claude choosing to
    call this tool at all, and the transport layer watching for it.

    The exception is an unresolved handover. The agent has told a customer a
    human will help them and has not yet arranged how, so ending the call
    strands them: this is exactly what happened live, where the agent asked
    "would you like me to find some callback slots?" and hung up on the same
    turn, before the customer could answer.

    Refused as a returned message rather than a raised exception, the same
    shape issue_refund uses for an outstanding confirmation — the model reads
    it as a tool result and works the problem, where an exception would just
    break the turn.
    """
    if escalation is not None and escalation.is_open:
        return (
            f"{END_REFUSED_PREFIX} You still need to sort out the handover to a colleague "
            "before saying goodbye. Offer the customer a callback time using "
            "find_available_slots and schedule_human_callback, or — if they would rather "
            "get in touch themselves — call record_customer_will_reach_out."
        )
    return "Session marked complete."
```

Add `from typing import Any` to the imports if not present. The parameter is typed `Any` deliberately: importing `EscalationState` here would make `agent.tools.summary` depend on `agent.tools.escalation`, and `escalation.py` has no need of the coupling.

In `agent/session.py`, wire the handler (line 94):

```python
        "end_conversation": lambda **kw: summary.end_conversation(**kw, escalation=gates.escalation),
```

And tighten `should_end_session` (line 107):

```python
def should_end_session(tool_calls: list[dict]) -> bool:
    """True if this turn's tool calls included the model deciding to sign off
    AND that decision was allowed to stand.

    The output check is load-bearing, not defensive. Phase 12 lets
    end_conversation REFUSE while a handover is unresolved, and a refusal
    that still ended the session would defeat the entire mechanism — the
    model would be told "not yet" while the call hung up anyway.
    """
    return any(
        call["name"] == "end_conversation"
        and not str(call.get("output", "")).startswith(summary.END_REFUSED_PREFIX)
        for call in tool_calls
    )
```

- [ ] **Step 4: Run the tests**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/session.py agent/tools/summary.py tests/test_session.py
git commit -m "Refuse to hang up on an unresolved handover"
```

---

### Task 5: Rewire `run_turn` — escalation no longer ends the call

**Files:**
- Modify: `agent/session.py:213-272` (`TurnOutcome`, `_escalation_notice`), `agent/session.py:370-400` (the escalation branch)
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `run_turn` no longer returns `ended=True` for `end_reason="escalated"`. `TurnOutcome.end_reason` values become `"model_ended"` only. New `_escalation_offer(reason) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_a_mandatory_trigger_no_longer_ends_the_call(monkeypatch):
    """The live failure, exactly: the agent asked "would you like me to find
    some available callback slots for you?" and the call ended on that same
    turn. The customer never got to answer.
    """
    session = _session_with_reply("Absolutely. Would you like me to find some callback slots?")
    _force_escalation(monkeypatch, "explicit request for a human", mandatory=True)

    outcome = await run_turn(session, "if it's necessary, I can't discuss this further")

    assert outcome.ended is False, "a reply ending in a question must not end the call"
    assert session.gates.escalation.is_open
    assert session.gates.escalation.escalation_id is not None, "the row is written at open"


@pytest.mark.asyncio
async def test_a_suggested_trigger_offers_instead_of_escalating(monkeypatch):
    """A mis-dictated order number is a cooperative repair, the healthiest
    signal a conversation can produce. It must not be treated as a verdict.
    """
    session = _session_with_reply("I'm still not finding that order.")
    _force_escalation(monkeypatch, "repeated failed lookups", mandatory=False)

    outcome = await run_turn(session, "let me try that number again")

    assert outcome.ended is False
    assert session.gates.escalation.status == "none", "an offer opens nothing"
    assert outcome.notice is not None and "?" in outcome.notice, "the customer must be asked"
    assert session.tracker.consecutive_failed_lookups == 0, "the offer resets its own counter"


@pytest.mark.asyncio
async def test_a_suggested_trigger_only_offers_once(monkeypatch):
    """Being asked over and over whether you want a human is its own failure."""
    session = _session_with_reply("Still not finding it.")
    _force_escalation(monkeypatch, "repeated failed lookups", mandatory=False)

    first = await run_turn(session, "try again")
    second = await run_turn(session, "and again")

    assert first.notice is not None
    assert second.notice is None, "one offer per trigger, per session"


@pytest.mark.asyncio
async def test_a_second_trigger_amends_rather_than_opening_a_second_escalation(monkeypatch):
    """One live call produced escalation rows 7, 8 and 9 for one problem —
    which with n8n connected would have been three Slack messages about one
    customer. Counted as rows, because that is the failure a reader misses.
    """
    reset_and_seed()
    session = _session_with_reply("Understood.")
    _force_escalation(monkeypatch, "explicit request for a human", mandatory=True)

    await run_turn(session, "get me a person")
    await run_turn(session, "seriously, a person")

    with get_connection() as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM escalations").fetchone()["n"]
    assert rows == 1, f"one problem, one escalation row — got {rows}"


@pytest.mark.asyncio
async def test_resolving_notifies_once_carrying_the_agreed_time(monkeypatch):
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent), raising=True)
    session = _session_with_reply("I'll get that booked.")
    _force_escalation(monkeypatch, "explicit request for a human", mandatory=True)
    await run_turn(session, "get me a person")
    assert sent == [], "nothing goes out at open"

    session.gates.escalation.record_resolution("customer_will_reach_out")
    _force_escalation(monkeypatch, None, mandatory=False)
    await run_turn(session, "I'll ring you back myself")

    assert len(sent) == 1
    assert sent[0]["resolution"] == "customer_will_reach_out"
```

Add these helpers to `tests/test_session.py`:

```python
def _force_escalation(monkeypatch, reason: str | None, mandatory: bool):
    """Drive run_turn's escalation branch without an API call. Patches
    check_escalation, which is the seam run_turn actually uses.
    """
    signal = None if reason is None else escalation.EscalationSignal(reason, mandatory=mandatory)

    async def _check(*args, **kwargs):
        return signal

    monkeypatch.setattr(escalation, "check_escalation", _check)

    async def _infer(customer_id, messages, client=None):
        return escalation.HandoffFields(
            customer_intent="wants a person",
            conversation_summary="asked for a human",
            verified_account_info=None,
            actions_taken=None,
            sentiment="neutral",
        )

    monkeypatch.setattr(escalation, "_infer_handoff_fields", _infer)
```

`_session_with_reply` should follow whatever stub-client pattern `tests/test_session.py` already uses for a canned reply — **read the file and reuse it**; do not invent a new helper.

- [ ] **Step 2: Run to verify they fail**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_session.py -q -k "mandatory or suggested or amends or resolving"`
Expected: FAIL — `outcome.ended` is still `True`, `EscalationSignal` unused by `run_turn`.

- [ ] **Step 3: Add the offer wording**

In `agent/session.py`, beside `_escalation_notice`:

```python
def _escalation_offer(reason: str) -> str:
    """What the customer hears when the agent SUSPECTS it is failing them.

    An offer, not an announcement, and it names a way to carry on — the live
    failure was a customer who had merely mis-dictated one digit of an order
    number, said so immediately, and was handed off and hung up on. Being
    one transposed digit from success is not a reason to end someone's call.

    Deliberately does not name the internal reason. "Sustained negative
    sentiment across multiple turns" read aloud tells an already frustrated
    customer they have been classified as angry.
    """
    if reason == "repeated failed lookups":
        return (
            "I'm still not finding that. Would you like to try the number once more, "
            "or shall I have a colleague call you back about it?"
        )
    return "Would it help if I arranged for a colleague to call you back about this?"
```

- [ ] **Step 4: Rewrite the escalation branch**

Replace `agent/session.py:381-399` (from `escalation_id: int | None = None` through the `elif should_end_session(...)` block):

```python
    state = session.gates.escalation
    escalation_id: int | None = state.escalation_id
    packet: dict[str, Any] | None = state.packet
    notice: str | None = notice  # preserve any hedge notice already set

    if signal and signal.mandatory:
        # The customer asked, or a rule requires it. No permission to seek.
        if state.status == "none":
            try:
                packet = await escalation.open_escalation(
                    session.customer_id, session.agent.messages, signal.reason
                )
                state.open(signal.reason)
                state.escalation_id = packet["escalation_id"]
                state.packet = packet
                escalation_id = state.escalation_id
                notice = _escalation_handover_notice()
            except Exception as exc:  # noqa: BLE001 — must never crash a live turn
                warnings.append(f"Escalation triggered ({signal.reason}) but couldn't be logged: {exc}")
        else:
            # Already escalating. One customer, one callback — this becomes
            # another item the same colleague can prepare for.
            state.amend(signal.reason)
    elif signal:
        # The agent's own inference. Offer, once, and reset the counter that
        # produced it so the customer gets a clean run at correcting
        # themselves rather than tripping the same threshold next breath.
        if signal.reason not in state.offered and not state.is_open:
            state.offered.add(signal.reason)
            notice = _escalation_offer(signal.reason)
        session.tracker.reset_streak(signal.reason)

    # Persist and notify whatever the tool loop resolved this turn. All the
    # async work lives here rather than in the tools, because dispatch_tool
    # calls handlers without awaiting them — a tool physically cannot do I/O.
    if state.pending_persist and state.status == "resolved":
        try:
            if state.packet is None:
                state.packet = await escalation.open_escalation(
                    session.customer_id, session.agent.messages, state.items[0]
                )
                state.escalation_id = state.packet["escalation_id"]
                escalation_id = state.escalation_id
            await escalation.resolve_escalation(
                state.packet,
                items=state.items,
                resolution=state.resolution,
                callback_time=state.callback_time,
            )
            packet = state.packet
            state.pending_persist = False
        except Exception as exc:  # noqa: BLE001 — must never crash a live turn
            warnings.append(f"Handover resolved but couldn't be recorded: {exc}")

    if should_end_session(result.tool_calls):
        outcome = TurnOutcome(
            reply=reply,
            ended=True,
            end_reason="model_ended",
            notice=notice,
            escalation_packet=packet,
            llm_latency_seconds=llm_latency,
            warnings=warnings,
        )
    else:
        outcome = TurnOutcome(
            reply=reply,
            notice=notice,
            escalation_packet=packet,
            llm_latency_seconds=llm_latency,
            warnings=warnings,
        )
```

Rename the `reason` variable assigned from `check_escalation` to `signal`, and change the telemetry call's `escalation_reason=reason` to `escalation_reason=signal.reason if signal else None` and `escalated=state.status != "none"`.

Add, beside `_escalation_offer`:

```python
def _escalation_handover_notice() -> str:
    """What the customer hears the moment a handover opens.

    No time is named yet — that is the whole change. Escalation used to
    announce a callback slot the customer had never agreed to and then end
    the call; now the agent says a colleague will help and then has to
    actually arrange it before it is allowed to say goodbye.
    """
    return (
        "Let me get one of my colleagues onto this for you. "
        "When would be a good time for them to give you a call?"
    )
```

Keep `_escalation_notice` and `_spoken_time` — `_spoken_time` is still used for speaking a confirmed slot aloud, and `_escalation_notice` is still reachable through the DTMF transfer path. If the task finds `_escalation_notice` genuinely has no remaining caller, delete it rather than leaving dead code.

- [ ] **Step 5: Run the tests**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
Expected: PASS. Existing tests asserting `end_reason == "escalated"` will fail — that behaviour is deliberately gone; update them to assert the new state machine instead of deleting them.

- [ ] **Step 6: Commit**

```bash
git add agent/session.py tests/test_session.py
git commit -m "Escalation opens a process instead of ending the call"
```

---

### Task 6: Notify an abandoned handover at `close_session`

**Files:**
- Modify: `agent/session.py:447-460` (`close_session`)
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: `close_session` notifies an unresolved escalation with `resolution="unresolved"`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_an_abandoned_handover_still_reaches_a_human(monkeypatch):
    """Customers hang up, sockets die, processes restart. A handover that
    never got resolved must still tell the human agent about the customer —
    and tell them honestly that no time was agreed.
    """
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent), raising=True)
    session = _session_with_reply("Understood.")
    _force_escalation(monkeypatch, "explicit request for a human", mandatory=True)
    await run_turn(session, "get me a person")
    monkeypatch.setattr(summary, "close_session", _stub_summary_close)

    await close_session(session)

    assert len(sent) == 1, "exactly one notification per escalation, even an abandoned one"
    assert sent[0]["resolution"] == "unresolved"
    assert sent[0]["callback_time"] is None, "no time was agreed — say so"


@pytest.mark.asyncio
async def test_a_resolved_handover_is_not_notified_again_at_close(monkeypatch):
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent), raising=True)
    session = _session_with_reply("Understood.")
    _force_escalation(monkeypatch, "explicit request for a human", mandatory=True)
    await run_turn(session, "get me a person")
    session.gates.escalation.record_resolution("customer_will_reach_out")
    _force_escalation(monkeypatch, None, mandatory=False)
    await run_turn(session, "I'll call back")
    monkeypatch.setattr(summary, "close_session", _stub_summary_close)

    await close_session(session)

    assert len(sent) == 1, "resolution already sent the one notification"
```

`_stub_summary_close` should mirror however `tests/test_session.py` already stubs the summary call — read the file and reuse it.

- [ ] **Step 2: Run to verify they fail**

Expected: FAIL — nothing is sent at close.

- [ ] **Step 3: Implement**

In `agent/session.py`, at the top of `close_session`, before the `if not session.agent.messages` guard:

```python
    # An escalation that never reached a resolution: the customer hung up,
    # the socket died, the process restarted. The human agent still needs to
    # hear about this customer — and needs to know that no time was agreed,
    # rather than being handed a callback slot nobody promised.
    #
    # Runs before the early return on purpose. An escalation cannot exist
    # without messages, but ordering the guard the other way would make that
    # a silent dependency rather than an obvious one.
    state = session.gates.escalation
    if state.is_open and state.packet is not None:
        try:
            await escalation.resolve_escalation(
                state.packet, items=state.items, resolution="unresolved", callback_time=None
            )
            state.pending_persist = False
        except Exception:  # noqa: BLE001 — an exit path must never crash
            logger.exception("could not notify an abandoned escalation")
```

Add a module-level `logger = logging.getLogger(__name__)` and `import logging` to `agent/session.py` if not already present.

- [ ] **Step 4: Run and commit**

```bash
ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q
git add agent/session.py tests/test_session.py
git commit -m "Notify an abandoned handover when the session closes"
```

---

### Task 7: Teach the model how to work an escalation

**Files:**
- Modify: `agent/prompts.py` (`SYSTEM_PROMPT`)
- Test: `tests/test_prompts.py` (or wherever prompt assertions already live — check first)

**Interfaces:** Consumes Tasks 3-5. Produces no new API.

- [ ] **Step 1: Add the guidance**

Add a section to `SYSTEM_PROMPT`, worded for a voice call:

```
## Handing over to a colleague

Sometimes you cannot finish something yourself — the customer asks for a
person, a policy needs a specialist, or you are simply not getting anywhere.
When that happens you are not finished: you have to arrange the handover
before the call can end.

- Ask when would suit them for a colleague to call back. Then call
  find_available_slots and offer them a real time.
- When they pick one, call schedule_human_callback. The first call asks them
  to confirm; call it again with the same time once they say yes.
- If they would rather get in touch themselves, call
  record_customer_will_reach_out. That is a perfectly good outcome.
- Never say goodbye before one of those two has gone through. If you try,
  you will be told so, and you should go back and arrange it.

Once the handover is arranged, the call is not over. Ask whether there is
anything else you can help with, and if there is, just help with it
normally — a refund you can process yourself gets processed, not handed
over. Only pass on something genuinely beyond you, and when you do, do not
arrange a second callback: the same colleague will cover it on the same
call. Say so, simply — "I'll add that to what they're calling you about."
```

- [ ] **Step 2: Add an assertion that the two tool names in the prompt actually exist**

```python
def test_the_prompt_only_names_tools_that_exist():
    """A prompt naming a tool that was renamed is a silent failure — the
    model calls it, gets "unknown tool", and improvises.
    """
    from agent.session import TOOLS

    names = {schema["name"] for schema in TOOLS}
    for named in ("find_available_slots", "schedule_human_callback", "record_customer_will_reach_out"):
        assert named in names, f"SYSTEM_PROMPT names {named}, which is not a registered tool"
        assert named in SYSTEM_PROMPT
```

- [ ] **Step 3: Run and commit**

```bash
ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q
git add agent/prompts.py tests/test_prompts.py
git commit -m "Tell the model how to work a handover"
```

---

### Task 8: The independent defect — `ended=True` did not stop the session

**Files:**
- Investigate: `transport/pipecat_processors.py`, `transport/browser.py`, `transport/text_cli.py`
- Test: `tests/test_pipecat_processors.py`

**This task is an investigation, not a prescription.** The live call showed `ended=True` returned at turn 7 with turns 9 and 10 proceeding normally afterwards — which is also why nothing was spoken at the real ending. Phase 12 removes the *specific* case that produced it (escalation no longer ends a call at detection), but the underlying "a turn said the call was over and it wasn't" behaviour would bite any `end_conversation` and needs its own test.

- [ ] **Step 1: Reproduce it in a test before changing anything**

Write a test that drives `ClaudeTurnProcessor` through a turn returning `ended=True`, then feeds it another transcript, and asserts the second transcript produces no reply. Follow the existing `_transcript` / `_started` helpers in `tests/test_pipecat_processors.py`.

If the test passes immediately, the defect is not in the processor — say so and move the investigation to the transport that hosted the live call (`transport/browser.py`), rather than declaring it fixed. A defect you cannot reproduce is not a defect you have fixed.

- [ ] **Step 2: Fix the smallest thing that makes the test pass**

Rule 5 applies: whatever this is, it is a transport concern. Nothing in `agent/` should need to change. If it does, stop and flag it — that means the abstraction broke.

- [ ] **Step 3: Commit**

```bash
git add transport tests/test_pipecat_processors.py
git commit -m "A turn that ends the call actually ends it"
```

---

## Checkpoint

**Automated:** the full suite passes offline with no API keys:
`ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`

**Manual (needs credits + the browser console):** one live call that

1. escalates on an explicit request for a human, and **does not hang up**;
2. agrees a callback time, confirms it, and books a real `appointments` row;
3. asks "anything else?", and handles a second, ordinary issue normally;
4. has a second escalation-worthy issue **amend** the existing handover rather than booking a second callback;
5. ends with a spoken goodbye;
6. delivers **exactly one** Slack message, carrying both items and the agreed time.

Then a second call that mis-dictates an order number twice and confirms the agent **offers** a callback rather than imposing one, and carries on normally when the customer says they would rather try again.

## Self-review notes

- **Spec coverage:** every section of the spec maps to a task — triggers (1), notification timing (2), resolution paths (3), `end_conversation` refusal (4), state machine + the no-tool-call counter bug + one-row-per-problem (1, 5), abandoned calls (6), prompt (7), the `ended=True` defect (8). Deviations are recorded as R1-R4 above rather than absorbed silently.
- **Type consistency:** `EscalationSignal` is produced in Task 1 and consumed in Task 5; `EscalationState` is produced in Task 1 and consumed in Tasks 3, 4, 5, 6; `END_REFUSED_PREFIX` is produced in Task 4 and consumed by `should_end_session` in the same task.
- **Known soft spot:** Task 5's diff is the largest and the only one touching a function with many existing callers. If a reviewer rejects one task, it will be that one.
