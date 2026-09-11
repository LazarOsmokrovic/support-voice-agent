# Phase 12 — Escalation as a Resolvable Process (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make escalation a state the agent must resolve — a callback the customer agrees to, or a record that they will make contact themselves — before the call can end, **without breaking a single thing that works today**.

**Architecture:** Escalation state becomes a mutable object on `SessionGates`, mutated by synchronous tool handlers and read by `run_turn`, which keeps all async work. Triggers split into *mandatory* (open directly) and *suggested* (offer first). `end_reason="escalated"` is **preserved** and moves from detection-time to end-of-call.

**Tech Stack:** Python 3.12, SQLite, Pydantic, `anthropic` SDK, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-10-phase-12-escalation-process-design.md`

**Supersedes:** `docs/superpowers/plans/2026-09-10-phase-12-escalation-process.md` — that plan silently broke Phase 10d and 6 eval scenarios (see C-1). Do not implement it.

---

## Global Constraints

- **Touch only these files.** Anything else is out of scope for this phase:
  `agent/tools/escalation.py`, `agent/tools/handoff.py` (new), `agent/tools/summary.py`,
  `agent/session.py`, `agent/prompts.py`, `data/mock_db.py`, and their tests.
- **`transport/` is untouched** (CLAUDE.md rule 5). If a task believes it must change a
  transport, **stop and flag it** — that means the abstraction broke.
- **`observability/turn_log.py` is untouched.** See D-2 for why, and what pays for it.
- **CLAUDE.md rule 6:** the callback booking goes through `PendingActionGate`.
- **CLAUDE.md rule 7:** resolution is a deterministic tool call. The model judges only
  genuinely ambiguous prose (did the customer accept?), never whether one has happened.
- **Never raise on an exit path.** `run_turn` and `close_session` already wrap every
  fallible call; new code follows the same shape.
- **No hard-coded values** — order IDs, slot times and customer IDs come from
  `data/mock_db.py` at runtime.
- Run the suite with keys blanked so no task spends API credit:
  `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
  (`env -u` does **not** work — `load_dotenv()` repopulates a *missing* variable; an
  empty string survives.) Baseline today: **359 passed, 3 skipped**.

---

## What the last plan got wrong

**C-1 (CRITICAL — the reason this replan exists).** The v1 plan removed the value
`end_reason="escalated"`. Two live consumers depend on that exact string:

- `transport/pipecat_processors.py:393` fires Phase 10d's Twilio warm-transfer hook on
  `outcome.end_reason == "escalated"`. Remove the value and the transfer silently never
  happens — an entire completed sub-phase becomes unreachable, discovered only whenever
  a Twilio number is finally funded.
- `eval/scenarios.py` pins it in **6** scenarios (lines 315, 431, 448, 486, 499, 519).

And v1's own Global Constraints forbade touching `transport/`, so it broke something it
was not allowed to repair.

**The fix is not to delete the value — it is to move when it is set.** See D-1.

**C-2 (CRITICAL).** v1 made `end_conversation` refuse *unboundedly* while an escalation
was open. A customer who says "just let me go" and never picks a callback could not
hang up, and the refusal text told the model to offer a callback again — reintroducing
"being asked repeatedly whether you want a human" at the exit. See Task 4.

---

## Design decisions

**D-1 — `end_reason="escalated"` is preserved, and moves to the end of the call.**

Today it means *"an escalation just happened, so the call is over."* After this phase it
means *"this call involved an escalation."* It is set on the turn that actually ends the
call, whenever the session's escalation state is not `none`.

This is strictly better for Phase 10d, not merely compatible: the Twilio hook now fires
with a packet that carries the **agreed callback time** and every item gathered, instead
of a packet assembled the instant a trigger fired and nothing agreed. The whisper to the
human agent gets better without one line of `transport/` changing.

**D-2 — `observability/turn_log.py` is not touched.**

The CEO review rated "no observability on the new state machine" HIGH, and it is a fair
finding: `TurnRecord` carries only detection-time fields. But the `escalations` table
gains `resolution`, `callback_time` and `resolved_at` in Task 1, so the durable answer to
"do escalations actually get resolved?" is one SQL query away. Adding turn-log fields
would touch a file this phase has no other reason to open. **Deferred, with the DB
carrying the weight.** If it later proves insufficient, it is a two-line change.

**D-3 — Escalation state lives on `SessionGates`, not `Session`.**

`build_dispatch_tool` (`agent/session.py:72`) builds the tool handlers **before** the
`Session` exists, so a handler cannot close over a field later set on `Session`.
`SessionGates` is already the mutable-per-session container the handlers close over, it
is already threaded in, and escalation state is literally a gate — it gates
`end_conversation`. Zero signature changes; the three existing 3-tuple unpackings
(`agent/session.py:199`, `tests/test_text_cli.py:43`, `tests/test_text_cli.py:126`) keep
working untouched.

**D-4 — All async work stays in `run_turn`; the resolution tools are synchronous.**

`dispatch_tool` calls `handler(**tool_input)` with no `await` (`agent/session.py:102`), so
a tool physically cannot do I/O. The tools record an outcome into the shared state;
`run_turn` persists and notifies — which is exactly where escalation already does this.

**D-5 — A suggested trigger never opens an escalation; it only offers.**

Implementing the spec literally would need `run_turn` to classify "did they decline?" —
another inference, the exact class of judgement that caused this phase. Instead a
suggested trigger resets its own counter, retires itself for the session, and appends an
offer. If the customer accepts, the model calls a resolution tool and *that* opens the
escalation. Nothing to detect.

**D-6 — `create_handoff_packet` keeps its name, signature and behaviour**, because
`transport/pipecat_processors.py:272` (DTMF zero) calls it and must not change. It
becomes "open and immediately resolve as a transfer" — which is what pressing zero means.

---

## File Structure

| File | Change |
|---|---|
| `agent/tools/escalation.py` | `EscalationState`, `EscalationSignal`, `MANDATORY_REASONS`, resolution constants, `reset_streak`, `open_escalation`, `mark_resolved`, `resolve_escalation`. `record_turn` returns a signal. `create_handoff_packet` becomes a wrapper. |
| `agent/tools/handoff.py` | **New.** Two sync resolution tools + schemas. |
| `agent/tools/summary.py` | `end_conversation` gains a **bounded** refusal. |
| `agent/session.py` | `SessionGates.escalation`; `TOOLS`; handlers; `should_end_session`; `run_turn`'s escalation branch; `close_session`. |
| `agent/prompts.py` | Handover guidance. |
| `data/mock_db.py` | Four `escalations` columns. |
| `tests/test_escalation.py`, `tests/test_handoff.py` (new), `tests/test_session.py` | Coverage. |

---

### Task 1: Escalation state, the trigger split, and the schema

**Files:**
- Modify: `agent/tools/escalation.py` (add above `class EscalationTracker`, ~line 135)
- Modify: `data/mock_db.py` (the `escalations` CREATE TABLE, ~line 67)
- Test: `tests/test_escalation.py`

**Interfaces produced:**
- `RESOLUTION_CALLBACK = "callback"`, `RESOLUTION_SELF = "customer_will_reach_out"`, `RESOLUTION_TRANSFER = "transfer"`, `RESOLUTION_UNRESOLVED = "unresolved"` — named constants, so the four magic strings never drift across four files.
- `STATUS_NONE = "none"`, `STATUS_OPEN = "open"`, `STATUS_RESOLVED = "resolved"`
- `MANDATORY_REASONS: frozenset[str]`
- `EscalationSignal(reason: str, mandatory: bool)` — frozen dataclass
- `EscalationState` — `status`, `escalation_id`, `packet`, `items: list[str]`, `resolution`, `callback_time`, `offered: set[str]`, `pending_persist: bool`, `refusals: int`; methods `open`, `amend`, `record_resolution`, property `is_open`
- `EscalationTracker.record_turn` → `EscalationSignal | None` (was `str | None`)
- `EscalationTracker.reset_streak(reason)` → `None`
- `check_escalation` → `EscalationSignal | None`

- [ ] **Step 1: Write the failing tests**

```python
def test_a_mandatory_trigger_is_marked_mandatory():
    """An explicit request for a human is the customer's decision, not the
    agent's inference, so it opens a handover without asking permission."""
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
    """The live failure: a customer mis-dictated an order number, corrected
    themselves immediately, and was escalated and hung up on. Failed lookups
    are the AGENT's inference that it is failing — sometimes true, sometimes
    a transposed digit. So it offers rather than imposes."""
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    failed = [{"name": "get_order_status", "output": {"found": False}}]
    for _ in range(escalation.FAILED_LOOKUP_ESCALATION_THRESHOLD):
        signal = tracker.record_turn(neutral, failed)
    assert signal is not None
    assert signal.reason == "repeated failed lookups"
    assert signal.mandatory is False


def test_sustained_negative_sentiment_is_only_a_suggestion():
    """This one hung up on a customer who was calmly cancelling an order.
    The classification prompt was fixed separately; making the trigger a
    suggestion is the structural half — a misread becomes an offer the
    customer can decline instead of a verdict."""
    tracker = escalation.EscalationTracker()
    upset = escalation.TurnClassification(
        intent="complaint", sentiment="negative", policy_restricted=False
    )
    for _ in range(escalation.NEGATIVE_SENTIMENT_ESCALATION_THRESHOLD):
        signal = tracker.record_turn(upset, [])
    assert signal is not None
    assert signal.mandatory is False


def test_resetting_a_streak_gives_the_customer_a_clean_run():
    """Declining an offer must reset the counter that produced it, or the
    agent simply asks again one turn later."""
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    failed = [{"name": "get_order_status", "output": {"found": False}}]
    for _ in range(escalation.FAILED_LOOKUP_ESCALATION_THRESHOLD):
        tracker.record_turn(neutral, failed)

    tracker.reset_streak("repeated failed lookups")

    assert tracker.consecutive_failed_lookups == 0
    assert tracker.record_turn(neutral, failed) is None


def test_a_turn_with_no_tool_calls_does_not_advance_the_lookup_counter():
    """Turn 10 of the live call escalated while the agent was merely asked to
    repeat a number back — no lookup happened at all."""
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    tracker.record_turn(neutral, [{"name": "get_order_status", "output": {"found": False}}])
    before = tracker.consecutive_failed_lookups

    tracker.record_turn(neutral, [])

    assert tracker.consecutive_failed_lookups == before


def test_the_state_machine_opens_amends_and_resolves():
    state = escalation.EscalationState()
    assert state.status == escalation.STATUS_NONE
    assert state.is_open is False

    state.open("explicit request for a human")
    assert state.status == escalation.STATUS_OPEN
    assert state.items == ["explicit request for a human"]

    state.amend("policy-restricted topic")
    assert state.status == escalation.STATUS_OPEN, "an amendment never reopens"
    assert state.items == ["explicit request for a human", "policy-restricted topic"]

    state.record_resolution(escalation.RESOLUTION_CALLBACK, callback_time="2026-09-14T09:00:00")
    assert state.status == escalation.STATUS_RESOLVED
    assert state.is_open is False
    assert state.pending_persist is True


def test_amending_a_resolved_escalation_keeps_it_resolved():
    """A second issue reaches the same human on the same callback. It must not
    drag the call back into an unresolvable state."""
    state = escalation.EscalationState()
    state.open("explicit request for a human")
    state.record_resolution(escalation.RESOLUTION_SELF)
    state.pending_persist = False

    state.amend("high-value refund requires approval")

    assert state.status == escalation.STATUS_RESOLVED
    assert len(state.items) == 2
    assert state.pending_persist is True, "an amendment after resolution sends an update"


def test_a_repeated_trigger_does_not_duplicate_an_item():
    """One tripped trigger firing every turn produced escalation rows 7, 8 and
    9 for one problem in a live call."""
    state = escalation.EscalationState()
    state.open("explicit request for a human")
    state.amend("explicit request for a human")
    assert state.items == ["explicit request for a human"]
```

- [ ] **Step 2: Run to verify they fail**

`ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_escalation.py -q`
Expected: FAIL — `EscalationState` does not exist; `record_turn` returns a string.

- [ ] **Step 3: Add the constants, signal, and state**

In `agent/tools/escalation.py`, above `class EscalationTracker`:

```python
# Named rather than repeated as literals: these four values travel across
# escalation.py, handoff.py, session.py and a DB column, and a typo in any one
# of them fails silently as "this escalation is somehow neither open nor
# resolved".
STATUS_NONE = "none"
STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"

RESOLUTION_CALLBACK = "callback"
RESOLUTION_SELF = "customer_will_reach_out"
RESOLUTION_TRANSFER = "transfer"
RESOLUTION_UNRESOLVED = "unresolved"

# Whose decision each trigger represents. A mandatory trigger is the
# customer's or a rule's — the agent has no standing to second-guess it. A
# suggested trigger is the agent's own inference that it is failing, which is
# exactly the judgement that hung up on a customer who had mis-dictated one
# digit, and on another who was calmly cancelling an order. Inferences get
# offered; they do not get imposed. This is CLAUDE.md rule 6's
# propose-then-confirm principle applied to handoffs.
MANDATORY_REASONS = frozenset(
    {
        "explicit request for a human",
        "policy-restricted topic",
    }
)


@dataclass(frozen=True)
class EscalationSignal:
    """A fired trigger, and whether the agent must act or merely offer."""

    reason: str
    mandatory: bool


@dataclass
class EscalationState:
    """One handover per session, for the whole life of the session.

    That is what happens on a real support line: a human ringing a customer
    back deals with everything that customer has, rather than booking three
    calls for three questions. So a second escalation-worthy issue becomes
    another ITEM on the same handover — which is why `items` is a list and not
    the single `reason` string it replaced.

    Mutated in place, never rebound, because the tool handlers in
    build_dispatch_tool close over it before the Session that owns it exists.
    """

    status: str = STATUS_NONE
    escalation_id: int | None = None
    packet: dict[str, Any] | None = None
    items: list[str] = field(default_factory=list)
    resolution: str | None = None
    callback_time: str | None = None
    # Suggested triggers that have already made their offer. Being asked over
    # and over whether you would like a human is its own kind of failure.
    offered: set[str] = field(default_factory=set)
    # Work run_turn still has to persist or notify. Cleared only on success,
    # so a turn that dies mid-persist retries on the next one.
    pending_persist: bool = False
    # How many times end_conversation has been refused. Bounded — see Task 4.
    refusals: int = 0

    @property
    def is_open(self) -> bool:
        return self.status == STATUS_OPEN

    def open(self, reason: str) -> None:
        if self.status == STATUS_NONE:
            self.status = STATUS_OPEN
        if reason not in self.items:
            self.items.append(reason)
        self.pending_persist = True

    def amend(self, reason: str) -> None:
        """A second trigger during an existing handover. Never reopens a
        resolved one and never books a second callback — it adds an item the
        human taking it over can prepare for.
        """
        if reason in self.items:
            return
        self.items.append(reason)
        self.pending_persist = True

    def record_resolution(self, resolution: str, callback_time: str | None = None) -> None:
        self.status = STATUS_RESOLVED
        self.resolution = resolution
        self.callback_time = callback_time
        self.pending_persist = True
```

Add `field` to the `from dataclasses import ...` line if absent.

- [ ] **Step 4: Return a signal, and fix the counter bug**

Rewrite `EscalationTracker.record_turn`'s body so every `return "<reason>"` becomes an
`EscalationSignal`, and **move the failed-lookup threshold check inside `if outcomes:`**:

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
            return EscalationSignal(
                "sustained negative sentiment across multiple turns", mandatory=False
            )

        outcomes = _turn_tool_outcomes(tool_calls)
        if outcomes:
            if any(outcomes):
                self.consecutive_failed_lookups = 0
            else:
                self.consecutive_failed_lookups += 1
            # Checked INSIDE this block on purpose. It used to sit outside,
            # re-reading the counter every turn — so once the streak tripped, a
            # turn containing no lookup at all still escalated. One live call
            # produced escalation rows 7, 8 and 9 for one problem that way, the
            # last from a turn that merely asked the agent to repeat a number.
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

Then add:

```python
    def reset_streak(self, reason: str) -> None:
        """Clear the counter behind a suggested trigger, so a declined offer
        gives the customer a clean run rather than tripping the same threshold
        on their next breath.
        """
        if reason == "repeated failed lookups":
            self.consecutive_failed_lookups = 0
        elif reason == "sustained negative sentiment across multiple turns":
            self.consecutive_negative_turns = 0
        elif reason == "repeated ungrounded replies":
            self.consecutive_ungrounded_replies = 0
```

Update `record_turn`'s and `check_escalation`'s return annotations to
`EscalationSignal | None`. `check_escalation`'s body needs no change.

- [ ] **Step 5: Add the schema columns**

In `data/mock_db.py`, inside `CREATE TABLE IF NOT EXISTS escalations`, after
`notified_at` and before the `FOREIGN KEY` line:

```sql
    -- Phase 12: a handover is a process, not an event. `items` is a
    -- newline-separated list of every reason attached to this ONE handover —
    -- one customer gets one callback, so a second escalation-worthy issue
    -- appends here rather than opening a second row. A reason is written by
    -- this codebase, never by a customer, so it cannot contain a newline.
    -- `resolution` is null while the handover is still open, which is also how
    -- an abandoned call is recognised at close_session.
    items                   TEXT,
    resolution              TEXT,
    callback_time           TEXT,
    resolved_at             TEXT,
```

The `.db` file is gitignored and regenerated: `python -m data.mock_db` picks the new
columns up, the same convention every prior phase used. **Run it as part of this step** —
a stale DB is exactly what produced `no such table: escalations` in a live call.

- [ ] **Step 6: Run the tests, and fix the call sites this breaks**

`ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`

Existing tests comparing `check_escalation`/`record_turn` to a string will fail. **Fix
them to read `signal.reason`** — do not weaken an assertion to make it pass. A test that
still passes while comparing a dataclass to a string has stopped checking anything.

- [ ] **Step 7: Commit**

```bash
git add agent/tools/escalation.py data/mock_db.py tests/
git commit -m "Split escalation triggers into mandatory and suggested"
```

---

### Task 2: Notify once, at resolution

**Files:**
- Modify: `agent/tools/escalation.py` (`create_handoff_packet`, ~line 294)
- Test: `tests/test_escalation.py`

**Interfaces produced:**
- `async open_escalation(customer_id, messages, reason, client=None) -> dict` — infers, logs the row, returns the packet. **Notifies nobody.**
- `mark_resolved(escalation_id, items, resolution, callback_time=None, resolved_at=None) -> None`
- `async resolve_escalation(packet, items, resolution, callback_time=None) -> bool` — persists, then notifies. Never raises.
- `create_handoff_packet` — unchanged signature and observable behaviour (D-6).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_opening_a_handover_writes_a_row_but_tells_nobody(monkeypatch):
    """Until an outcome is known there is nothing useful to tell a human. "A
    customer needs help, we don't know what about or when to ring" is a
    message they have to chase.

    The row is still written immediately, so a dropped call leaves a record."""
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)

    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "get me a person"}],
        "explicit request for a human",
    )

    assert packet["escalation_id"] is not None
    assert sent == [], "opening must not notify"
    with get_connection() as conn:
        row = conn.execute(
            "SELECT resolution FROM escalations WHERE escalation_id = ?",
            (packet["escalation_id"],),
        ).fetchone()
    assert row["resolution"] is None


@pytest.mark.asyncio
async def test_resolving_notifies_exactly_once_with_the_agreed_time(monkeypatch):
    """"Notifies twice" is the failure a reader cannot see, so this counts
    calls rather than inspecting the last one."""
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)
    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "human please"}],
        "explicit request for a human",
    )
    slot = "2026-09-14T09:00:00"

    delivered = await escalation.resolve_escalation(
        packet,
        items=["explicit request for a human"],
        resolution=escalation.RESOLUTION_CALLBACK,
        callback_time=slot,
    )

    assert delivered is True
    assert len(sent) == 1, f"exactly one notification per handover, got {len(sent)}"
    assert sent[0]["resolution"] == escalation.RESOLUTION_CALLBACK
    assert sent[0]["callback_time"] == slot
    assert sent[0]["items"] == ["explicit request for a human"]


@pytest.mark.asyncio
async def test_a_notification_failure_still_leaves_the_handover_resolved(monkeypatch):
    """Delivery has never been allowed to affect persistence (Phase 11)."""
    async def _explode(packet, **kwargs):
        raise RuntimeError("webhook down")

    monkeypatch.setattr(escalation, "notify_escalation", _explode)
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)
    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "human please"}],
        "explicit request for a human",
    )

    delivered = await escalation.resolve_escalation(
        packet, items=["explicit request for a human"],
        resolution=escalation.RESOLUTION_SELF,
    )

    assert delivered is False
    with get_connection() as conn:
        row = conn.execute(
            "SELECT resolution, resolved_at FROM escalations WHERE escalation_id = ?",
            (packet["escalation_id"],),
        ).fetchone()
    assert row["resolution"] == escalation.RESOLUTION_SELF
    assert row["resolved_at"] is not None


@pytest.mark.asyncio
async def test_create_handoff_packet_still_opens_and_resolves_together(monkeypatch):
    """transport/pipecat_processors.py:272 (DTMF zero) calls this and must not
    change — CLAUDE.md rule 5. Pressing zero IS the resolution: the caller is
    being put through right now, so there is no process to work through."""
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)

    packet = await escalation.create_handoff_packet(
        CUSTOMER_ID, [{"role": "user", "content": "0"}], "caller pressed 0 for a human"
    )

    assert packet["escalation_id"] is not None
    assert "callback_time" in packet, "the existing contract includes a callback time"
    assert len(sent) == 1
```

Helpers (add near the top of `tests/test_escalation.py` if not already present):

```python
def _recording_notifier(sink: list[dict]):
    async def _notify(packet, **kwargs):
        sink.append(packet)
        return True
    return _notify


async def _stub_infer_fields(customer_id, messages, client=None):
    """Keeps these tests offline; the inference is covered separately."""
    return escalation.HandoffFields(
        customer_intent="wants a person",
        conversation_summary="asked for a human",
        verified_account_info=None,
        actions_taken=None,
        sentiment="neutral",
    )
```

- [ ] **Step 2: Run to verify they fail** — `open_escalation` does not exist.

- [ ] **Step 3: Split the function**

Replace `create_handoff_packet` with `open_escalation`, `mark_resolved`,
`resolve_escalation`, and a thin `create_handoff_packet` wrapper. Key points:

- `open_escalation` is today's `create_handoff_packet` **minus** the notify and
  `mark_notified` calls, plus `"items": [reason]` and `"resolution": None` in the packet.
  Keep the existing `redact_fields` call and its comment verbatim — redaction happens
  once so the DB row and the webhook carry identical text.
- `resolve_escalation` **persists before notifying**: a webhook that hangs for its full
  retry budget must not leave the row claiming the handover is still open. Keep the two
  separate `try/except` blocks around `mark_resolved`/`notify_escalation` and the third
  around `mark_notified`, matching the existing code's shape and comments.
- `create_handoff_packet` = `open_escalation` then `resolve_escalation(...,
  resolution=RESOLUTION_TRANSFER, callback_time=packet["callback_time"])`, returning the
  packet.

- [ ] **Step 4: Run the full suite. Step 5: Commit.**

```bash
git commit -am "Notify once, at resolution, not at detection"
```

---

### Task 3: The two resolution tools

**Files:**
- Create: `agent/tools/handoff.py`
- Modify: `agent/session.py` — `TOOLS` (~line 38), `SessionGates` (~line 56), `handlers` (~line 83)
- Test: `tests/test_handoff.py`

**Interfaces produced:**
- `SCHEDULE_CALLBACK_SCHEMA`, `RECORD_CALLBACK_DECLINED_SCHEMA`
- `schedule_human_callback(slot_time, state, gate, customer_id) -> dict` — **sync**
- `record_customer_will_reach_out(escalation) -> dict` — **sync**
- `SessionGates.escalation: EscalationState`

- [ ] **Step 1: Write the failing tests** (`tests/test_handoff.py`)

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
from agent.tools.escalation import (
    RESOLUTION_CALLBACK, RESOLUTION_SELF, STATUS_NONE, STATUS_OPEN, STATUS_RESOLVED,
    EscalationState,
)
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
    """From the real calendar. A literal date here would quietly start failing
    once it fell outside the booking window."""
    return scheduling.find_available_slots()["slots"][0]


def test_scheduling_a_callback_proposes_before_it_books():
    """CLAUDE.md rule 6: booking is irreversible, so the first call proposes
    and only a later confirmed turn commits."""
    state, gate, slot = _open_state(), PendingActionGate(), _free_slot()

    first = handoff.schedule_human_callback(
        slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID
    )

    assert first["status"] == "pending_confirmation"
    assert state.status == STATUS_OPEN, "a proposal resolves nothing"
    with get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM appointments").fetchone()["n"] == 0


def test_a_confirmed_callback_books_a_real_appointment_and_resolves():
    """The time Slack shows has to be genuinely held, or two customers get
    promised the same slot."""
    state, gate, slot = _open_state(), PendingActionGate(), _free_slot()

    handoff.schedule_human_callback(slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID)
    gate.turn += 1  # the customer's "yes" arrives on a later turn
    result = handoff.schedule_human_callback(
        slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID
    )

    assert result["scheduled"] is True
    assert state.status == STATUS_RESOLVED
    assert state.resolution == RESOLUTION_CALLBACK
    assert state.callback_time == slot
    assert state.pending_persist is True, "run_turn still has to notify"
    with get_connection() as conn:
        assert conn.execute("SELECT scheduled_time FROM appointments").fetchone()["scheduled_time"] == slot


def test_an_unavailable_slot_leaves_the_handover_open():
    """The agent has to offer another time, not carry on as if something was
    arranged."""
    state, gate, slot = _open_state(), PendingActionGate(), _free_slot()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO appointments (customer_id, scheduled_time, reason, status) "
            "VALUES (?, ?, 'taken', 'scheduled')",
            (CUSTOMER_ID, slot),
        )

    result = handoff.schedule_human_callback(
        slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID
    )

    assert result["scheduled"] is False
    assert result["error"] == "slot_unavailable"
    assert state.status == STATUS_OPEN


def test_a_declined_callback_is_a_real_resolution_and_books_nothing():
    """"I'll call back when I know my schedule" is a legitimate ending, not a
    failure — and holding a slot they never agreed to is wrong."""
    state = _open_state()

    result = handoff.record_customer_will_reach_out(escalation=state)

    assert result["recorded"] is True
    assert state.status == STATUS_RESOLVED
    assert state.resolution == RESOLUTION_SELF
    assert state.callback_time is None
    with get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM appointments").fetchone()["n"] == 0


def test_accepting_an_offer_opens_a_handover_that_never_existed():
    """A suggested trigger only offers — nothing is open until the customer
    says yes, so the resolution tool has to open it."""
    state = EscalationState()
    assert state.status == STATUS_NONE

    handoff.record_customer_will_reach_out(escalation=state)

    assert state.status == STATUS_RESOLVED
    assert state.items, "a handover opened by acceptance still needs a reason"


def test_resolving_twice_does_not_book_a_second_callback():
    """One customer, one callback, by design."""
    state, gate, slot = _open_state(), PendingActionGate(), _free_slot()
    handoff.schedule_human_callback(slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID)
    gate.turn += 1
    handoff.schedule_human_callback(slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID)

    second = handoff.record_customer_will_reach_out(escalation=state)

    assert second["recorded"] is False
    assert state.resolution == RESOLUTION_CALLBACK, "the first resolution stands"
    with get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM appointments").fetchone()["n"] == 1
```

- [ ] **Step 2: Run to verify they fail** — no module `agent.tools.handoff`.

- [ ] **Step 3: Write `agent/tools/handoff.py`**

Module docstring must state the sync constraint (D-4) and why these two tools exist.
`schedule_human_callback`:

1. If `state.status == STATUS_RESOLVED` → return `{"scheduled": False, "error":
   "already_resolved", "message": ...}` naming the existing `callback_time` and saying
   anything else raised goes to the same colleague.
2. Otherwise call `scheduling.book_appointment(slot_time=slot_time, reason="callback from
   a human agent", state=gate, customer_id=customer_id)`.
3. `status == "pending_confirmation"` → pass the message through unchanged.
4. `not booked` → pass `error`/`message` through, so the model sees `slot_unavailable`
   rather than a generic failure. **State stays open.**
5. Booked → `_ensure_open(state)`, `state.record_resolution(RESOLUTION_CALLBACK,
   callback_time=slot_time)`, return `{"scheduled": True, "appointment_id": ...,
   "callback_time": slot_time, "message": ...}`.

`record_customer_will_reach_out(escalation)`: refuse if already resolved, else
`_ensure_open` + `record_resolution(RESOLUTION_SELF)`.

`_ensure_open(state)`: if `status == STATUS_NONE`, `state.open(ACCEPTED_OFFER_REASON)`
where `ACCEPTED_OFFER_REASON = "customer accepted an offer of a callback"` — a suggested
trigger never opened one, and a handover with no reason tells the human nothing.

- [ ] **Step 4: Register**

`agent/session.py` — add `handoff` to the `from agent.tools import ...` line, then:

`TOOLS`, after `scheduling.CANCEL_APPOINTMENT_SCHEMA`:
```python
    handoff.SCHEDULE_CALLBACK_SCHEMA,
    handoff.RECORD_CALLBACK_DECLINED_SCHEMA,
```

`SessionGates`:
```python
    # Phase 12. Not a PendingActionGate, but a gate in the most literal sense —
    # it is what refuses end_conversation while a handover is unresolved. Here
    # rather than on Session because the tool handlers close over it, and they
    # are built before the Session that owns them exists.
    escalation: escalation.EscalationState = field(default_factory=escalation.EscalationState)
```

`handlers`:
```python
        "schedule_human_callback": lambda **kw: handoff.schedule_human_callback(
            **kw, state=gates.escalation, gate=gates.scheduling, customer_id=customer_id
        ),
        "record_customer_will_reach_out": lambda **kw: handoff.record_customer_will_reach_out(
            **kw, escalation=gates.escalation
        ),
```

The callback shares `gates.scheduling` because it **is** a booking — a customer cannot
have a pending appointment and a pending callback at once without one clobbering the other.

- [ ] **Step 5: Run the full suite. Step 6: Commit.**

---

### Task 4: A *bounded* refusal to hang up

**Files:**
- Modify: `agent/tools/summary.py` (`end_conversation`, ~line 52)
- Modify: `agent/session.py` (`should_end_session`, ~line 107; handler wiring)
- Test: `tests/test_session.py`

**This task fixes C-2.** An unbounded refusal traps the customer on the call.

**Interfaces produced:**
- `END_REFUSED_PREFIX = "Not yet."`
- `MAX_END_REFUSALS = 2`
- `end_conversation(escalation=None) -> str`
- `should_end_session` now requires the call to have **succeeded**.

- [ ] **Step 1: Write the failing tests**

```python
def test_end_conversation_is_refused_while_a_handover_is_open():
    """The bug that prompted this phase: the agent asked "would you like me to
    find some callback slots?" and hung up on the same turn."""
    from agent.tools import summary
    from agent.tools.escalation import EscalationState

    state = EscalationState()
    state.open("explicit request for a human")

    result = summary.end_conversation(escalation=state)

    assert not should_end_session([{"name": "end_conversation", "output": result}])


def test_the_refusal_gives_up_rather_than_trapping_the_customer():
    """A refusal must be a nudge, not a cage.

    If the customer says "just let me go" and the model never calls a
    resolution tool, an unbounded refusal means they cannot hang up — and the
    refusal text tells the model to offer a callback again, which is the
    "asked repeatedly whether you want a human" failure this phase exists to
    prevent, relocated to the exit. After MAX_END_REFUSALS the call ends and
    close_session records it as unresolved."""
    from agent.tools import summary
    from agent.tools.escalation import EscalationState

    state = EscalationState()
    state.open("explicit request for a human")

    for _ in range(summary.MAX_END_REFUSALS):
        refused = summary.end_conversation(escalation=state)
        assert not should_end_session([{"name": "end_conversation", "output": refused}])

    allowed = summary.end_conversation(escalation=state)
    assert should_end_session([{"name": "end_conversation", "output": allowed}]), (
        "the customer must always be able to leave"
    )


def test_end_conversation_is_allowed_once_the_handover_is_resolved():
    from agent.tools import summary
    from agent.tools.escalation import RESOLUTION_SELF, EscalationState

    state = EscalationState()
    state.open("explicit request for a human")
    state.record_resolution(RESOLUTION_SELF)

    assert should_end_session(
        [{"name": "end_conversation", "output": summary.end_conversation(escalation=state)}]
    )


def test_end_conversation_still_works_with_no_handover_at_all():
    """The overwhelming majority of calls. Nothing here may make an ordinary
    goodbye harder."""
    from agent.tools import summary

    assert should_end_session([{"name": "end_conversation", "output": summary.end_conversation()}])
```

- [ ] **Step 2: Run to verify they fail** — `end_conversation()` takes no arguments.

- [ ] **Step 3: Implement**

`agent/tools/summary.py`:

```python
# What a refused end_conversation returns. Recognised by agent/session.py's
# should_end_session, so a refusal can never be mistaken for the model signing
# off.
END_REFUSED_PREFIX = "Not yet."

# How many times the agent may decline to hang up before giving in.
#
# A refusal is a nudge, not a cage. A customer who says "just let me go" and
# never picks a callback must still be able to leave — otherwise the refusal
# text keeps telling the model to offer a callback, which is "being asked
# repeatedly whether you want a human" moved to the exit. close_session
# records the handover as unresolved, so the human still hears about them.
MAX_END_REFUSALS = 2


def end_conversation(escalation: Any | None = None) -> str:
    """Normally there is no real work to do — the point is Claude choosing to
    call this tool at all, and the transport watching for it.

    The exception is an unresolved handover. The agent has told a customer a
    human will help and has not yet arranged how, so ending the call strands
    them — which is exactly what happened live, where it asked "would you like
    me to find some callback slots?" and hung up on the same turn.

    Refused as a returned message rather than a raised exception, the same
    shape issue_refund uses for an outstanding confirmation: the model reads it
    as a tool result and works the problem, where an exception would break the
    turn.
    """
    if escalation is not None and escalation.is_open:
        if escalation.refusals < MAX_END_REFUSALS:
            escalation.refusals += 1
            return (
                f"{END_REFUSED_PREFIX} Sort out the handover before saying goodbye. "
                "Offer the customer a callback time with find_available_slots and "
                "schedule_human_callback, or — if they would rather get in touch "
                "themselves — call record_customer_will_reach_out."
            )
        # Asked twice, still unresolved. The customer wants to go; let them.
    return "Session marked complete."
```

Add `from typing import Any` if absent. The parameter is typed `Any` deliberately —
importing `EscalationState` would couple `summary` to `escalation` for no benefit.

`agent/session.py` handler:
```python
        "end_conversation": lambda **kw: summary.end_conversation(**kw, escalation=gates.escalation),
```

`should_end_session`:
```python
def should_end_session(tool_calls: list[dict]) -> bool:
    """True if this turn's tool calls included the model deciding to sign off
    AND that decision was allowed to stand.

    The output check is load-bearing, not defensive: Phase 12 lets
    end_conversation REFUSE while a handover is unresolved, and a refusal that
    still ended the session would defeat the whole mechanism.
    """
    return any(
        call["name"] == "end_conversation"
        and not str(call.get("output", "")).startswith(summary.END_REFUSED_PREFIX)
        for call in tool_calls
    )
```

- [ ] **Step 4: Run the full suite. Step 5: Commit.**

---

### Task 5: Rewire `run_turn` — **preserving `end_reason="escalated"`**

**Files:**
- Modify: `agent/session.py` — `_escalation_notice` area (~line 232), the escalation
  branch (~line 380-405), `close_session` (~line 447)
- Test: `tests/test_session.py`

**This task fixes C-1.** Read D-1 before writing a line.

**Interfaces produced:**
- `run_turn` no longer ends the call at escalation *detection*.
- `end_reason="escalated"` is **still produced** — on the turn that ends a call whose
  escalation state is not `none`.
- `_escalation_handover_notice()`, `_escalation_offer(reason)`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_a_mandatory_trigger_no_longer_ends_the_call(monkeypatch, tmp_path):
    """The live failure exactly: the agent asked "would you like me to find
    some available callback slots for you?" and the call ended on that same
    turn. The customer never got to answer."""
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "t.db")
    mock_db.reset_and_seed()
    session = _session_replying("Of course. When would suit you for a call?")
    _force_signal(monkeypatch, "explicit request for a human", mandatory=True)

    outcome = await run_turn(session, "can I speak to a person")

    assert outcome.ended is False, "a reply ending in a question must not end the call"
    assert session.gates.escalation.is_open
    assert session.gates.escalation.escalation_id is not None, "the row is written at open"


@pytest.mark.asyncio
async def test_end_reason_escalated_survives_for_the_twilio_hook(monkeypatch, tmp_path):
    """transport/pipecat_processors.py:393 fires Phase 10d's warm-transfer
    hook on end_reason == "escalated", and eval/scenarios.py pins it in six
    scenarios. The value must not disappear — it moves from detection-time to
    end-of-call, so the hook now fires with a packet carrying the AGREED
    callback time instead of one assembled before anything was agreed."""
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "t.db")
    mock_db.reset_and_seed()
    session = _session_replying("Speak to you then.")
    _force_signal(monkeypatch, "explicit request for a human", mandatory=True)
    await run_turn(session, "can I speak to a person")
    session.gates.escalation.record_resolution(escalation.RESOLUTION_SELF)

    _force_signal(monkeypatch, None, mandatory=False)
    outcome = await _run_turn_ending_the_call(session, "that's all, thanks")

    assert outcome.ended is True
    assert outcome.end_reason == "escalated", (
        "a call that involved a handover must still report end_reason='escalated'"
    )
    assert outcome.escalation_packet is not None, "the hook needs the packet"


@pytest.mark.asyncio
async def test_an_ordinary_call_still_ends_as_model_ended(monkeypatch):
    """The value must not start appearing on calls that never escalated."""
    session = _session_replying("Take care!")
    _force_signal(monkeypatch, None, mandatory=False)

    outcome = await _run_turn_ending_the_call(session, "thanks, bye")

    assert outcome.end_reason == "model_ended"


@pytest.mark.asyncio
async def test_a_suggested_trigger_offers_instead_of_escalating(monkeypatch):
    """A mis-dictated order number is a cooperative repair, the healthiest
    signal a conversation can produce. It must not be treated as a verdict."""
    session = _session_replying("I'm still not finding that order.")
    _force_signal(monkeypatch, "repeated failed lookups", mandatory=False)

    outcome = await run_turn(session, "let me try that number again")

    assert outcome.ended is False
    assert session.gates.escalation.status == escalation.STATUS_NONE, "an offer opens nothing"
    assert outcome.notice is not None and "?" in outcome.notice, "the customer must be asked"
    assert session.tracker.consecutive_failed_lookups == 0, "the offer resets its own counter"


@pytest.mark.asyncio
async def test_a_suggested_trigger_only_offers_once(monkeypatch):
    """Being asked over and over whether you want a human is its own failure."""
    session = _session_replying("Still not finding it.")
    _force_signal(monkeypatch, "repeated failed lookups", mandatory=False)

    first = await run_turn(session, "try again")
    second = await run_turn(session, "and again")

    assert first.notice is not None
    assert second.notice is None


@pytest.mark.asyncio
async def test_one_problem_produces_one_escalation_row(monkeypatch, tmp_path):
    """A live call produced escalation rows 7, 8 and 9 for one problem —
    which with n8n connected would have been three Slack messages about one
    customer. Counted as rows, because that is the failure a reader misses."""
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "t.db")
    mock_db.reset_and_seed()
    session = _session_replying("Understood.")
    _force_signal(monkeypatch, "explicit request for a human", mandatory=True)

    await run_turn(session, "get me a person")
    await run_turn(session, "seriously, a person")

    with get_connection() as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM escalations").fetchone()["n"]
    assert rows == 1, f"one problem, one row — got {rows}"


@pytest.mark.asyncio
async def test_an_abandoned_handover_still_reaches_a_human(monkeypatch, tmp_path):
    """Customers hang up, sockets die. A handover that never resolved must
    still tell the human — and say honestly that no time was agreed."""
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "t.db")
    mock_db.reset_and_seed()
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    session = _session_replying("Understood.")
    _force_signal(monkeypatch, "explicit request for a human", mandatory=True)
    await run_turn(session, "get me a person")
    assert sent == [], "nothing goes out at open"
    monkeypatch.setattr(summary, "close_session", _stub_summary_close)

    await close_session(session)

    assert len(sent) == 1
    assert sent[0]["resolution"] == escalation.RESOLUTION_UNRESOLVED
    assert sent[0]["callback_time"] is None


@pytest.mark.asyncio
async def test_a_resolved_handover_is_not_notified_again_at_close(monkeypatch, tmp_path):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "t.db")
    mock_db.reset_and_seed()
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    session = _session_replying("Understood.")
    _force_signal(monkeypatch, "explicit request for a human", mandatory=True)
    await run_turn(session, "get me a person")
    session.gates.escalation.record_resolution(escalation.RESOLUTION_SELF)
    _force_signal(monkeypatch, None, mandatory=False)
    await run_turn(session, "I'll ring you back myself")
    monkeypatch.setattr(summary, "close_session", _stub_summary_close)

    await close_session(session)

    assert len(sent) == 1, "resolution already sent the one notification"
```

Helper to add:

```python
def _force_signal(monkeypatch, reason: str | None, mandatory: bool):
    """Drive run_turn's escalation branch without an API call, patching the
    seam run_turn actually uses."""
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

`_session_replying(text)`, `_run_turn_ending_the_call(session, text)` and
`_stub_summary_close` must follow the **existing** `MagicMock` / `_text_response` /
`_tool_use_response` patterns already in `tests/test_session.py` (see
`test_run_turn_detects_the_model_ending_the_conversation`, line ~80). **Read that file
and reuse them — do not invent a new harness.**

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Add the two notice builders**

Beside `_escalation_notice` (keep that function — `create_handoff_packet`'s DTMF path
still reaches it):

```python
def _escalation_handover_notice() -> str:
    """What the customer hears the moment a handover opens.

    No time is named yet — that is the whole change. Escalation used to
    announce a callback slot the customer had never agreed to and then end the
    call; now the agent says a colleague will help and has to actually arrange
    it before it is allowed to say goodbye.
    """
    return (
        "Let me get one of my colleagues onto this for you. "
        "When would be a good time for them to give you a call?"
    )


def _escalation_offer(reason: str) -> str:
    """What the customer hears when the agent SUSPECTS it is failing them.

    An offer, not an announcement, and it names a way to carry on — the live
    failure was a customer one transposed digit from success, and another
    calmly cancelling an order. Deliberately never names the internal reason:
    "sustained negative sentiment across multiple turns" read aloud tells an
    already frustrated customer they have been classified as angry.
    """
    if reason == "repeated failed lookups":
        return (
            "I'm still not finding that. Would you like to try the number once more, "
            "or shall I have a colleague call you back about it?"
        )
    return "Would it help if I arranged for a colleague to call you back about this?"
```

- [ ] **Step 4: Rewrite the escalation branch**

Rename the variable bound from `check_escalation` to `signal`. Replace the branch with:

```python
    state = session.gates.escalation
    escalation_id: int | None = state.escalation_id
    packet: dict[str, Any] | None = state.packet

    if signal and signal.mandatory:
        if state.status == escalation.STATUS_NONE:
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
                warnings.append(
                    f"Escalation triggered ({signal.reason}) but couldn't be logged: {exc}"
                )
        else:
            # Already handing over. One customer, one callback — this becomes
            # another item the same colleague can prepare for.
            state.amend(signal.reason)
    elif signal:
        # The agent's own inference. Offer once, and reset the counter that
        # produced it so the customer gets a clean run at correcting themselves
        # rather than tripping the same threshold on their next breath.
        if signal.reason not in state.offered and not state.is_open:
            state.offered.add(signal.reason)
            notice = _escalation_offer(signal.reason)
        session.tracker.reset_streak(signal.reason)

    # Persist and notify whatever the tool loop resolved this turn. The async
    # work lives here because dispatch_tool calls handlers without awaiting
    # them — a tool physically cannot do I/O (see agent/tools/handoff.py).
    if state.pending_persist and state.status == escalation.STATUS_RESOLVED:
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
        # end_reason KEEPS the value "escalated" — see the plan's D-1.
        # transport/pipecat_processors.py:393 fires Phase 10d's Twilio transfer
        # on it, and eval/scenarios.py pins it in six scenarios. What changed
        # is WHEN it is set: not the instant a trigger fires, but when a call
        # that involved a handover actually ends. The hook now gets a packet
        # carrying the agreed callback time rather than one assembled before
        # anything was agreed.
        escalated = state.status != escalation.STATUS_NONE
        outcome = TurnOutcome(
            reply=reply if reply.strip() else farewell(int(session.session_id[:8], 16)),
            ended=True,
            end_reason="escalated" if escalated else "model_ended",
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

**Preserve the existing empty-reply farewell fallback exactly as written** — it is a
live fix from today and a call must never end in silence.

In the `log_turn` call, change `escalation_reason=reason` to
`escalation_reason=signal.reason if signal else None` and `escalated=outcome.end_reason
== "escalated"`.

- [ ] **Step 5: Notify an abandoned handover in `close_session`**

At the top of `close_session`, before the `if not session.agent.messages` guard:

```python
    # A handover that never reached a resolution: the customer hung up, the
    # socket died, the process restarted. The human still needs to hear about
    # this customer — and needs to know no time was agreed, rather than being
    # handed a slot nobody promised.
    state = session.gates.escalation
    if state.is_open and state.packet is not None:
        try:
            await escalation.resolve_escalation(
                state.packet,
                items=state.items,
                resolution=escalation.RESOLUTION_UNRESOLVED,
                callback_time=None,
            )
            state.pending_persist = False
        except Exception:  # noqa: BLE001 — an exit path must never crash
            logger.exception("could not notify an abandoned handover")
```

Add `import logging` and a module-level `logger = logging.getLogger(__name__)` if absent.

- [ ] **Step 6: Run the full suite**

Existing tests asserting `end_reason == "escalated"` *at detection* will fail — that
behaviour is deliberately gone. **Update them to the new state machine; do not delete
them.**

- [ ] **Step 7: Commit**

---

### Task 6: Prompt guidance, and the eval scenarios

**Files:**
- Modify: `agent/prompts.py` (`_SYSTEM_PROMPT_TEMPLATE`)
- Test: `tests/test_escalation.py`

- [ ] **Step 1: Add a handover section to the system prompt**

Worded for a voice call, and consistent with the prompt's existing line-continuation
style (`\` at end of line):

```
## Handing over to a colleague

Sometimes you cannot finish something yourself — the customer asks for a
person, a policy needs a specialist, or you are simply not getting anywhere.
When that happens you are not finished: you have to arrange the handover
before the call can end.

- Ask when would suit them for a colleague to call back, then call
  find_available_slots and offer a real time.
- When they pick one, call schedule_human_callback. The first call asks them
  to confirm; call it again with the same time once they say yes.
- If they would rather get in touch themselves, call
  record_customer_will_reach_out. That is a perfectly good outcome.
- Never say goodbye before one of those two has gone through.
- If they clearly want to go and will not settle either, let them. Say a
  colleague will be in touch and close warmly. Do not keep asking.

Once the handover is arranged the call is not over. Ask whether there is
anything else, and if there is, just help with it normally — a refund you can
process yourself gets processed, not handed over. Only pass on something
genuinely beyond you, and when you do, do not arrange a second callback: the
same colleague covers it on the same call. Say so simply — "I'll add that to
what they're calling you about."
```

- [ ] **Step 2: Assert the prompt only names tools that exist**

```python
def test_the_prompt_only_names_handover_tools_that_exist():
    """A prompt naming a renamed tool is a silent failure — the model calls it,
    gets "unknown tool", and improvises."""
    from agent.prompts import SYSTEM_PROMPT
    from agent.session import TOOLS

    names = {schema["name"] for schema in TOOLS}
    for named in ("find_available_slots", "schedule_human_callback", "record_customer_will_reach_out"):
        assert named in names, f"SYSTEM_PROMPT names {named}, which is not a registered tool"
        assert named in SYSTEM_PROMPT
```

- [ ] **Step 3: Re-record the affected eval scenarios**

Six scenarios in `eval/scenarios.py` pin `end_reason="escalated"` (lines 315, 431, 448,
486, 499, 519). The **value** still exists (D-1), but the conversations change: an
escalation no longer ends on the detection turn, so the recorded transcripts are stale.

`eval/scenarios.py` is outside this plan's file allow-list, and re-recording needs live
API credit. **Do not edit or re-record in this phase.** Instead, verify the value still
exists and hand the re-record to the 10c live checkpoint, which already needs credit:

```bash
grep -c 'end_reason="escalated"' eval/scenarios.py   # expect 6, unchanged
```

Add a line to `PROGRESS.md` under Phase 12 noting the six scenarios need re-recording
before the eval suite is trusted again.

- [ ] **Step 4: Run the full suite. Step 5: Commit.**

---

## Checkpoint

**Automated:** the full suite passes offline with no API keys. Baseline before this
phase is **359 passed, 3 skipped**; it must not go down.

**Manual (needs credit):** one live call that

1. escalates on an explicit request for a human and **does not hang up**;
2. agrees a callback time, confirms it, and writes a real `appointments` row;
3. is asked "anything else?", and handles a second, ordinary issue normally;
4. has a second escalation-worthy issue **amend** the existing handover;
5. ends with a spoken goodbye;
6. delivers **exactly one** Slack message carrying both items and the agreed time.

Then a second call that asks for a human and then says "actually, forget it, goodbye" —
it must be allowed to end (C-2), and Slack must show it as `unresolved`.

Then a third that mis-dictates an order number twice: the agent **offers** rather than
imposes, and carries on normally when the customer says they would rather try again.

## Self-review

- **Every spec section maps to a task**: triggers (1), notification timing (2),
  resolution paths (3), refusal (4), state machine + counter bug + one-row-per-problem
  (1, 5), abandoned calls (5), prompt (6).
- **CEO findings addressed**: C-1/F-3 → D-1 + Task 5; C-2/F-1 → Task 4; F-4 → Task 1
  constants; F-2 → D-2 (deferred, with reasons); F-5 → column comment in Task 1.
- **Type consistency**: `EscalationSignal` produced in Task 1, consumed in Task 5;
  `EscalationState` produced in Task 1, consumed in 3/4/5; `END_REFUSED_PREFIX` produced
  and consumed in Task 4; the `RESOLUTION_*` constants used in 1, 2, 3, 5.
- **Files touched**: 6 source + 3 test. `transport/`, `observability/`, `eval/`,
  `guardrails/` untouched.
- **Known soft spot**: Task 5 is the largest diff and the only one touching a function
  with many existing callers. If a reviewer rejects one task, it will be that one.
