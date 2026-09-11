# Phase 12 — Escalation as a Resolvable Process (v3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make escalation a state the agent must resolve — a callback the customer agrees to, or a record that they will make contact themselves — before the call can end, without breaking anything that works today.

**Architecture:** Escalation state lives on `SessionGates`, mutated by **async** resolution tools that persist and notify inline. `run_turn` decides whether the call may end *after* the escalation state is known. `end_reason="escalated"` survives, narrowed to outcomes that still mean "transfer this call now".

**Tech Stack:** Python 3.12, SQLite, Pydantic, `anthropic` SDK, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-10-phase-12-escalation-process-design.md`

**Supersedes:** v1 (`2026-09-10-…`) and v2 (`2026-09-11-…-v2.md`). **Do not implement either.** v1 deleted `end_reason="escalated"` and silently killed Phase 10d. v2 kept the string but broke both its consumers, rested on a fabricated constraint, and left the original bug intact. What each got wrong is recorded below, because the same mistakes are easy to re-make.

---

## Global Constraints

- **Touch only these files:**
  `agent/tools/escalation.py`, `agent/tools/handoff.py` (new), `agent/tools/summary.py`,
  `agent/session.py`, `agent/prompts.py`, `data/mock_db.py`,
  `observability/turn_log.py`, `eval/scenarios.py`, and their tests.
  The last two are **not** scope creep: without them this phase silently changes what
  `escalation_reason` means (F-8) and breaks six eval scenarios (F-7).
- **`transport/` is untouched** (CLAUDE.md rule 5). If a task believes it must change a
  transport, **stop and flag it** — the abstraction broke.
- **CLAUDE.md rule 6:** every irreversible action keeps a confirmation turn of its own.
- **CLAUDE.md rule 7:** resolution is a deterministic tool call.
- **Never raise on an exit path.**
- **No hard-coded values** — IDs, slots and customers come from `data/mock_db.py` at runtime.
- Suite command (keys blanked so no task spends credit; `env -u` does **not** work —
  `load_dotenv()` repopulates a *missing* var, an empty string survives):
  `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
  **Baseline: 359 passed, 3 skipped. It must not go down.**

### Verified line references

Checked against the working tree at the time of writing. v2 cited three of these wrongly
and its central argument rested on two of them.

| What | Where |
|---|---|
| Twilio warm-transfer hook fires here | `transport/pipecat_processors.py:418` |
| DTMF-zero calls `create_handoff_packet` | `transport/pipecat_processors.py:227` |
| DTMF writes its own `end_reason="escalated"` | `transport/pipecat_processors.py:266` |
| Tool loop runs here | `agent/session.py:291` |
| Escalation is checked here (**after** the loop) | `agent/session.py:372` |
| `core.py` **awaits** awaitable tool output | `agent/core.py:147` |
| Booking gate key | `agent/tools/scheduling.py:172` |
| History-edit precedent | `agent/session.py:129` |

---

## What v1 and v2 got wrong

**C-1 — v2 left the original bug in place.** `agent.send()` (`agent/session.py:291`) runs
the *entire* tool loop; escalation is checked afterwards (`:372`). So on the turn a
trigger fires, the state is still `none` for the whole loop: `end_conversation` is never
refused, and v2's own code then labelled that turn `ended=True, end_reason="escalated"`.
That is the spec's failing transcript unchanged. **Fix: D-1.**

**C-2 — v2's central premise was false.** It claimed tool handlers cannot do async work.
`agent/core.py:147` does `if hasattr(output, "__await__"): output = await output`. Async
tools work today. **Fix: D-2.**

**C-3 — v2 could lose a booked callback entirely.** Slot reserved, resolution recorded in
memory, then the turn raises before the deferred persist; `close_session`'s flush was
guarded on `is_open` while the status was `resolved`, so nothing fired. **Fix: D-2 + D-5.**

**F-5 — v2's Twilio "fix" was worse than the bug it fixed.** The hook
(`transport/pipecat_processors.py:418`) makes `transport/telephony.py` say *"Connecting
you now. Please hold."* and dial a human. Under v2, a customer who booked a callback for
next Tuesday got bridged to a live human at goodbye. **Fix: D-3.**

**F-6 — v2's stated benefit did not exist.** `render_whisper` reads only `escalation_id`,
`reason`, `customer_intent`, `verified_account_info`, `actions_taken`, `sentiment`. Never
`items`, never `callback_time`. **Fix: D-4.**

**F-9 — the notice asks a question the model has no record of asking.** `run_turn` never
writes `notice` into `session.agent.messages`, but every transport speaks it. The
customer answers a question that is not in the history — the spec's own complaint #1,
made worse. **Fix: D-6.**

---

## Design decisions

**D-1 — The end is decided AFTER the escalation state is known.**

`should_end_session` stays exactly as it is. `run_turn` computes `ending` once, *after*
the escalation branch has run, and an open handover suppresses it regardless of when the
trigger was detected. This is what actually fixes the live bug; the `end_conversation`
refusal (D-7) is the model-facing half that makes the agent *do something about it*, not
the mechanism that stops the hang-up.

**D-2 — The resolution tools are `async` and do their own I/O.**

`agent/core.py:147` awaits awaitable tool output, so an `async def` handler works today
with no change to `agent/` or `transport/`. The tools book, persist and notify inside a
single await, at the moment the customer agrees. This deletes v2's `pending_persist`
deferral and the window where a booked callback could vanish (C-3).

**D-3 — `end_reason="escalated"` is narrowed to outcomes that still mean "transfer now".**

```python
transferable = state.resolution in (None, RESOLUTION_TRANSFER, RESOLUTION_UNRESOLVED)
end_reason = "escalated" if (state.status != STATUS_NONE and transferable) else "model_ended"
```

A booked callback or a customer-will-reach-out ends as `model_ended`: the human is
already being told out of band by the resolution notification, and bridging a live call
to them is wrong. An unresolved or transfer-shaped handover still means transfer, so the
Phase 10d hook keeps meaning what it has always meant. The DTMF path is untouched — it
writes its own `end_reason` at `transport/pipecat_processors.py:266`.

**D-4 — The whisper is improved through `reason`, the field it actually reads.**

`resolve_escalation` folds the outcome into the packet's `reason` string, e.g.
`"explicit request for a human; also: high-value refund — callback agreed for Thursday at
9am"`. `render_whisper` already renders `reason`, so the human hears the agreed time with
zero transport changes. v2 claimed this benefit while writing to fields nobody reads.

**D-5 — `close_session` flushes on `pending_persist`, not on `is_open`.**

Anything left unsent gets sent, whatever the status, and the flush is idempotent.

**D-6 — A spoken notice that asks a question is written into the model's history.**

Using the same defensive shape as `_substitute_hedge_in_history` (`agent/session.py:129`).
Otherwise the customer answers a question the model never sees itself ask. The prompt's
handover section must then NOT also tell the model to ask it, or the customer hears it twice.

**D-7 — The refusal budget is per TURN, not per call.**

`agent/core.py:91` allows 8 tool iterations, so a model can call `end_conversation` three
times inside one `agent.send()` and burn a call-counted budget with no customer utterance
in between. Budget is consumed at most once per `session.turn`; repeat calls within a turn
are refused without consuming it.

**D-8 — Escalation state lives on `SessionGates`.** `build_dispatch_tool`
(`agent/session.py:72`) builds handlers before the `Session` exists, so a handler cannot
close over a field set later on `Session`. `SessionGates` is already the container the
handlers close over, and this is literally a gate. The 3-tuple return is unchanged.

**D-9 — A suggested trigger only offers; it never opens a handover.** If the customer
accepts, the model calls a resolution tool and *that* opens it. Nothing to detect, no
second classifier.

**D-10 — `create_handoff_packet` keeps its name, signature and behaviour** for
`transport/pipecat_processors.py:227` (rule 5). It is "open and immediately resolve as a
transfer", which is what pressing zero means.

---

## File Structure

| File | Change |
|---|---|
| `agent/tools/escalation.py` | Status/resolution constants, `EscalationSignal`, `EscalationState`, `MANDATORY_REASONS`, `reset_streak`, `open_escalation`, `mark_resolved`, `resolve_escalation`. `record_turn` returns a signal. `create_handoff_packet` becomes a wrapper. |
| `agent/tools/handoff.py` | **New.** Two **async** resolution tools + schemas. |
| `agent/tools/summary.py` | Per-turn bounded refusal. |
| `agent/session.py` | `SessionGates.escalation`; `TOOLS`; handlers; the reordered end decision; notice-into-history; `close_session` flush. |
| `agent/prompts.py` | Handover guidance. |
| `data/mock_db.py` | Four `escalations` columns. |
| `observability/turn_log.py` | `escalation_offered` field. |
| `eval/scenarios.py` | Rewrite six scenarios. |

---

### Task 1: State, trigger split, and schema

**Files:** `agent/tools/escalation.py`, `data/mock_db.py`, `tests/test_escalation.py`

**Produces:** `STATUS_NONE/OPEN/RESOLVED`; `RESOLUTION_CALLBACK/SELF/TRANSFER/UNRESOLVED`;
`MANDATORY_REASONS`; `SUGGESTED_REASONS`; `EscalationSignal(reason, mandatory)`;
`EscalationState`; `EscalationTracker.record_turn -> EscalationSignal | None`;
`EscalationTracker.reset_streak(reason)`; `check_escalation -> EscalationSignal | None`.

- [ ] **Step 1: Back up the live-call data before anything else**

`python -m data.mock_db` runs `reset_and_seed()`, which **drops every table**
(`data/mock_db.py:194-202`). That is necessary — `CREATE TABLE IF NOT EXISTS` will not
add columns to an existing table — but it destroys the escalation, ticket and refund rows
produced by the Phase 9-11 live calls, which is the evidence this phase was designed from.

```bash
sqlite3 data/mock_data.db .dump > /tmp/pre-phase12-backup.sql
```

- [ ] **Step 2: Write the failing tests**

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
    assert signal is not None and signal.mandatory is True
    assert signal.reason == "explicit request for a human"


def test_inferred_triggers_are_only_suggestions():
    """Both of these hung up on customers who were fine: one had mis-dictated
    a digit and corrected themselves, the other was calmly cancelling an
    order. They are the AGENT's inference that it is failing — sometimes
    true, sometimes not. Inferences get offered, not imposed."""
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    failed = [{"name": "get_order_status", "output": {"found": False}}]
    for _ in range(escalation.FAILED_LOOKUP_ESCALATION_THRESHOLD):
        signal = tracker.record_turn(neutral, failed)
    assert signal.reason == "repeated failed lookups"
    assert signal.mandatory is False
    assert signal.reason in escalation.SUGGESTED_REASONS

    tracker2 = escalation.EscalationTracker()
    upset = escalation.TurnClassification(
        intent="complaint", sentiment="negative", policy_restricted=False
    )
    for _ in range(escalation.NEGATIVE_SENTIMENT_ESCALATION_THRESHOLD):
        signal2 = tracker2.record_turn(upset, [])
    assert signal2.mandatory is False


def test_every_reason_is_classified_exactly_once():
    """A reason in neither set is silently treated as suggested; a reason in
    both is a contradiction. Either way the behaviour would be decided by
    accident."""
    assert escalation.MANDATORY_REASONS & escalation.SUGGESTED_REASONS == set()


def test_resetting_a_streak_gives_the_customer_a_clean_run():
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
    assert state.status == escalation.STATUS_NONE and state.is_open is False

    state.open("explicit request for a human")
    assert state.status == escalation.STATUS_OPEN
    assert state.items == ["explicit request for a human"]

    state.amend("policy-restricted topic")
    assert state.status == escalation.STATUS_OPEN, "an amendment never reopens"
    assert state.items == ["explicit request for a human", "policy-restricted topic"]

    state.record_resolution(escalation.RESOLUTION_CALLBACK, callback_time="2026-09-14T09:00:00")
    assert state.status == escalation.STATUS_RESOLVED and state.is_open is False


def test_amending_a_resolved_handover_keeps_it_resolved():
    state = escalation.EscalationState()
    state.open("explicit request for a human")
    state.record_resolution(escalation.RESOLUTION_SELF)
    state.amend("high-value refund requires approval")
    assert state.status == escalation.STATUS_RESOLVED
    assert len(state.items) == 2


def test_a_repeated_trigger_does_not_duplicate_an_item():
    """One tripped trigger firing every turn produced escalation rows 7, 8
    and 9 for one problem in a live call."""
    state = escalation.EscalationState()
    state.open("explicit request for a human")
    state.amend("explicit request for a human")
    assert state.items == ["explicit request for a human"]


def test_the_refusal_budget_is_spent_once_per_turn():
    """agent/core.py:91 allows 8 tool iterations, so a model that reads the
    refusal and simply retries can call end_conversation three times inside
    ONE agent.send(). A call-counted budget is exhausted with no customer
    utterance in between, and the call ends on the trigger turn with the
    handover open — which is the bug this phase exists to fix."""
    state = escalation.EscalationState()
    state.open("explicit request for a human")

    assert state.consume_refusal(turn=4) is True
    assert state.consume_refusal(turn=4) is True, "a repeat within one turn still refuses"
    assert state.refusals == 1, "but it must not spend budget"
```

- [ ] **Step 3: Run to verify they fail.**

- [ ] **Step 4: Add constants, signal and state**

Above `class EscalationTracker` in `agent/tools/escalation.py`:

```python
# Named rather than repeated as literals: these values travel across
# escalation.py, handoff.py, session.py, turn_log.py and a DB column, and a
# typo in any one fails silently as "this handover is somehow neither open
# nor resolved".
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
# the judgement that hung up on a customer who had mis-dictated one digit,
# and on another who was calmly cancelling an order. Inferences get offered;
# they do not get imposed. CLAUDE.md rule 6's principle, applied to handoffs.
MANDATORY_REASONS = frozenset(
    {"explicit request for a human", "policy-restricted topic"}
)
SUGGESTED_REASONS = frozenset(
    {
        "repeated failed lookups",
        "sustained negative sentiment across multiple turns",
        "repeated ungrounded replies",
    }
)


@dataclass(frozen=True)
class EscalationSignal:
    reason: str
    mandatory: bool


@dataclass
class EscalationState:
    """One handover per session, for the life of the session.

    That is what happens on a real support line: a human ringing a customer
    back deals with everything that customer has, rather than booking three
    calls for three questions. So a second escalation-worthy issue becomes
    another ITEM on the same handover — which is why `items` is a list and not
    the single `reason` string it replaced.

    Mutated in place, never rebound: the tool handlers in build_dispatch_tool
    close over it before the Session that owns it exists.
    """

    status: str = STATUS_NONE
    escalation_id: int | None = None
    packet: dict[str, Any] | None = None
    items: list[str] = field(default_factory=list)
    resolution: str | None = None
    callback_time: str | None = None
    # Suggested triggers that have already made their offer. Being asked over
    # and over whether you want a human is its own kind of failure.
    offered: set[str] = field(default_factory=set)
    # Anything still unsent. close_session flushes on this, NOT on is_open —
    # a resolution recorded and then lost to a crashing turn is not open, and
    # guarding on is_open would drop it silently.
    pending_persist: bool = False
    refusals: int = 0
    _last_refusal_turn: int | None = None

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
        colleague taking it over can prepare for.
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

    def consume_refusal(self, turn: int) -> bool:
        """True if end_conversation should be refused. Budget is spent at most
        once per TURN.

        agent/core.py:91 allows 8 tool iterations, so a model that reads the
        refusal and retries can call end_conversation three times inside one
        agent.send(). Counting calls would exhaust the budget with no customer
        utterance in between — the nudge becomes a rubber stamp on exactly the
        turn it was meant to catch.
        """
        if self.refusals >= MAX_END_REFUSALS and turn != self._last_refusal_turn:
            return False
        if turn != self._last_refusal_turn:
            self.refusals += 1
            self._last_refusal_turn = turn
        return True
```

`MAX_END_REFUSALS = 2` lives here too (not in `summary.py`), since `EscalationState` is
what enforces it.

Add `field` to the `dataclasses` import if absent.

- [ ] **Step 5: Return a signal; fix the counter bug**

Rewrite `EscalationTracker.record_turn` so every `return "<reason>"` becomes
`EscalationSignal(<reason>, mandatory=<bool>)`, with the three immediate triggers
mandatory and the three streak triggers not — and **move the failed-lookup threshold
check inside the `if outcomes:` block**:

```python
        outcomes = _turn_tool_outcomes(tool_calls)
        if outcomes:
            if any(outcomes):
                self.consecutive_failed_lookups = 0
            else:
                self.consecutive_failed_lookups += 1
            # INSIDE this block on purpose. It used to sit outside, re-reading
            # the counter every turn — so once the streak tripped, a turn with
            # no lookup at all still escalated. One live call produced rows 7,
            # 8 and 9 for one problem that way, the last from a turn that
            # merely asked the agent to repeat a number back.
            if self.consecutive_failed_lookups >= FAILED_LOOKUP_ESCALATION_THRESHOLD:
                return EscalationSignal("repeated failed lookups", mandatory=False)
```

Add `reset_streak(reason)` clearing the counter behind each suggested reason. Update the
return annotations on `record_turn` and `check_escalation`.

- [ ] **Step 6: Add the schema columns**

In `data/mock_db.py`, inside `CREATE TABLE IF NOT EXISTS escalations`, after
`notified_at`, before `FOREIGN KEY`:

```sql
    -- Phase 12: a handover is a process, not an event. `items` is a
    -- newline-separated list of every reason on this ONE handover — one
    -- customer gets one callback, so a second escalation-worthy issue appends
    -- here rather than opening a second row. Reasons are written by this
    -- codebase, never by a customer, so none can contain a newline.
    -- `resolution` is null while the handover is open, which is also how an
    -- abandoned call is recognised at close_session.
    items                   TEXT,
    resolution              TEXT,
    callback_time           TEXT,
    resolved_at             TEXT,
```

Then `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m data.mock_db`.

- [ ] **Step 7: Run the suite and fix the call sites this breaks**

Existing tests comparing `check_escalation`/`record_turn` to a string will fail. Fix them
to read `signal.reason`. **Do not weaken an assertion to make it pass** — a test that
still passes while comparing a dataclass to a string has stopped checking anything.

- [ ] **Step 8: Commit.**

---

### Task 2: Open, resolve, and notify exactly once

**Files:** `agent/tools/escalation.py`, `tests/test_escalation.py`

**Produces:** `async open_escalation(...) -> dict`; `mark_resolved(...)`;
`async resolve_escalation(packet, items, resolution, callback_time=None) -> bool`;
`create_handoff_packet` unchanged externally.

**Packet-mutation contract (state it in the docstrings — v2 left it implicit and its own
tests would have failed a compliant implementation):**

- `open_escalation` returns a packet with `items=[reason]`, `resolution=None`, and
  `callback_time` from `_next_callback_slot()` — a *suggested, unbooked* time, kept only
  because `create_handoff_packet`'s existing contract (D-10) returns one.
- `resolve_escalation` **mutates the packet it is given** — writing final `items`,
  `resolution`, `callback_time`, and folding the outcome into `reason` (D-4) — *before*
  persisting and notifying. The packet is what `render_whisper` consumes, so the object
  the caller holds must match what was sent.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_opening_a_handover_writes_a_row_but_tells_nobody(monkeypatch):
    """Until an outcome is known there is nothing useful to tell a human. "A
    customer needs help, we don't know what about or when to ring" is a
    message they have to chase. The row is still written immediately, so a
    dropped call leaves a record."""
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)

    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "human please"}],
        "explicit request for a human",
    )

    assert packet["escalation_id"] is not None
    assert sent == []
    with get_connection() as conn:
        row = conn.execute(
            "SELECT resolution FROM escalations WHERE escalation_id = ?",
            (packet["escalation_id"],),
        ).fetchone()
    assert row["resolution"] is None


@pytest.mark.asyncio
async def test_resolving_notifies_once_and_puts_the_time_where_a_human_reads_it(monkeypatch):
    """"Notifies twice" is the failure a reader cannot see, so this counts.

    And the agreed time is folded into `reason` deliberately:
    transport/telephony.py's render_whisper reads escalation_id, reason,
    customer_intent, verified_account_info, actions_taken and sentiment — it
    never reads `items` or `callback_time`. Writing the time only to those
    fields would mean the human hearing the whisper never learns it."""
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
        items=["explicit request for a human", "refund eligibility question"],
        resolution=escalation.RESOLUTION_CALLBACK,
        callback_time=slot,
    )

    assert delivered is True
    assert len(sent) == 1, f"exactly one notification per handover, got {len(sent)}"
    assert sent[0]["resolution"] == escalation.RESOLUTION_CALLBACK
    assert sent[0]["callback_time"] == slot
    assert sent[0]["items"] == ["explicit request for a human", "refund eligibility question"]
    assert "refund eligibility question" in sent[0]["reason"], (
        "every item must reach `reason`, the only field the whisper renders"
    )
    assert packet["resolution"] == escalation.RESOLUTION_CALLBACK, (
        "the caller's packet is what the transport whispers — it must be updated in place"
    )


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
    """transport/pipecat_processors.py:227 (DTMF zero) calls this and must not
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
    assert packet["resolution"] == escalation.RESOLUTION_TRANSFER
    assert len(sent) == 1
```

Helpers:

```python
def _recording_notifier(sink: list[dict]):
    async def _notify(packet, **kwargs):
        sink.append(dict(packet))   # copy: resolve_escalation mutates in place
        return True
    return _notify


async def _stub_infer_fields(customer_id, messages, client=None):
    """Keeps these offline; the inference is covered separately."""
    return escalation.HandoffFields(
        customer_intent="wants a person",
        conversation_summary="asked for a human",
        verified_account_info=None,
        actions_taken=None,
        sentiment="neutral",
    )
```

- [ ] **Step 2: Run to verify they fail. Step 3: Implement.**

`open_escalation` is today's `create_handoff_packet` **minus** the notify and
`mark_notified` calls, plus `items`/`resolution` in the returned packet. Keep the existing
`redact_fields` call and its comment verbatim.

`resolve_escalation` order matters: **mutate the packet, persist, then notify.** A webhook
that hangs for its full retry budget must not leave the row claiming the handover is open.
Keep the three separate `try/except` blocks the existing code uses, each with its own
comment — they cover genuinely different failures.

`create_handoff_packet` = `open_escalation` then `resolve_escalation(...,
resolution=RESOLUTION_TRANSFER, callback_time=packet["callback_time"])`, returning the packet.

- [ ] **Step 4: Run the suite. Step 5: Commit.**

---

### Task 3: Two async resolution tools

**Files:** `agent/tools/handoff.py` (new), `agent/tools/scheduling.py`, `agent/session.py`,
`tests/test_handoff.py`

**Produces:** `SCHEDULE_CALLBACK_SCHEMA`, `RECORD_CALLBACK_DECLINED_SCHEMA`;
`async schedule_human_callback(slot_time, state, gate, customer_id) -> dict`;
`async record_customer_will_reach_out(escalation, customer_id, messages) -> dict`;
`SessionGates.escalation`; `book_appointment(..., key_prefix="book")`.

**Both tools are `async` (D-2).** `agent/core.py:147` awaits awaitable tool output, so an
`async def` handler works today. They book, persist and notify inside one await — the
customer agrees and the record is complete before the turn can fail.

- [ ] **Step 1: Write the failing tests** (`tests/test_handoff.py`)

Cover, at minimum:

```python
@pytest.mark.asyncio
async def test_scheduling_a_callback_proposes_before_it_books():
    """CLAUDE.md rule 6: booking is irreversible, so the first call proposes
    and only a later confirmed turn commits."""
    # assert result["status"] == "pending_confirmation"
    # assert state.status == STATUS_OPEN  (a proposal resolves nothing)
    # assert zero appointments rows


@pytest.mark.asyncio
async def test_a_confirmed_callback_books_persists_and_notifies_in_one_await():
    """The whole point of the tools being async. When this returns, the slot
    is held, the escalations row says `callback`, and the human has been told
    — so a turn that dies immediately afterwards cannot lose any of it."""
    # assert result["scheduled"] is True
    # assert state.resolution == RESOLUTION_CALLBACK and state.callback_time == slot
    # assert state.pending_persist is False   <-- nothing deferred
    # assert exactly one notification, carrying the slot
    # assert the appointments row exists with scheduled_time == slot


@pytest.mark.asyncio
async def test_a_callback_does_not_share_a_confirmation_key_with_an_appointment():
    """agent/tools/scheduling.py:172 keys the gate on ("book", slot_time) and
    nothing else. Without a distinct prefix: the customer proposes an ordinary
    appointment at slot X on turn 3 and never confirms it; a handover opens;
    the model proposes a CALLBACK at slot X on turn 5; the gate sees a
    matching pending proposal from an earlier turn and commits — booking an
    action the customer was never asked to confirm. A rule 6 violation."""
    gate = PendingActionGate()
    slot = _free_slot()
    scheduling.book_appointment(
        slot_time=slot, reason="a haircut", state=gate, customer_id=CUSTOMER_ID
    )                              # proposal, turn 0
    gate.turn += 1

    result = await handoff.schedule_human_callback(
        slot_time=slot, state=_open_state(), gate=gate, customer_id=CUSTOMER_ID
    )

    assert result["scheduled"] is False, "a callback must need its own confirmation"
    assert result["status"] == "pending_confirmation"


@pytest.mark.asyncio
async def test_an_unavailable_slot_leaves_the_handover_open():
    # assert result["error"] == "slot_unavailable" and state.status == STATUS_OPEN


@pytest.mark.asyncio
async def test_a_declined_callback_is_a_real_resolution_and_books_nothing():
    """"I'll call back when I know my schedule" is a legitimate ending, not a
    failure — and holding a slot they never agreed to is wrong."""


@pytest.mark.asyncio
async def test_the_tools_refuse_to_manufacture_a_handover_from_nothing():
    """Both tools are registered for EVERY session. A customer saying "I'll
    get back to you when I know my schedule" in an entirely ordinary call is a
    plausible prompt for the model to call record_customer_will_reach_out —
    which would write an escalations row, fire a Slack notification, and flip
    end_reason. Only a real offer or an open handover authorises opening one."""
    state = EscalationState()          # nothing open, nothing offered

    result = await handoff.record_customer_will_reach_out(
        escalation=state, customer_id=CUSTOMER_ID, messages=[]
    )

    assert result["recorded"] is False
    assert result["error"] == "no_handover"
    assert state.status == STATUS_NONE


@pytest.mark.asyncio
async def test_accepting_an_offer_opens_the_handover_the_offer_implied():
    """A suggested trigger only offers — nothing is open until the customer
    says yes, so the resolution tool has to open it. The offer is what makes
    this legitimate; see the test above for the case with no offer."""
    state = EscalationState()
    state.offered.add("repeated failed lookups")

    result = await handoff.record_customer_will_reach_out(
        escalation=state, customer_id=CUSTOMER_ID, messages=[]
    )

    assert result["recorded"] is True
    assert state.status == STATUS_RESOLVED and state.items


@pytest.mark.asyncio
async def test_resolving_twice_does_not_book_a_second_callback():
    """One customer, one callback, by design."""
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Give `book_appointment` a key prefix**

In `agent/tools/scheduling.py`, add a keyword-only `key_prefix: str = "book"` and use
`state.check(key=(key_prefix, slot_time))`. Default preserves every existing caller and
every existing test. Extend the comment at `:161-171` to say why a callback passes a
different prefix: what identifies the booking is still the time, but a callback and an
ordinary appointment are different actions, and rule 6 asks the customer to confirm the
action, not the slot.

- [ ] **Step 4: Write `agent/tools/handoff.py`**

Module docstring states: these are async because `agent/core.py:147` awaits awaitable tool
output, so they can persist and notify at the moment the customer agrees rather than
deferring it — a deferred resolution is one a failing turn can lose.

`schedule_human_callback`:
1. Already resolved → `{"scheduled": False, "error": "already_resolved", ...}` naming the
   existing `callback_time`, and saying anything else raised goes to the same colleague.
2. `_ensure_open` guard (see below) — no offer and nothing open → `no_handover`.
3. `scheduling.book_appointment(..., key_prefix="callback", state=gate, ...)`.
4. `pending_confirmation` → pass the message through unchanged.
5. `not booked` → pass `error`/`message` through so the model sees `slot_unavailable`.
   **State stays open.**
6. Booked → `_ensure_open`, `state.record_resolution(RESOLUTION_CALLBACK, slot_time)`,
   then **await the persist**: open the packet if there is none, then
   `await escalation.resolve_escalation(...)`, then `state.pending_persist = False`.
   Wrap that in `try/except` — a tool must never raise into the loop — and on failure
   leave `pending_persist` True so `close_session` retries (D-5).
7. Return `{"scheduled": True, "appointment_id": ..., "callback_time": slot_time,
   "spoken_time": ...}` — include the spoken form so the model can read it back.

`record_customer_will_reach_out(escalation, customer_id, messages)`: same guard, then
`record_resolution(RESOLUTION_SELF)` and the same awaited persist.

```python
ACCEPTED_OFFER_REASON = "customer accepted an offer of a callback"


def _may_open(state: EscalationState) -> bool:
    """Only a real offer, or an already-open handover, authorises opening one.

    Both tools are registered for every session, so without this a model could
    call one during an entirely ordinary conversation and manufacture an
    escalation row, a Slack message, and a changed end_reason out of nothing.
    """
    return state.status != STATUS_NONE or bool(state.offered)
```

- [ ] **Step 5: Register the tools**

`agent/session.py`: add `handoff` to the tools import; two entries in `TOOLS`; a field on
`SessionGates`:

```python
    # Phase 12. Not a PendingActionGate, but a gate in the most literal sense —
    # it is what stops the call ending while a handover is unresolved. Here
    # rather than on Session because the tool handlers close over it, and they
    # are built before the Session that owns them exists.
    escalation: escalation.EscalationState = field(default_factory=escalation.EscalationState)
```

and two handlers. `schedule_human_callback` gets `state=gates.escalation`,
`gate=gates.scheduling`, `customer_id=customer_id`. `record_customer_will_reach_out` needs
`messages` for `open_escalation`'s inference — pass a zero-argument accessor or the
session's list by reference; **decide and write it down**, do not leave it implicit.

- [ ] **Step 6: Run the suite. Step 7: Commit.**

---

### Task 4: The per-turn bounded refusal

**Files:** `agent/tools/summary.py`, `agent/session.py`, `tests/test_session.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_end_conversation_is_refused_while_a_handover_is_open():
    state = EscalationState()
    state.open("explicit request for a human")
    result = summary.end_conversation(escalation=state, turn=1)
    assert not should_end_session([{"name": "end_conversation", "output": result}])


def test_the_refusal_budget_survives_a_model_that_retries_inside_one_turn():
    """agent/core.py:91 allows 8 tool iterations. A model that reads "Not yet."
    and simply calls end_conversation again burns a call-counted budget with
    no customer utterance in between — and the call ends on the trigger turn
    with the handover open, which is the bug this phase exists to fix."""
    state = EscalationState()
    state.open("explicit request for a human")

    for _ in range(5):
        result = summary.end_conversation(escalation=state, turn=7)
        assert not should_end_session([{"name": "end_conversation", "output": result}])
    assert state.refusals == 1


def test_the_refusal_gives_up_rather_than_trapping_the_customer():
    """A refusal is a nudge, not a cage. A customer who says "just let me go"
    and never picks a callback must still be able to leave; close_session
    records the handover as unresolved, so the human still hears about them."""
    state = EscalationState()
    state.open("explicit request for a human")

    for turn in range(1, escalation.MAX_END_REFUSALS + 1):
        refused = summary.end_conversation(escalation=state, turn=turn)
        assert not should_end_session([{"name": "end_conversation", "output": refused}])

    allowed = summary.end_conversation(escalation=state, turn=escalation.MAX_END_REFUSALS + 1)
    assert should_end_session([{"name": "end_conversation", "output": allowed}])


def test_end_conversation_is_allowed_once_resolved_and_with_no_handover():
    """The second case is the overwhelming majority of calls — nothing here
    may make an ordinary goodbye harder."""
```

- [ ] **Step 2-3: Implement**

`agent/tools/summary.py` gains `END_REFUSED_PREFIX = "Not yet."` and

```python
def end_conversation(escalation: Any | None = None, turn: int = 0) -> str:
    """Normally there is no real work to do — the point is Claude choosing to
    call this tool at all, and the transport watching for it.

    The exception is an unresolved handover: the agent has told a customer a
    human will help and has not arranged how, so ending strands them. That is
    exactly what happened live, where it asked "would you like me to find some
    callback slots?" and hung up on the same turn.

    Refused as a returned message rather than a raised exception — the same
    shape issue_refund uses for an outstanding confirmation. The model reads it
    as a tool result and works the problem; an exception would break the turn.

    This refusal is the model-facing nudge. It is NOT what stops the call
    ending — run_turn does that (see D-1), because the escalation for this very
    turn is not detected until after the tool loop has finished.
    """
    if escalation is not None and escalation.is_open and escalation.consume_refusal(turn):
        return (
            f"{END_REFUSED_PREFIX} Sort out the handover before saying goodbye. "
            "Offer a callback time with find_available_slots and "
            "schedule_human_callback, or — if they would rather get in touch "
            "themselves — call record_customer_will_reach_out."
        )
    return "Session marked complete."
```

`agent/session.py`: handler becomes
`lambda **kw: summary.end_conversation(**kw, escalation=gates.escalation, turn=session_turn())`
— the turn number must come from the live session, so **decide how** (a small mutable
holder set by `run_turn`, or a closure over `gates`) and write it down. `should_end_session`
gains the output check:

```python
    return any(
        call["name"] == "end_conversation"
        and not str(call.get("output", "")).startswith(summary.END_REFUSED_PREFIX)
        for call in tool_calls
    )
```

Verified safe against existing callers: `tests/test_text_cli.py:111-113` passes tool_calls
with no `"output"` key (`.get` handles it) and `tests/test_validators.py:18` passes
`"output": "done"`.

- [ ] **Step 4: Run the suite. Step 5: Commit.**

---

### Task 5: Rewire `run_turn` and `close_session`

**Files:** `agent/session.py`, `tests/test_session.py`

**This is the largest diff and the one most likely to need a second round.** Read D-1,
D-3, D-5 and D-6 before writing.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_the_call_does_not_end_on_the_turn_a_handover_opens(monkeypatch, tmp_path):
    """THE regression test for this phase.

    v2 shipped a test for this that used a plain-text reply with no
    end_conversation call — it passed with the bug fully present. This one
    scripts the model calling end_conversation on the SAME turn the trigger
    fires, which is the real sequence: agent.send() runs the whole tool loop
    (agent/session.py:291) before escalation is checked (:372), so the state
    is still `none` for the entire loop and the refusal cannot fire."""
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "t.db")
    mock_db.reset_and_seed()
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("end_conversation", {}),
            _text_response("Of course. When would suit you for a call?"),
        ]
    )
    _force_signal(monkeypatch, "explicit request for a human", mandatory=True)
    session = create_session("CUST-1001", client=fake_client)

    outcome = await run_turn(session, "I can't discuss this further right now")

    assert outcome.ended is False, (
        "a handover opened this very turn must stop the call ending on it"
    )
    assert session.gates.escalation.is_open


@pytest.mark.asyncio
async def test_a_booked_callback_does_not_trigger_a_twilio_transfer(monkeypatch, tmp_path):
    """transport/pipecat_processors.py:418 fires the warm transfer on
    end_reason == "escalated", and transport/telephony.py answers with
    "Connecting you now. Please hold." and dials a human.

    So a customer who agreed to a call next Tuesday and then said goodbye must
    NOT end as "escalated" — they would be bridged to a live human on a call
    they were finishing. The value survives for outcomes that still mean
    transfer now; a settled callback is not one of them."""
    # ... open a handover, resolve it as RESOLUTION_CALLBACK, then end the call
    assert outcome.end_reason == "model_ended"


@pytest.mark.asyncio
async def test_an_unresolved_handover_still_ends_as_escalated(monkeypatch, tmp_path):
    """The other half: a caller who would settle nothing still needs the
    Twilio hook, because a live transfer is the only handover left."""
    # ... open a handover, exhaust MAX_END_REFUSALS, end the call
    assert outcome.end_reason == "escalated"
    assert outcome.escalation_packet is not None, "the hook needs the packet"


@pytest.mark.asyncio
async def test_an_ordinary_call_still_ends_as_model_ended(monkeypatch):
    """The value must not start appearing on calls that never escalated."""


@pytest.mark.asyncio
async def test_an_ordinary_turn_does_not_raise(monkeypatch):
    """v2's snippet passed notice=notice in both TurnOutcome branches without
    ever initialising it — UnboundLocalError on every turn with no signal.
    The cheapest possible test for the cheapest possible mistake."""
    _force_signal(monkeypatch, None, mandatory=False)
    session = _session_replying("Sure, it's out for delivery.")
    outcome = await run_turn(session, "where is my order")
    assert outcome.notice is None


@pytest.mark.asyncio
async def test_a_suggested_trigger_offers_and_resets_its_own_counter(monkeypatch):
    """A mis-dictated order number is a cooperative repair — the healthiest
    signal a conversation can produce. It must not be treated as a verdict.

    The counter is PRIMED first: _force_signal replaces check_escalation
    wholesale, so tracker.record_turn never runs and a fresh tracker's counter
    is 0 anyway. Without priming, "assert counter == 0" passes whether or not
    reset_streak was ever called."""
    session = _session_replying("I'm still not finding that order.")
    session.tracker.consecutive_failed_lookups = escalation.FAILED_LOOKUP_ESCALATION_THRESHOLD
    _force_signal(monkeypatch, "repeated failed lookups", mandatory=False)

    outcome = await run_turn(session, "let me try that number again")

    assert outcome.ended is False
    assert session.gates.escalation.status == escalation.STATUS_NONE, "an offer opens nothing"
    assert outcome.notice is not None and "?" in outcome.notice
    assert session.tracker.consecutive_failed_lookups == 0
    assert "repeated failed lookups" in session.gates.escalation.offered


@pytest.mark.asyncio
async def test_a_spoken_question_reaches_the_model_history(monkeypatch):
    """The notice is spoken by every transport but was never written into
    session.agent.messages. So the agent asks "when would be a good time?",
    the customer answers "Tuesday at two", and the model has no record of
    asking — which is the spec's own complaint #1, made worse, because the
    appended text is a question the conversation is waiting on."""
    session = _session_replying("I'll get a colleague onto this.")
    _force_signal(monkeypatch, "repeated failed lookups", mandatory=False)

    outcome = await run_turn(session, "still no luck")

    history = str(session.agent.messages)
    assert outcome.notice in history, "a question the agent asks must be in its own history"


@pytest.mark.asyncio
async def test_a_suggested_trigger_only_offers_once(monkeypatch):
    """Being asked over and over whether you want a human is its own failure."""


@pytest.mark.asyncio
async def test_one_problem_produces_one_escalation_row(monkeypatch, tmp_path):
    """A live call produced escalation rows 7, 8 and 9 for one problem — with
    n8n connected that is three Slack messages about one customer. Counted as
    rows, because that is the failure a reader misses."""


@pytest.mark.asyncio
async def test_a_resolution_lost_to_a_crashing_turn_is_flushed_at_close(monkeypatch, tmp_path):
    """The C-3 window. The tools persist inline now, so this covers the
    remaining case: the persist itself failed, leaving pending_persist True on
    a RESOLVED state. close_session must flush on pending_persist, NOT on
    is_open — guarding on is_open drops a resolved-but-unsent handover
    silently, with the slot held and nobody told."""
    # ... force resolve_escalation to fail once inside the tool, then close
    assert len(sent) == 1
    assert sent[0]["resolution"] == escalation.RESOLUTION_CALLBACK


@pytest.mark.asyncio
async def test_an_abandoned_handover_notifies_as_unresolved(monkeypatch, tmp_path):
    """Customers hang up, sockets die. The human still needs to hear about
    them — and to know no time was agreed, rather than being handed a slot
    nobody promised."""


@pytest.mark.asyncio
async def test_closing_twice_notifies_once(monkeypatch, tmp_path):
    """close_session must be idempotent: it is called from a finally block in
    every transport, and a retried teardown must not produce a second Slack
    message about the same customer."""


@pytest.mark.asyncio
async def test_a_transferred_caller_is_not_also_reported_unresolved(monkeypatch, tmp_path):
    """DTMF zero (transport/pipecat_processors.py:227) opens and resolves its
    OWN row via create_handoff_packet and never touches gates.escalation. If
    the session already had an open handover, the caller is transferred
    (notification 1) and then close_session reports the session's own
    escalation as unresolved (notification 2, different id) — two messages
    about one caller, one of them wrong."""
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Add the notice helpers and the history writer**

```python
def _escalation_handover_notice() -> str:
    """What the customer hears the moment a handover opens.

    No time is named yet — that is the whole change. Escalation used to
    announce a slot the customer had never agreed to and then end the call.
    """
    return (
        "Let me get one of my colleagues onto this for you. "
        "When would be a good time for them to give you a call?"
    )


def _escalation_offer(reason: str) -> str:
    """What the customer hears when the agent SUSPECTS it is failing them.

    An offer, not an announcement, naming a way to carry on. Deliberately
    never names the internal reason: "sustained negative sentiment across
    multiple turns" read aloud tells an already frustrated customer they have
    been classified as angry.
    """
    if reason == "repeated failed lookups":
        return (
            "I'm still not finding that. Would you like to try the number once more, "
            "or shall I have a colleague call you back about it?"
        )
    return "Would it help if I arranged for a colleague to call you back about this?"


def _append_to_last_assistant_message(messages, text) -> str | None:
    """Record something the agent SAID but did not generate.

    Notices are spoken by every transport and were never written back into
    history, so a notice that ends in a question left the model waiting on an
    answer to a question it has no record of asking.

    Same defensive shape as _substitute_hedge_in_history above: never raises,
    returns a warning string instead of guessing when the message is not the
    text-only assistant turn it expects.
    """
```

- [ ] **Step 4: Rewrite the escalation branch**

Order matters — this is D-1.

```python
    notice: str | None = None          # M-10: unbound otherwise on ordinary turns
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
                notice = None
                warnings.append(
                    f"Escalation triggered ({signal.reason}) but couldn't be logged: {exc}"
                )
        else:
            state.amend(signal.reason)
    elif signal:
        if signal.reason not in state.offered and not state.is_open:
            state.offered.add(signal.reason)
            notice = _escalation_offer(signal.reason)
        session.tracker.reset_streak(signal.reason)

    if notice:
        # The customer is about to hear this. If the model has no record of
        # saying it, their answer arrives as a non-sequitur.
        history_warning = _append_to_last_assistant_message(session.agent.messages, notice)
        if history_warning:
            warnings.append(history_warning)

    # THE END DECISION — computed here, after the state is known, never before.
    # agent.send() ran the whole tool loop above, so on the turn a trigger
    # fires the handover did not exist while end_conversation was called and
    # could not have been refused. This is what actually stops the hang-up.
    ending = should_end_session(result.tool_calls)
    if ending and state.is_open:
        ending = False
        warnings.append("end_conversation suppressed: the handover is still open")
        if not notice:
            notice = _escalation_handover_notice()
            history_warning = _append_to_last_assistant_message(session.agent.messages, notice)
            if history_warning:
                warnings.append(history_warning)

    if ending:
        # D-3. The value survives for outcomes that still mean "transfer this
        # call now" — transport/pipecat_processors.py:418 bridges a live human
        # on it. A booked callback or a customer-will-reach-out ends as
        # model_ended: the colleague has already been told out of band, and
        # bridging someone who agreed to a call next Tuesday is wrong.
        transferable = state.resolution in (
            None, escalation.RESOLUTION_TRANSFER, escalation.RESOLUTION_UNRESOLVED
        )
        escalated = state.status != escalation.STATUS_NONE and transferable
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

**Keep the empty-reply farewell fallback exactly as written** — a live fix, and a call
must never end in silence.

- [ ] **Step 5: Fix the turn log's meaning (F-8)**

`observability/turn_log.py` gains `escalation_offered: str | None`. In `run_turn`:

```python
                escalated=state.status != escalation.STATUS_NONE,
                # Only a reason that actually opened or amended a handover. A
                # suggested trigger that merely OFFERED is not an escalation,
                # and writing it here would make eval/scoring.py's
                # score_escalation ("did this scenario escalate?" = any row
                # with an escalation_reason) fail every never-escalates
                # scenario, and would make the turn-log baseline count offers
                # as escalations.
                escalation_reason=signal.reason if (signal and signal.mandatory) else None,
                escalation_offered=signal.reason if (signal and not signal.mandatory) else None,
```

- [ ] **Step 6: Rewrite `close_session`'s flush**

```python
    # Flush anything the turn loop could not. Guarded on pending_persist, NOT
    # on is_open: a resolution recorded and then lost to a failing persist is
    # RESOLVED, not open, and guarding on is_open drops it silently — slot
    # held, row still open, nobody told.
    state = session.gates.escalation
    if state.pending_persist and state.packet is not None:
        if state.resolution == escalation.RESOLUTION_TRANSFER:
            # DTMF zero already opened, resolved and notified its own row.
            state.pending_persist = False
        else:
            try:
                await escalation.resolve_escalation(
                    state.packet,
                    items=state.items,
                    resolution=state.resolution or escalation.RESOLUTION_UNRESOLVED,
                    callback_time=state.callback_time,
                )
                state.status = escalation.STATUS_RESOLVED   # idempotent: no second message
                state.pending_persist = False
            except Exception as exc:  # noqa: BLE001 — an exit path must never crash
                close_error = f"Could not record the handover: {exc}"
```

Surface `close_error` through `SessionCloseResult.error` the way every other close failure
is surfaced (`agent/session.py:473-474`), rather than only logging it.

- [ ] **Step 7: Run the suite.** Existing tests asserting `end_reason == "escalated"` *at
detection* will fail — that behaviour is deliberately gone. Update them to the new state
machine; **do not delete them**.

- [ ] **Step 8: Delete or repurpose the dead helpers**

After this task `_escalation_notice` and `_spoken_time` have no callers
(`transport/pipecat_processors.py:231` builds its own DTMF string). **Repurpose rather
than delete:** `schedule_human_callback` returns `spoken_time`, so use `_spoken_time(slot)`
to read the agreed time back — "Thursday at 9am" rather than an ISO timestamp. That is the
one place this phase genuinely gains from them.

- [ ] **Step 9: Commit.**

---

### Task 6: Prompt guidance

**Files:** `agent/prompts.py`, `tests/test_escalation.py`

- [ ] **Step 1: Add a handover section** (line-continuation style, like the rest of the prompt)

```
## Handing over to a colleague

Sometimes you cannot finish something yourself — the customer asks for a
person, a policy needs a specialist, or you are simply not getting anywhere.
When that happens you are not finished: you have to arrange the handover
before the call can end.

- Call find_available_slots and offer a real time.
- When they pick one, call schedule_human_callback. The first call asks them
  to confirm; call it again with the same time once they say yes. Read the
  agreed time back to them.
- If they would rather get in touch themselves, call
  record_customer_will_reach_out. That is a perfectly good outcome.
- Never say goodbye before one of those two has gone through.
- If they clearly want to go and will not settle either, let them. Say a
  colleague will be in touch, and close warmly. Do not keep asking.

Once the handover is arranged the call is not over. Ask whether there is
anything else, and if there is, help with it normally — a refund you can
process yourself gets processed, not handed over. Only pass on something
genuinely beyond you, and when you do, do not arrange a second callback: the
same colleague covers it on the same call. Say so simply — "I'll add that to
what they're calling you about."
```

**Do not include "ask when would suit them for a callback"** — `_escalation_handover_notice`
already says exactly that and is now written into history (D-6). Instructing the model to
ask it too makes the customer hear the same question twice in one turn.

- [ ] **Step 2: Assert the prompt only names real tools**

```python
def test_the_prompt_only_names_handover_tools_that_exist():
    """A prompt naming a renamed tool is a silent failure — the model calls it,
    gets "unknown tool", and improvises."""
    from agent.prompts import SYSTEM_PROMPT
    from agent.session import TOOLS

    names = {schema["name"] for schema in TOOLS}
    for named in ("find_available_slots", "schedule_human_callback",
                  "record_customer_will_reach_out"):
        assert named in names, f"SYSTEM_PROMPT names {named}, which is not a registered tool"
        assert named in SYSTEM_PROMPT
```

- [ ] **Step 3: Run the suite. Step 4: Commit.**

---

### Task 7: Rewrite the six eval scenarios

**Files:** `eval/scenarios.py`, `tests/` (a new structural test)

These **cannot** be fixed by re-recording. `eval/scoring.py:292-294` reads
`result.observed[-1].end_reason`, and `eval/harness.py:139-155` breaks the turn loop on
`outcome.ended`. Editing definitions needs no credit; only the re-record does.

**Group A — single-turn scenarios that can no longer produce an ending turn.** With
detection no longer ending the call, the last observed turn's `end_reason` is `None`
forever, so re-recording yields the same `None`.

| Scenario | Line | Change |
|---|---|---|
| `refund_high_value_escalates` | 315 | add a closing turn; expect a handover to open, and the call to end only after it resolves |
| `triage_explicit_human_request` | 431 | same |
| `triage_policy_restricted_topic` | 499 | same |

**Group B — scenarios asserting behaviour this phase deletes.** Each pins
`escalation_turn=2` + a suggested `escalation_reason` + `end_reason="escalated"`. Under
D-9 these triggers never escalate; they offer.

| Scenario | Line | Change |
|---|---|---|
| `triage_sustained_frustration` | 448 | assert an **offer**, a non-ended call, and no `escalations` row |
| `triage_repeated_failed_lookups` | 486 | same |
| `guardrail_ungrounded_ladder_escalates` | 519 | same |

Group B is arguably the most valuable new coverage in this phase: it pins the exact
behaviour whose absence hung up on a real customer twice.

- [ ] **Step 1: Add a structural test that cannot pass while the scenarios are wrong**

v2's check was `grep -c 'end_reason="escalated"' eval/scenarios.py # expect 6` — which
passes while all six are broken. Replace it with something that knows the semantics:

```python
def test_no_scenario_expects_a_suggested_trigger_to_escalate():
    """Suggested triggers offer; they do not escalate. A scenario still
    pinning end_reason="escalated" on one of them encodes deleted behaviour
    and will fail at the next live eval run — after credit has been spent."""
    from agent.tools.escalation import SUGGESTED_REASONS
    from eval.scenarios import SCENARIOS

    for scenario in SCENARIOS:
        if scenario.expect.escalation_reason in SUGGESTED_REASONS:
            assert scenario.expect.end_reason != "escalated", (
                f"{scenario.name} expects a suggested trigger to escalate"
            )
            assert scenario.expect.escalation_turn is None, (
                f"{scenario.name} pins an escalation turn for a trigger that only offers"
            )


def test_no_scenario_expects_a_settled_callback_to_transfer_the_call():
    """D-3: end_reason="escalated" still bridges a live Twilio call
    (transport/pipecat_processors.py:418). A scenario that both books a
    callback and expects "escalated" is asserting that a customer who agreed
    to a call next Tuesday gets connected to a human immediately."""
    from eval.scenarios import SCENARIOS

    for scenario in SCENARIOS:
        booked = any(
            tool.name == "schedule_human_callback" for tool in scenario.expect.tools_called
        )
        if booked:
            assert scenario.expect.end_reason != "escalated", (
                f"{scenario.name} books a callback and still expects a live transfer"
            )
```

- [ ] **Step 2: Rewrite the six. Step 3: Run the suite. Step 4: Commit.**

- [ ] **Step 5: Note the re-record in `PROGRESS.md`** — the recordings must be regenerated
at the 10c live checkpoint before the eval suite is trusted again. `eval/recordings/` holds
only `.gitkeep` today, so nothing offline is affected in the meantime.

---

## Checkpoint

**Automated:** the full suite passes offline with no API keys, at **≥ 359 passed, 3 skipped**.

**Manual (needs credit):** three live calls.

1. **The happy path.** Ask for a human. The call **does not end**. Agree a time, confirm
   it, hear it read back. Get asked "anything else?", raise an ordinary refund, have it
   processed normally. Raise a second escalation-worthy issue and hear it **added** to the
   existing handover rather than booking a second call. Say goodbye and hear one. Confirm
   Slack shows **exactly one** message carrying both items and the agreed time.
2. **The refusenik.** Ask for a human, then "actually, forget it, goodbye." The call must
   be allowed to end after at most two nudges, and Slack must show it as `unresolved`.
3. **The cooperative repair.** Mis-dictate an order number twice. The agent **offers**
   rather than imposes, and carries on normally when you say you'd rather try again.
   No `escalations` row is written.

## Self-review

- **Every critical and high from the v2 review is addressed:** C-1→D-1+Task 5;
  C-2→D-2+Task 3; C-3→D-2+D-5; F-4→D-7; F-5→D-3; F-6→D-4; F-7→Task 7; F-8→Task 5 Step 5;
  F-9→D-6.
- **Mediums addressed:** M-10 (`notice` init), M-11 (repurpose the helpers), M-12
  (`key_prefix`), M-13 (`_may_open`), M-14 (skip transfer at close), M-15 (idempotent +
  surfaced error), M-16 (DB backup), M-17 (primed counter), M-18 (packet contract, and
  every line reference re-verified).
- **Two deliberate scope additions**, both forced: `observability/turn_log.py` (F-8 —
  otherwise this phase silently changes what `escalation_reason` means) and
  `eval/scenarios.py` (F-7 — otherwise six scenarios break at the next paid eval run).
- **Known soft spots:** Task 5 is the largest diff. Task 3's `messages` accessor and Task
  4's turn accessor are the two places the plan says "decide and write it down" rather than
  dictating a mechanism — both are small, but an implementer must not leave them implicit.
