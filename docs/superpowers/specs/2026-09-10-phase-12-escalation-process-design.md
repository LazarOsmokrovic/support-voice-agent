# Phase 12 — Escalation as a resolvable process

Date: 2026-09-10
Status: approved, not yet implemented
Depends on: Phase 4 (escalation), Phase 5 (scheduling), Phase 11 (n8n notification).

## Why this phase exists

Escalation today is an **event that ends the call**. One turn: a trigger fires,
`create_handoff_packet` writes a row and notifies Slack, and `run_turn` returns
`ended=True`. There is no process, no resolution, and no way back.

Found by talking to it. From the turn log:

> **Customer:** "if it's necessary. I'm not able to discuss further at the moment."
> **Agent:** "Absolutely, that makes sense. **Would you like me to find some available
> callback slots for you?**"
> *— and the call ended on that same turn, reason `explicit request for a human`.*

The agent asked a question and hung up before the answer. Three separate failures
are visible in that one exchange:

1. **The reply and the escalation decision do not know about each other.** The model
   composes a reply; `check_escalation` then independently decides to hand off. So
   the agent can ask a question it will never hear answered — incoherent whether or
   not the escalation was warranted.
2. **No callback time was ever agreed**, so the customer left not knowing when anyone
   would ring, and the human agent had nothing actionable. The escalation "succeeded"
   while helping nobody.
3. **A misjudgement was fatal.** "I'm not able to discuss further" is not a request
   for a human, but once classified as one there was no path back.

The fix is not a better classifier. It is that **escalation should be a state the
agent has to work its way out of**, with the call unable to end until it has.

## Not every trigger deserves to be imposed

A second live call exposed a deeper problem than the one above. The customer
mis-dictated an order number, said so immediately — *"I'm sorry, that's not what I
said, let me give you a new one"* — and the agent escalated and hung up.

`FAILED_LOOKUP_ESCALATION_THRESHOLD = 2`, and the counter resets **only on a
successful lookup**. A customer correcting themselves resets nothing, because the
tracker reads tool outcomes and never sees the conversation. Turn 3 invalid → 1.
Turn 7 not-found → 2 → escalate. Turn 9 → escalate again. Turn 10 contained *no
lookup at all* and escalated a third time, because the counter is re-checked every
turn regardless of whether anything was looked up.

The trigger was built to catch *"the agent is failing this customer"*, which is a fair
thing to catch. But it counts failures without asking whose they are, and cannot
distinguish:

- **the agent failing** — orders that exist are not found, something is broken;
- **the customer self-correcting** — a mis-heard digit, immediately acknowledged.

The second is a **cooperative repair**, the healthiest signal a conversation can
produce. The customer was one transposed digit from success and got hung up on
instead.

So triggers divide by whose decision they represent:

| Trigger | Class | Why |
|---|---|---|
| Explicit request for a human | **mandatory** | The customer asked |
| Policy-restricted topic | **mandatory** | A rule, not a judgement |
| Tool-signalled (high-value refund) | **mandatory** | A specialist must approve |
| Repeated failed lookups | **suggested** | The agent's inference |
| Sustained negative sentiment | **suggested** | The agent's inference |
| Repeated ungrounded replies | **suggested** | A guardrail's suspicion |

**Mandatory triggers open an escalation directly.** **Suggested triggers make the
agent offer one** — *"I'm still not finding that. Would you like me to have a
colleague call you, or shall we try the number once more?"* — and an escalation opens
only if the customer accepts.

This is the principle rule 6 already applies to refunds, extended to handoffs: the
agent proposes, the customer decides. An inference is a suggestion, not a verdict.

**Declining resets the counter that produced the offer**, so the customer gets a clean
run at correcting themselves rather than tripping the same threshold on their next
breath. Without the reset an offer is meaningless — the agent would simply ask again
one turn later.

**A suggested trigger fires at most one offer per state.** Being asked repeatedly
whether you want a human is its own kind of failure.

## The state machine

A session holds **at most one escalation for its entire life**.

| State | Meaning | May the call end? |
|---|---|---|
| `none` | Nothing escalated | yes |
| `open` | Detected, no resolution agreed | **no** |
| `resolved` | Callback booked, or customer declined | yes |

**Transitions**

- **none → open.** A trigger fires. The `escalations` row is written immediately, so
  a dropped call still leaves a record. **Nothing is sent to Slack yet** — there is
  nothing useful to tell a human until the outcome is known. The call does *not* end.
- **open → resolved.** Either `schedule_human_callback(slot_time)` or
  `record_customer_will_reach_out()`. **This is when the notification fires**, carrying
  the resolution, the agreed time, and every item gathered.
- **any → amended.** A second trigger fires later in the same call. It does **not**
  reopen the process or schedule a second callback. It appends an item to the existing
  packet; if the escalation was already resolved, an *update* notification goes out
  against the same `escalation_id`.

## Why one escalation per session

Because that is what actually happens on a support line. A human agent ringing a
customer back resolves everything that customer has — they do not schedule three
separate calls for three questions. So a second escalation-worthy issue should reach
the same human, on the same call, as an additional item they can prepare for.

This is also why the packet stops being one `reason` string and becomes a **list of
items**. "Cancellation query, plus a refund eligibility question" is what a human
taking a handoff needs; a single sentence is not.

## The conversation continues

Resolution is not the end of the call. Once the callback is arranged the agent asks
whether there is anything else, and then **handles it normally** — a refund it is
perfectly capable of processing gets processed, not escalated. Only genuinely
escalation-worthy things amend the packet.

This is the behaviour a human agent has: *"I've booked that call for you. Was there
anything else while I have you?"*

## Two resolution paths, both first-class

**Scheduled callback.** The customer picks a time from `find_available_slots`, and it
becomes a **real appointment** via Phase 5's `book_appointment` — reserved in the
`appointments` table, and confirmed through rule 6's propose-then-confirm turn. So the
time Slack shows is one that is genuinely held, and two customers cannot be promised
the same slot. It costs one extra turn to confirm, which is correct: booking is
irreversible.

**Customer declines.** "I'll call back when I know my schedule" is a legitimate ending,
not a failure. `record_customer_will_reach_out()` resolves the escalation, the human
agent is told the customer will initiate contact, and no slot is held.

## What stops the call ending

`end_conversation` returns an error while an escalation is `open`, naming why — the
same propose-then-refuse shape `issue_refund` already uses when a confirmation is
outstanding. The model sees the refusal and works the problem instead of hanging up.

**But a call can still drop.** The customer hangs up, the socket dies, the process
restarts. So `close_session` checks for an unresolved escalation and notifies with
status `unresolved`, so the human agent still hears about the customer — and also
learns that no time was agreed.

That gives **exactly one notification per escalation, always carrying the truth**:
resolved with a time, resolved as customer-initiated, or unresolved because the call
ended first. Amendments after resolution add an update; they never duplicate.

## Components

| File | Change |
|---|---|
| `agent/session.py` | `Session` gains `escalation: OpenEscalation \| None`. `run_turn`'s escalation branch stops setting `ended=True`; it opens or amends state instead. `close_session` notifies an unresolved escalation. |
| `agent/tools/escalation.py` | `create_handoff_packet` splits: writing the row at open, notifying at resolve/amend/abandon. Packet carries `items`, `resolution`, `callback_time`. |
| `agent/tools/handoff.py` (new) | `schedule_human_callback`, `record_customer_will_reach_out` — the two resolution tools, plus their schemas. |
| `agent/session.py` (`TOOLS`) | Registers the two new tools; `end_conversation` gains its refusal path. |
| `agent/prompts.py` | Escalation-mode guidance: arrange a callback, then ask what else; do not re-escalate, amend. |
| `data/mock_db.py` | `escalations` gains `items`, `resolution`, `callback_time`, `resolved_at`. |

**Rule 5 holds** — nothing here touches a transport. Rule 7 holds too: resolution is a
deterministic tool call, not a judgement inferred from prose. Rule 6 holds because the
callback is a real booking behind a real confirmation.

## Error handling

- **A trigger fires while an escalation is already open** — amend, never reopen. No
  second callback, no second process. Today this re-fires every single turn once
  tripped: one live call produced escalation rows 7, 8 and 9 for one problem, which
  with n8n connected would have been three Slack messages about one customer.
- **A suggested trigger fires but the customer declines** — no escalation opens, the
  counter resets, and no further offer is made for that trigger until something
  changes. The conversation continues normally.
- **A turn with no tool calls at all** must not re-evaluate a lookup counter. Turn 10
  of the call above escalated while merely being asked to repeat a number back.
- **The customer never resolves and the call drops** — `close_session` notifies as
  `unresolved`. The record is never silently lost.
- **`schedule_human_callback` with an unavailable slot** — `book_appointment` already
  returns `slot_unavailable`; the escalation stays `open` and the agent offers another.
- **The notification fails** — the escalation is still resolved and the row still
  records it. Delivery has never been allowed to affect persistence (Phase 11), and
  that does not change.
- **`end_conversation` while open** — refused with a message the model can act on, not
  an exception.
- **The classifier misfires** — now recoverable: the customer declines, the escalation
  resolves as customer-initiated, and the conversation continues.

## Testing

- The state machine directly: none → open → resolved; open → amended; resolved →
  amended; and that a second trigger never opens a second escalation.
- `end_conversation` is refused while open and permitted once resolved.
- **Exactly one notification per escalation**, and never at open. Asserted by counting
  calls, because "notifies twice" is the failure mode a reader cannot see.
- An abandoned call notifies as `unresolved`.
- A resolved escalation writes a real `appointments` row, and a declined one does not.
- The packet's `items` accumulate in order and survive redaction with identifiers intact.
- **Regression for the bug that prompted this:** a turn whose reply ends in a question
  must not end the call when escalation fires.
- **Regression for the second live failure:** two failed lookups make the agent *offer*
  rather than escalate; declining resets the counter; and a third consecutive failure
  after a decline does not silently escalate behind the customer's back.
- A turn with no tool calls does not advance or re-trigger a lookup counter.
- One tripped trigger produces **one** escalation row, not one per subsequent turn.
  Asserted by counting rows, since the live failure produced three for one problem.
- A mandatory trigger still opens without asking — the offer path must not weaken an
  explicit request for a human or a policy-restricted topic.

## A related defect this phase must also fix

The same call showed `ended=True` returned at turn 7, with turns 9 and 10 continuing
normally afterwards. Whatever the transport does with `EndFrame`, the session did not
actually stop — which is also why nothing was spoken at the real ending.

This is independent of the escalation redesign and would bite anywhere `ended=True` is
returned, including a normal `end_conversation`. Phase 12 removes the specific case
that produced it (escalation no longer ends a call at detection), but the underlying
"a turn said the call was over and it wasn't" behaviour needs its own investigation and
its own test, not merely to be routed around.

Values come from `data/mock_db.py` at runtime, never hard-coded.

## Checkpoint

**Automated:** the full suite passes offline with no API keys.

**Manual:** a live call that escalates, agrees a callback time, hears the agent ask
what else it can help with, then has a *second* issue handled normally — and a second
escalation-worthy issue amend the existing handoff rather than starting a new one.
Confirm Slack receives one notification carrying both items and the agreed time.

## Out of scope

- Notifying a human in real time at `open` — deliberately deferred; the value is in
  the resolution, and two messages per escalation is noise.
- Multiple escalations per session. One customer, one callback, by design.
- Editing the original Slack message in place rather than posting an update. That is an
  n8n workflow concern, not an agent one.
- Escalation surviving across sessions. A new call is a new customer.
