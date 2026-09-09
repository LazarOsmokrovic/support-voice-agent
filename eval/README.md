# Eval suite

20 scripted scenarios across all six capabilities, recorded once against the
real API and replayed offline forever after.

## Running it

    python -m eval.run_eval                    # everything, offline, no API key
    python -m eval.run_eval --scenario NAME    # one
    python -m eval.run_eval --json             # the whole report as one object
    python -m eval.run_eval --strict           # a release gate: stale fixtures fail too

Exit codes: `0` all pass · `1` a behavioural regression · `2` the fixtures
need refreshing (STALE / DRIFT / MISSING / ERROR) with no behavioural
failure. Three codes rather than two because the fixes differ, and
conflating them trains people to ignore the signal. `--strict` collapses 2
into 1.

## The outcomes

| Outcome | Means | What to do |
|---|---|---|
| PASS | Behaviour matched the expectations. | Nothing. |
| FAIL | Behaviour did not match. | Read the failure lines; this is a regression. |
| STALE | A prompt, tool schema or the seed changed since recording. | Re-record the named scenario. Nothing is scored, because scoring a stale recording scores a fiction. |
| DRIFT | Replay re-ran the tools and got different output than recording saw. | Usually Chroma: an ONNX/chromadb bump changes embeddings, hence top-k, hence policy text. Occasionally a real tool regression. |
| MISSING | No recording on disk. | Record it. Never a silent skip. |
| ERROR | The scenario raised. | Read the traceback; the run continued without it. |

The runner **never** re-records anything itself, in any of these cases —
not even STALE, which names the exact recording that needs it. Re-recording
is always an explicit, named action, because doing it automatically would
spend money nobody asked to spend and destroy the very signal STALE exists
to give you: you'd never see "this changed," only ever a fresh, silent pass.

**Why STALE is not a code failure.** A STALE outcome means the world moved —
a prompt was edited, a tool schema changed, the seed data was refreshed —
not that the agent misbehaved. Scoring turns recorded against an old prompt
tells you nothing about the current one, so `run_eval` refuses to score them
at all and reports STALE instead. That is why it costs exit code 2 rather
than 1 by default: it is a "go do something" signal, not a "something is
broken" signal, and `--strict` exists for the one context (a release gate)
where you want staleness to block anyway.

## Adding a scenario

1. Add a `Scenario` to `SCENARIOS` in `eval/scenarios.py`. **Resolve every
   identifier — customer IDs, order IDs — from `data/mock_db.py` at runtime,
   via the `_customer_id(name_fragment)` / `_order_id(item_fragment)`
   helpers already at the top of that file. Never type a seeded value by
   hand**, not even one you copied from a real query output.

   This is not a style preference. This project has shipped two Critical
   bugs from exactly this defect class, in two different phases:

   - Phase 11's outbound-webhook redactor destroyed its own 17-digit order
     IDs, because every test that exercised it used an invented value
     (`4111 1111 1111 1111`) instead of a real seeded one, so nothing ever
     ran the redactor against the actual shape it needed to preserve.
   - Phase 10a's storage-boundary redactor did the same thing a second time
     — destroying real ISO dates and `TBA…US` tracking numbers — for the
     identical reason: hand-typed fixture values that merely *looked*
     plausible, rather than values `data/mock_db.py` actually produces.

   `score_redactor_preserves_identifiers()` is the standing guard against a
   third occurrence. It reads every seeded order ID, tracking number, order
   date and appointment time at runtime and asserts `redact_text(value) ==
   value` — 30 values today, and more automatically if the seed grows. It runs
   once per evaluation and reports through the `redactor:` line at the foot of
   the report; a failure exits **1**, not 2, because a redactor eating the
   store's own data is a behavioural regression rather than a stale fixture.

   It is deliberately a property of the redactor rather than of the stored
   records. An earlier version tried to infer destruction from records — if a
   record named one identifier, omitted another, and contained any
   `[redacted-` marker, it called the absent one destroyed. That both
   false-positived (an unrelated masked email made an unmentioned tracking
   number look eaten) and, more seriously, never covered dates at all, so
   deleting the ISO-date exemption from `guardrails/pii.py` left the suite
   green. Testing the redactor directly cannot false-positive, needs no
   anchoring, and fires even when no scenario happens to store the value.

   The stored-record check still exists alongside it, as a second net: within
   a record, an order's `item` text anchors the check, because `item` is free
   text none of the redaction patterns can match. If a record mentions the
   item and carries a redaction marker but has lost the order ID or tracking
   number, that is flagged. Absence alone never is — a record that simply does
   not concern an order is not evidence that anything was destroyed.

   A hand-typed order ID or date in a scenario reintroduces exactly that
   failure mode here: it can look right, pass review, and quietly stop
   testing the thing it claims to test the moment the seed changes shape
   underneath it. Resolving through `_customer_id`/`_order_id` at runtime is
   what keeps a scenario honest — and it is not a one-time precaution: the
   seed dates were already refreshed once, on 2026-09-08, specifically
   because scenarios anchored to a real calendar date go stale on their own
   schedule, and they will be refreshed again.

2. Write `expect` **before** recording. It is the test; writing it
   afterwards turns the eval into a description of whatever happened to
   come back, not a check of what should have.
3. Record it: `python -m eval.record --scenario your_scenario_name`.
4. Read the printed worksheet, assign each turn a grounding label, and paste
   the `grounding_truth=(...)` block back into the scenario.
5. `python -m eval.run_eval --scenario your_scenario_name` should now pass.

`grounding_truth` is authored **after** recording on purpose, and `expect`
**before**. Mixing the two orders is how an eval quietly stops testing
anything: an `expect` written after the fact just restates whatever the
recording contains, and a `grounding_truth` assigned before it can be
compared against the real reply is a guess wearing the clothes of a
measurement.

## Re-recording

`python -m eval.record --scenario NAME` (repeatable) or `--all`. This is the
**only** entry point that needs `ANTHROPIC_API_KEY`, and the only one that
costs money — every other command in this package, including the full
`run_eval` suite, runs entirely offline against fixtures already on disk.

Recordings go STALE **by design** the moment a system prompt, a tool schema,
or the seed changes — the runner detects this via hashes stored alongside
each recording (see the `STALE` row above) and reports exactly which field
moved and the exact re-record command to fix it. It never re-records on its
own initiative. If it did, a prompt edit would silently produce a fresh
"passing" recording instead of the loud STALE signal that tells you your
fixtures no longer describe the current system — and it would spend real
money doing it, unasked, every time you ran the suite.

## What the false-positive number does and does not mean

This is the most important section of this document, because it is the
number most likely to be misread.

The report prints the grounding detector's false-positive rate as `n/N`
with raw counts, **never** a bare percentage, and reports `0/0 —
insufficient data` rather than `0%` when nothing was labellable. Read a
percentage-free `n/N` here as a deliberate refusal to imply more precision
than the sample supports.

**Why the denominator is small.** Only a turn carrying both a
`search_policy` tool call and a numeric claim in the reply is labellable at
all (`guardrails/validators.py`'s own trigger condition —
`GROUNDING_TRIGGER_TOOLS = ("search_policy",)` — is the same gate the
false-positive count inherits). Across 20 scenarios that is realistically a
single-digit to low-teens count of turns, not hundreds. A 95% confidence
interval on a rate computed from a single-digit denominator is roughly ±25
percentage points — wide enough that a measured "1/6 false positives" and a
true rate anywhere from near-zero to over half are both consistent with the
same observation.

**What it can do:** prove, end to end, that `check_reply_grounding` fires on
a genuine fabrication and stays quiet on ordinary correct replies — a
working-machine check, not a rate estimate — and give a reproducible
baseline that a later prompt or regex change can be measured against. The
number that matters is the **delta** between two runs of this suite, not the
absolute level of either one.

**What it categorically cannot do: settle `UNGROUNDED_REPLY_ESCALATION_THRESHOLD`
on statistical grounds.** The denominator is too small to estimate a real
false-positive rate, every scenario here is authored by the same person who
wrote the detector (so it cannot surface a failure mode nobody scripted),
and it says nothing about real customer traffic. Anyone tempted to point at
this number to argue the threshold should move from 2 to 3 (or back) is
reading more into a single-digit sample than a single-digit sample can
support — use it to confirm the detector still behaves the same way after a
change, not to pick a number.

## Known limitations

These are real, were found during review rather than papered over, and are
documented here rather than left for someone to rediscover.

- **The unreachable-claims blind spot.** `agent/tools/refunds.py`'s
  `issue_refund` calls `search_policy` internally to compute
  `policy_reference`, but that internal call never lands in the turn's
  `tool_calls` — only tool calls the model itself issues do. So a reply like
  "you're eligible for a $349.99 refund, and you're within the 30-day
  window" carries no `search_policy` in `tool_calls` for that turn, and the
  grounding detector's trigger condition never fires: **that turn is never
  grounding-checked at all**, not passed, not flagged — simply invisible to
  the detector. Phase 10c measures this gap as a first-class metric
  (`unreachable_claims` in the report) rather than fixing it; fixing it now
  would move the baseline in the same run where the baseline is first being
  established, which defeats the point of establishing one.
- **`flagged`, `hedged`, and `unreachable_claims` are counted over ALL
  turns, including `not_applicable` ones** — unlike the false-positive and
  false-negative counts, which count only turns that carry a `grounding_truth`
  label. Read `flagged` as "flagged, out of every turn in the suite," never
  as "flagged, among the turns that were actually labelled" — those are
  different denominators and the report does not collapse them into one.
- **The turn-log invariant is enforced in both places, and reacts differently
  in each.** `len(log_lines) == len(records)` is checked in `eval/run_eval.py`'s
  `evaluate()` and again in `eval/record.py` before a recording is saved.
  `log_turn()` never raises, so without this a silently dropped write is
  indistinguishable from a disabled log. The runner reports a mismatch as that
  one scenario's `ERROR` and carries on, because losing the other 19 results to
  one bad write helps nobody. The recorder *raises* instead: it is the path that
  writes the fixture, so an incomplete recording would be inherited by every
  later replay with nothing left to reveal the loss. Saving a knowingly
  incomplete recording is worse than recording nothing.

## `guardrail_ungrounded_ladder_escalates` may not be scriptable

This scenario needs the model to fabricate a policy number, twice in a row,
in response to scripted prompting — and no prompt can *guarantee* a model
hallucinates on demand. It may pass cleanly during recording, or the model
may (correctly, and to its credit) decline to invent numbers even when
pushed toward it.

If it fails on a recording pass: **reword the scenario's turns and
re-record.** Do **not** relabel a genuinely grounded reply as `ungrounded`
to make the scenario pass. `grounding_truth` is the only ground truth this
suite has for the false-negative count — hand-editing a label to match a
desired outcome rather than the actual reply corrupts that ground truth
silently, and every measurement downstream of it (false positives, false
negatives, the delta a future change is checked against) becomes unusable
without anyone knowing it happened.
