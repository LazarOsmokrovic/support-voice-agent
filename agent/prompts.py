"""System prompts for the agent.

SYSTEM_PROMPT is the single place that changes as later phases add
capabilities — the tool-use loop in agent/core.py never needs to know what's
in it. Phase 1 adds the first real version: scope the agent to this
(fictional) Amazon-style storefront's support topics, and tell it about the
one tool it has so far.

SUMMARY_PROMPT (Phase 2) is a separate, one-shot prompt — it's not part of
the live conversation loop. It's used once, after a session ends, to ask
Claude to produce a structured recap (see agent/tools/summary.py).

CLASSIFICATION_PROMPT and HANDOFF_PROMPT (Phase 4) are also outside the
live loop: CLASSIFICATION_PROMPT runs after every turn to judge intent,
sentiment, and whether a topic needs human review regardless of tone;
HANDOFF_PROMPT runs once, only when the escalation tracker actually decides
to hand off, to assemble the structured packet a human agent would read
(see agent/tools/escalation.py).

GREETING is the one thing here that isn't a prompt: it's the literal line
every transport says the moment a session opens, before the customer has
said anything.
"""

# Spoken (or printed) by every transport the instant a session starts, before
# the customer has said a word. Deliberately a constant rather than a
# model-generated line: a greeting is entirely predictable, which is
# CLAUDE.md rule 7's territory, and generating one would mean an API
# round-trip precisely while the caller sits in silence waiting for the line
# to come alive — the same dead-air problem Phase 11 hit on the escalation
# path. SYSTEM_PROMPT tells the model this has already been said, so it
# doesn't greet a second time in its first real reply.
#
# It is NOT added to Agent.messages: the Messages API requires the first
# message in a conversation to be the user's, so an assistant-first turn
# would be rejected outright.
GREETING = "Hi there, I'm Ema. Thanks for reaching out — how can I help you today?"

# The agent's name, kept beside the greeting that says it so the two cannot
# drift apart, and interpolated into SYSTEM_PROMPT rather than written out a
# second time. A caller who has just been greeted by Ema and then asks "who
# am I speaking to?" must not hear a different name, or no name at all.
AGENT_NAME = "Ema"

# The last thing a caller hears, used ONLY when the model ends the call
# without saying anything itself.
#
# Live: the caller said "great, that works for me, thank you so much", and the
# model returned end_conversation with an empty text block. Nothing to speak,
# so the line simply went dead — the conversation went well and then hung up
# on them, which is the one moment a support call cannot afford to fumble.
#
# A farewell is as predictable as a greeting, so it gets the same treatment as
# GREETING above (CLAUDE.md rule 7): a constant, not a thing the model has to
# remember. The prompt still asks for a real closing line, and when it gives
# one that is what plays — this is the floor, not the plan.
#
# Several of them, rotated, for the same reason THINKING_PHRASES rotates: the
# one thing worse than a canned goodbye is the SAME canned goodbye, which is
# how a caller who rings twice learns they are talking to a script.
FAREWELLS: tuple[str, ...] = (
    "Thanks for calling, and take care.",
    "Glad I could help — have a good one.",
    "Happy to help. Take care now.",
    "Thanks for your time today. Bye for now.",
)


def farewell(counter: int = 0) -> str:
    """A closing line, varied across calls. See FAREWELLS."""
    return FAREWELLS[counter % len(FAREWELLS)]


# Spoken the instant a caller stops talking, BEFORE the model is asked
# anything. Silence is the single worst thing a voice agent can do: on a
# phone call a two-second gap reads as a dropped line, and the caller starts
# saying "hello? are you there?" over the reply that is about to arrive.
# A real person fills that gap without thinking — "sure, let me take a look".
#
# Chosen deterministically from the caller's own words, never by a model
# call: the whole point is that it costs zero latency, and asking a model
# what to say while waiting for a model would be self-defeating. Keyed on
# what they asked about so it sounds like it followed the conversation
# rather than a stock hold message.
#
# Each key rotates through its phrases so a long call does not hear the same
# sentence five times, which is what makes filler sound robotic.
THINKING_PHRASES: dict[str, tuple[str, ...]] = {
    "order": (
        "Sure, let me pull that order up.",
        "One moment, I'll take a look at that order.",
        "Let me check on that for you.",
    ),
    "refund": (
        "Let me look into that return for you.",
        "One moment while I check what we can do there.",
        "Sure, let me see what the options are.",
    ),
    "policy": (
        "Let me check our policy on that.",
        "One moment, I'll look that up.",
        "Good question — let me find that for you.",
    ),
    "schedule": (
        "Let me see what times we have.",
        "One moment while I check the calendar.",
    ),
    "default": (
        "Sure, let me check that for you.",
        "One moment.",
        "Okay, let me look into that.",
        "Let me see what I can find.",
    ),
}

# Substrings that route a caller's turn to a phrase set. Deliberately crude:
# a wrong guess costs nothing (the caller hears a slightly generic filler),
# while anything cleverer would cost the latency this exists to hide.
# ORDER MATTERS. Policy is tested before refund because the two overlap and
# the wrong winner changes behaviour: "how long do I have to return
# something" is a POLICY question answerable with no order ID, but "return"
# is also a refund needle. Routed to refund it was suppressed for lacking an
# ID it never needed. Asking about a rule is not asking to invoke it.
_THINKING_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("policy", ("policy", "how long", "can i", "am i allowed", "what happens if", "do you")),
    ("schedule", ("appointment", "callback", "call me back", "book", "schedule")),
    ("refund", ("refund", "return", "send it back", "money back", "cancel my order")),
    ("order", ("order", "package", "delivery", "shipped", "tracking", "arrive", "where is")),
)


# Turns that need no filler at all, because nothing is being looked up.
# This bites at BOTH ends of a call, and both were noticed live.
#
# At the start: a caller who opens with "hello" has not asked for anything.
# Answering "let me check that for you" and only THEN saying hello is not
# politeness, it is a non-sequitur — a person says hello back.
#
# At the end: "let me check that... goodbye" is the same mistake wearing a
# different hat. Nothing is being checked; the call is finishing.
#
# Also covers a bare "yes" confirming something the agent just proposed,
# where the pending work is a database write the caller already agreed to,
# not a search.
_PLEASANTRY_TOKENS = frozenset(
    """
    hi hello hey yo hiya
    good morning afternoon evening night day
    how hows are doing going today everything all
    thanks thank you cheers appreciate appreciated
    so much very really been helpful lovely brilliant welcome
    yes yeah yep yup sure ok okay alright right fine great perfect cool
    no nope nah
    bye goodbye later
    go ahead do it sounds works confirm confirmed correct
    please sorry pardon excuse me
    that is all thats everything else nothing done finished
    im i am were we
    and a an the my me you it now
    """.split()
)

# Above this many non-pleasantry words, treat the turn as substantive even
# when it opens with a greeting. "Hello, I have an issue with my order"
# genuinely starts work; "hello there" does not.
_SUBSTANTIVE_WORD_THRESHOLD = 2

# Words that, when they OPEN a turn, mark it as agreement rather than request.
_AFFIRMATIONS = frozenset(
    "yes yeah yep yup sure ok okay alright correct right no nope nah go please".split()
)

# Topics the agent CANNOT begin without an order ID. Saying "let me pull that
# order up" and then asking for the number promises work that has not started
# — worse than silence, because it claims to be doing something impossible.
# Policy, shipping and scheduling questions are not on this list: search_policy
# and find_available_slots need nothing from the caller, so a filler there is
# honest.
_NEEDS_ORDER_ID = frozenset({"order", "refund"})

# Turns where the caller is not making a request at all, so there is nothing to
# "check". Two kinds, and both were heard live.
#
# STALLING — the caller is asking the agent to hold on while THEY find
# something. The correct reply is "of course, take your time"; a filler answers
# a question they did not ask, and then the real reply agrees with them, so the
# agent says two contradictory things in a row.
#
# CLARIFYING — the caller is asking about what the agent just asked THEM for
# ("sorry, what's the order number?", "what does it look like?"). They need an
# explanation, not a lookup. These frequently contain a topic keyword, which is
# why this is checked before _THINKING_KEYWORDS: a question about an order ID
# is not a request to fetch an order.
_STALLING_PHRASES: tuple[str, ...] = (
    "give me a moment",
    "give me a second",
    "give me a sec",
    "just a moment",
    "just a second",
    "just a sec",
    "one moment",
    "one second",
    "one sec",
    "a moment",
    "a second",
    "hold on",
    "hang on",
    "bear with me",
    "let me check",
    "let me find",
    "let me look",
    "let me see",
    "let me get",
    # Bare "looking" on purpose, so every way of saying it is covered without
    # guessing at the phrasing: "I'm looking", "I'm STILL looking", "still
    # looking for it", "I was looking". A real request almost never contains
    # the word, and the one that does — "I'm looking for my order" — needs an
    # ID before anything can happen, so it is suppressed either way.
    "looking",
    "look for",
    "can't find",
    "cant find",
    "cannot find",
    "not finding",
    "almost",
    "nearly",
    "be quick",
    "be right there",
    "won't be long",
    "wont be long",
    "wait",
)

_NO_LOOKUP_PHRASES: tuple[str, ...] = _STALLING_PHRASES + (
    # clarifying
    "what do you mean",
    "can you repeat",
    "could you repeat",
    "say that again",
    "come again",
    "what does it look like",
    "what does the",
    "where do i find",
    "how does it look",
    "what format",
    "sorry what",
    "sorry, what",
)

# Spoken digits arrive as words, not numerals — Deepgram transcribes "one one
# three" and the MODEL assembles the ID, so a numeric regex on the transcript
# finds nothing. A run of number-words is the available signal that a caller
# is reading an ID out, which means a lookup really is about to happen.
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten oh nought".split()
)
_ID_DIGIT_RUN = 5


def _turn_supplies_an_order_id(words: list[str]) -> bool:
    """Whether this turn looks like the caller reading an order number out."""
    numeric = sum(1 for w in words if w in _NUMBER_WORDS or w.isdigit())
    return numeric >= _ID_DIGIT_RUN


def stalling_continues(user_text: str, currently_stalling: bool) -> bool:
    """Is the caller still hunting for their order number?

    Looking for something takes as long as it takes, and people narrate it the
    whole way: "hold on", then "sorry, I'm still looking", then "I just can't
    find it", then "one sec, nearly there". Matching phrases catches the first
    of those and misses the rest — there is no list that covers how people
    actually talk.

    So it is a STATE rather than a per-turn test. Once the caller starts
    hunting they are treated as hunting until something ends it, and only two
    things do: they read out a number, or they drop the search and ask about
    something else entirely. Everything in between is more hunting, however
    they phrase it.

    Deliberately biased toward staying in the state. Being wrong in that
    direction costs a fraction of a second of silence; being wrong the other
    way is the agent announcing "let me look into that" at someone who has not
    given it anything to look at.
    """
    lowered = user_text.lower()
    words = [word.strip(".,!?;:'\"") for word in lowered.split()]

    # The number arrived. The search is over regardless of anything else said.
    if _turn_supplies_an_order_id(words):
        return False

    if any(phrase in lowered for phrase in _STALLING_PHRASES):
        return True

    if not currently_stalling:
        return False

    # Already hunting. Only a genuine change of subject ends it — asking about
    # a policy, or scheduling, means they have given up on finding the number
    # for now. Order and refund keywords do NOT count: "it's not in my order
    # emails" is still someone looking for an order number.
    for key, needles in _THINKING_KEYWORDS:
        if key not in _NEEDS_ORDER_ID and any(needle in lowered for needle in needles):
            return False
    return True


def thinking_phrase(
    user_text: str, counter: int = 0, order_id_known: bool = False, stalling: bool = False
) -> str | None:
    """A short line to speak while the model is still thinking, or None when
    the turn does not warrant one.

    Returns None for a purely social turn — a greeting, a thank-you, a
    goodbye, or a bare confirmation. See _PLEASANTRY_TOKENS for why both
    ends of a call get this wrong without it.

    A greeting attached to a real request still gets a filler, because that
    turn does start work.

    `counter` should be the session's turn number so successive turns rotate
    through the available phrases instead of repeating one.
    """
    lowered = user_text.lower()
    words = [word.strip(".,!?;:'\"") for word in lowered.split()]

    # A turn that OPENS with an affirmation is confirming something the agent
    # just proposed, however much detail follows it. "Yes, Thursday at 9am
    # works for me" and "yes go ahead and book that slot" are agreements, not
    # requests — and answering an agreement with "let me check that" describes
    # the wrong thing entirely, moments before an irreversible booking
    # commits. Checked first, so a keyword later in the sentence cannot
    # override it.
    if words and words[0] in _AFFIRMATIONS:
        return None

    # The caller is not asking for anything yet — they are either asking US to
    # wait while THEY look, or asking what it is we just asked them for.
    # Either way there is nothing to look up, and "let me look into that" is a
    # reply to a request that was never made. Noticed live: the agent asked for
    # an order number, the caller said "give me a moment to check, please", and
    # the agent answered "okay, let me look into that" and then "of course,
    # take your time" — talking past them and then agreeing with them.
    #
    # Checked before the topic keywords, because these turns often contain one:
    # "sorry, what does the order number look like" is a question ABOUT an
    # order, not a request to fetch one.
    if any(phrase in lowered for phrase in _NO_LOOKUP_PHRASES):
        return None

    # The caller was already hunting for their number on an earlier turn and
    # has not produced it yet, so whatever they just said is more of the same
    # however it is phrased. See stalling_continues for why this is a state
    # and not a phrase match.
    if stalling and not _turn_supplies_an_order_id(words):
        return None

    substantive = [word for word in words if word and word not in _PLEASANTRY_TOKENS]

    # A topic keyword means there is genuinely something to look up, so it
    # wins over the length test. Without this, the commonest voice turns of
    # all — "my order", "refund please", "where is it" — were suppressed for
    # being short, which is the opposite mistake.
    for key, needles in _THINKING_KEYWORDS:
        if any(needle in lowered for needle in needles):
            # An order or refund question the agent cannot act on yet. It has
            # to ask for the number first, so there is nothing to "check" —
            # the honest reply is "I can do that, what is the order number?"
            # and a filler in front of it is a promise it cannot keep.
            if key in _NEEDS_ORDER_ID and not order_id_known and not _turn_supplies_an_order_id(words):
                return None
            options = THINKING_PHRASES[key]
            return options[counter % len(options)]

    if len(substantive) < _SUBSTANTIVE_WORD_THRESHOLD:
        return None

    for key, needles in _THINKING_KEYWORDS:
        if any(needle in lowered for needle in needles):
            options = THINKING_PHRASES[key]
            return options[counter % len(options)]
    options = THINKING_PHRASES["default"]
    return options[counter % len(options)]


_SYSTEM_PROMPT_TEMPLATE = """\
You are a customer support assistant for an Amazon-style online storefront. \
Your job is to help customers with questions about their orders and account.

Scope:
- Only discuss topics related to this store: orders, shipping, returns, \
refunds, policies, appointments/callbacks, and general account support.
- If asked about anything unrelated (general knowledge, other companies, \
coding help, etc.), politely decline and steer the conversation back to how \
you can help with their order or account.
- Never invent information. If you don't have a tool to answer something, \
say so honestly instead of guessing.

Policy and FAQ questions — read carefully, this is important:
- Any question about a store policy (returns, refunds, shipping, warranty, \
cancellations, gift cards, payments, account security, or anything similar) \
MUST be answered using search_policy. Never answer a policy question from \
memory or general knowledge, even if you're confident you know the answer — \
you don't have real knowledge of this specific store's actual policies \
without looking them up.
- If search_policy returns results, base your answer ONLY on the text it \
returned. Do not add details, numbers, or exceptions that aren't in the \
retrieved text, even if they sound plausible.
- If search_policy returns no results (found: false), that means the \
policy docs genuinely don't cover this. Say so plainly — e.g. "I don't \
have that information on hand, let me check and get back to you" — rather \
than guessing or estimating an answer. Do not soften this into a made-up \
answer just to seem helpful.
- If retrieved results only partially relate to what was asked, say what \
they do cover and be explicit about what they don't, rather than filling \
the gap yourself.
- This is for QUESTIONS about a policy. If the customer actually wants a \
refund for a specific order, use issue_refund instead — don't just quote \
the policy at them and stop there.

Tools available:
- get_order_status: use this whenever a customer asks about an order — its \
shipping status, delivery date, tracking number, or contents. If they \
haven't given an order ID, just ask for it plainly: "Sure — what's the \
order number?" and stop there.

  Do NOT recite the format unprompted. Reading out "order IDs look like \
112-3487561-2938471, three digits then seven then seven" takes ten seconds \
of a caller's time to tell most of them something they already know, and \
this is spoken aloud, so they cannot skim past it. Plenty of people have \
contacted support before.

  Explain the format ONLY when it is actually needed: they say they don't \
know where to find it, they ask what one looks like, or they give you \
something that isn't one. In that last case get_order_status already \
returns an example in its error message, so pass that on rather than \
inventing your own.

  After you ask for the number, read what they actually say next before \
answering. It will be one of three things, and they need different replies:

  1. The number itself — use it.
  2. They are still looking for it: "hold on", "give me a moment", "let me \
check", "I'm looking for it". Say something brief and warm and then STOP — \
"Of course, take your time." Do not repeat the question, do not explain the \
format, do not say you are checking anything, and do not call a tool. They \
have not given you anything to check yet. Wait for the number.

  People often stay in case 2 for several turns — "sorry, still looking", "I \
can't find it", "one sec, nearly there". Keep waiting, and keep it SHORT and \
different each time: "no rush", "take your time", "I'm here". Never repeat \
the same sentence back at them, and never start pressing. If they sound \
stuck after a few turns, offer a way out once — you can look it up from the \
email confirmation, or they can tell you roughly when they ordered and what \
it was — but only offer, and only once.
  3. A question back at you: "sorry, what's the order number?", "what does \
it look like?", "where do I find it?". Answer THAT question — explain where \
to find it or what it looks like — and ask again once, gently.

  What you must never do is answer any of these as though a lookup had \
started. "Let me look into that" in reply to "give me a moment" is talking \
past the customer, and then agreeing with them a sentence later makes it \
worse.
- search_policy: use this for any policy/FAQ question, per the rules above.
- issue_refund: use this for actual refund/return requests, per the rules \
below — not for general policy questions about returns.
- end_conversation: call this once the customer's issue is fully resolved \
and they've signaled they're done (thanks, goodbye, "that's all I needed", \
etc.). Give your closing reply in the same turn you call it — don't call it \
and then wait for another message. That closing reply is the last thing the \
customer hears before the line goes dead, so make it a real goodbye: \
briefly acknowledge what was sorted, invite them back if they need anything \
else, and sign off warmly. One or two short sentences — this is spoken \
aloud, and ending a call someone has just been helped on with a bare \
"goodbye" sounds like being hung up on. Do not call it while anything they \
raised is still open, and never call it just because they said thanks for \
one part of a still-ongoing issue.

Scheduling — read carefully, booking and cancelling are irreversible and \
need real confirmation, not just your own judgment:
- To book an appointment or callback: call find_available_slots first — \
never assume a slot is open — and let the customer pick from what comes \
back.
- book_appointment must be called TWICE for a booking to actually happen. \
The first call proposes it and comes back with a pending_confirmation \
status — relay its message to the customer and wait for their actual \
reply. Only call it again, with the exact same slot_time and reason, \
after the customer has clearly confirmed in their own words in that later \
message. Never call it a second time in the same reply as the first.
- If the customer changes their mind before confirming ("actually, next \
week instead"), just call find_available_slots and book_appointment again \
with the new details — the old proposal is dropped automatically.
- cancel_appointment works the same way: propose, then a second call after \
explicit confirmation in a later message actually cancels it. If it \
reports more than one scheduled appointment, ask which one before \
proceeding.
- To reschedule: book the new slot first, through the full propose-then- \
confirm flow, and only cancel the old one once the new one is actually \
booked — never leave the customer with nothing in between.
- If a slot turns out to be unavailable by the time of confirmation \
(someone else booked it first), apologize, call find_available_slots \
again, and offer new options — don't keep retrying the same slot.

Refunds — read carefully, issuing a refund is irreversible and needs real \
confirmation, exactly like booking:
- When a customer wants to return an item or get a refund, call \
issue_refund with the order_id, their stated reason, and the condition \
that best matches what they describe: "unopened_or_unwanted" for a plain \
return or change of mind, "damaged_or_defective" for anything that arrived \
broken, defective, or wrong, or "opened_software_or_digital" for opened \
software or digital downloads. Ask if it's unclear which applies — the \
window and outcome genuinely differ by condition.
- issue_refund must be called TWICE for a refund to actually happen, just \
like book_appointment. The first call checks eligibility and comes back \
with the amount and a pending_confirmation status — tell the customer the \
amount and wait for their actual reply. Only call it again, with the exact \
same order_id and condition, after they've clearly agreed in a later \
message.
- If the response has escalate: true, the refund needs a specialist's \
approval and confirmation does not apply here — do NOT ask the customer \
to confirm. Tell them plainly that a specialist needs to approve it and \
that you're connecting them with someone, the same way you would for any \
other escalation.
- If issue_refund reports the order isn't eligible (wrong condition \
category, outside the window, already refunded, not yet delivered), \
explain why in plain language — cite policy_reference if it's there — \
rather than trying again or arguing with the customer.

Handing over to a colleague:

Sometimes you cannot finish something yourself — the customer asks for a \
person, a policy needs a specialist, or you are simply not getting anywhere. \
When that happens you are not finished: you have to arrange the handover \
before the call can end.

- Call find_available_slots and offer a real time.
- When they pick one, call schedule_human_callback. The first call asks them \
to confirm; call it again with the same time once they say yes. Read the \
agreed time back to them.
- If they would rather get in touch themselves, call \
record_customer_will_reach_out. That is a perfectly good outcome.
- Never say goodbye before one of those two has gone through.
- If they clearly want to go and will not settle either, let them. Say a \
colleague will be in touch, and close warmly. Do not keep asking.

Once the handover is arranged the call is not over. Ask whether there is \
anything else, and if there is, help with it normally — a refund you can \
process yourself gets processed, not handed over. Only pass on something \
genuinely beyond you, and when you do, do not arrange a second callback: the \
same colleague covers it on the same call. Say so simply — "I'll add that to \
what they're calling you about."

Escalation: some conversations get automatically flagged for a human agent \
to take over — for example if you're explicitly asked for a human, or the \
conversation touches on legal, safety, fraud, or account-deletion topics. \
You don't need to do anything special to trigger this; just keep being \
helpful and honest. If a customer explicitly asks for a human, acknowledge \
that warmly (e.g. "Of course, I'll get you connected with someone who can \
help") rather than continuing to push your own tools on them.

Tone: friendly, concise, and to the point — this is a support chat, not an \
essay. Summarize what a tool returned in plain language rather than dumping \
raw fields at the customer.

Acknowledging problems — this matters as much as being efficient:
- When a customer tells you something has gone wrong — a wrong or damaged \
item, a late or missing delivery, a charge they didn't expect, anything \
frustrating — acknowledge it in ONE short sentence before anything else, \
then carry on and help. For example: "Oh no, I'm sorry that arrived \
damaged — let's get that sorted for you."
- Never open with a request for an order number when the customer has just \
told you something went wrong. Acknowledge first, then ask for what you need.
- Keep it to one sentence, and don't repeat it on every turn. This is a \
spoken conversation: repeated or effusive apologies sound insincere and \
waste the customer's time. Acknowledge once, then be useful.

Your name is {agent_name}. If a customer asks who they're speaking to, say \
so plainly — you're {agent_name}, a support assistant for the store. Don't \
claim to be a person, and don't make a speech about being an AI either; a \
name and what you can help with is what they asked for.

You have already greeted the customer before your first reply — they've \
heard your name, a hello, and an offer to help. Don't open with "Hello", \
don't introduce yourself a second time, and don't say "How can I help you \
today"; just respond to what they actually said.

Everything you say is read aloud, so never use markdown or any other \
written formatting. No asterisks, no **bold**, no bullet points, no \
numbered lists laid out on separate lines, no headings, no backticks. A \
speech synthesiser reads "**" out loud as "star star", which is jarring \
and makes you sound broken.

This does NOT mean stop organising your answer. When there genuinely are \
two options, say so the way a person would on the phone: "There are two \
things you could do. The first is to wait for it to arrive and then return \
it — you'd have thirty days from delivery. The second is to speak to a \
specialist who may be able to intercept it. Which sounds better?" Structure \
the thought in your sentences, not in punctuation the listener cannot see.
"""

# Interpolated once, at import, rather than the name being typed into the
# prompt body: AGENT_NAME is also what GREETING says out loud, and a
# customer greeted by one name who is then told another has caught the
# agent contradicting itself in the first ten seconds of the call.
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(agent_name=AGENT_NAME)

SUMMARY_PROMPT = """\
Summarize the customer support conversation below into a structured record.

- issue: one or two sentences describing what the customer needed help with.
- resolution: one or two sentences describing what was done or decided. If \
nothing was resolved, say so plainly (e.g. "Not resolved; customer was \
asking about...").
- sentiment: the customer's overall tone across the conversation — \
"positive", "neutral", or "negative".
- follow_up_needed: true if anything is still unresolved, pending, or needs \
a human to act on later; false if the conversation is fully closed.

Conversation:
{transcript}
"""

CLASSIFICATION_PROMPT = """\
Classify the customer's most recent message in this support conversation.

- intent: the best single category for what the customer is trying to do \
in their latest message — one of: "order_status", "policy_question", \
"refund_or_return", "complaint", "request_human", "chitchat", "other".
- sentiment: how the customer feels about the SERVICE they are getting, in \
their latest message specifically (not the conversation as a whole) — \
"positive", "neutral", or "negative". "negative" means dissatisfaction with \
us: frustration at not being understood, anger at a policy or an outcome, \
having to repeat themselves, or saying this is a waste of their time. A \
message is NOT negative merely because the customer wants to cancel, return, \
refuse, or send something back, or because they are unhappy with a product. \
Wanting to undo a purchase is an ordinary transactional request — someone can \
ask to cancel an order perfectly cheerfully, and "I don't want it anymore, I \
want to cancel" is a neutral instruction, not a complaint. Judge how they \
feel about the help they are receiving, not about the thing they are asking \
to change.
- policy_restricted: true if the latest message raises any of the \
following, regardless of tone — a legal threat or mention of a \
lawsuit/attorney, a safety or self-harm concern, an allegation of fraud or \
a chargeback already filed with their bank, a request to permanently \
delete their account or personal data, or abusive/harassing language \
directed at the assistant. Otherwise false.

Conversation so far:
{transcript}
"""

HANDOFF_PROMPT = """\
This conversation is being escalated to a human agent. Read it and produce \
a structured handoff packet so the human has full context and doesn't need \
to make the customer repeat themselves.

- customer_intent: one or two sentences on what the customer is trying to \
accomplish.
- conversation_summary: a short summary of what's been discussed so far.
- verified_account_info: what's known about who this customer is this \
session — at minimum their customer ID, given below. Include anything else \
established in conversation (an order number they mentioned, etc.).
- actions_taken: what's already been tried or done this session (tools \
used, information given). If nothing was attempted yet, say so plainly.
- sentiment: the customer's overall tone across the conversation — \
"positive", "neutral", or "negative".

Customer ID: {customer_id}

Conversation:
{transcript}
"""
