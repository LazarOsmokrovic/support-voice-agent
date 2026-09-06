"""Phase 10a: guardrails/validators.py — post-LLM grounding detection.

A detector, not a prover: these tests pin the common hallucination shape
(an invented window or fee after a policy lookup), not entailment.
"""

from __future__ import annotations

from guardrails.validators import HEDGE_PHRASES, check_reply_grounding, hedge_for


def _policy_call(found: bool, text: str = ""):
    output = {"found": True, "results": [{"text": text}]} if found else {"found": False}
    return {"name": "search_policy", "input": {"query": "returns"}, "output": output}


def test_no_findings_when_no_grounding_tool_was_called():
    calls = [{"name": "end_conversation", "input": {}, "output": "done"}]
    assert check_reply_grounding("You have 30 days to return it.", calls) == []


def test_flags_a_number_absent_from_the_retrieved_policy():
    calls = [_policy_call(True, "Most items can be returned within 30 days of delivery.")]
    findings = check_reply_grounding("You have 90 days to return that.", calls)
    assert findings
    assert "90" in findings[0]


def test_clean_when_the_number_appears_in_the_retrieved_policy():
    calls = [_policy_call(True, "Most items can be returned within 30 days of delivery.")]
    assert check_reply_grounding("You have 30 days to return it.", calls) == []


def test_flags_a_claim_after_retrieval_returned_nothing():
    calls = [_policy_call(False)]
    findings = check_reply_grounding("There's a 15% restocking fee on that.", calls)
    assert findings


def test_honest_abstention_is_never_flagged():
    calls = [_policy_call(False)]
    reply = "I don't have that information on hand — let me check and get back to you."
    assert check_reply_grounding(reply, calls) == []


def test_numbers_grounded_in_a_different_tools_output_are_accepted():
    calls = [
        {"name": "get_order_status", "input": {}, "output": {"found": True, "price": 34.99, "quantity": 1}},
        _policy_call(True, "Refunds are issued within 3-5 business days."),
    ]
    assert check_reply_grounding("Your $34.99 refund arrives in 3 to 5 business days.", calls) == []


def test_non_claim_numbers_are_not_flagged():
    """Only policy-shaped claims (durations, percentages, money) are checked —
    an incidental number like 'two ways' must not trip the detector."""
    calls = [_policy_call(True, "Most items can be returned within 30 days of delivery.")]
    assert check_reply_grounding("I can help with that in 2 ways, and you have 30 days.", calls) == []


def test_hedge_for_rotates_deterministically():
    assert hedge_for(0) == HEDGE_PHRASES[0]
    assert hedge_for(1) == HEDGE_PHRASES[1 % len(HEDGE_PHRASES)]
    assert hedge_for(len(HEDGE_PHRASES)) == HEDGE_PHRASES[0]


def test_hedge_phrases_are_all_non_empty():
    assert HEDGE_PHRASES and all(phrase.strip() for phrase in HEDGE_PHRASES)


class _HostileStr:
    """A tool output value whose __str__ itself raises.

    Simulates the kind of buggy object a tool could hand back that survives
    neither json.dumps nor the str() fallback — the detector must still fail
    open rather than crash the turn it's meant to protect.
    """

    def __str__(self):
        raise RuntimeError("boom")


def test_never_raises_on_a_tool_output_whose_str_itself_raises(caplog):
    calls = [{"name": "search_policy", "input": {}, "output": _HostileStr()}]
    with caplog.at_level("WARNING", logger="guardrails.validators"):
        # Deliberately unguarded: an escaping exception must fail this test.
        result = check_reply_grounding("There's a 15% restocking fee on that.", calls)
    assert result == []
    assert "grounding check failed" in caplog.text


def test_never_raises_on_a_non_dict_tool_call_entry(caplog):
    calls = ["not-a-dict"]
    with caplog.at_level("WARNING", logger="guardrails.validators"):
        # Deliberately unguarded: an escaping exception must fail this test.
        result = check_reply_grounding("There's a 15% restocking fee on that.", calls)
    assert result == []
    assert "grounding check failed" in caplog.text
