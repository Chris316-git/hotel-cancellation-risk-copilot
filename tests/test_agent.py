import json

import pytest

from hotel_risk.agent import HotelAgent, HotelTools
from hotel_risk.agent.core import ungrounded_numbers
from hotel_risk.agent.evaluate import run_cases
from hotel_risk.agent.providers import AgentProviderError, ClaudeProvider
from hotel_risk.data import clean
from hotel_risk.train import build_bundle
from tests.test_predict import synthetic


@pytest.fixture(scope="module")
def tools():
    df = synthetic(2500)
    full = df.assign(assigned_room_type="A", booking_changes=0, days_in_waiting_list=0,
                     required_car_parking_spaces=0, total_of_special_requests=0,
                     reservation_status="x", reservation_status_date="2017-01-01")
    bundle = build_bundle(clean(full))
    return HotelTools(bundle, df.drop(columns="is_canceled"))


def a_date(tools):
    return str(tools.clean["arrival_date"].iloc[0].date())


def test_tools_return_valid_probabilities(tools):
    d = a_date(tools)
    top = tools.list_riskiest_bookings(d, 3)["bookings"]
    probs = [b["cancel_probability"] for b in top]
    assert probs == sorted(probs, reverse=True) and all(0 <= p <= 1 for p in probs)
    s = tools.overbooking_summary(d)
    assert 0 <= s["oversell_recommended"] <= s["bookings"]
    risk = tools.get_booking_risk(top[0]["booking_id"])
    assert risk["cancel_probability"] == top[0]["cancel_probability"]
    clf = tools.bundle.model.named_steps["clf"]
    if hasattr(clf, "booster_"):  # LightGBM present: native contributions are available
        assert len(risk["drivers"]) == 3


def test_bad_inputs_are_errors_not_crashes(tools):
    agent = HotelAgent(tools)
    r = agent.ask("riskiest bookings on 2030-01-01")
    assert "error" in r.tool_calls[0] and "could not" in r.answer.lower()
    assert "error" in agent.ask("why is booking 99999999 risky").tool_calls[0]


def test_unsupported_question_calls_no_tool(tools):
    r = HotelAgent(tools).ask("what is the weather in Lisbon?")
    assert r.tool_calls == [] and r.grounded


def test_router_answers_are_grounded(tools):
    d = a_date(tools)
    agent = HotelAgent(tools)
    for q in (f"top 3 riskiest on {d}", f"overbooking summary for {d}", "why is booking 5 risky",
              "how does the overbooking policy work"):
        r = agent.ask(q)
        assert r.tool_calls and r.grounded, (q, r.ungrounded)


def test_grounding_check_flags_invented_numbers():
    calls = [{"name": "t", "args": {}, "result": {"cancel_probability": 0.4321, "n": 12}}]
    assert ungrounded_numbers("It is 43% likely, 12 bookings.", calls) == []
    assert ungrounded_numbers("It is 77% likely, 12 bookings.", calls) == ["77%"]
    assert ungrounded_numbers("Booking 7 on 2017-07-15", calls, "why is booking 7 risky") == []


def test_grounding_handles_signs_and_numeric_categories():
    calls = [{"name": "t", "args": {}, "result": {"drivers": [{"feature": "agent", "value": "9", "log_odds": -1.07}]}}]
    # LLMs write a typographic minus and quote category codes such as agent 9
    assert ungrounded_numbers("agent 9 pulls toward staying (\u22121.07) and (-1.07)", calls) == []
    assert ungrounded_numbers("agent 9 (-1.08)", calls) == ["1.08"] or ungrounded_numbers("agent 9 (-1.08)", calls) == ["-1.08"]


class Boom:
    name = "boom"

    def run(self, *a):
        raise AgentProviderError("no key")


class Liar:
    name = "liar"

    def run(self, message, history, tools):
        from hotel_risk.agent.providers import ProviderResult
        return ProviderResult("Cancellation chance is 99.9%.", [])


def test_fallback_when_provider_fails(tools):
    r = HotelAgent(tools, provider=Boom()).ask("how does the overbooking policy work")
    assert r.fallback_used and r.provider == "rules" and r.fallback_reason == "no key" and r.tool_calls


def test_ungrounded_llm_answer_is_flagged(tools):
    r = HotelAgent(tools, provider=Liar()).ask("anything")
    assert not r.grounded and "Warning" in r.answer


def test_claude_provider_without_key_falls_back(tools, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = HotelAgent(tools, provider=ClaudeProvider()).ask("how does the overbooking policy work")
    assert r.fallback_used and r.tool_calls


def test_golden_cases_pass_with_rule_router(tools):
    d = a_date(tools)
    cases = [
        {"q": f"top 3 riskiest bookings on {d}", "expect_tool": "list_riskiest_bookings",
         "expect_args": {"arrival_date": d, "top_n": 3}},
        {"q": "why is booking 5 risky", "expect_tool": "get_booking_risk", "expect_args": {"booking_id": 5}},
        {"q": "riskiest bookings on 2030-01-01", "expect_tool": "list_riskiest_bookings", "expect_error": True},
        {"q": "tell me a joke", "expect_tool": None},
    ]
    rep = run_cases(HotelAgent(tools), cases)
    assert rep["pass_rate"] == 1.0, json.dumps(rep["results"], indent=1)
