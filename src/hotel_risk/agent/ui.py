"""Streamlit front end for the copilot: streamlit run src/hotel_risk/agent/ui.py

Shows the answer AND the audit trail (which tools ran, with what arguments, what they returned, and
whether every number in the answer was found in tool output).
"""
from __future__ import annotations

import json
import os

import streamlit as st

from hotel_risk.agent import HotelAgent, HotelTools
from hotel_risk.agent.providers import ClaudeProvider
from hotel_risk.data import load_raw
from hotel_risk.predict import load_bundle

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
BOOKINGS_PATH = os.environ.get("BOOKINGS_PATH", "data/raw/hotel_bookings.csv")
EXAMPLES = [
    "Which bookings arriving 2017-07-15 are most likely to cancel?",
    "How many should we oversell on 2017-07-15?",
    "Why is booking 100 risky?",
    "How does the overbooking policy work?",
]

st.set_page_config(page_title="Hotel cancellation copilot", page_icon="🏨", layout="wide")


@st.cache_resource(show_spinner="Loading model and scoring bookings...")
def get_tools() -> HotelTools:
    tools = HotelTools(load_bundle(MODEL_PATH), load_raw(BOOKINGS_PATH))
    tools._all_scores()  # score once up front so the first question is fast
    return tools


def render_trail(msg: dict) -> None:
    """Badges plus an expandable list of tool calls under an assistant message."""
    badge = f"provider: **{msg['provider']}**" + (" (fallback)" if msg["fallback_used"] else "")
    st.caption(badge + (" · numbers verified against tools" if msg["grounded"]
                        else " · :warning: some numbers not found in tool output"))
    with st.expander(f"Tool calls ({len(msg['tool_calls'])})"):
        if not msg["tool_calls"]:
            st.write("No tool was called.")
        for c in msg["tool_calls"]:
            st.markdown(f"`{c['name']}`")
            st.json({"args": c["args"], **({"error": c["error"]} if "error" in c else {"result": c["result"]})},
                    expanded=False)
    if msg.get("fallback_reason"):
        st.caption(f"Fallback reason: {msg['fallback_reason']}")


try:
    tools = get_tools()
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nTrain the model first:\n\n"
             "`python -m hotel_risk.train data/raw/hotel_bookings.csv models/model.joblib`")
    st.stop()

with st.sidebar:
    st.header("Hotel cancellation copilot")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    mode = st.radio("Who chooses the tools?", ["Rule router (no key)", "Claude"],
                    index=1 if has_key else 0, help="Claude needs ANTHROPIC_API_KEY in the environment.")
    if mode == "Claude" and not has_key:
        st.warning("No ANTHROPIC_API_KEY found; answers will fall back to the rule router.")
    st.markdown("**Try one**")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state["pending"] = q
    if st.button("Clear chat"):
        st.session_state["messages"] = []
    st.caption("Data: public Hotel Booking Demand dataset (two Portuguese hotels, 2015-2017), replayed as if "
               "bookings were arriving. Dollar figures use illustrative assumptions, not real finances.")

st.session_state.setdefault("messages", [])
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_trail(m)

question = st.chat_input("Ask about a date (YYYY-MM-DD) or a booking id") or st.session_state.pop("pending", None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]]
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    agent = HotelAgent(tools, ClaudeProvider() if mode == "Claude" else None)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = agent.ask(question, history)
        st.markdown(r.answer)
        msg = {"role": "assistant", "content": r.answer, "provider": r.provider,
               "fallback_used": r.fallback_used, "fallback_reason": r.fallback_reason,
               "grounded": r.grounded, "tool_calls": json.loads(json.dumps(r.tool_calls, default=str))}
        render_trail(msg)
    st.session_state["messages"].append(msg)
