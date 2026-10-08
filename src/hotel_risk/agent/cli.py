"""Chat in the terminal: python -m hotel_risk.agent.cli   (add --claude to use the Claude provider)."""
from __future__ import annotations

import sys

from ..data import load_raw
from ..predict import load_bundle
from .core import HotelAgent
from .providers import ClaudeProvider
from .tools import HotelTools


def main() -> None:
    agent = HotelAgent(HotelTools(load_bundle("models/model.joblib"), load_raw("data/raw/hotel_bookings.csv")),
                       ClaudeProvider() if "--claude" in sys.argv else None)
    history: list[dict] = []
    print("Hotel cancellation copilot (historical data, 2015-2017). Empty line to quit.")
    while (q := input("> ").strip()):
        r = agent.ask(q, history)
        print(r.answer, f"\n  [provider={r.provider}{' (fallback)' if r.fallback_used else ''}, "
                        f"tools={[c['name'] for c in r.tool_calls]}]\n")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r.answer}]


if __name__ == "__main__":
    main()
