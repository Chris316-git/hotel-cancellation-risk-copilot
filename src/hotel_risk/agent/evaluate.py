"""Golden-case evaluation for the agent: python -m hotel_risk.agent.evaluate [--provider claude]

Checks, per case: did it call the right tool with the right arguments, did it handle errors, and was every
number in the answer produced by a tool. CI runs this with the rule router (no API key needed).
Writes reports/agent_eval.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..data import load_raw
from ..predict import load_bundle
from .core import HotelAgent
from .providers import ClaudeProvider
from .tools import HotelTools


def run_cases(agent: HotelAgent, cases: list[dict]) -> dict:
    rows = []
    for c in cases:
        r = agent.ask(c["q"])
        calls = r.tool_calls
        want = c.get("expect_tool")
        tool_ok = (not calls) if want is None else any(
            k["name"] == want and all(k["args"].get(a) == v for a, v in c.get("expect_args", {}).items())
            for k in calls)
        errored = any("error" in k for k in calls)
        error_ok = errored == bool(c.get("expect_error", False))
        rows.append({"question": c["q"], "tool_ok": tool_ok, "error_handling_ok": error_ok,
                     "grounded": r.grounded, "provider": r.provider, "fallback_used": r.fallback_used,
                     "passed": tool_ok and error_ok and r.grounded})
    n = len(rows)
    return {"cases": n, "pass_rate": round(sum(x["passed"] for x in rows) / n, 3),
            "tool_accuracy": round(sum(x["tool_ok"] for x in rows) / n, 3),
            "grounded_rate": round(sum(x["grounded"] for x in rows) / n, 3),
            "fallback_rate": round(sum(x["fallback_used"] for x in rows) / n, 3), "results": rows}


def main(argv: list[str]) -> dict:
    provider = ClaudeProvider() if "--provider" in argv and argv[argv.index("--provider") + 1] == "claude" else None
    bundle = load_bundle("models/model.joblib")
    tools = HotelTools(bundle, load_raw("data/raw/hotel_bookings.csv"))
    cases = [json.loads(line) for line in Path("evals/agent_cases.jsonl").read_text().splitlines() if line.strip()]
    report = run_cases(HotelAgent(tools, provider), cases)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/agent_eval.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = main(sys.argv[1:])
    print({k: v for k, v in rep.items() if k != "results"})
    for r in rep["results"]:
        print("PASS" if r["passed"] else "FAIL", "|", r["question"])
