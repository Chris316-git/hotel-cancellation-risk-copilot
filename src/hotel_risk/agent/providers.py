"""Who decides which tool to call: a deterministic rule router (default, free, used in CI) or Claude."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .tools import TOOL_SPECS, HotelTools, ToolError

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

SYSTEM_PROMPT = """You are a revenue-management assistant for a hotel booking cancellation model.
Rules:
- Use the tools for every number. Never estimate, recompute or invent figures.
- Bookings come from a historical dataset (Portuguese hotels, 2015-2017); say so if asked about 'today'.
- Dollar figures rest on illustrative assumptions, not real finances; mention that when you quote them.
- Drivers are model sensitivities, not causal effects. Do not promise outcomes.
- If a tool returns an error, relay it and suggest a valid input. Keep answers short."""


class AgentProviderError(RuntimeError):
    """The provider could not produce an answer (no key, API error...). The agent then falls back."""


@dataclass
class ProviderResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)  # {"name", "args", "result" | "error"}


def _run_tool(tools: HotelTools, name: str, args: dict) -> dict:
    """Execute one tool and always return a record, so errors are visible in the audit trail."""
    try:
        return {"name": name, "args": args, "result": tools.call(name, args)}
    except (ToolError, TypeError) as exc:  # TypeError: the model sent arguments the tool doesn't take
        return {"name": name, "args": args, "error": str(exc)}


class RuleRouter:
    """Keyword/regex intent routing with templated answers. No LLM, fully reproducible."""
    name = "rules"

    HELP = ("I can answer: 'riskiest bookings on 2017-07-15', 'overbooking summary for 2017-07-15', "
            "'why is booking <id> risky', and 'how does the overbooking policy work'.")

    def run(self, message: str, history: list[dict], tools: HotelTools) -> ProviderResult:
        m = message.lower()
        date = DATE_RE.search(message)
        booking = re.search(r"(?:booking|订单|#)\s*(?:id\s*)?[:#]?\s*(\d+)", m)
        top = re.search(r"top\s*(\d+)", m)
        if booking:
            call = _run_tool(tools, "get_booking_risk", {"booking_id": int(booking.group(1))})
        elif date and re.search(r"overbook|oversell|summary|how many|超售|汇总", m):
            call = _run_tool(tools, "overbooking_summary", {"arrival_date": date.group(1)})
        elif date and re.search(r"risk|top|highest|likely|cancel|风险|最高", m):
            args = {"arrival_date": date.group(1)} | ({"top_n": int(top.group(1))} if top else {})
            call = _run_tool(tools, "list_riskiest_bookings", args)
        elif re.search(r"polic|assum|threshold|rule|how .*(decid|work)|阈值|假设|策略", m):
            call = _run_tool(tools, "explain_policy", {})
        else:
            return ProviderResult(self.HELP)
        return ProviderResult(self._render(call), [call])

    @staticmethod
    def _render(call: dict) -> str:
        if "error" in call:
            return f"I could not do that: {call['error']}"
        r, name = call["result"], call["name"]
        if name == "list_riskiest_bookings":
            lines = [f"Riskiest bookings arriving {r['arrival_date']} ({r['bookings_that_day']} bookings that day):"]
            for b in r["bookings"]:
                action = "oversell" if b["oversell_recommended"] else "hold"
                lines.append(f"- booking {b['booking_id']}: {b['cancel_probability']:.0%} cancel chance "
                             f"({b['risk_band']}), lead time {b['lead_time']} days, {b['deposit_type']}; {action}")
            return "\n".join(lines) + f"\n{r['note']}"
        if name == "overbooking_summary":
            return (f"{r['arrival_date']}: {r['bookings']} bookings, about {r['expected_cancellations']} expected "
                    f"cancellations, {r['high_risk_bookings']} high-risk, and the cost-aware policy would "
                    f"oversell {r['oversell_recommended']}. {r['note']}")
        if name == "get_booking_risk":
            drivers = "; ".join(f"{d['feature']}={d['value']} {d['pushes']}" for d in r["drivers"]) or "n/a"
            return (f"Booking {r['booking_id']} ({r['hotel']}, arriving {r['arrival_date']}): "
                    f"{r['cancel_probability']:.0%} cancel chance ({r['risk_band']}), "
                    f"{'oversell' if r['oversell_recommended'] else 'hold'}. Top drivers: {drivers}. {r['driver_note']}")
        lines = [r["rule"], f"Gain: {r['gain']}. Cost: {r['cost']}.",
                 "Assumptions: " + ", ".join(f"{k}={v}" for k, v in r["assumptions"].items()) + "."]
        if "captured_share_of_oracle" in r:
            lines.append(f"On the test period the cost-aware policy captured {r['captured_share_of_oracle']:.1%} "
                         "of the perfect-foresight upper bound.")
        return " ".join(lines) + f" {r['note']}"


class ClaudeProvider:
    """Anthropic tool-use loop. Needs `pip install anthropic` and ANTHROPIC_API_KEY."""
    name = "claude"
    MAX_STEPS = 5

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.environ.get("HOTEL_AGENT_MODEL", "claude-sonnet-4-5")
        self._client = client

    def _get_client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise AgentProviderError("ANTHROPIC_API_KEY is not set")
            try:
                import anthropic
            except ImportError as exc:
                raise AgentProviderError("pip install anthropic to use the Claude provider") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def run(self, message: str, history: list[dict], tools: HotelTools) -> ProviderResult:
        client = self._get_client()
        messages = [{"role": h["role"], "content": h["content"]} for h in history] + \
                   [{"role": "user", "content": message}]
        calls: list[dict] = []
        try:
            for _ in range(self.MAX_STEPS):
                resp = client.messages.create(model=self.model, max_tokens=800, system=SYSTEM_PROMPT,
                                              tools=TOOL_SPECS, messages=messages)
                if resp.stop_reason != "tool_use":
                    text = "".join(b.text for b in resp.content if b.type == "text").strip()
                    return ProviderResult(text, calls)
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for block in (b for b in resp.content if b.type == "tool_use"):
                    call = _run_tool(tools, block.name, dict(block.input))
                    calls.append(call)
                    payload = call.get("result", {"error": call.get("error")})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(payload, default=str)})
                messages.append({"role": "user", "content": results})
        except AgentProviderError:
            raise
        except Exception as exc:  # network, auth, rate limit: never crash the request
            raise AgentProviderError(f"Claude call failed: {exc}") from exc
        raise AgentProviderError("tool loop did not finish within the step limit")
