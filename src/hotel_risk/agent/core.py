"""The agent: provider first, deterministic fallback second, grounding check on every answer."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .providers import AgentProviderError, DATE_RE, ProviderResult, RuleRouter
from .tools import HotelTools

log = logging.getLogger(__name__)
NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")
CAUTION = "\n[Warning: some numbers above were not found in tool output; verify before using them.]"


def _tool_numbers(value, out: list[float]) -> None:
    """Collect every numeric leaf from tool results (bools excluded)."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            _tool_numbers(v, out)
    elif isinstance(value, list):
        for v in value:
            _tool_numbers(v, out)


def ungrounded_numbers(answer: str, tool_calls: list[dict], question: str = "") -> list[str]:
    """Numbers in the answer that no tool produced (and the user did not supply).

    A number is grounded if it equals a tool value, or a tool probability shown as a percentage, after
    rounding to the precision written in the answer. Dates are ignored; numbers the user typed
    (booking ids, 'top 3') and 0/1 (formula constants) count as grounded.
    """
    source: list[float] = []
    for c in tool_calls:
        _tool_numbers(c.get("result"), source)
        _tool_numbers(c.get("args"), source)
        # numbers inside a tool's error message (e.g. the valid id range) are tool output too
        source += [float(x.replace(",", "")) for x in NUM_RE.findall(DATE_RE.sub("", str(c.get("error", ""))))
                   if "%" not in x]
    typed = {0.0, 1.0} | {float(x.replace(",", "")) for x in NUM_RE.findall(DATE_RE.sub("", question)) if "%" not in x}
    bad = []
    for tok in NUM_RE.findall(DATE_RE.sub("", answer)):
        raw = tok.rstrip("%").replace(",", "")
        n = float(raw)
        dec = len(raw.split(".")[1]) if "." in raw else 0
        pct = tok.endswith("%")
        candidates = [v * 100 for v in source] if pct else source + [v * 100 for v in source]
        if n in typed or any(round(c, dec) == n for c in candidates):
            continue
        bad.append(tok)
    return bad


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    provider: str = "rules"
    fallback_used: bool = False
    fallback_reason: str | None = None
    grounded: bool = True
    ungrounded: list[str] = field(default_factory=list)


class HotelAgent:
    def __init__(self, tools: HotelTools, provider=None, fallback=None):
        self.tools = tools
        self.provider = provider or RuleRouter()
        self.fallback = fallback or RuleRouter()

    def ask(self, message: str, history: list[dict] | None = None) -> AgentResult:
        history = history or []
        used, reason, fb = self.provider, None, False
        try:
            res: ProviderResult = self.provider.run(message, history, self.tools)
        except AgentProviderError as exc:
            log.warning("provider %s failed (%s); using rule router", self.provider.name, exc)
            used, reason, fb = self.fallback, str(exc), True
            res = self.fallback.run(message, history, self.tools)
        bad = ungrounded_numbers(res.answer, res.tool_calls, message)
        answer = res.answer + (CAUTION if bad else "")
        return AgentResult(answer, res.tool_calls, used.name, fb, reason, not bad, bad)
