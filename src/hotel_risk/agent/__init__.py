"""A small, auditable tool-calling agent on top of the cancellation model.

The LLM (or the rule router) only decides WHICH tool to call and how to phrase the answer.
Every number comes from a tool, and a grounding check flags numbers that no tool produced.
"""
from .core import AgentResult, HotelAgent  # noqa: F401
from .tools import HotelTools  # noqa: F401
