"""FastAPI service: uvicorn hotel_risk.api:app  (set MODEL_PATH, default models/model.joblib)."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import load_bundle, score

app = FastAPI(title="Hotel Cancellation Risk API", version="0.1.0")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_bundle(os.environ.get("MODEL_PATH", "models/model.joblib"))
    return _bundle


class Booking(BaseModel):
    hotel: str
    lead_time: int = Field(ge=0)
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=1)
    children: int = Field(default=0, ge=0)
    babies: int = Field(default=0, ge=0)
    meal: str = "BB"
    country: Optional[str] = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int = Field(default=0, ge=0, le=1)
    previous_cancellations: int = Field(default=0, ge=0)
    previous_bookings_not_canceled: int = Field(default=0, ge=0)
    reserved_room_type: str
    deposit_type: str
    agent: Optional[float] = None
    company: Optional[float] = None
    customer_type: str
    adr: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score_bookings(bookings: list[Booking]):
    try:
        res = score(get_bundle(), [b.model_dump() for b in bookings])
    except FileNotFoundError:
        raise HTTPException(503, "Model file not found; run `python -m hotel_risk.train`.")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return res.to_dict(orient="records")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[dict] = Field(default_factory=list, max_length=20)


_agent = None


def get_agent():
    """Built lazily: loads the model and the booking table the agent can look things up in."""
    global _agent
    if _agent is None:
        from .agent import HotelAgent, HotelTools
        from .agent.providers import ClaudeProvider
        from .data import load_raw
        tools = HotelTools(get_bundle(), load_raw(os.environ.get("BOOKINGS_PATH", "data/raw/hotel_bookings.csv")))
        use_claude = os.environ.get("AGENT_PROVIDER", "rules") == "claude"
        _agent = HotelAgent(tools, ClaudeProvider() if use_claude else None)
    return _agent


@app.post("/chat")
def chat(req: ChatRequest):
    """Ask the copilot a question. Numbers in answers come from tools; see `grounded` and `tool_calls`."""
    try:
        r = get_agent().ask(req.message, req.history)
    except FileNotFoundError:
        raise HTTPException(503, "Model or booking data not found; run train and check BOOKINGS_PATH.")
    return {"answer": r.answer, "provider": r.provider, "fallback_used": r.fallback_used,
            "grounded": r.grounded, "tool_calls": r.tool_calls}
