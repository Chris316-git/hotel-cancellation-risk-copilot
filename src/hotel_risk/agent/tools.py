"""The narrow set of capabilities the agent may use. Each tool returns plain JSON-serialisable data.

There is deliberately no free-form SQL/code tool: the agent can only score bookings that exist in the
loaded table and read the validated reports. All dollar figures use the illustrative assumptions in
`impact.Assumptions`; they are not real hotel financials.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import booking_time_features, clean
from ..predict import Bundle

ASSUMPTION_NOTE = ("Dollar figures use illustrative assumptions (resell fill rate, walk cost), "
                   "not real hotel financials.")

TOOL_SPECS = [
    {"name": "list_riskiest_bookings",
     "description": "Rank bookings arriving on one date by calibrated cancellation probability.",
     "input_schema": {"type": "object", "properties": {
         "arrival_date": {"type": "string", "description": "YYYY-MM-DD"},
         "top_n": {"type": "integer", "minimum": 1, "maximum": 20}},
         "required": ["arrival_date"]}},
    {"name": "get_booking_risk",
     "description": "Cancellation probability, decision and top drivers for one booking_id.",
     "input_schema": {"type": "object", "properties": {"booking_id": {"type": "integer"}},
                      "required": ["booking_id"]}},
    {"name": "overbooking_summary",
     "description": "For one arrival date: bookings, expected cancellations, and how many the "
                    "cost-aware policy would oversell.",
     "input_schema": {"type": "object", "properties": {"arrival_date": {"type": "string"}},
                      "required": ["arrival_date"]}},
    {"name": "explain_policy",
     "description": "The overbooking rule, its assumptions, and the validated model/policy results.",
     "input_schema": {"type": "object", "properties": {}}},
]


class ToolError(ValueError):
    """Bad arguments from the caller (unknown id, date with no bookings...). Safe to show."""


class HotelTools:
    def __init__(self, bundle: Bundle, raw_bookings: pd.DataFrame, reports_dir: str | Path = "reports"):
        self.bundle = bundle
        self.reports_dir = Path(reports_dir)
        self.raw = raw_bookings.reset_index(drop=True)
        self.clean = clean(self.raw)
        # clean() may drop invalid rows, so keep a stable id that survives it
        self.clean["booking_id"] = self.clean.index
        self._scored: pd.DataFrame | None = None

    # ---- scoring -------------------------------------------------------------------------
    def _all_scores(self) -> pd.DataFrame:
        """Score every loaded booking once (calibrated p, risk band, oversell decision)."""
        if self._scored is None:
            X = booking_time_features(self.clean)
            p = self.bundle.calibrator.predict(self.bundle.model.predict_proba(X)[:, 1])
            a = self.bundle.assumptions
            nights = np.maximum(self.clean["total_nights"].to_numpy(), 1)
            gain = a.resell_fill_rate * self.clean["adr"].to_numpy() * nights
            cost = a.walk_cost_nights * self.clean["adr"].to_numpy() + a.walk_fixed
            df = self.clean[["booking_id", "hotel", "arrival_date", "lead_time", "adr", "deposit_type",
                             "market_segment", "country"]].copy()
            df["cancel_probability"] = np.round(p, 4)
            df["risk_band"] = pd.cut(p, [-0.01, 0.25, 0.6, 1.0], labels=["low", "medium", "high"]).astype(str)
            df["oversell_recommended"] = p > cost / (gain + cost + 1e-9)
            self._scored = df
        return self._scored

    @staticmethod
    def _date(value: str) -> pd.Timestamp:
        try:
            return pd.Timestamp(str(value)[:10])
        except (ValueError, TypeError):
            raise ToolError(f"'{value}' is not a valid date; use YYYY-MM-DD.")

    def _on_date(self, arrival_date: str) -> pd.DataFrame:
        d = self._date(arrival_date)
        rows = self._all_scores()
        rows = rows[rows["arrival_date"] == d]
        if rows.empty:
            lo, hi = self.clean["arrival_date"].min().date(), self.clean["arrival_date"].max().date()
            raise ToolError(f"No bookings arrive on {d.date()}. Loaded data covers {lo} to {hi}.")
        return rows

    # ---- tools ---------------------------------------------------------------------------
    def list_riskiest_bookings(self, arrival_date: str, top_n: int = 5) -> dict:
        rows = self._on_date(arrival_date)
        top_n = int(min(max(top_n, 1), 20))
        top = rows.sort_values("cancel_probability", ascending=False).head(top_n)
        return {"arrival_date": str(self._date(arrival_date).date()), "bookings_that_day": int(len(rows)),
                "bookings": [{
                    "booking_id": int(r.booking_id), "hotel": r.hotel, "lead_time": int(r.lead_time),
                    "adr": round(float(r.adr), 2), "deposit_type": r.deposit_type,
                    "cancel_probability": float(r.cancel_probability), "risk_band": r.risk_band,
                    "oversell_recommended": bool(r.oversell_recommended)} for r in top.itertuples()],
                "note": ASSUMPTION_NOTE}

    def overbooking_summary(self, arrival_date: str) -> dict:
        rows = self._on_date(arrival_date)
        return {"arrival_date": str(self._date(arrival_date).date()), "bookings": int(len(rows)),
                "expected_cancellations": round(float(rows["cancel_probability"].sum()), 1),
                "high_risk_bookings": int((rows["risk_band"] == "high").sum()),
                "oversell_recommended": int(rows["oversell_recommended"].sum()),
                "assumptions": dict(vars(self.bundle.assumptions)), "note": ASSUMPTION_NOTE}

    def get_booking_risk(self, booking_id: int) -> dict:
        scored = self._all_scores()
        match = scored[scored["booking_id"] == int(booking_id)]
        if match.empty:
            raise ToolError(f"booking_id {booking_id} not found (valid ids 0 to {len(scored) - 1}).")
        r = match.iloc[0]
        return {"booking_id": int(r.booking_id), "hotel": r.hotel, "arrival_date": str(r.arrival_date.date()),
                "lead_time": int(r.lead_time), "adr": round(float(r.adr), 2),
                "cancel_probability": float(r.cancel_probability), "risk_band": r.risk_band,
                "oversell_recommended": bool(r.oversell_recommended),
                "drivers": self._drivers(int(booking_id)),
                "driver_note": "Model sensitivities (log-odds contributions), not causal effects.",
                "note": ASSUMPTION_NOTE}

    def _drivers(self, booking_id: int, top: int = 3) -> list[dict]:
        """Top feature contributions for one booking via LightGBM's native pred_contrib."""
        pre, clf = self.bundle.model.named_steps["pre"], self.bundle.model.named_steps["clf"]
        if not hasattr(clf, "booster_"):
            return []  # model type without native contributions
        row = booking_time_features(self.clean.loc[self.clean["booking_id"] == booking_id])
        contrib = clf.predict(pre.transform(row), pred_contrib=True)[0][:-1]
        names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        out = []
        for k in np.argsort(-np.abs(contrib))[:top]:
            val = row.iloc[0][names[k]]
            out.append({"feature": names[k], "value": val.item() if hasattr(val, "item") else val,
                        "pushes": "towards cancel" if contrib[k] > 0 else "towards stays",
                        "log_odds": round(float(contrib[k]), 2)})
        return out

    def explain_policy(self) -> dict:
        a = self.bundle.assumptions
        out = {"rule": "Oversell a booking when p*gain > (1-p)*cost, i.e. p > cost / (gain + cost).",
               "gain": "resell_fill_rate * adr * nights (the room is re-sold after a cancellation)",
               "cost": "walk_cost_nights * adr + walk_fixed (relocate the guest if everyone shows up)",
               "assumptions": dict(vars(a)), "trained_through": self.bundle.trained_through,
               "note": ASSUMPTION_NOTE}
        for name, keep in (("phase4_impact.json", ("net_value_usd", "captured_share_of_oracle")),
                           ("phase3_calibration.json", ("threshold_policy",))):
            f = self.reports_dir / name
            if f.exists():
                data = json.loads(f.read_text())
                out |= {k: data[k] for k in keep if k in data}
        return out

    def call(self, name: str, args: dict) -> dict:
        """Single dispatch point so every call is validated and logged the same way."""
        allowed = {s["name"] for s in TOOL_SPECS}
        if name not in allowed:
            raise ToolError(f"unknown tool '{name}'")
        return getattr(self, name)(**args)
