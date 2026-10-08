# Hotel Cancellation Risk System

Roughly 4 in 10 hotel bookings in this dataset never turn into a stay. This project tries to guess
which ones, **at the moment the booking is made**, and then asks the question the hotel actually
cares about: is it worth overselling that room?

It covers the whole trip: honest data cleaning, a model, a model that is also *calibrated* (so "70%"
actually means about 70%), a dollars-and-cents decision layer, and a small API you can poke at.

## The short version

- Predicts cancellations using only what the hotel knows when the booking comes in. No peeking at
  how the stay ended.
- Tests on bookings that arrive *after* the training period, the way a real deployment would.
  A random split flatters the model (AUC 0.924 vs 0.860). We report the less flattering number.
- Turns probabilities into an overbooking policy worth about **$1.25M** on the test period
  under illustrative cost assumptions, roughly 38.5% of what a perfect fortune teller would earn.

## Data

The project uses the public **Hotel Booking Demand** dataset: 119,390 bookings from two hotels
(a city hotel and a resort hotel), arriving between July 2015 and August 2017. It has 32 columns,
including the `is_canceled` label. The file lives in `data/raw/hotel_bookings.csv`, so you can run
everything without hunting for data.

Reference: N. Antonio, A. de Almeida, L. Nunes, "Hotel booking demand datasets", *Data in Brief*, 22, 2019
([ISCTE repository](https://repositorio.iscte-iul.pt/handle/10071/16929)). Check the original
publication for terms of use before reusing or redistributing the data.

**Why this one**

- One row per reservation, with a real cancellation outcome. That is exactly what a cancellation
  model needs, and aggregate hotel stats can't give you.
- Arrival dates and lead times make a proper time-based split possible, which this project leans on.
- It comes with a peer-reviewed description, so you can check where it came from.
- It also includes columns that are only known *after* the outcome (`reservation_status`,
  `assigned_room_type`, `booking_changes`...), which makes it a good playground for catching leakage.

**What it is not**

- It is not recent and it is not US data: two hotels, 2015-2017. Results show the method works,
  not what your hotel's numbers will look like. Booking-level US data with cancellations is
  mostly proprietary, and I couldn't find a verifiable recent public alternative.
- Some copies of this dataset online come with relabelled hotel names, shifted dates or extra columns.
  I skipped those: the origin is unclear, and altered dates would break the time-based split.
- The money numbers in Phase 4 come from assumptions I made up and labelled as such, not from these hotels.

## How it avoids cheating

- **Booking-time features only.** Columns like `reservation_status`, `assigned_room_type` and
  `booking_changes` reveal the outcome or only exist after it, so they are excluded
  (see `LEAKAGE_COLUMNS` in `src/hotel_risk/data.py`, each with the reason).
- **Temporal validation.** Train on the past, test on the future.

## Run it yourself

```bash
python -m venv .venv && source .venv/bin/activate     # or use conda
pip install -r requirements.txt && pip install -e .

pytest -q                                   # tests (the API test runs if FastAPI is installed)
python -m hotel_risk.audit                  # data checks + the leakage column list
python -m hotel_risk.evaluate               # Phase 2: models, temporal vs random split, confusion matrix
python -m hotel_risk.calibrate              # Phase 3: calibration + thresholds
python -m hotel_risk.impact                 # Phase 4: overbooking policy value
python -m hotel_risk.explain                # feature importance + SHAP (pip install shap) + plain-words insights
python -m hotel_risk.compare_models         # logistic vs forest vs gradient boosting

python -m hotel_risk.train data/raw/hotel_bookings.csv models/model.joblib
uvicorn hotel_risk.api:app --port 8000      # then open http://localhost:8000/docs
```

Your numbers may differ in the third decimal depending on library versions.

## Results

### Models (temporal split: train before 2017-03, test after)

| Split | Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Temporal | Logistic baseline | 0.836 | 0.789 | 0.161 |
| Temporal | LightGBM | **0.860** | **0.813** | **0.150** |
| Random (for comparison) | Logistic baseline | 0.875 | 0.828 | 0.135 |
| Random | LightGBM | 0.924 | 0.893 | 0.107 |

The random split looks better because it lets the model learn from bookings that happen later than
the ones it is tested on. Real hotels don't get to do that, so the temporal numbers are the ones to trust.

### What the model gets right and wrong (confusion matrix)

At a plain 0.5 cut-off on the test period (about 32,800 bookings, 40% of them cancelled):

| | Flagged and cancelled (TP) | Flagged but stayed (FP) | Missed cancellations (FN) | Correctly cleared (TN) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Logistic regression | 6,613 | 1,972 | 6,514 | 17,679 | 0.770 | 0.504 | 0.609 |
| **LightGBM** | 7,331 | 2,170 | 5,796 | 17,481 | 0.772 | 0.559 | 0.648 |

![Confusion matrices](reports/confusion_matrix.png)

Reading it: when LightGBM says "this one will cancel", it is right about 77% of the time, but at 0.5 it
still misses 44% of real cancellations. That is the usual precision/recall trade-off, and it is why the
threshold is chosen on a separate calibration slice (next section) instead of defaulting to 0.5. LightGBM
beats the baseline mostly by catching more cancellations (recall 0.50 to 0.56) at the same precision.

### Calibration and thresholds

Data is split chronologically into train (before 2016-12), calibration (2016-12 to 2017-02) and
test (2017-03 onward). Calibrators and the decision threshold are fit on the calibration slice only.

| Test scores | Brier | Log loss | ECE |
|---|---|---|---|
| Raw LightGBM | 0.154 | 0.451 | 0.052 |
| Platt (sigmoid) | 0.152 | 0.448 | 0.045 |
| Isotonic | 0.153 | 0.450 | 0.044 |

Calibration makes the probabilities more honest without hurting ranking (AUC stays about 0.85).

Two things worth knowing:

- The threshold picked for 80% precision on the calibration slice (0.46) gave 74% precision and
  62% recall on test. The cancel rate drifted from 34% to 40% between the two periods, so a
  threshold tuned on the past doesn't transfer perfectly. Real data does this.
- The riskiest 10% of bookings were 100% cancellations on test, largely non-refundable-deposit
  bookings, a well-known quirk of this dataset. Don't read that as superhuman skill. The model
  is most useful in the messy middle.

### Business impact

The policy: oversell a booking's room when `p * gain > (1 - p) * cost`, where `p` is the calibrated
cancel probability, *gain* is what you earn if the room is re-sold (resell rate x ADR x nights), and
*cost* is what it takes to relocate a guest who shows up anyway.

The assumptions are made up on purpose and easy to change in `Assumptions`: 50% chance a freed room
is re-sold, relocation costs 2 nights of ADR plus $50.

| Policy (test period, 32,778 bookings) | Net value |
|---|---|
| No overbooking | $0 |
| Flag p >= 0.5 | $667k |
| Flag p >= 0.8 | $538k |
| **Cost-aware model** | **$1.25M** |
| Perfect foresight (upper bound) | $3.25M |

The cost-aware policy beats fixed thresholds because it accounts for each booking's price and length of
stay. Across the scenarios in `reports/phase4_impact.json` the value ranges from $0.4M to $3.5M,
so the real answer depends on real operating costs. It also ignores hotel capacity and demand limits
and treats bookings as independent, so treat the dollars as an illustration, not a forecast.

### Which features matter, and which models were tried

Permutation importance on the test period (shuffle one feature, see how much ROC-AUC drops):

| Feature | AUC drop |
|---|---|
| `country` | 0.107 |
| `deposit_type` | 0.083 |
| `agent` | 0.074 |
| `market_segment` | 0.067 |
| `lead_time` | 0.032 |

What the raw data says about the top drivers (cancel rate on the test period, no model involved, so you can
check the model's story against it):

| Feature | Highest-cancel group | Lowest-cancel group |
|---|---|---|
| `deposit_type` | Non Refund: 100% | No Deposit: 33% |
| `lead_time` | 181+ days: 50% | 0-7 days: 10% |
| `market_segment` | Groups: 61% | Direct: 17% |
| `customer_type` | Transient: 45% | Contract: 12% |
| `country` | PRT: 63% | DEU: 22% |

Reading it: booking far ahead, booking as a group or through an agent, and having no deposit at stake all go
with more cancellations. The 100% for "Non Refund" is odd (you'd expect a non-refundable booking to be the
*least* likely to cancel) and is a known quirk of this dataset, so treat that column as a data artifact
rather than a business rule. `python -m hotel_risk.explain` prints this table, plus the SHAP direction for
numeric features, and writes everything to `reports/feature_importance.json`.

SHAP agrees with the raw table on direction for the numeric features: longer lead time pushes towards
cancel, and more previous bookings that were *not* cancelled pushes towards staying (a repeat guest who
kept their last bookings is a safer bet). Two to treat carefully: higher `adr` (price per night) pushes
towards cancel, which may reflect room type and season rather than price itself, and
`arrival_date_day_of_month` shows a strong direction even though it has no sensible causal story, so read it
as noise the model picked up rather than a real driver. Correlation in the data is not a reason to act.

The plot is in `reports/feature_importance.png`, and a SHAP summary (which direction each feature pushes
the prediction) is in `reports/shap_summary.png`. Two honest observations: `deposit_type`, `lead_time` and
`market_segment` are the kind of drivers you'd expect, but `country` and `agent` are also near the top.
Those are specific to *these* hotels (where guests come from, which travel agents book them), so a model
trained here would not carry over to another hotel unchanged. That is a limitation of the data, not
something more tuning would fix.

Same temporal split, different model families:

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Logistic regression | 0.836 | 0.789 | 0.161 |
| Random forest | 0.848 | 0.803 | 0.155 |
| HistGradientBoosting (scikit-learn) | 0.852 | 0.807 | 0.156 |
| XGBoost | 0.858 | 0.811 | 0.151 |
| **LightGBM** | **0.860** | **0.813** | **0.150** |

LightGBM came out on top, so it is the model used everywhere else (calibration, business impact, API).
The gaps between the boosted models are small, so the ranking among them matters less than the gap to
the logistic baseline. XGBoost and SHAP are optional extras (`pip install -r requirements-extra.txt`) and are not part of the CI
run. On macOS, LightGBM and XGBoost need the `libomp` library (`conda install -c conda-forge llvm-openmp`).

## The API

`POST /score` takes a list of booking-time records and returns, for each, a calibrated cancel
probability, a risk band (low / medium / high) and whether overselling the room is recommended
under the Phase 4 assumptions. Records with no guests or a negative ADR are rejected. `GET /health`
says whether it's alive.

A Dockerfile is included, and GitHub Actions runs the tests on every push
(`.github/workflows/ci.yml`). If this ran in production, the first thing to watch would be cancel-rate
drift (see the 34% to 40% jump above), with scheduled retraining.

```bash
docker build -t hotel-risk . && docker run -p 8000:8000 -v $PWD/models:/app/models hotel-risk
```

## The copilot (agent)

You can also ask questions in plain English, for example "which bookings arriving 2017-07-15 are most likely
to cancel?", "how many should we oversell that day?" or "why is booking 100 risky?".

```bash
python -m hotel_risk.train data/raw/hotel_bookings.csv models/model.joblib   # once
python -m hotel_risk.agent.cli                    # chat in the terminal (rule router, no key needed)
python -m hotel_risk.agent.cli --claude           # let Claude pick the tools (needs ANTHROPIC_API_KEY)
python -m hotel_risk.agent.evaluate               # golden questions -> reports/agent_eval.json
streamlit run src/hotel_risk/agent/ui.py        # chat UI that also shows every tool call (pip install streamlit)
AGENT_PROVIDER=claude uvicorn hotel_risk.api:app  # POST /chat with {"message": "..."}
```

![Copilot UI answering an overbooking question, with the tool calls visible](reports/copilot_ui.png)

How it is built, and why it is boring on purpose:

- **Four tools, no free-form code or SQL.** `list_riskiest_bookings`, `overbooking_summary`,
  `get_booking_risk` (with the top three drivers from LightGBM's native `pred_contrib`) and `explain_policy`.
  They wrap the same calibrated model and cost-aware policy as everything above.
- **The model never does the maths.** The LLM only chooses a tool and phrases the answer. Every number must
  come from a tool.
- **A grounding check on every answer.** Numbers in the reply that no tool produced get flagged
  (`grounded: false` plus a visible warning). Tool errors, such as a date with no bookings, are returned as
  errors instead of being made up around.
- **Deterministic fallback.** With no API key, or if the API fails, a regex rule router answers from the same
  tools, so tests and CI never depend on a paid service.
- **Auditable UI.** The Streamlit page shows, under every answer, which tools ran, their arguments and raw
  results, which provider answered, and whether the grounding check passed.
- **Evaluated, not just demoed.** `evals/agent_cases.jsonl` checks tool choice, arguments, error handling and
  grounding. Tests run the same cases against a small synthetic model.

Limits worth knowing: bookings come from the historical dataset (a replay, not a live feed), the dollar
figures use the illustrative assumptions described above, and drivers are model sensitivities, not causes.

## Roadmap

- [x] Phase 1: scaffold, data cleaning, leakage audit, temporal split
- [x] Phase 2: baseline and gradient boosting, temporal evaluation
- [x] Phase 3: calibration and threshold selection
- [x] Phase 4: business-impact simulation
- [x] Phase 5: model packaging, API, tests, CI

## License

The code is under the MIT License (see `LICENSE`). That covers the code only. The dataset in
`data/raw/` is subject to the terms of its original publication (see the Data section).
