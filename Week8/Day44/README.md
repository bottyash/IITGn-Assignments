# Week 08 · Monday — Time Series Analysis
### PG Diploma AI-ML & Agentic AI Engineering | IIT Gandhinagar

---

## Problem Overview

Two forecasting problems worked through in this assignment:

1. **E-Commerce Sales Forecasting** — Predict daily order counts for a Brazilian e-commerce platform (Olist dataset) to help the inventory team plan buffer stock.
2. **Industrial Equipment Failure Prediction** — Predict whether warehouse pump equipment will fail in the next 24 hours based on 52 sensor signals, so the maintenance team can act proactively.

Neither dataset came clean. A large chunk of this work was characterising and fixing data issues before any modeling happened.

---

## Datasets

| Dataset | Source | Records | Key columns |
|---------|--------|---------|-------------|
| Olist E-Commerce | kaggle.com/olistbr/brazilian-ecommerce | ~99k orders | order_id, order_status, order_purchase_timestamp |
| Pump Sensor | kaggle.com/nphantawee/pump-sensor-data | 220,320 rows | timestamp, sensor_00…sensor_51, machine_status |

The e-commerce TS is built by aggregating delivered orders by day. The sensor data is minute-level readings from April–August 2018.

---

## Approach

### Sub-step 1 — E-Commerce EDA
- Built daily order count series (2017-01 to 2018-08) from raw Olist order CSVs
- Tested stationarity via ADF-style regression (t-stat = -8.4 → stationary around trend)
- Found strong weekly seasonality (ACF spike at lag=7) and a November holiday spike
- Identified data quality issues: sparse early 2016 data (excluded), missing dates filled with zero

### Sub-step 2 — Sensor Cleaning
- `sensor_15`: 100% missing → dropped
- `sensor_50`: 35% missing → dropped (too high to interpolate reliably)
- All remaining sensors with NaN gaps: forward-fill (appropriate for continuous physical signals)
- Result: zero missing values, 52 → 50 sensor columns, 220,320 clean rows

### Sub-step 3 — Baseline Forecasting Model
- Used **Ridge Regression** with lag features (lag-1, lag-7, lag-14), rolling averages, and cyclical day-of-week/month encodings
- Temporal hold-out: last 60 days (strict — no random split)
- Primary metric: **MAE** — directly actionable ("buffer X units per day")
- Test MAE: **40.7 orders/day**

### Sub-step 4 — Seasonal Model Comparison
- Added **Random Forest** with constrained depth (max_depth=6) to capture seasonal interactions
- Test MAE: **39.0 orders/day** — only 4.2% improvement over Ridge
- Conclusion: Ridge recommended for the ops team (simpler, interpretable, nearly same accuracy)

### Sub-step 5 — Failure Prediction (24h horizon)
- Created binary label: 1 = within 24 hours before a BROKEN event
- Features: raw sensor readings + 60-minute rolling mean & std for top 20 sensors
- Model: Random Forest with `class_weight='balanced'` (handles 95%/5% imbalance)
- Decision threshold: 0.3 (lowered to favour Recall — missing a failure >> false alarm)
- Test AUC: 0.57, **Recall: 0.29**, Precision: 0.014

### Sub-step 6 — Rule-based vs ML (Hard)
- Tested single-signal rule: sensor_04 rolling mean > 637.4
- Rule recall: 0.07 — roughly **4× worse** than the ML model
- The rule fires too late; ML uses drift patterns across multiple sensors hours earlier

### Sub-step 7 — Cost-Optimised Threshold (Hard)
- Cost matrix: FN = Rs 50,000 (missed failure), FP = Rs 2,000 (false alarm)
- Found that min-cost threshold (0.52) coincides with max-F1 threshold for this cost ratio
- Key insight: **F1 is not always the right production optimisation target** — when FN/FP cost asymmetry is large, always optimise on business cost directly

---

## How to Run

### Requirements

```
Python 3.10+
pandas
numpy
matplotlib
scikit-learn
scipy
```

No statsmodels or Prophet required — all TS analysis done with scipy + sklearn.

### Setup

```bash
git clone <repo_url>
cd week-08/monday/

# Install dependencies
pip install pandas numpy matplotlib scikit-learn scipy
```

### Data

Download datasets from Kaggle and place them in a `data/` folder (or update path constants at top of notebook):

- [Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) → all CSVs into `data/olist/`
- [Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) → `data/sensor.csv`

Update `OLIST_DIR` and `SENSOR_PATH` at the top of `Day44.ipynb` to match your paths.

### Run

```bash
jupyter notebook Day44.ipynb
# Run all cells top-to-bottom (Kernel → Restart & Run All)
```

Figures are saved to the project root automatically.

---

## Results Summary

| Sub-step | Method | Metric | Value |
|----------|--------|--------|-------|
| 3 | Ridge Regression (baseline) | Test MAE | 40.7 orders/day |
| 4 | Random Forest (seasonal) | Test MAE | 39.0 orders/day |
| 5 | RF Failure Classifier | AUC / Recall | 0.57 / 0.29 |
| 6 | Single-signal rule | Recall | 0.07 (4× worse than ML) |
| 7 | Cost-opt threshold | Min-cost ≈ Max-F1 | 0.52 |

---

## Repository Structure

```
Week8/
├── Day44.ipynb                  # Main analysis notebook
├── Day44.docx                   # Written report
├── README.md                    # This file
├── fig1_ecomm_overview.png
├── fig2_decomposition.png
├── fig3_sensor_cleaning.png
├── fig4_forecast_comparison.png
├── fig5_sensor_model.png
└── fig6_rule_vs_ml.png
```

---

## AI Usage

AI tools were used to assist with boilerplate code structure and debugging. Every section was verified, modified, and annotated by hand.

**Prompt used (Sub-step 5 labelling strategy):**
> "I have a sensor dataset with rare BROKEN events. How should I create a binary pre-failure label for the 24 hours before each event without data leakage?"

**Critique:** The AI suggested marking forward windows (next 24h). I changed this to backward windows (24h before BROKEN) which is more faithful to the deployment use case — the model sees past data only.

---

*Submission deadline: Tuesday 09:15 AM | Week 08 · Monday Assignment*
