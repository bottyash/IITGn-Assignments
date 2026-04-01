# Day 32 PM — Decision Trees & Random Forest: Applied
## Insurance Fraud Detection Case Study

**Week 6 | Machine Learning & AI**

---

## Overview

This assignment builds an end-to-end fraud detection system for an insurance company,
balancing recall optimisation (FN cost = 10× FP cost) with regulatory interpretability requirements.

---

## Files in This Submission

| File | Description |
|------|-------------|
| `D32_PM_Solution.ipynb` | Main solution notebook (all 4 parts) |
| `README_PM.md` | This file |

### Generated outputs (created when notebook runs)
| File | Description |
|------|-------------|
| `dt_fraud_tree.png` | Decision Tree (max_depth=5) visualization |
| `roc_curves.png` | ROC curves comparison (DT vs RF) |
| `cost_analysis.png` | FP vs FN cost breakdown per model |
| `n_estimators_tradeoff.png` | AUC and train time vs number of trees |

---

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn scipy
jupyter notebook D32_PM_Solution.ipynb
```

**Python version:** 3.8+  
**Key libraries:** scikit-learn ≥ 1.1, numpy ≥ 1.21, pandas ≥ 1.3

---

## Part Breakdown

### Part A — Concept Application (40%)

**Scenario:** Insurance fraud prediction with cost-sensitive evaluation

**Synthetic Dataset (3000 records):**

| Feature | Description |
|---------|-------------|
| `claim_amount` | Dollar value of the insurance claim |
| `policy_age_months` | How long the policy has been active |
| `num_prev_claims` | Number of prior claims by this customer |
| `days_since_policy` | Days since the policy was issued |
| `premium_amount` | Annual premium paid |
| `customer_age` | Age of the policyholder |
| `num_witnesses` | Witnesses at the incident |
| `claim_to_premium` | Ratio of claim amount to premium |
| `incident_hour` | Hour of day the incident was reported |
| `is_night_claim` | Binary: filed between 10 PM – 6 AM |

**Steps:**
1. Synthetic data with realistic fraud rate (~12%)
2. Decision Tree (max_depth=5, class_weight='balanced') + top 3 fraud rules
3. RF tuned with RandomizedSearchCV (scoring=**recall**, 5-fold CV)
4. Comprehensive metrics table (accuracy, precision, recall, F1, ROC-AUC, CV scores)
5. Cost-sensitive analysis (FN=10×FP)
6. 2-paragraph deployment recommendation

**Key Results:**

| Model | Recall | ROC-AUC | Total Cost |
|-------|--------|---------|------------|
| Decision Tree (d=5) | ~0.71 | ~0.84 | Higher |
| Random Forest (tuned, recall) | ~0.84 | ~0.89 | Lower |

**Top 3 Fraud Indicator Rules (DT):**
```
Rule 1: IF claim_to_premium > 10 AND num_prev_claims > 3 → FLAG FRAUD
Rule 2: IF days_since_policy <= 30 AND claim_amount > 15000 → FLAG FRAUD
Rule 3: IF num_witnesses == 0 AND is_night_claim == 1 → FLAG FRAUD
```

**Recommendation:** RF for automated scoring (AUC ≈ 0.89, Recall ≈ 0.84) + DT rules for human review.
Use threshold ~0.35 (below default 0.5) to maximise recall given the 10× FN cost penalty.

---

### Part B — Stretch Problem (30%): Gradient Boosting Preview

**Bagging vs Boosting in one paragraph:**

> Bagging (Random Forest) builds many trees **in parallel**, each on a random bootstrap sample,
> and averages predictions — this reduces **variance**. Boosting builds trees **sequentially**,
> each correcting the residual errors of the previous ensemble — this reduces **bias** but risks
> overfitting without careful regularisation.

**Recommended resource:**
- StatQuest: *"Gradient Boost Part 1: Regression Main Ideas"*
  → https://www.youtube.com/watch?v=3CC4N4z3GJc

---

### Part C — Interview Ready (20%)

**Q1 — 1000 Trees vs 100 Trees:**

| Factor | 100 Trees | 1000 Trees |
|--------|-----------|------------|
| ROC-AUC | ~0.890 | ~0.892 (+0.002) |
| Train time | 1× | ~10× |
| Prediction latency | 1× | ~10× |
| RAM usage | 1× | ~10× |

**Verdict:** 100–200 trees is the production sweet spot. Plot the learning curve and pick the elbow.
1000 trees only justified when diagnosing instability (large σ in CV scores).

**Q2 — `compare_models(X, y, models_dict)`:**
```python
def compare_models(X, y, models_dict, n_splits=5, seed=42):
    # 5-fold stratified CV per model
    # Returns DataFrame with mean/std of accuracy, F1, training time
    ...
```

**Q3 — Debug (feature importances differ between runs):**
- **Root cause:** No `random_state` set → different bootstrap samples + random feature subsets each run
- With only 10 trees, randomness dominates; importance scores are non-deterministic
- **Fix:** Always pass `random_state=<int>`
- **Bonus:** Even with fixed seed, MDI importances are biased for high-cardinality features — use permutation importance as the reliable alternative

---

### Part D — AI-Augmented Task (10%)

**OOB Error Explanation (Non-technical analogy):**
> Imagine hiring 500 teachers, each marking a random 63% of exam papers. The papers each
> teacher *didn't* see are their "out-of-bag" samples. After training, each tree only predicts
> on its OOB samples — averaging these gives a validation estimate without needing a held-out set.

**AI Response Verification:**
- ✅ ~63% bootstrap coverage confirmed (1 - 1/e ≈ 63.2%)
- ✅ OOB predictions aggregate only trees that excluded that sample
- ✅ OOB score ≈ cross-validation error (empirically verified in notebook)

**When OOB differs significantly from test error:**
1. Small datasets (noisy OOB sets)
2. Distribution shift between train and test periods
3. Data leakage in features
4. Severe class imbalance

---

## Key Concepts Covered

| Concept | Where in notebook |
|---------|-------------------|
| DT vs RF comparison | Part A |
| Interpretability vs accuracy tradeoff | Part A, Recommendation |
| Hyperparameter tuning (RandomizedSearchCV) | Part A — A3 |
| Recall-optimised tuning | Part A — A3 |
| Cost-sensitive evaluation | Part A — A5 |
| MDI vs Permutation importance | Part A |
| Gradient Boosting (preview) | Part B |
| n_estimators tradeoff | Part C — Q1 |
| compare_models utility function | Part C — Q2 |
| Reproducibility (random_state) | Part C — Q3 |
| OOB error estimation | Part D |

