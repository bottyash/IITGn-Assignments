# Day 32 AM — Decision Trees & Random Forest
## Bank Loan Approval System

**Week 6 | Machine Learning & AI**

---

## Overview

This assignment builds a complete loan approval system using Decision Trees and Random Forest,
balancing model accuracy with regulatory interpretability requirements.

---

## Files in This Submission

| File | Description |
|------|-------------|
| `Day32-AM.ipynb` | Main solution notebook (all 4 parts) |
| `extra_trees_comparison.md` | Part B stretch: ExtraTrees vs RF research doc |
| `README.md` | This file |

### Generated outputs (created when notebook runs)
| File | Description |
|------|-------------|
| `dt_tree.png` | Decision Tree visualization (max_depth=4) |
| `confusion_matrices.png` | Side-by-side confusion matrices (DT vs RF) |
| `feature_importance.png` | MDI vs Permutation importance comparison |
| `rf_vs_et.png` | Random Forest vs Extra Trees benchmark |
| `bias_variance.png` | Bias-variance tradeoff diagram + overfitting curve |
| `overfitting_curve.png` | Train vs test accuracy across max_depths |
| `model_comparison_infographic.png` | AI-assisted non-technical comparison infographic |

---

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn scipy
jupyter notebook D32_AM_Solution.ipynb
```

**Python version:** 3.8+  
**Key libraries:** scikit-learn ≥ 1.1, numpy ≥ 1.21, pandas ≥ 1.3

---

## Part Breakdown

### Part A — Concept Application (40%)

**Task:** Bank loan approval system (2000 synthetic records)

**Features used:**
- `annual_income` — applicant's yearly income
- `credit_score` — 300–850 FICO-style score
- `loan_amount` — requested loan amount
- `employment_years` — years in current employment
- `debt_to_income` — ratio of monthly debt to monthly income
- `num_credit_cards` — number of open credit cards

**Steps:**
1. Synthetic data generation with realistic business-rule approval logic
2. Decision Tree (max_depth=4) + extract top 3 decision rules
3. Random Forest tuned with RandomizedSearchCV (5-fold CV, scoring=roc_auc)
4. Model comparison: accuracy, F1, ROC-AUC, interpretability
5. MDI vs Permutation importance analysis
6. 1-paragraph deployment recommendation

**Key results:**

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Decision Tree (d=4) | ~0.87 | ~0.86 | ~0.93 |
| Random Forest (tuned) | ~0.93 | ~0.92 | ~0.97 |

**Top 3 Decision Rules (DT):**
```
Rule 1: IF credit_score > 700 AND debt_to_income <= 0.35 → APPROVE
Rule 2: IF credit_score <= 700 AND employment_years > 5 → APPROVE
Rule 3: IF credit_score <= 650 → REJECT
```

**Recommendation:** Deploy RF as scoring engine + DT rules for regulatory explanations (hybrid architecture).

---

### Part B — Stretch Problem (30%): Extra Trees

See `extra_trees_comparison.md` for full analysis.

**Key findings:**
- Extra Trees is ~2.4× faster than Random Forest (random thresholds, no bootstrap)
- Accuracy/AUC gap is small (~1%) — negligible for most use cases
- Prefer Extra Trees for real-time/large-scale pipelines; prefer RF when accuracy is paramount

---

### Part C — Interview Ready (20%)

**Q1 — Bias-Variance Tradeoff:**
- A single full-depth Decision Tree has HIGH variance (changes dramatically with different training data)
- Random Forest reduces variance via bagging: averaging many uncorrelated trees smooths out individual tree errors
- Empirically demonstrated: bootstrap simulation shows RF reduces prediction variance by ~60% vs single DT

**Q2 — `plot_overfitting_curve(X, y, max_depths)`:**
```python
def plot_overfitting_curve(X, y, max_depths, test_size=0.2, seed=42):
    # Trains DTs at each depth, plots train vs test accuracy
    # Returns optimal depth (max test accuracy)
    ...
```

**Q3 — Debug (identical train/test accuracy 0.95):**
- NOT a problem — it is the ideal outcome (well-generalising model)
- `max_depth=3` strongly regularises each tree (prevents memorisation)
- Bagging further reduces variance
- **Investigate if:** suspiciously high (data leakage), or model seems too simple (try removing max_depth)

---

### Part D — AI-Augmented Task (10%)

Generated infographic comparing Logistic Regression / Decision Tree / Random Forest across:
- Interpretability, Accuracy, Training Speed, Handles Non-linearity

**Critique of AI output:**
- Interpretability scores are context-dependent (shallow DT is interpretable; deep DT is not)
- Added "Handles Non-linearity" dimension (missing from AI version)
- Added "Use When" guidance for non-technical readers

---

## Concepts Covered

| Concept | Where in notebook |
|---------|-------------------|
| Gini impurity / Entropy / Information Gain | Part A — DT training |
| Decision rule extraction | Part A — A2 |
| Overfitting & pruning (max_depth) | Part C — Q1, Q2 |
| Bootstrap & bagging | Part B, Part C — Q1 |
| Feature randomness (max_features) | Part A — RF tuning |
| RandomizedSearchCV + 5-fold CV | Part A — A3 |
| MDI vs Permutation importance | Part A — A5 |
| ExtraTreesClassifier | Part B |
| Bias-variance tradeoff | Part C — Q1 |
