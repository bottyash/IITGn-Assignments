# Day 40 — Week 7 Friday: NLP Evaluation & Production Readiness

**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**

---

## Problem Overview

ShopSense is preparing to launch a review intelligence feature. A previous team's sentiment classifier reported 94% test accuracy and was greenlit for production — but the customer support team has since flagged that it's predicting "Positive" for nearly every review, including confirmed 1-star complaints.

This assignment walks through:
- Why accuracy alone is a dangerous metric for imbalanced datasets
- How to properly evaluate a sentiment classifier
- Constraint testing (new categories, multilingual, latency)
- Building a business cost model for production deployment
- Reproducing and diagnosing the broken pipeline
- Writing a production monitoring specification

---

## Dataset Description

**ShopSense Reviews** (`shopsense_reviews__1_.csv`)
- 10,199 product reviews across 6 categories: Home, Books, Electronics, Clothing, Beauty, Food
- 20 columns including: `review_text`, `sentiment_label`, `rating`, `language`, `category`
- Sentiment classes: Positive (69.9%), Negative (20.4%), Neutral (9.7%) — imbalanced
- Languages: English (80.4%), Hindi (9.8%), Code-mixed (9.8%)
- Some reviews contain HTML tags (scraped data) — cleaned in preprocessing

**ShopSense Customers** (`shopsense_customers__1_.csv`)
- 50,000 customer profiles — used for context (acquisition channel, churn, spend, etc.)

---

## Approach

Honestly, when I first looked at the data I wasn't surprised the previous team got tripped up — 70% of the reviews are Positive, so any model that learns to just predict Positive would score well on accuracy. The real challenge is catching the 20% Negative reviews, because those are the complaints that matter to the business.

**My pipeline:**
1. Strip HTML from review text, remove special characters, lowercase
2. TF-IDF with bigrams (captures phrases like "not good", "very bad")
3. Train two models: Logistic Regression and Naive Bayes
4. Evaluate with Macro F1 (treats all 3 classes equally), not accuracy
5. Test both models against the three engineering constraints
6. Build a cost model to translate errors into business impact
7. Reproduce the broken pipeline and show before/after metrics

---

## Steps to Run

**Python version:** 3.10+

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Place the CSV files in the same folder as the notebook:**
```
week-07/friday/
├── Day40.ipynb
├── shopsense_reviews__1_.csv
└── shopsense_customers__1_.csv
```

**Run the notebook:**
```bash
jupyter notebook Day40.ipynb
```

Or run all cells in order (Kernel → Restart & Run All).

---

## Results Summary

| Metric | Logistic Regression | Naive Bayes |
|--------|--------------------:|------------:|
| Accuracy | 98.83% | 98.33% |
| Macro F1 | **0.9829** | 0.9777 |
| Negative Recall | 96.72% | 97.54% |
| Neutral Recall | 94.86% | 94.86% |
| Positive Recall | 100.00% | 99.04% |
| Mean Inference (ms) | 0.87 | 0.92 |
| P99 Latency (ms) | < 5 | < 5 |
| Passes 20ms constraint | ✓ | ✓ |

**Constraint Testing:**

| Constraint | LR Result | NB Result |
|-----------|-----------|-----------|
| New categories (F1 drop) | 0.0008 | 0.0018 |
| Hindi Macro F1 | 0.9767 | 0.9706 |
| Code-mixed Macro F1 | 0.9878 | 0.9805 |

**Cost Model (100k reviews/day):**

| Model | FN Cost/Day | FP Cost/Day | Total/Day |
|-------|------------:|------------:|----------:|
| Broken Pipeline | ~$16,500+ | — | > $16,500 |
| Logistic Regression | $10,033 | $0 | **$10,033** |
| Naive Bayes | $7,525 | $1,336 | $8,861 |

**Recommendation:** Deploy Logistic Regression. Zero false-positive rate means no unnecessary disruption to happy customers. Monitor Macro F1 weekly (retrain threshold: 0.90) and Negative Recall (retrain threshold: 0.85).

---

## Sample Outputs

- `fig1_class_dist.png` — Sentiment class distribution (bar + pie)
- `fig3_constraints.png` — Constraint testing results across 3 engineering constraints
- `fig4_cost_model.png` — Daily misclassification cost breakdown
- `fig5_pipeline_fix.png` — Broken vs fixed pipeline confusion matrix comparison
- `fig6_final_comparison.png` — Macro F1 vs daily cost scatter plot

---

## The Broken Pipeline — What Went Wrong

The 94% accuracy number was real, but misleading. The team:
1. Used accuracy as the sole evaluation metric (hides imbalance)
2. Didn't strip HTML from review text
3. Didn't check per-class recall before deployment
4. Didn't monitor prediction distribution post-launch

A simple Macro F1 check during evaluation would have shown the model's Negative Recall was poor — and the problem would never have made it to production.

---

## AI Usage

Parts of this notebook were developed with AI assistance (code suggestions, phrasing of non-technical summaries).

**Prompt used for Sub-step 4 cost model framing:**
> "What are realistic business cost estimates for false negatives vs false positives in an e-commerce review sentiment system? FN = genuine complaint predicted positive, FP = positive review predicted negative."

**Critique:** The AI suggested $10-20 for FN and $1-3 for FP — I used $15 and $2 after adjusting for ShopSense's scale (100k reviews/day means even small cost differences compound). The core ratio (FN ~7-8x more expensive than FP) matched my own intuition about complaint handling costs.

---

## Commit History

```
feat: add data loading, preprocessing, and class distribution analysis
feat: train LR and NB models, add macro F1 evaluation  
feat: add constraint tests for category/language/latency
feat: add cost model and daily misclassification projections
feat: add broken pipeline reproduction and vulnerability check
```
