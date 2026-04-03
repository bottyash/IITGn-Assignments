# D33 | PM Session — SVM, KNN & Full Week 6 Algorithm Comparison

**IIT Gandhinagar | PG Diploma in AI-ML & Agentic AI Engineering**  
Week 6, Day 33 | Machine Learning & AI

---

## Problem Overview

Build a comprehensive ML cheat-sheet notebook covering **all 8 algorithms** from Week 6. Each algorithm gets a structured "card" — when to use, key hyperparameters, pros/cons, and a working 5-line code snippet. All 8 models are then tested on a single dataset under fair, reproducible conditions. The assignment also covers text classification with SVM + TF-IDF, and three interview-style deep-dive questions.

---

## Dataset

### Part A — Breast Cancer Wisconsin
- **Source:** `sklearn.datasets.load_breast_cancer`
- **Size:** 569 samples, 2 classes (malignant / benign)
- **Features:** 30 numerical features (tumor measurements)
- **Split:** 80/20 stratified train-test

Why this dataset? It's clean, well-balanced, and large enough that all 8 algorithms produce meaningful results. Medical diagnosis context makes the comparison story easier to explain.

### Part B — 20 Newsgroups (Text)
- **Source:** `sklearn.datasets.fetch_20newsgroups`
- **Categories:** alt.atheism, comp.graphics, sci.med, soc.religion.christian
- **Train:** 2,257 documents | **Test:** 1,502 documents
- **Features:** TF-IDF vectors (~35K dimensions after filtering)

---

## The 8 Algorithms (Week 6 Cheat Sheet)

| # | Algorithm | When to Use | Key Params |
|---|-----------|------------|-----------|
| 1 | Logistic Regression | Need probabilities, linear data, fast baseline | C, penalty, solver |
| 2 | Decision Tree | Interpretability, mixed types | max_depth, min_samples_split |
| 3 | Random Forest | Robust baseline, feature importance | n_estimators, max_features |
| 4 | Gradient Boosting | Best tabular accuracy, Kaggle | n_estimators, learning_rate |
| 5 | AdaBoost | Weak learner boosting, clean data | n_estimators, learning_rate |
| 6 | SVM (RBF) | Small/medium data, non-linear | C, gamma, kernel |
| 7 | KNN | Instance-based, quick prototype | n_neighbors, metric |
| 8 | Naive Bayes | Text, high-dimensional, speed | var_smoothing |

---

## My Approach

### Part A — Fair Comparison
The key word here is **fair**. All 8 models were evaluated using the exact same setup:
- Same 5-fold Stratified CV (same fold splits via `random_state=42`)
- Scale-sensitive algorithms (LR, SVM, KNN) got StandardScaler; tree-based ones did not
- Same train-test split throughout

I also built a `model_selection_report()` function (Part C Q2) that automates this pipeline for any dataset — it returns a formatted DataFrame with CV stats and paired t-test p-values to identify statistically significant differences between models.

### Part B — Text Classification
Text is where SVM really shines. TF-IDF vectors are extremely high-dimensional and sparse — SVM's LinearSVC handles this efficiently without needing dense kernel computation. The pipeline is just two steps: TfidfVectorizer → LinearSVC, with headers/footers stripped to make it a genuine content-based task.

### Part C — Deep-dive Questions
The Q1 (100 features, 50 samples) was interesting — the p>>n regime breaks most of the algorithms we use routinely. KNN completely fails (curse of dimensionality), tree-based methods memorise, and only regularised linear models remain viable.

The Q2 function includes a paired t-test against the best model's fold scores — this tells you not just *who won* but whether the differences are statistically meaningful or just noise.

---

## Steps to Run

**Prerequisites:**
```bash
pip install scikit-learn numpy pandas matplotlib seaborn scipy
```

**Run the notebook:**
```bash
jupyter notebook D33_PM_SVM_KNN_CS.ipynb
```

All cells should be run in order. The 20 Newsgroups fetch requires an internet connection on first run (it downloads ~14 MB and caches). Subsequent runs are offline.

---

## Results Summary

### Part A — 8-Algorithm Ranking (Breast Cancer Dataset)

| Rank | Model | CV Accuracy | Test Accuracy |
|------|-------|------------|--------------|
| 1 | **Gradient Boosting** | **97.36% ± 1.28%** | 97.37% |
| 2 | SVM (RBF) | 97.14% ± 1.43% | 96.49% |
| 3 | Random Forest | 96.92% ± 1.62% | 97.37% |
| 4 | Logistic Regression | 96.70% ± 1.92% | 97.37% |
| 5 | AdaBoost | 96.48% ± 2.01% | 96.49% |
| 6 | KNN (K=5) | 95.82% ± 1.83% | 95.61% |
| 7 | Decision Tree | 93.41% ± 3.12% | 92.98% |
| 8 | Naive Bayes | 92.53% ± 2.25% | 93.86% |

**Recommendation:** Gradient Boosting wins on CV mean with the lowest variance. However, if deployment speed matters (production inference), Logistic Regression achieves the same test accuracy in 0.021s vs 1.42s.

### Part B — Text Classification

| Model | Test Accuracy |
|-------|--------------|
| TF-IDF + LinearSVC | **90.21%** |
| TF-IDF + Logistic Regression | 89.35% |

LinearSVC wins by 0.86% — consistent with the literature (SVM was THE standard for text classification before deep learning).

---

## Key Takeaways

1. **No single best algorithm** — the winner depends on data size, feature type, whether you need probabilities, and inference speed constraints
2. **Gradient Boosting** is the most reliable general-purpose algorithm for tabular data
3. **LinearSVC** is uniquely suited for text (high-dimensional, sparse TF-IDF)
4. **p >> n scenario** (100 features, 50 samples): only regularised linear models survive — tree-based and KNN fail badly
5. **Paired t-test** is the right tool for determining if model differences are statistically significant, not just "Model A got 97.5% and Model B got 97.1%"
6. **AI-generated decision guides are useful starting points** but miss important edge cases: imbalanced classes, ordinal targets, concept drift in streaming data

---

## Sample Outputs

Generated plots (saved as PNG):
- `all8_comparison.png` — Grouped bar chart with CV error bars for all 8 algorithms
- `text_classification.png` — Per-class F1 + overall accuracy for LinearSVC vs LogReg
- `algorithm_selection_guide.png` — Visual decision tree for algorithm selection

---

## File Structure

```
D33_PM_SVM_KNN_CS/
├── D33_PM_SVM_KNN_CS.ipynb         # Main notebook
├── README_PM.md                    # This file
├── D33_PM_SVM_KNN_CS_Report.docx   # Full written report
└── *.png                           # Generated visualisations
```

---

*IIT Gandhinagar | PG Diploma in AI-ML & Agentic AI Engineering | Week 6*
