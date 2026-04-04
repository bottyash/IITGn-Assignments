# D34 PM | PCA, Clustering & Week 6 Comprehensive Review

**IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**  
**Week 6 | Day 34 | Machine Learning & AI**

---

## Problem Overview

This notebook is a complete **Week 6 personal ML reference** covering all 13 algorithms/techniques taught this week, plus a stretch task on image compression with PCA. The goal is to have a single, working reference notebook that can serve as interview prep and a quick-start template for future projects.

---

## Algorithms Covered

| # | Algorithm | Type | Use Case |
|---|-----------|------|----------|
| 1 | Logistic Regression | Supervised | Churn prediction |
| 2 | Decision Tree | Supervised | Medical diagnosis |
| 3 | Random Forest | Supervised | House price prediction |
| 4 | AdaBoost | Supervised | Face detection |
| 5 | XGBoost | Supervised | Credit scoring |
| 6 | LightGBM | Supervised | Real-time recommendations |
| 7 | Voting Classifier | Ensemble | Medical imaging |
| 8 | Stacking | Ensemble | Kaggle competitions |
| 9 | SVM | Supervised | Text classification |
| 10 | KNN | Supervised | Recommendation systems |
| 11 | K-Means | Unsupervised | Customer segmentation |
| 12 | DBSCAN | Unsupervised | Geospatial hotspot detection |
| 13 | PCA | Dimensionality Reduction | Eigenfaces, compression |

---

## Dataset

**Wine Dataset** (from `sklearn.datasets.load_wine`) for code verification

| Property | Value |
|----------|-------|
| Samples | 178 |
| Features | 13 (alcohol, ash, flavanoids, etc.) |
| Classes | 3 wine cultivars |
| Preprocessing | StandardScaler |

---

## Approach

### Part A — Algorithm Reference Notebook
- Each algorithm has: 1-line description, when to use, 3-line code example, 2 key hyperparameters, 1 use case
- Text-based algorithm selection flowchart
- 3 algorithms (LR, RF, XGBoost) tested on Wine with 5-fold CV to verify all snippets work

### Part B — Image Compression with PCA (Stretch)
- Synthetic 128×128 RGB image created with gradients and noise
- PCA applied separately to each color channel
- Compressed with n_components = 5, 20, 50, 100
- MSE and compression ratio computed for each level

### Part C — Interview Prep
- Full ML pipeline (1000 samples, 200 features) with 3 algorithm choices explained
- `weekly_model_comparison()` function with optional PCA preprocessing
- Analysis of why PCA can hurt accuracy (3 reasons)

### Part D — AI-Verified Study Guide
- Prompted for structured study guide, verified each section, identified missing concepts

---

## How to Run

```bash
# Clone repo
git clone <your-repo-url>

# Install dependencies  
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn

# Run notebook
jupyter notebook Day-34-PM.ipynb
```

---

## Results Summary

### Wine Dataset — 5-Fold CV Comparison

| Model | CV Accuracy | Std |
|-------|------------|-----|
| XGBoost | ~0.98 | ±0.02 |
| Random Forest | ~0.98 | ±0.02 |
| Logistic Regression | ~0.97 | ±0.02 |

### PCA Image Compression Trade-off

| Components | MSE | Compression Ratio |
|-----------|-----|------------------|
| 5 | High | ~12x |
| 20 | Medium | ~4x |
| 50 | Low | ~1.8x |
| 100 | Very Low | ~1.0x |

---

## Sample Outputs

- `pca_scree_wine.png` — Scree plot showing Wine dataset PCA (bar + cumulative line)
- `wine_comparison.png` — Bar chart comparing LR vs RF vs XGBoost CV accuracy
- `pca_image_compression.png` — Grid showing original vs 4 compression levels

---

## Key Takeaways

1. **Always scale** before LR, SVM, KNN, PCA, K-Means — they're all distance-sensitive
2. **XGBoost and LightGBM** are the best out-of-box choices for tabular data
3. **PCA at 95% variance** is usually safe, but watch out: discriminative information ≠ high-variance information
4. **Stacking > Voting** for accuracy; **Voting** for simplicity and robustness
5. For clustering: K-Means when K is known; DBSCAN when shape is arbitrary and noise matters

---

*Submitted as part of Day 34 take-home assignment — IIT Gandhinagar AI-ML PG Diploma Program*
