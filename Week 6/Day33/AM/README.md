# D33 | AM Session — SVM & KNN: Handwritten Digit Classifier

**IIT Gandhinagar | PG Diploma in AI-ML & Agentic AI Engineering**  
Week 6, Day 33 | Machine Learning & AI

---

## Problem Overview

Build and evaluate a handwritten digit classifier using two classic ML algorithms — **SVM with RBF kernel** and **K-Nearest Neighbours** — on sklearn's `load_digits` dataset (a smaller 8×8 pixel version of the famous MNIST benchmark). The assignment also benchmarks FAISS (Facebook's similarity search library) against sklearn's KNN for speed.

---

## Dataset

- **Source:** `sklearn.datasets.load_digits`
- **Size:** 1,797 samples, 10 classes (digits 0–9)
- **Features:** 64 pixel intensity values (8×8 grayscale image, flattened)
- **Class balance:** ~178–183 samples per class (well balanced)
- **Pixel range:** 0–16 (not the usual 0–255, this is a lower-res version)

The dataset is small enough to train on a laptop but representative enough to reveal interesting confusion patterns between visually similar digits.

---

## My Approach

I wanted to go beyond just fitting models — the goal was to understand *why* the models behave the way they do. Here's how I structured the solution:

### Part A — Building the Classifiers

**Preprocessing:**  
Split the data into 80/20 train-test first, *then* fit the scaler only on train data. This is important — fitting the scaler on the full dataset would be data leakage. StandardScaler brings all 64 pixel features to mean=0, std=1, which is essential for SVM's RBF kernel and KNN's distance computations.

**SVM (RBF kernel) with GridSearchCV:**  
Tuned `C` ∈ {0.1, 1, 10, 100} and `gamma` ∈ {0.0001, 0.001, 0.01, 0.1} using 5-fold cross-validation. Best params: **C=10, gamma=0.001**.

**KNN:**  
Swept K from 1 to 20 using 5-fold CV on the training set. K=3 gave the best cross-validation accuracy (0.9819). Used Euclidean distance throughout.

**Confusion Analysis:**  
After getting predictions, I looked specifically at which digit pairs get mixed up most. The patterns (3↔8, 4↔9, 1↔7) make intuitive sense when you look at the 8×8 images — these pairs genuinely look similar at low resolution.

### Part B — FAISS Benchmark

Implemented KNN search using FAISS's `IndexFlatL2` (exact L2 search) and compared query time against sklearn's `BallTree`-based KNN for 1000 queries.

Key finding: **FAISS is ~7.6× faster** with essentially identical accuracy. The real advantage of FAISS shows up at scale — for millions of vectors it uses approximate indexing (IVF, HNSW) for 50–100× speedup.

### Part C — Interview Questions

- Explained the SVM vs LR distinction (margin maximisation vs likelihood maximisation)
- Implemented KNN from scratch using only NumPy (vectorised distance computation)
- Debugged a broken SVM — root cause was missing feature scaling (salary vs age scale mismatch)

### Part D — AI-Augmented Visualisation

Generated a 5-panel visualisation showing how SVM's decision boundary and margin change as C goes from 0.01 to 100 (using PCA-reduced 2D data for the digit 0 vs 8 binary problem). Verified the kernel trick analogy for accuracy.

---

## Steps to Run

**Prerequisites:**
```bash
pip install scikit-learn numpy matplotlib seaborn faiss-cpu
```

**Run the notebook:**
```bash
jupyter notebook D33_AM_SVM_KNN.ipynb
```

Run all cells in order. The notebook is designed to execute sequentially — each section builds on the previous one.

**Note on FAISS:** If `faiss-cpu` installation fails (rare on some systems), the benchmark section will print a fallback message. The rest of the notebook runs fine without it.

---

## Results Summary

| Model | Configuration | Test Accuracy |
|-------|--------------|---------------|
| SVM | RBF kernel, C=10, gamma=0.001 | **98.33%** |
| KNN | K=3, Euclidean distance | **97.22%** |

### Most Confused Digit Pairs (both models)

| Pair | Why it's confusing |
|------|-------------------|
| 3 ↔ 8 | Similar curved strokes, especially in low-res |
| 4 ↔ 9 | Closed loop at top looks similar |
| 1 ↔ 7 | Near-vertical strokes, hard to distinguish |

### FAISS vs sklearn KNN (1000 queries)

| Method | Time | Accuracy |
|--------|------|----------|
| sklearn KNN | 31.2 ms | 97.40% |
| FAISS (FlatL2) | 4.1 ms | 97.20% |
| **Speedup** | **~7.6×** | 0.2% loss |

---

## Key Takeaways

1. **Always scale features** before SVM or KNN — unscaled data breaks both algorithms  
2. **SVM > KNN** on this dataset (~1.1% accuracy gap) because the RBF kernel creates a richer decision surface  
3. **FAISS** is the production choice for similarity search — same accuracy, dramatically faster  
4. **Confused digit pairs** are not random — they reflect real visual ambiguity  

---

## Sample Outputs

The notebook generates the following plots (saved as PNG):
- `sample_digits.png` — Example images from each class
- `gridsearch_heatmap.png` — C vs gamma CV accuracy heatmap
- `knn_k_selection.png` — K vs accuracy curve with optimal K marked
- `confusion_matrices.png` — Side-by-side confusion matrices for SVM and KNN
- `f1_comparison.png` — Per-class F1 score comparison
- `misclassified_examples.png` — Actual misclassified digit images
- `faiss_benchmark.png` — Speed and accuracy comparison bar chart
- `svm_c_boundary.png` — Decision boundary visualisation for 5 C values

---

## File Structure

```
D33_AM_SVM_KNN/
├── Day-33-AM.ipynb     # Main notebook
├── README.md                # This file
├── Day-33-AM-Report.docx  # Full written report
└── *.png                    # Generated visualisations
```

---

*IIT Gandhinagar | PG Diploma in AI-ML & Agentic AI Engineering | Week 6*
