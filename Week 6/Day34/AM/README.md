# D34 AM | Clustering (K-Means & DBSCAN) on Iris Dataset

**IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**  
**Week 6 | Day 34 | Unsupervised Learning**

---

## Problem Overview

Can an unsupervised algorithm rediscover biological species just from measurements — with no labels at all?

That's the core question here. We take the Iris dataset, strip the species labels, and apply K-Means, DBSCAN, and Hierarchical Clustering. Then we compare what the algorithms found with the actual ground truth species and measure how much they agree.

---

## Dataset

**Iris Dataset** (from `sklearn.datasets.load_iris`)

| Property | Value |
|----------|-------|
| Samples | 150 |
| Features | 4 (sepal length, sepal width, petal length, petal width) |
| True classes | 3 (setosa, versicolor, virginica) |
| Preprocessing | StandardScaler (zero mean, unit variance) |

Labels are **only used for evaluation**, never for training.

---

## Approach

### Part A — K-Means & DBSCAN
1. Load Iris, drop labels, apply `StandardScaler`
2. Run elbow method to confirm K=3
3. Apply `KMeans(n_clusters=3)` and evaluate with ARI + NMI
4. Build a cross-tabulation of predicted clusters vs true species
5. Visualize using PCA-projected 2D scatter (side-by-side comparison)
6. Apply DBSCAN with multiple eps values; analyze cluster count & noise

### Part B — Hierarchical Clustering (Self-Study)
7. Apply `AgglomerativeClustering(n_clusters=3, linkage='ward')`
8. Generate dendrogram using `scipy.cluster.hierarchy.dendrogram` (sample of 45 points)
9. Compare ARI with K-Means

### Part C — Interview Prep
- K-Means as a "greedy" algorithm + local minima explanation
- K-Means implemented from scratch using only NumPy
- Silhouette score interpretation (0.25 analysis)

### Part D — AI Analogy (Fruit Sorting)
- Prompted AI for a fruit-sorting analogy
- Verified accuracy, added critique, extended with failure cases

---

## How to Run

```bash
# Clone the repo
git clone <your-repo-url>

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn scipy

# Run the notebook
jupyter notebook Day-34-AM.ipynb
```

No additional datasets needed — everything uses sklearn's built-in Iris dataset.

---

## Results Summary

| Method | Clusters Found | ARI | Silhouette |
|--------|---------------|-----|------------|
| K-Means (K=3) | 3 | **0.6201** | 0.4599 |
| DBSCAN (eps=0.5) | 2 | N/A (2 clusters) | — |
| Agglomerative (Ward) | 3 | 0.6153 | — |

**Key finding:** Setosa is perfectly recovered by K-Means (50/50 samples in one cluster). Versicolor and Virginica overlap in feature space, which limits the ARI ceiling for any unsupervised method. DBSCAN identifies only 2 groups because these two species are not density-separated.

---

## Sample Outputs

- `elbow_method.png` — Inertia vs K curve with elbow at K=3
- `kmeans_vs_true.png` — PCA 2D scatter: true labels (left) vs K-Means (right)  
- `dbscan_vs_kmeans.png` — DBSCAN results vs K-Means side by side
- `dendrogram.png` — Hierarchical dendrogram (45-sample subset, Ward linkage)

---

## Key Observations

1. An ARI of 0.62 is strong for an algorithm that never saw the labels
2. The partial confusion between versicolor and virginica is a **data property**, not an algorithm failure
3. DBSCAN struggles here because the iris feature space has connected density — no clear density gap between species 1 and 2
4. The Ward-linkage dendrogram visually shows setosa branching off first, far from the other two

---

*Submitted as part of Day 34 take-home assignment — IIT Gandhinagar AI-ML PG Diploma Program*
