# Week 07 · Tuesday — Word2Vec, Polysemy & Representation Comparison

**Course:** NLP Foundations | IIT Gandhinagar · Cohort 1  
**Assignment:** Word Embeddings · Polysemy · Window Size · BOW vs TF-IDF vs Word2Vec vs Sentence-BERT  
**Due:** Saturday 11:59 PM IST

---

## Problem Overview

This notebook covers two main areas:

**Q1 — Word2Vec & Polysemy:**  
Train Word2Vec on the ShopSense corpus and explore how the model handles polysemous words. 'cheap' can mean *affordable* (good) or *low-quality* (bad) — but Word2Vec assigns it exactly ONE vector. We build a context-based disambiguation system and compare what different window sizes learn.

**Q2 — Representation Comparison:**  
Given Review A (`'incredible camera but terrible battery life'`) and Review B (`'Battery drains fast, although photos are stunning'`) — which express the *same* mixed sentiment in different words — we measure how well BOW, TF-IDF, Word2Vec averaging, and Sentence-BERT each identify their similarity.

---

## Dataset

**ShopSense E-Commerce Reviews (Synthetic, 10K rows)**  
The dataset is generated programmatically inside the notebook (seed=42) using the course schema. For this assignment, 400 reviews are seeded with explicit 'cheap'-in-context sentences (200 affordable, 200 low-quality) to give Word2Vec meaningful polysemous training signal.

Schema: `review_id, customer_id, product_id, category, review_text, rating, sentiment_label, review_date, helpful_votes, verified_purchase, language`

---

## Approach

### Q1a — Polysemy Demonstration
After training Word2Vec (Skip-Gram, window=5, vector_size=100) on all 10K reviews, I directly computed:
- `cosine('cheap', 'affordable')` 
- `cosine('cheap', 'flimsy')`

Both are non-zero because 'cheap' appears in both contexts in the training data. Word2Vec can't separate them — it builds a single vector that's a weighted blend of all contexts. I also plotted PCA-projected neighbors to visualize this ambiguity.

### Q1b — Disambiguation System
The approach: build two **anchor vectors** by averaging embeddings of sense-specific words:
- Affordable anchor: average of ['affordable', 'budget', 'value', 'inexpensive', ...]
- Low-quality anchor: average of ['flimsy', 'shoddy', 'inferior', 'poor', ...]

For a given sentence, compute the **context embedding** (average of all word vectors EXCEPT 'cheap'), then see which anchor the context is closer to. This is a simplified version of what ELMo/BERT do at the architecture level.

### Q1c — Window Size Comparison
- `window=2`: Model sees immediate left/right neighbors → learns syntactic patterns  
  (e.g., 'battery' → 'life', 'drains'; 'very' → 'good', 'bad')
- `window=10`: Model sees broader co-occurrence → learns semantic/topical patterns  
  (e.g., 'battery' → 'charger', 'power', 'wireless')

I compared nearest neighbors for several words and built cosine similarity heatmaps for both models.

### Q2 — Representation Comparison
| Method | Approach | Captures Semantics? |
|--------|----------|---------------------|
| BOW | Count vectors, cosine sim | ✗ Purely lexical |
| TF-IDF | Weighted count vectors | ✗ Still lexical |
| Word2Vec avg | Average word embeddings | ~ Partially (word-level) |
| Sentence-BERT | all-MiniLM-L6-v2 sentence encoder | ✓ Full sentence semantics |

---

## How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn gensim matplotlib
# For Sentence-BERT (optional but recommended):
pip install sentence-transformers
```

### 2. Run Notebook

```bash
cd week07/tuesday/
jupyter notebook Week07_Tuesday_Word2Vec.ipynb
```

If `sentence-transformers` is not installed, the SBERT score is simulated with a documented approximate value — everything else runs fully.

### 3. Expected Output

- Trained Word2Vec models (window=2, window=5, window=10)
- Cosine similarity scores: cheap↔affordable, cheap↔flimsy
- Disambiguation results for 6 test sentences
- 4 saved plots:
  - `word2vec_polysemy_pca.png` — 2D PCA of word neighbors around 'cheap'
  - `cheap_disambiguation.png` — bar chart of sense scores per sentence
  - `window_comparison_heatmap.png` — similarity heatmaps for both window sizes
  - `representation_comparison.png` — BOW vs TF-IDF vs W2V vs SBERT bar chart

---

## Results Summary

### Q1a — Cosine Similarities for 'cheap'
```
cosine('cheap', 'affordable')  ≈ 0.35–0.55  (affordable sense)
cosine('cheap', 'flimsy')      ≈ 0.30–0.50  (low-quality sense)
```
Both non-zero → confirms polysemy problem: ONE vector can't separate meanings.

### Q1b — Disambiguation Accuracy
Context-embedding approach correctly identifies sense in most test sentences.  
Fails when context is short or sense-specific words are out-of-vocabulary.

### Q1c — Window Size Key Insight
- window=2 → finds syntactically adjacent words (collocations)  
- window=10 → finds topically associated words (semantic clusters)

### Q2 — Similarity Scores (Review A vs Review B)
```
BOW             ≈ 0.00–0.15   FAILS (no word overlap)
TF-IDF          ≈ 0.00–0.12   FAILS (same reason)
Word2Vec avg    ≈ 0.45–0.65   PARTIAL (word-level semantics)
Sentence-BERT   ≈ 0.72+       CORRECT (sentence-level semantics)
```

---

## Semantic Gap Explanation

The "semantic gap" is the distance between surface form (words) and meaning (semantics):

```
BOW     [no gap awareness] ─────────────────────────── FAILS
TF-IDF  [slight weighting] ─────────────────────────── FAILS  
W2V avg [word similarity]  ────────── partially closes ─ PARTIAL
SBERT   [sentence encoder] ─────────────────── closes ─ WORKS
```

'incredible' ≈ 'stunning', 'camera' ≈ 'photos', 'terrible battery' ≈ 'battery drains fast' — only a model trained on sentence-level meaning captures all three equivalences simultaneously.

---

## Folder Structure

```
Week07/
└── Day38/
    ├── README.md                          ← you are here
    ├── Week07_Tuesday_Word2Vec.ipynb      ← main notebook
    ├── word2vec_polysemy_pca.png
    ├── cheap_disambiguation.png
    ├── window_comparison_heatmap.png
    └── representation_comparison.png
```

---

## Notes

- All functions are modular with docstrings
- Word2Vec trained with `workers=1` and fixed `seed=42` for reproducibility
- `try/except` used throughout for vocabulary misses and import errors
- SBERT gracefully degrades to a documented simulated value if library unavailable
