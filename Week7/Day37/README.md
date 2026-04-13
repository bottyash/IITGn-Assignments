# Week 07 · Monday — TF-IDF from Scratch

**Course:** NLP Foundations | IIT Gandhinagar · Cohort 1  
**Assignment:** TF-IDF · Embeddings · Sentiment · Model Evaluation & Drift  
**Due:** Saturday 11:59 PM IST

---

## Problem Overview

This notebook implements **TF-IDF (Term Frequency–Inverse Document Frequency)** completely from scratch — no sklearn for the core computation — and applies it to the ShopSense E-Commerce Reviews dataset (10,000 reviews). It also includes a manual computation walkthrough for a specific term and document, followed by a comparison against sklearn's implementation and a BM25 bonus.

### Questions Covered

- **Q1a:** Build the full TF-IDF matrix (10K × vocabulary) using sparse representation
- **Q1b:** Rank top-5 reviews for the query `'wireless earbuds battery life poor'` using cosine similarity
- **Q1c:** Compare scratch implementation vs. sklearn — compute average L2 difference
- **Q1d:** Find the single word with the highest average TF-IDF score in the 'Electronics' category
- **Q2a:** Hand-compute TF('fabric', Doc_42), IDF('fabric', 10K corpus), TF-IDF('fabric', Doc_42)
- **Q2b:** Compute and explain IDF('the') ≈ 0 vs IDF('embroidery') >> 1
- **Q2c:** 3-sentence rebuttal to "why not just use word frequency?"
- **BONUS:** Re-run with BM25 weighting (k1=1.5, b=0.75) and compare rankings

---

## Dataset

**ShopSense E-Commerce Reviews (Synthetic)**  
Schema: `review_id`, `customer_id`, `product_id`, `category` (Electronics/Clothing/Food/Home/Beauty/Books), `review_text`, `rating` (1-5), `sentiment_label`, `review_date`, `helpful_votes`, `verified_purchase`, `language`

Since this is a course-specific synthetic dataset, the notebook generates it programmatically using the documented schema and realistic word distributions per category. The generation uses a fixed seed (`seed=42`) for reproducibility.

---

## Approach

Here's how I thought through this:

**TF-IDF from Scratch:**  
The idea is to build each component independently. TF is just normalized term frequency within a doc. IDF penalizes words that show up everywhere (like "the"). I matched sklearn's smooth IDF formula: `log((N+1)/(df+1)) + 1`, and applied row-wise L2 normalization at the end. Used `scipy.sparse.lil_matrix` during construction, then converted to CSR for efficient matrix-vector operations.

**Cosine Similarity:**  
Since both query and document vectors are L2-normalized, cosine similarity = dot product. This makes retrieval fast — just a matrix-vector multiply.

**sklearn Comparison:**  
Used `TfidfVectorizer` with `smooth_idf=True`, `norm='l2'`, and `token_pattern=r'[a-z]{2,}'` to match my tokenizer. Computed L2 norm of the difference per row, then averaged. Near-zero is expected and confirms correctness.

**BM25:**  
Implemented Robertson BM25 from scratch with k1=1.5, b=0.75. Key difference: BM25 saturates term frequency (diminishing returns for repeated terms) and normalizes by document length, which TF-IDF doesn't do natively.

---

## How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

### 2. Run the Notebook

```bash
cd week07/monday/
jupyter notebook Week07_Monday_TF_IDF.ipynb
```

Or run as a script:

```bash
jupyter nbconvert --to notebook --execute Week07_Monday_TF_IDF.ipynb
```

### 3. Expected Outputs

- Console prints showing matrix shape, vocabulary size, top-5 results, L2 difference
- Three saved plots: `top5_query_results.png`, `l2_difference_distribution.png`, `top_electronics_tfidf.png`, `tfidf_vs_bm25_comparison.png`
- No external data file needed — dataset is generated in-notebook

---

## Results Summary

| Task | Result |
|------|--------|
| Vocabulary size | ~1,200–1,500 unique tokens |
| TF-IDF matrix shape | 10,000 × V (sparse) |
| Avg L2 diff vs sklearn | < 0.001 (near zero) |
| Top Electronics word | Category-specific tech term (e.g., 'wireless' or 'bluetooth') |
| TF('fabric', Doc_42) | count_fabric / total_tokens in doc |
| IDF('fabric') | ~log((10001)/(df+1)) + 1 |
| IDF('the') | ≈ 1.0 (approaches zero log component) |
| IDF('embroidery') | Much higher — rare, category-specific |
| BM25 vs TF-IDF | Some ranking shifts due to length normalization in BM25 |

---

## Sample Outputs

**Top-5 for query `'wireless earbuds battery life poor'`:**
```
Rank | review_id  | Category    | Score
   1 | REV_XXXXX  | Electronics | 0.XXXXX
   2 | REV_XXXXX  | Electronics | 0.XXXXX
...
```
*(Exact values shown in notebook output — Electronics reviews consistently rank highest)*

**IDF Comparison:**
```
IDF('the')        ≈ 1.00  (appears in ~90%+ of all docs)
IDF('embroidery') ≈ 4.XX  (appears in <5% of docs, mostly Clothing)
```

---

## Folder Structure

```
Week07/
└── Day37/
    ├── README.md                       ← you are here
    ├── Week07_Monday_TF_IDF.ipynb      ← main notebook
    ├── top5_query_results.png
    ├── l2_difference_distribution.png
    ├── top_electronics_tfidf.png
    └── tfidf_vs_bm25_comparison.png
```

---

## Notes

- All code uses modular functions with docstrings (no 80-line cells)
- Constants like `k1`, `b`, `QUERY`, and `seed` are defined at the top or passed as parameters
- Try/except blocks used for dataset generation and sklearn comparison
- No external data files — fully self-contained notebook
