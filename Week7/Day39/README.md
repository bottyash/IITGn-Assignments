# Week 07 – Wednesday: Hard NLP Patterns in Indian E-Commerce Reviews

**IIT Gandhinagar | Cohort 1 | NLP Foundations**  
**Assignment:** Daily Take-Home – Week 07, Wednesday  
**Topics:** NLP Patterns · Aspect-Based Sentiment · Model Comparison  

---

## Problem Overview

Indian e-commerce reviews don't behave like standard English text. This assignment tackles the five hardest NLP patterns found in the ShopSense dataset and analyzes the gap between review-level and aspect-level sentiment classification.

**Q1 – 5 Hard NLP Patterns:**
- (a) Negation: *"not bad at all"* → actually POSITIVE
- (b) Sarcasm: *"Wow great! Broke on day 1"* → actually NEGATIVE
- (c) Code-mixing: *"Product bahut accha hai lekin delivery late thi"* → MIXED
- (d) Implicit: *"Returned it within 2 hours"* → NEGATIVE (no explicit sentiment word!)
- (e) Comparative: *"Way better than my previous Samsung"* → POSITIVE

For each pattern: preprocessing + feature engineering + baseline failure mode.

**Q2 – Aspect-Level vs Review-Level Classification:**
- Review-level F1: 88% | Aspect-level F1: 71%
- Why the 17-point gap exists
- How to close it to 80%+
- Aspect-sentiment extraction from: *"Amazing camera quality but the battery is atrocious and customer support was unhelpful."*

---

## Dataset

**ShopSense E-Commerce Reviews** (synthetic, mirrors assignment schema):
- Reviews: 10K rows → `review_id, customer_id, product_id, category, review_text, rating (1-5), sentiment_label, language`
- Categories: Electronics / Clothing / Food / Home / Beauty / Books
- Language: English / Hindi / Code-mixed
- Since the actual data file isn't in the repo, `generate_shopsense_reviews()` creates a representative 500-row subset that covers all 5 pattern types

---

## Approach

### Q1 – Pattern-by-Pattern

**Negation:**  
Scope-based negation tagging – any word within 4 tokens of a negator gets a `_NEG` suffix. So "not bad" becomes "not bad_NEG" and the TF-IDF model learns `bad_NEG` ≠ `bad`. Also engineered binary features for "not bad at all" pattern specifically.

**Sarcasm:**  
Two-signal detection: (1) exclamatory/hyperbolic opener + (2) factual negative event in a later clause. Also checks for polarity flip across sentence boundaries. Assigns a weighted sarcasm score and thresholds at 0.4.

**Code-mixing:**  
Built a manually curated Hindi sentiment lexicon (positive: bahut accha, badhiya, shandar; negative: bekar, kharab, bakwaas). Transliterates Hindi tokens to `HINDI_POS_GOOD` style tokens so standard English models can process them. Also measures the code-mixing ratio to weight Hindi signals appropriately.

**Implicit Sentiment:**  
Regex-based behavioral cue detection. Key insight: actions reveal sentiment. "Returned within 2 hours" matches `returned.*within.*hours` → negative. "Ordered again" → positive. No lexicon needed because the sentiment comes from the *action* not a descriptor.

**Comparative:**  
Comparative phrase regex (e.g. `way better than`, `much worse than`). The model then maps: positive comparative about this product → POSITIVE review; negative comparative → NEGATIVE. Also extracts what's being compared against (e.g., "Samsung", "Philips version").

### Q2 – Aspect-Level Analysis

**Why aspect-level is harder:**  
It's fundamentally two tasks: (1) identify WHAT aspect is being discussed (boundary detection), (2) classify sentiment for that aspect. Review-level is just one task (2). Multi-polarity in a single review, implicit aspects, and aspect-opinion alignment all add complexity.

**Improvement path to 80%+:**
1. BERT-based ABSA with `[CLS] aspect [SEP] review` input format (+7.5%)
2. Domain pre-training on ShopSense corpus (+3.2%)
3. Aspect-aware multi-head attention (+4.1%)
4. Back-translation data augmentation (+2.8%)
5. Dependency parsing for opinion-aspect alignment (+2.5%)
6. Multi-task learning (shared encoder) (+3.0%)

**Projected total: ~94%** (from 71% base, with all strategies combined)

---

## How to Run

```bash
# 1. Clone the repo and navigate to this folder
cd week07/wednesday/

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch Jupyter
jupyter notebook week07_wednesday_nlp_patterns.ipynb

# 4. Run All Cells (Kernel → Restart & Run All)
```

**No external data download needed** – the notebook generates synthetic ShopSense data internally.

---

## Results Summary

### Q1 – Baseline vs Enhanced Accuracy

| Pattern | True Label | Baseline Pred | Baseline Acc | Enhanced Acc | Improvement |
|---------|-----------|---------------|-------------|-------------|-------------|
| Negation | positive | negative | 42% | 78% | +36pp |
| Sarcasm | negative | positive | 38% | 65% | +27pp |
| Code-mixing | mixed | neutral | 35% | 71% | +36pp |
| Implicit | negative | neutral | 28% | 72% | +44pp |
| Comparative | positive | neutral | 55% | 83% | +28pp |

**Hardest pattern: Implicit** (28% baseline – no sentiment words at all!)  
**Easiest to fix with features: Comparative** (83% enhanced with regex)

### Q2 – Aspect-Sentiment Extraction

Target: *"Amazing camera quality but the battery is atrocious and customer support was unhelpful."*

| Aspect | Opinion Word | Sentiment |
|--------|-------------|-----------|
| camera quality | amazing | **POSITIVE** |
| battery | atrocious | **NEGATIVE** |
| customer support | unhelpful | **NEGATIVE** |

One review → simultaneously positive AND negative. This is why aspect-level matters for the product team.

---

## Sample Outputs

- `pattern_accuracy_comparison.png` – Baseline vs Enhanced bar chart + improvement deltas
- `aspect_difficulty.png` – F1 score comparison + sources of difficulty
- `improvement_strategies.png` – Waterfall chart of path to 80%+
- `aspect_vs_review_level.png` – Multi-aspect sentiment breakdown

---

## Folder Structure

```
Week07/Day39/
├── week07-Day39.ipynb   ← Main notebook
├── README.md                             ← This file
├── pattern_accuracy_comparison.png       ← Figure 1
├── aspect_difficulty.png                 ← Figure 2
├── improvement_strategies.png            ← Figure 3
└── aspect_vs_review_level.png            ← Figure 4
```

---

*IIT Gandhinagar – NLP Foundations | Cohort 1 | Week 07 Wednesday*
