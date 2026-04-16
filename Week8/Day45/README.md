# Day 45 — Hospital Stay Prediction: Data Cleaning + Neural Network from Scratch

**Week 08 · Tuesday | PG Diploma AI-ML & Agentic AI Engineering · IIT Gandhinagar**

---

## Problem Overview

Dr. Priya Anand, Head of Analytics at a hospital chain, needs a system to predict which patients will have **prolonged hospital stays (>30 days)** — a critical problem for bed management and resource planning.

This notebook solves the full pipeline:
- Clean a messy real-world hospital dataset
- Build a **3-layer neural network from scratch in NumPy** (no PyTorch/TensorFlow)
- Evaluate with clinically meaningful metrics
- Optimise the decision threshold to minimise expected clinical cost
- Expose the "94% accuracy" trap using a trivial classifier
- Use the trained NN as a feature extractor (Hard)

---

## Dataset Description

**Source:** Healthcare Analytics II (Kaggle / AV Healthcare Analytics)  
**File:** `train_data.csv` (318,438 rows × 18 columns)

| Column | Type | Description |
|--------|------|-------------|
| `case_id` | int | Patient case identifier (dropped — ID column) |
| `Hospital_code` | int | Hospital identifier |
| `Hospital_type_code` | str | Type of hospital (a/b/c/d/e/f/g) |
| `City_Code_Hospital` | int | Hospital city code |
| `Hospital_region_code` | str | Region (X/Y/Z) |
| `Available Extra Rooms in Hospital` | int | Spare capacity |
| `Department` | str | Department (radiotherapy, anesthesia, etc.) |
| `Ward_Type` | str | Ward classification |
| `Ward_Facility_Code` | str | Facility code |
| `Bed Grade` | float | Bed condition grade (has **113 missing values**) |
| `patientid` | int | Patient ID (dropped — ID column) |
| `City_Code_Patient` | float | Patient's home city (has **4,532 missing values**) |
| `Type of Admission` | str | Emergency / Trauma / Urgent |
| `Severity of Illness` | str | Extreme / Moderate / Minor |
| `Visitors with Patient` | int | Number of visitors |
| `Age` | str | Age range (stored as strings e.g., '41-50') |
| `Admission_Deposit` | float | Deposit amount at admission |
| `Stay` | str | **Target** — 11 stay duration categories |

**Engineered Target:** `LongStay` = 1 if Stay > 30 days, else 0  
Class distribution: 59.4% short stay / 40.6% long stay

---

## Approach

### Step 1 — Data Quality Audit

Examined all 18 columns systematically:
- Found 2 columns with missing values (`Bed Grade`: 113, `City_Code_Patient`: 4,532)
- Identified `Age` stored as string ranges — needs conversion
- Noted `case_id` and `patientid` are ID columns with no predictive value
- Observed the target `Stay` has 11 categories — binarised for classification

### Step 2 — Data Cleaning

| Issue | Fix | Rationale |
|-------|-----|-----------|
| Missing `Bed Grade` (0.04%) | Median imputation | Robust to outliers; very small missingness |
| Missing `City_Code_Patient` (1.42%) | Mode imputation | Categorical feature — mean not applicable |
| `Age` stored as strings | Map to midpoints (e.g., '41-50' → 45) | Preserves ordinal structure |
| ID columns (`case_id`, `patientid`) | Drop | Would cause data leakage |
| Categorical codes | LabelEncoder | Keeps feature matrix compact for NumPy NN |
| Multi-class `Stay` (11 classes) | Binarise at 30 days | Clinically meaningful threshold |

### Step 3 — Neural Network Architecture

Built entirely in NumPy — no autograd libraries:

```
Input (15 features)
      ↓
Dense(64, ReLU)    ← Layer 1
      ↓
Dense(32, ReLU)    ← Layer 2
      ↓
Dense(1, Sigmoid)  ← Output layer
```

Key implementation details:
- **He initialisation** (`W ~ N(0, sqrt(2/n_prev))`) for ReLU compatibility
- **Numerically stable sigmoid** — conditional formula to avoid `exp` overflow
- **Mini-batch SGD** (batch_size=256) — full batch GD was too slow to converge
- **Binary Cross-Entropy loss** with epsilon clipping

### Step 4 — Training & Evaluation

- **Primary metric: AUC-ROC** — chosen over accuracy due to moderate class imbalance
- **Secondary metric: F1-Score** — balances precision/recall for the positive class
- Training: 400 epochs, LR=0.005, 8,000 sampled rows

### Step 5 — Clinical Cost Optimisation

Defined asymmetric costs:
- FN (missed long-stay patient) = **10 units** — bed shortage, emergency staffing
- FP (over-flagged short-stay) = **1 unit** — minor over-allocation

Swept thresholds from 0.05 to 0.95, found optimal at **0.095**.

### Step 6 — The 94% Accuracy Trap (Hard)

Demonstrated how a trivial "always predict majority class" classifier achieves 92.6% accuracy on the original multi-class framing (only 7.4% of patients have 0-10 day stays). F1 = 0.00.

### Step 7 — NN as Feature Extractor (Hard)

Extracted 32-dimensional embeddings from the penultimate layer → trained logistic regression on top. AUC = 0.7940 vs direct NN AUC = 0.7953.

---

## Steps to Run

### Requirements

```
Python >= 3.10
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Run the Notebook

```bash
# Clone/navigate to the folder
cd week-08/tuesday/

# Make sure data files are present
ls *.csv
# train_data.csv  test_data.csv  sample_sub.csv  train_data_dictionary.csv

# Launch notebook
jupyter notebook Day45.ipynb
```

> **Note:** The notebook uses a relative path `train_data.csv`. Do NOT hardcode absolute paths — the TA runs notebooks from a clean environment.

### Python Version

Tested on Python 3.11. All dependencies are standard — no GPU required.

---

## Results Summary

| Model | AUC-ROC | F1-Score | Accuracy |
|-------|---------|----------|----------|
| NumPy Neural Network (ours) | **0.7953** | 0.6708 | 0.769 |
| Sklearn Logistic Regression | 0.7918 | 0.6893 | 0.788 |
| NN Embedding + Logistic Reg | 0.7940 | 0.6631 | — |

**Clinical Cost Optimisation:**
- Default threshold (0.5): Cost = 1,208 units
- Optimal threshold (0.095): Cost = 936 units  
- **22.5% cost reduction**

**The 94% Accuracy Trap:**
- Trivial classifier accuracy: 92.6%
- Trivial classifier F1: 0.00
- This is why AUC-ROC and F1 matter for imbalanced medical datasets

---

## Sample Outputs

The notebook generates 6 figures:
1. `fig1_distribution.png` — Original vs binarised stay distribution
2. `fig2_missing.png` — Missing values audit
3. `fig3_loss.png` — Training/validation loss curve (400 epochs)
4. `fig4_confusion.png` — Confusion matrices (NN vs Logistic Reg)
5. `fig5_cost.png` — Clinical cost vs threshold sweep
6. `fig6_roc.png` — ROC curve comparison (all 3 models)

---

## Repository Structure

```
week-08/
└── tuesday/
    ├── Day45.ipynb              ← Main notebook
    ├── README.md                ← This file
    ├── train_data.csv           ← Training data (place here)
    ├── test_data.csv            ← Test data
    ├── sample_sub.csv           ← Submission format
    └── train_data_dictionary.csv ← Column descriptions
```

---

## AI Usage Policy

**Prompt used:** *"Help me understand how He initialisation works for ReLU networks and why it's preferred over Xavier."*

**Critique:** The AI explanation was conceptually correct (variance preservation argument). I added the explicit formula and implemented it directly. The AI's sigmoid implementation wasn't numerically stable — I rewrote it with the conditional formula.

---

*Submitted for Week 08 · Tuesday · IIT Gandhinagar PG Diploma AI-ML*
