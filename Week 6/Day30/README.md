# SUV Purchase Prediction using Logistic Regression

## Overview

This project implements a complete machine learning pipeline to predict whether a customer will purchase an SUV based on their Age and Estimated Salary.

The solution follows a structured approach: data loading, preprocessing, model training, evaluation, and visualization. The goal was to not just make the model work, but to keep the code clean, readable, and easy to extend.

---

## Dataset

* File: suv_data.csv
* Features used:

  * Age
  * EstimatedSalary
* Target:

  * Purchased (0 or 1)

---

## Steps Performed

1. Data Loading

   * Loaded dataset using pandas
   * Checked shape, columns, and data types

2. Data Preprocessing

   * Encoded Gender column (Male → 0, Female → 1)
   * Selected relevant features (Age, EstimatedSalary)
   * Separated features (X) and target (y)

3. Train-Test Split

   * Used 80/20 split with random_state=42

4. Feature Scaling

   * Applied StandardScaler
   * Ensured no data leakage (fit on train only)

5. Model Training

   * Used LogisticRegression from sklearn
   * Trained on scaled training data

6. Model Evaluation

   * Calculated Accuracy
   * Generated Confusion Matrix
   * Visualized confusion matrix

7. Visualization

   * Plotted decision boundary
   * Observed how model separates classes

8. Improvement

   * Tried different splits (70/30, 75/25)
   * Compared accuracy results

---

## How to Run

1. Install dependencies:
   pip install pandas numpy matplotlib scikit-learn

2. Place files in same folder:

   * suv_data.csv
   * SUV_FullMarks.ipynb

3. Run notebook:

   * Open Jupyter Notebook
   * Run all cells sequentially

---

## Results

* Model achieves ~85–87% accuracy (varies slightly with split)
* Older users with higher salaries are more likely to purchase SUVs
* Model performs well but has some false negatives

---

## Project Structure

* SUV_FullMarks.ipynb → Main notebook
* suv_data.csv → Dataset
* README.txt → Project documentation

---

## Key Learnings

* Logistic Regression is a classification algorithm despite its name
* Feature scaling significantly impacts performance
* Confusion matrix gives deeper insight than accuracy alone
* Clean code structure makes debugging and extension easier

---

## Notes

This project was built with focus on clarity and engineering quality, not just output.
Functions are modular, naming is consistent, and logic is easy to follow.

---

## Author

Yash
PG Diploma – AI/ML & Agentic AI Engineering
IIT Gandhinagar
