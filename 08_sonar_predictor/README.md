# Submarine Sonar: Rock vs. Mine Prediction

A binary classification system that analyzes 60-band sonar chirp signals to distinguish between underwater rocks and metal naval mines.

---

## 📌 Project Overview
* **Notebook:** [Rock_vs_Mine_Prediction.ipynb](Rock_vs_Mine_Prediction.ipynb)
* **Technologies:** Python, Pandas, Scikit-Learn, Logistic Regression, Seaborn
* **Author:** Muhammad Roman Khan

---

## 🚀 Workflow & Key Steps
1. Loading 60-frequency sonar return signal data (`sonar.all-data`)
2. Statistical profiling and target value counts (Mine 'M' vs Rock 'R')
3. Data visualization: target distribution and frequency correlations
4. Stratified train-test split
5. Logistic Regression model training
6. Accuracy evaluation on training and test sets
7. Building an automated predictive system that classifies new sensor readings

---

## 💻 How to Run
1. Open this directory in VS Code or Jupyter Notebook:
   ```bash
   jupyter notebook Rock_vs_Mine_Prediction.ipynb
   ```
2. Run all cells sequentially to view data visualizations, model training results, and predictive tests.
