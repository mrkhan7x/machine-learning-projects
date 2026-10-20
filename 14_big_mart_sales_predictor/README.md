# Big Mart Retail Sales Forecasting

A commercial retail sales prediction model that forecasts item outlet sales across various store formats using missing value imputation, Label Encoding, and Random Forest regression.

---

## 📌 Project Overview
* **Notebook:** [big_mart_sales_prediction.ipynb](big_mart_sales_prediction.ipynb)
* **Technologies:** Python, Pandas, Scikit-Learn, LabelEncoder, Random Forest Regressor, Seaborn
* **Author:** Muhammad Roman Khan

---

## 🚀 Workflow & Key Steps
1. Retail sales dataset loading (`Train.csv`)
2. Handling missing values (Mean for `Item_Weight`, Mode for `Outlet_Size`)
3. Visualizing feature distributions (MRP, Weight, Establishment Years)
4. Categorical feature transformation using LabelEncoder
5. Train-test split (80/20)
6. Random Forest Regressor training
7. Evaluation with R² Score and Mean Absolute Error
8. Predictive system calculating expected sales revenue

---

## 💻 How to Run
1. Open this directory in VS Code or Jupyter Notebook:
   ```bash
   jupyter notebook big_mart_sales_prediction.ipynb
   ```
2. Run all cells sequentially to view data visualizations, model training results, and predictive tests.
