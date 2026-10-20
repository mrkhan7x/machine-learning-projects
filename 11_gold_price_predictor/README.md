# Gold ETF Price Prediction

A financial regression model that forecasts Gold ETF (GLD) prices using financial market indicators (S&P 500, Crude Oil, Silver, EUR/USD) and Random Forest Regressor.

---

## 📌 Project Overview
* **Notebook:** [gold_price_prediction.ipynb](gold_price_prediction.ipynb)
* **Technologies:** Python, Scikit-Learn, Random Forest Regressor, Pandas, Seaborn
* **Author:** Muhammad Roman Khan

---

## 🚀 Workflow & Key Steps
1. Financial time series dataset loading (`gld_price_data.csv`)
2. Multi-asset correlation heatmap analysis
3. Gold price distribution histogram with KDE
4. Feature selection and train-test split
5. Random Forest Regressor training with 100 estimators
6. Metric evaluation (R² Score and Mean Absolute Error)
7. Actual vs. Predicted price line plot comparison
8. Predictive system estimating gold price from market inputs

---

## 💻 How to Run
1. Open this directory in VS Code or Jupyter Notebook:
   ```bash
   jupyter notebook gold_price_prediction.ipynb
   ```
2. Run all cells sequentially to view data visualizations, model training results, and predictive tests.
