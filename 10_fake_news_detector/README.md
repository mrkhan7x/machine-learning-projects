# Fake News Detection using Machine Learning

A natural language processing pipeline that detects fabricated and clickbait news articles using Porter Stemming, TF-IDF Vectorization, and Logistic Regression.

---

## 📌 Project Overview
* **Notebook:** [fake_news_prediction.ipynb](fake_news_prediction.ipynb)
* **Technologies:** Python, NLTK, PorterStemmer, TF-IDF Vectorizer, Logistic Regression, Scikit-Learn
* **Author:** Muhammad Roman Khan

---

## 🚀 Workflow & Key Steps
1. News dataset loading and metadata consolidation (Author + Title)
2. Text preprocessing: lowercase, regex symbol removal, stopwords removal, Porter Stemming
3. Class distribution visualization
4. TF-IDF vector matrix construction
5. Logistic Regression model fitting
6. Accuracy scoring on train and test data
7. Predictive verification on test news headlines

---

## 💻 How to Run
1. Open this directory in VS Code or Jupyter Notebook:
   ```bash
   jupyter notebook fake_news_prediction.ipynb
   ```
2. Run all cells sequentially to view data visualizations, model training results, and predictive tests.
