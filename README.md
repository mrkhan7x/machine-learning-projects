# Machine Learning & Deep Learning Projects Portfolio

A comprehensive, hands-on portfolio of **Machine Learning**, **Natural Language Processing**, and **Deep Learning** projects built by **Muhammad Roman Khan**. 

Each project follows a clean, beginner-friendly, step-by-step implementation standard inspired by **Andrew Ng's Machine Learning Specialization** and **Siddhardhan's Hands-On ML Series**. Every notebook includes exploratory data analysis, data visualizations (Seaborn/Matplotlib), model training, evaluation metrics, and an interactive predictive system.

---

## 🚀 Projects Overview

| # | Project Name | Domain / Focus | Key Algorithms & Tools | Notebook | Guide |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **01** | **SMS Spam Detection** | NLP / Text Classification | TF-IDF, Logistic Regression | [Notebook](01_spam_detector/sms_spam_prediction.ipynb) | [Guide](01_spam_detector/README.md) |
| **02** | **CLI Task Manager** | Software / Python OOP | OOP Classes, JSON Persistence | [Notebook](02_task_manager/task_manager.ipynb) | [Guide](02_task_manager/README.md) |
| **03** | **Weather Dashboard** | Data Analytics & Viz | Pandas, Seaborn Visualizations | [Notebook](03_weather_dashboard/weather_dashboard.ipynb) | [Guide](03_weather_dashboard/README.md) |
| **04** | **FastAPI Task API** | Backend Engineering | FastAPI, Pydantic, TestClient | [Notebook](04_fastapi_api/fastapi_task_api.ipynb) | [Guide](04_fastapi_api/README.md) |
| **05** | **Semantic Search Engine** | Information Retrieval | TF-IDF, Cosine Similarity | [Notebook](05_semantic_search/semantic_search_engine.ipynb) | [Guide](05_semantic_search/README.md) |
| **06** | **Diabetes Prediction** | Healthcare Diagnostic | Logistic Regression, Scaling | [Notebook](06_diabetes_predictor/diabetes_prediction.ipynb) | [Guide](06_diabetes_predictor/README.md) |
| **07** | **Model Benchmark** | Model Comparison | LogReg vs Decision Tree vs RF vs SVM | [Notebook](07_model_benchmark/model_benchmark_comparison.ipynb) | [Guide](07_model_benchmark/README.md) |
| **08** | **Sonar: Rock vs. Mine** | Signal Classification | 60-Band Sonar, Logistic Regression | [Notebook](08_sonar_predictor/Rock_vs_Mine_Prediction.ipynb) | [Guide](08_sonar_predictor/README.md) |
| **09** | **Diabetes Prediction (SVM)** | Healthcare Diagnostic | Support Vector Machine (Linear SVC) | [Notebook](09_diabetes_svm_predictor/diabetes_svm_prediction.ipynb) | [Guide](09_diabetes_svm_predictor/README.md) |
| **10** | **Fake News Prediction** | NLP / Text Classification | Porter Stemming, TF-IDF, LogReg | [Notebook](10_fake_news_detector/fake_news_prediction.ipynb) | [Guide](10_fake_news_detector/README.md) |
| **11** | **Gold Price Prediction** | Financial Regression | Random Forest Regressor | [Notebook](11_gold_price_predictor/gold_price_prediction.ipynb) | [Guide](11_gold_price_predictor/README.md) |
| **12** | **Heart Disease Prediction** | Cardiovascular Diagnostic | Logistic Regression | [Notebook](12_heart_disease_predictor/Heart_Disease_Prediction.ipynb) | [Guide](12_heart_disease_predictor/README.md) |
| **13** | **Credit Card Fraud Detection**| Anomaly Detection | Random Under-Sampling, LogReg | [Notebook](13_credit_card_fraud_detector/credit_card_fraud_detection.ipynb) | [Guide](13_credit_card_fraud_detector/README.md) |
| **14** | **Big Mart Sales Prediction** | Retail Sales Forecasting | Random Forest Regressor, Label Encoding | [Notebook](14_big_mart_sales_predictor/big_mart_sales_prediction.ipynb) | [Guide](14_big_mart_sales_predictor/README.md) |

---

## 🧠 PyTorch Deep Learning Track

Dedicated deep learning implementations built from scratch using **PyTorch** (`torch.nn`, `torch.optim`):

| # | Project Name | Architecture | Framework | Notebook | Guide |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **01** | **Diabetes Classifier** | Multi-Layer Perceptron (MLP) | PyTorch Binary Cross Entropy | [Notebook](pytorch-projects/01_diabetes_classifier/diabetes_pytorch.ipynb) | [Guide](pytorch-projects/01_diabetes_classifier/README.md) |
| **02** | **Sonar Rock vs. Mine** | Deep Neural Network Classifier | PyTorch Sequential Layers | [Notebook](pytorch-projects/02_sonar_classifier/sonar_pytorch.ipynb) | [Guide](pytorch-projects/02_sonar_classifier/README.md) |
| **03** | **Gold Price Regressor** | Deep Neural Network Regressor | PyTorch MSE Loss & Adam | [Notebook](pytorch-projects/03_gold_price_predictor/gold_price_pytorch.ipynb) | [Guide](pytorch-projects/03_gold_price_predictor/README.md) |
| **04** | **MNIST Digit Classifier** | Computer Vision MLP / CNN | PyTorch Torchvision MNIST | [Notebook](pytorch-projects/04_mnist_digit_classifier/mnist_pytorch.ipynb) | [Guide](pytorch-projects/04_mnist_digit_classifier/README.md) |

---

## 🛠️ Project Design Standard

Every project in this repository follows a clean, professional structure:
1. **Importing the Dependencies:** Clean imports of standard libraries without over-engineering.
2. **Data Collection & Exploration:** Inspecting shape, statistical measures (`.describe()`), data types (`.info()`), and null values.
3. **Data Visualization:** Exploratory plots using Seaborn and Matplotlib (countplots, correlation heatmaps, feature distributions).
4. **Feature & Target Splitting:** Separating feature matrix `X` and target vector `Y`.
5. **Train-Test Split:** Standardized train/test partitioning with stratification where appropriate.
6. **Model Training:** Clean initialization and fitting of Scikit-Learn or PyTorch models.
7. **Model Evaluation:** Training accuracy vs test accuracy, $R^2$ scores, and confusion matrices.
8. **Building a Predictive System:** End-to-end predictive code block accepting custom test inputs, reshaping them, and printing plain-English predictions.

---

## 💻 Getting Started

Clone the repository and install the dependencies:

```bash
git clone https://github.com/mrkhan7x/machine-learning-projects.git
cd machine-learning-projects

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Run any notebook directly in VS Code or Jupyter Lab:
```bash
jupyter notebook
```

---

## 👤 Author
**Muhammad Roman Khan (M R KHAN)**
* Website: [portfolio.mrkhan.qzz.io](https://portfolio.mrkhan.qzz.io)
* GitHub: [@mrkhan7x](https://github.com/mrkhan7x)
