<div align="center">

# 📉 Customer Churn Prediction

### An End-to-End Machine Learning Pipeline for Predicting Customer Churn

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**A production-ready machine learning application that predicts customer churn using an automated ML pipeline, MLflow experiment tracking, and a Flask web interface.**

</div>

---

# 📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses. Identifying customers who are likely to leave allows organizations to improve customer retention through proactive engagement.

This project builds an end-to-end Machine Learning pipeline that automates data ingestion, preprocessing, model training, evaluation, and prediction through an interactive Flask application.

---

# ✨ Features

- End-to-End Machine Learning Pipeline
- Automated Data Ingestion & Preprocessing
- Feature Engineering
- Model Training & Evaluation
- MLflow Experiment Tracking
- Flask Web Application
- Prediction Pipeline
- Docker Support
- Modular Project Structure
- Logging & Exception Handling

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Flask
- MLflow
- Docker
- Git & GitHub

---

# 📂 Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── artifacts/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── model_columns.pkl
│   ├── metrics.json
│   └── feature_importance.json
│
├── src/
│   └── customer_churn_prediction/
│       ├── components/
│       ├── pipelines/
│       ├── logger.py
│       ├── exception.py
│       └── utils.py
│
├── templates/
├── static/
├── app.py
├── main.py
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

# ⚙️ ML Pipeline

```text
Raw Dataset
      │
      ▼
Data Ingestion
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
MLflow Tracking
      │
      ▼
Best Model Saved
      │
      ▼
Prediction Pipeline
      │
      ▼
Flask Dashboard
```

---

# 📊 Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains customer information including:

- Customer demographics
- Service subscriptions
- Contract details
- Payment methods
- Monthly & Total Charges
- Customer tenure

**Target Variable**

- Churn (Yes / No)

---

# 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/aggarwal-navya/Customer_Churn_Prediction.git

cd Customer_Churn_Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Train the Model (Optional)

```bash
python main.py
```

### Run the Flask Application

```bash
python app.py
```

Visit:

```
http://127.0.0.1:5001
```

---

# 🐳 Docker

Build

```bash
docker build -t churnguard-pro .
```

Run

```bash
docker run -p 5001:5001 churnguard-pro
```

---

# 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

The best-performing model is automatically saved and used by the prediction pipeline.

---

# 📷 Application Preview

> Add screenshots of:
>
> - Dashboard
> - Prediction Form
> - Prediction Result
> - MLflow Experiment Tracking

---

# 👩‍💻 Author

**Navya Aggarwal**

AI & Machine Learning | Data Analytics

- GitHub: https://github.com/aggarwal-navya
- LinkedIn: https://www.linkedin.com/in/navya-aggarwal14/
- Portfolio: https://navyaaggarwal.vercel.app/

---

If you found this project helpful, consider giving it a ⭐ on GitHub.
