# AQI Predictor – Okara (3-Day AQI Forecasting System)
### Try Live Here: https://aqi-prediction-okaracity.streamlit.app/


A complete end-to-end MLOps project that predicts **Air Quality Index (AQI) for the next 3 days (72 hours)** for **Okara, Pakistan**.

The system includes:

* Automated data scraping
* Feature engineering with lag features
* Feature storage in MongoDB
* Model training with MLflow tracking
* registry for version control
* CI/CD pipelines using GitHub Actions
* Web application deployment using Streamlit

---

# Project Overview

Air pollution is a serious environmental issue. This project builds a production-ready machine learning system that:

1. Collects AQI data hourly
2. Stores processed features in MongoDB
3. Trains ML models automatically
4. Tracks experiments using MLflow
5. Predicts AQI for the next 72 hours
6. Deploys a web app for real-time predictions

City Covered: **Okara, Pakistan**

Forecast Horizon: **Next 3 Days (72 Hours)**

---

# System Architecture

Data Flow:

Scraping → Feature Engineering → Feature Store (MongoDB) → Model Training → MLflow Tracking, Dagshub → Streamlit App

Automation:

* Hourly Data Pipeline (Feature updates)
* Daily Training Pipeline (Model retraining)

---

# Project Structure

```
AQI_PREDICTOR/
│
├── .github/workflows/
│   ├── daily_training_pipeline.yml
│   └── hourly_data_pipeline.yml
│
├── automation/
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── feature_store_writer.py
│   ├── run_hourly_pipeline.py
│   └── training_pipeline.py
│
├── backend/
│   ├── data.csv
│   ├── EDA.ipynb
│   ├── feature_engineering.ipynb
│   ├── feature_store.py
│   ├── final_features.csv
│   ├── scrapping.py
│   └── training.ipynb
│
├── frontend/
│   └── app.py
│
├── .env
├── .gitignore
├── LICENSE
└── mlflow.db
```

---

# Folder Explanation

## .github/workflows/

Contains CI/CD pipelines:

### hourly_data_pipeline.yml

* Runs every hour
* Scrapes AQI data
* Performs feature engineering
* Stores transformed features into MongoDB

### daily_training_pipeline.yml

* Runs daily
* Pulls latest features
* Trains ML models
* Logs experiments to MLflow

---

## automation/

Production pipeline scripts.

### data_ingestion.py

Scrapes real-time AQI data for Okara.

### feature_engineering.py

Creates:

* Lag features (AQI_t-1, AQI_t-2, etc.)
* Rolling averages
* Time-based features

### feature_store_writer.py

Stores transformed features into MongoDB feature store.

### run_hourly_pipeline.py

Entry point for hourly pipeline automation.

### training_pipeline.py

* Loads features from MongoDB
* Creates 72-hour target
* Splits dataset
* Trains models (XGBoost, MLP, Random Forest Regressor)
* Logs metrics & models to Dagshub

---

## backend/

Contains development notebooks and initial experimentation files:

* EDA.ipynb → Exploratory Data Analysis
* training.ipynb → Model experimentation
* feature_engineering.ipynb → Feature experiments
* scrapping.py → Initial scraping logic
* final_features.csv → Engineered dataset snapshot

---

## frontend/

### app.py

Streamlit application that:

* Connects to MongoDB
* Loads trained model
* Predicts AQI for next 3 days
* Displays results interactively

---

# Machine Learning Details

Problem Type: Time Series Forecasting
Target: AQI
Forecast Window: 72 Hours

Features:

* Historical AQI lag values
* Rolling mean features
* Time-based features (hour, day, etc.)

Model Used:

* XGBoost Regressor, Multilayer Perceptron, Random Forest Regressor

Evaluation Metrics:

* MAE
* RMSE
* R² Score

Experiment Tracking:

* MLflow
* Remote tracking server (DagsHub)

---

# Database

Feature Store: MongoDB Atlas

Used For:

* Storing transformed feature dataset
* Serving data for training
* Supporting Streamlit app

---

# CI/CD Pipelines

Two automated workflows:

1. Hourly Data Pipeline
   Updates feature store continuously.

2. Daily Training Pipeline
   Retrains model automatically to adapt to new AQI patterns.

Manual trigger enabled via workflow_dispatch.

---

# Deployment

Frontend deployed on Streamlit Cloud.

Backend:

* MongoDB Atlas (Cloud database)
* MLflow tracking server

Environment Variables Required:

```
MONGO_URI
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

---

# Key MLOps Concepts Implemented

* Feature Store Design
* model registry for version control
* Automated Data Pipeline
* Scheduled Training
* Experiment Tracking
* Cloud Deployment
* Secret Management
* CI/CD Integration

---

# How to Run Locally

1. Clone repository

2. Install dependencies

```
pip install -r requirements.txt
```

3. Add .env file

```
MONGO_URI=your_mongo_uri
MLFLOW_TRACKING_URI=your_mlflow_uri
```

4. Run hourly pipeline

```
python automation/run_hourly_pipeline.py
```

5. Run training pipeline

```
python automation/training_pipeline.py
```

6. Run Streamlit app

```
streamlit run frontend/app.py
```

---

# Future Improvements

* Add data validation (Great Expectations)
* Add Docker containerization
* Add monitoring dashboard

---

# Conclusion

This project demonstrates a complete production-ready machine learning pipeline for forecasting AQI in Okara for the next 3 days.

It integrates:

* Data Engineering
* Feature Engineering
* Model Training
* Experiment Tracking
* CI/CD Automation
* Cloud Deployment

This is a full MLOps lifecycle implementation suitable for real-world deployment.

## ✍️ Author

**Najma Razzaq**  
AI Engineer | [LinkedIn](https://www.linkedin.com/in/najmarazzaq)

--- 

## **License**  
📜 APACHE License – Open for contributions!  