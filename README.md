# AQI Predictor – Okara (3-Day AQI Forecasting System)

### 🔗 Try Live Here: https://aqi-prediction-okaracity.streamlit.app/

https://github.com/user-attachments/assets/8e536073-52e9-4328-b217-66d792e9e922

A complete end-to-end MLOps project that predicts **Air Quality Index (AQI) for the next 3 days (72 hours)** for **Okara, Pakistan**.

## Why I Built This Project

Air pollution is one of Pakistan's most serious public health and environmental challenges. Okara, like many cities in Punjab, experiences dangerous AQI levels - especially during crop-burning season — yet there was no easy way for residents to know what the next few days would look like.

I chose this project for two reasons:

**1. Real-world environmental impact**
Air quality directly affects respiratory health, especially for children and the elderly. A 3-day forecast can help people plan outdoor activity, mask usage, and precautions.

**2. To learn the complete ML lifecycle — end to end**
Most academic projects stop at model training in a notebook. I wanted to experience the full production path: data collection from a live API, feature engineering, feature storage, model training, experiment tracking, CI/CD automation, and finally deployment to a live web application that anyone can use.


## Project Overview

This project builds a production-ready machine learning system that:

1. Collects AQI data hourly
2. Stores processed features in MongoDB
3. Trains ML models automatically
4. Tracks experiments using MLflow
5. Predicts AQI for the next 72 hours
6. Deploys a web app for real-time predictions

- **City Covered:** Okara, Pakistan
- **Forecast Horizon:** Next 3 Days (72 Hours)

## System Architecture

```
Scraping → Feature Engineering → Feature Store (MongoDB) → Model Training 
→ MLflow Tracking (DagsHub) → Streamlit App
```

**Automation:**
- Hourly Data Pipeline (feature updates)
- Daily Training Pipeline (model retraining)

---

## Project Structure

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

**Key Folders:**

- **`.github/workflows/`** — Two CI/CD pipelines: hourly data pipeline (scrape → engineer → store) and daily training pipeline (retrain → log to MLflow)
- **`automation/`** — Production pipeline scripts (ingestion, feature engineering, feature store writer, training)
- **`backend/`** — Development notebooks and experimentation (EDA, feature engineering, training)
- **`frontend/`** — Streamlit app for real-time predictions


## Machine Learning Details

- **Problem Type:** Time Series Forecasting
- **Target:** AQI
- **Forecast Window:** 72 Hours
- **Features:** Historical AQI lag values, rolling mean features, time-based features (hour, day)
- **Evaluation Metrics:** MAE, RMSE, R² Score
- **Experiment Tracking:** MLflow + DagsHub

## Why I Chose These Models

I benchmarked three models to compare a gradient boosting approach, an ensemble approach, and a neural network approach.

### XGBoost Regressor (Best Performer)
- **What it is:** Gradient boosting algorithm that builds trees sequentially, each new tree correcting the errors of the previous ones
- **Why I chose it:** Excellent for tabular and time-series data, handles missing values well, captures non-linear patterns, and is fast to train
- **Result:** Lowest error (MAPE 18.37%, RMSE 0.75)

### Random Forest Regressor
- **What it is:** Ensemble of many decision trees, averaging their predictions
- **Why I chose it:** Robust to overfitting, easy to interpret, and a strong baseline
- **Result:** Good performance but slightly higher error than XGBoost

### Multilayer Perceptron (MLP)
- **What it is:** A simple feed-forward neural network
- **Why I chose it:** To test whether a deep learning approach could capture patterns that tree-based models missed
- **Result:** Performed reasonably but did not beat XGBoost — likely because the dataset size favored tree-based methods


## Challenges & How I Solved Them

### 1. Data Leakage (The Biggest Challenge)

**The problem:**
My model was performing extremely well during training (very low error) but gave poor predictions on new, unseen data. Something was wrong.

**Root cause:**
I was using *future* values to predict *past* targets. Specifically, rolling averages and lag features were being computed in a way that accidentally included the target value itself — a classic case of **data leakage**.

**How I fixed it:**
- Shifted all lag and rolling features by the correct time offset so only past data was used for each prediction point
- Used a **time-based train/test split** instead of a random split — critical for time-series problems
- Re-validated the model on completely unseen future data to confirm the fix

**Result:** Model error dropped to a realistic, generalizable level (MAPE 18.37%, RMSE 0.75).

### 2. Overfitting from Too Many Features

**The problem:**
After adding many engineered features, the model started memorizing noise instead of learning patterns.

**How I fixed it:**
Used **SHAP analysis** to identify which features actually contributed to predictions. Removed low-impact and noisy features, which improved generalization on unseen data.

### 3. Automating a Live Data Pipeline

**The problem:**
Manually scraping and retraining was not scalable or reliable.

**How I fixed it:**
Built two GitHub Actions workflows — an **hourly data pipeline** (scrape → engineer → store in MongoDB) and a **daily training pipeline** (retrain → log to MLflow). This turned a manual notebook into a self-updating production system.


## Database

- **Feature Store:** MongoDB Atlas
- **Used For:** Storing transformed feature dataset, serving data for training, supporting the Streamlit app


## CI/CD Pipelines

1. **Hourly Data Pipeline** — Updates feature store continuously
2. **Daily Training Pipeline** — Retrains model automatically to adapt to new AQI patterns

Manual trigger enabled via `workflow_dispatch`.


## Deployment

- **Frontend:** Streamlit Cloud
- **Backend:** MongoDB Atlas (cloud database) + MLflow tracking server (DagsHub)

**Environment Variables Required:**

```
MONGO_URI
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

---

## Key MLOps Concepts Implemented

- Feature Store Design
- Model Registry for Version Control
- Automated Data Pipeline
- Scheduled Training
- Experiment Tracking (MLflow + DagsHub)
- Cloud Deployment
- Secret Management
- CI/CD Integration
- SHAP for Model Interpretability


## How to Run Locally

**1. Clone the repository**

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add `.env` file**
```
MONGO_URI=your_mongo_uri
MLFLOW_TRACKING_URI=your_mlflow_uri
```

**4. Run hourly pipeline**
```bash
python automation/run_hourly_pipeline.py
```

**5. Run training pipeline**
```bash
python automation/training_pipeline.py
```

**6. Run Streamlit app**
```bash
streamlit run frontend/app.py
```


## Future Improvements

- Add data validation (Great Expectations)
- Add Docker containerization
- Add monitoring dashboard
- Expand to multiple cities in Punjab

## Conclusion

This project demonstrates a complete production-ready machine learning pipeline for forecasting AQI in Okara for the next 3 days. It integrates **Data Engineering, Feature Engineering, Model Training, Experiment Tracking, CI/CD Automation, and Cloud Deployment** — a full MLOps lifecycle implementation suitable for real-world deployment.


## ✍️ Author

**Najma Razzaq**
AI Engineer | [LinkedIn](https://www.linkedin.com/in/najmarazzaq)


## 📜 License

Apache License – Open for contributions!
