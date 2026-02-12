import streamlit as st
import pandas as pd
import os
from datetime import timedelta

import mlflow
from mlflow.tracking import MlflowClient

# ===================== UI =====================
st.title("Real Time AQI Predictor")
st.write("Predicts next 3 days AQI using latest trained models")

# ===================== MLFLOW CONFIG =====================
MONGO_URI = os.getenv("MONGO_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

client = MlflowClient()
experiment = client.search_experiments()[0]

# ===================== HELPERS =====================
def get_latest_run(run_name):
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    return runs[0] if runs else None

# ===================== LOAD FEATURE STORE =====================
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import urlparse, quote_plus

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

parsed = urlparse(mongo_uri)
if parsed.username:
    escaped_user = quote_plus(parsed.username)
    escaped_pass = quote_plus(parsed.password) if parsed.password else ''
    netloc = f"{escaped_user}:{escaped_pass}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    mongo_uri = parsed._replace(netloc=netloc).geturl()

mongo_client = MongoClient(mongo_uri)
db = mongo_client["aqi_feature_store"]
collection = db["hourly_features"]

df = pd.DataFrame(list(collection.find({}, {"_id": 0})))

# ===================== MODEL METRICS =====================
models_config = [
    ("XGBoost", "XGBoost_72hr"),
    ("MLP", "MLP_72hr"),
    ("RandomForest", "rf_72hr")
]

models_info = []

for display_name, run_name in models_config:
    run = get_latest_run(run_name)

    if run:
        metrics = run.data.metrics
        models_info.append({
            "Model": display_name,
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "MAPE": metrics.get("MAPE"),
            "run_id": run.info.run_id
        })

df_metrics = pd.DataFrame(models_info)

st.subheader("Latest Model Performance")
st.dataframe(df_metrics[["Model", "MAE", "RMSE", "MAPE"]])

# ===================== SELECT BEST MODEL =====================
best_row = df_metrics.sort_values("RMSE").iloc[0]
best_model_name = best_row["Model"]
best_run_id = best_row["run_id"]

st.success(f"Best Model Selected: {best_model_name}")

# ===================== LOAD BEST MODEL =====================
model_uri = f"runs:/{best_run_id}/model"
best_model = mlflow.pyfunc.load_model(model_uri)

# ===================== FUTURE FEATURE CREATION =====================
def create_future_features(df, model):
    feature_cols = [
        "seasons", "hour", "month", "year", "day_of_week",
        "timeof_day", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
        "aqi_lag_6", "aqi_lag_12", "aqi_lag_24", "aqi_24hr_avg"
    ]

    last_row = df.iloc[-1]

    X_future = pd.DataFrame([{
        "seasons": last_row["seasons"],
        "hour": last_row["hour"],
        "month": last_row["month"],
        "year": last_row["year"],
        "day_of_week": last_row["day_of_week"],
        "timeof_day": last_row["timeof_day"],
        "aqi_lag_1": last_row["aqi_lag_1"],
        "aqi_lag_2": last_row["aqi_lag_2"],
        "aqi_lag_3": last_row["aqi_lag_3"],
        "aqi_lag_6": last_row["aqi_lag_6"],
        "aqi_lag_12": last_row["aqi_lag_12"],
        "aqi_lag_24": last_row["aqi_lag_24"],
        "aqi_24hr_avg": last_row["aqi_24hr_avg"],
    }])[feature_cols]

    preds_72 = model.predict(X_future)[0]
    return preds_72

# ===================== PREDICTION =====================
if st.button("Predict Next 3 Days AQI"):

    preds_72 = create_future_features(df, best_model)

    day1 = sum(preds_72[:24]) / 24
    day2 = sum(preds_72[24:48]) / 24
    day3 = sum(preds_72[48:72]) / 24

    st.subheader("AQI Forecast (Next 3 Days)")
    st.write(f"Day 1 AQI: {day1:.2f}")
    st.write(f"Day 2 AQI: {day2:.2f}")
    st.write(f"Day 3 AQI: {day3:.2f}")
