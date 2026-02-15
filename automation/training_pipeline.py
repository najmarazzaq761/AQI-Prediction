import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor


import xgboost as xgb
load_dotenv()

# Set environment variables FIRST
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
MONGO_URI = os.getenv("MONGO_URI")

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

EXPERIMENT_NAME = "AQI_Training"
mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 1. Fetch features from MongoDB
# --------------------------------------------------
def load_features():
    client = MongoClient(MONGO_URI)
    db = client["aqi_feature_store"]
    collection = db["hourly_features"]

    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# --------------------------------------------------
# 2. Create 72-step direct multi-output target
# --------------------------------------------------
def create_multi_output_target(df, horizon=72):
    df = df.copy()

    for i in range(1, horizon + 1):
        df[f"target_t+{i}"] = df["aqi"].shift(-i)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "seasons",
        "hour",
        "month",
        "year",
        "day_of_week",
        "timeof_day",
        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_12",
        "aqi_lag_24",
        "aqi_24hr_avg",
    ]

    X = df[feature_cols]
    y = df[[f"target_t+{i}" for i in range(1, horizon + 1)]]

    return X, y


# --------------------------------------------------
# 3. Time-based split
# --------------------------------------------------
def time_based_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 4. Evaluation metrics
# --------------------------------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


# --------------------------------------------------
# 5. Train and log model
# --------------------------------------------------
def train_and_log(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae, rmse, mape = evaluate(y_test, preds)

    with mlflow.start_run(run_name=f"{model_name}_72hr"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("horizon", 72)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)

        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="AQI_Forecaster"
        )


# --------------------------------------------------
# 6. Main training pipeline
# --------------------------------------------------
def main():
    print("Loading features...")
    df = load_features()

    print("Creating targets...")
    X, y = create_multi_output_target(df, horizon=72)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = time_based_split(X, y)


    print("Training XGBoost...")
    XGBoost = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
    )
    train_and_log(XGBoost, "XGBoost", X_train, X_test, y_train, y_test)

    print("Training MLP...")

    MLP = MultiOutputRegressor(
        MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=300,
            random_state=42
        )
    )

    train_and_log(MLP, "MLP", X_train, X_test, y_train, y_test)
    
    print("Training Random Forest...")
    rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=42)
    )
    train_and_log(rf, "RandomForest", X_train, X_test, y_train, y_test)

    print("Training completed.")


if __name__ == "__main__":
    main()
