import pandas as pd
import requests
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
API_URL = os.getenv("AQI_API_KEY")

def fetch_latest_data():
    client = MongoClient(MONGO_URI)
    db = client["aqi_feature_store"]
    collection = db["hourly_features"]

    existing_timestamps = set(
        collection.distinct("timestamp")
    )

    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)

    if "timestamp" not in df.columns:
        raise ValueError("timestamp column missing from API data")

    df = df[~df["timestamp"].isin(existing_timestamps)]

    return df
