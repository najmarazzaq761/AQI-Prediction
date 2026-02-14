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
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat=30.746&lon=73.331&appid={API_URL}"
    
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    records = data.get("list", [])
    rows = []

    for record in records:
        row = {
            "timestamp": record["dt"],
            "aqi": record["main"]["aqi"],
            **record["components"]
        }
        rows.append(row)


    df = pd.DataFrame(data)

    if "timestamp" not in df.columns:
        raise ValueError("timestamp column missing from API data")

    df = df[~df["timestamp"].isin(existing_timestamps)]

    return df
