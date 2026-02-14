import pandas as pd
import requests
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
API_KEY = os.getenv("AQI_API_KEY")

LAT = 30.746
LON = 73.331


def get_mongo_client():
    return MongoClient(MONGO_URI)


# -------- Fetch new raw data from API --------
def fetch_latest_data():
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

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

    df = pd.DataFrame(rows)
    return df


# -------- Fetch recent historical data from feature store --------
def fetch_recent_raw_data(limit=30):
    client = get_mongo_client()
    db = client["aqi_feature_store"]
    collection = db["hourly_features"]   # changed here

    data = list(
        collection.find({}, {"_id": 0, "timestamp": 1, "aqi": 1})
        .sort("timestamp", -1)
        .limit(limit)
    )

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)
