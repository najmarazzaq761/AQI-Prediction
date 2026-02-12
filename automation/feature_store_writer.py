from pymongo import MongoClient, UpdateOne
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

def write_features_to_store(feature_df):
    client = MongoClient(MONGO_URI)
    db = client["aqi_feature_store"]
    collection = db["hourly_features"]

    collection.create_index("timestamp", unique=True)

    operations = []

    for record in feature_df.to_dict("records"):
        operations.append(
            UpdateOne(
                {"timestamp": record["timestamp"]},
                {"$setOnInsert": record},
                upsert=True
            )
        )

    if operations:
        collection.bulk_write(operations)
