from pymongo import MongoClient 
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, quote_plus
import pandas as pd

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# Parse and escape username/password in URI
parsed = urlparse(mongo_uri)
if parsed.username:
    escaped_user = quote_plus(parsed.username)
    escaped_pass = quote_plus(parsed.password) if parsed.password else ''
    netloc = f"{escaped_user}:{escaped_pass}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    mongo_uri = parsed._replace(netloc=netloc).geturl()

client = MongoClient(mongo_uri)

# print(client.list_database_names())

db = client["aqi_feature_store"]
collection = db["hourly_features"]

df = pd.read_csv("final_features.csv")

records = df.to_dict(orient="records")
collection.insert_many(records)

# Prevent duplicates
collection.create_index("timestamp", unique=True)
print("Data inserted into MongoDB successfully.")

