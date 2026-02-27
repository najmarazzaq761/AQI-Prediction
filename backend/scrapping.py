from datetime import datetime, timezone, timedelta
import pytz
import os
from dotenv import load_dotenv
import requests
import csv
import time

load_dotenv()

def pkt_to_utc_unix(time_str, fmt="%Y-%m-%d %H-%M-%S"):
    pkt = pytz.timezone("Asia/Karachi")
    dt_pkt = pkt.localize(datetime.strptime(time_str, fmt))
    return int(dt_pkt.astimezone(timezone.utc).timestamp())

# time range 
start_unix = pkt_to_utc_unix("2025-07-01 00-00-00")
end_unix   = pkt_to_utc_unix("2026-01-25 00-00-00") 

lat = 30.746
lon = 73.331
API_KEY = os.getenv("AQI_API_KEY")

file_name = "data.csv"
file_exists = os.path.isfile(file_name)

seen = set()

with open(file_name, mode="a", newline="") as f:
    writer = None
    current = start_unix

    while current < end_unix:
        next_hour = current + 3600
        url = (
            "http://api.openweathermap.org/data/2.5/air_pollution/history"
            f"?lat={lat}&lon={lon}&start={current}&end={next_hour}&appid={API_KEY}"
        )

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            for record in data.get("list", []):
                ts = record["dt"]
                if ts in seen:
                    continue
                seen.add(ts)

                row = {
                    "timestamp": ts,
                    "aqi": record["main"]["aqi"],
                    **record["components"]
                }

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if not file_exists:
                        writer.writeheader()

                writer.writerow(row)

        else:
            print("API error:", response.status_code)

        current = next_hour
        time.sleep(1)  # avoid rate limits
