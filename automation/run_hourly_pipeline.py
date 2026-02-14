from data_ingestion import (
    fetch_latest_data,
    fetch_recent_raw_data
)
from feature_engineering import create_features
from feature_store_writer import write_features_to_store
import pandas as pd


def main():
    print("Fetching latest raw data...")
    new_raw_df = fetch_latest_data()

    if new_raw_df.empty:
        print("No new data found.")
        return

    print("Fetching historical data from feature store...")
    history_df = fetch_recent_raw_data(limit=30)

    combined_df = pd.concat([history_df, new_raw_df])
    combined_df = combined_df.drop_duplicates("timestamp")
    combined_df = combined_df.sort_values("timestamp")

    print("Creating features...")
    feature_df = create_features(combined_df)

    if feature_df.empty:
        print("Not enough history for lag features yet.")
        return

    latest_feature = feature_df.sort_values("timestamp").tail(1)

    print("Writing features to MongoDB...")
    write_features_to_store(latest_feature)

    print("Hourly pipeline completed successfully.")


if __name__ == "__main__":
    main()
