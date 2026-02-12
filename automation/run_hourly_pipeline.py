from data_ingestion import fetch_latest_data
from feature_engineering import create_features
from feature_store_writer import write_features_to_store

def main():
    print("Fetching latest raw data...")
    raw_df = fetch_latest_data()

    if raw_df.empty:
        print("No new data found.")
        return

    print("Creating features...")
    feature_df = create_features(raw_df)

    if feature_df.empty:
        print("No valid feature rows after engineering.")
        return

    print("Writing features to MongoDB...")
    write_features_to_store(feature_df)

    print("Hourly pipeline completed successfully.")

if __name__ == "__main__":
    main()
