import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

       # Time based features
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Time components
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday

    # Time of day classification
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    df['time_of_day'] = df['hour'].apply(get_time_of_day)

    # Seasonal classification based on month
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'

    df['season'] = df['month'].apply(get_season)

    # Encode categorical features 
    season_map = {
        "Spring": 0,
        "Summer": 1,
        "Autumn": 2,
        "Winter": 3
    }

    time_map = {
        "Morning": 0,
        "Afternoon": 1,
        "Evening": 2,
        "Night": 3
    }

    df["seasons"] = df["season"].map(season_map)
    df["timeof_day"] = df["time_of_day"].map(time_map)

    df.drop(columns=["season", "time_of_day"], inplace=True)

    #  Sort by time 
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lag features
    lags = [1, 2, 3, 6, 12, 24]
    for lag in lags:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    #  Rolling feature 
    df["aqi_24hr_avg"] = df["aqi"].rolling(24).mean().round(2)

    #  Final clean dataset 
    df = df.dropna().reset_index(drop=True)

    final_df = df[
        [
            "timestamp",
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
            "aqi"
        ]
    ]

    return final_df
