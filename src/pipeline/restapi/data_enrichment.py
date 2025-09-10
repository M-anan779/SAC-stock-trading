import pandas as pd
import datetime as dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils.feature_generation import TechnicalIndicators

def compute_features(file, csv_dtypes, training_dir, validation_dir):
    print(f"Working on: {file.name}")

    ticker_df = pd.read_csv(file, dtype=csv_dtypes, parse_dates=["timestamp"]).set_index("timestamp")
    ti = TechnicalIndicators(ticker_df)
    ticker_df = ti.generate_all_indicators().reset_index().dropna()

    dt_series = pd.to_datetime(ticker_df["timestamp"])
    date_series = dt_series.dt.date
    time_series = dt_series.dt.time

    ticker_df.insert(0, "time", time_series)
    ticker_df.insert(0, "date", date_series)
    ticker_df.insert(0, "ticker", file.stem)
    ticker_df = ticker_df.drop(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # split training and validation data sets
    training_df = ticker_df[ticker_df["date"].between(dt.date(2015,9,1), dt.date(2023,9,1))]
    validation_df = ticker_df[ticker_df["date"].between(dt.date(2023,9,1), dt.date(2025,9,1))]

    # save both training + validation sets as CSV
    training_df.to_csv(training_dir / file.name, index=False)
    validation_df.to_csv(validation_dir / file.name, index=False)

def run(config):
    input_dir = Path(config["input_dir"])
    training_dir = Path(config["training_dir"])
    validation_dir = Path(config["validation_dir"])
    
    training_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Explicit dtypes
    csv_dtypes = {
        "timestamp": "string",
        "open":      "float32",
        "high":      "float32",
        "low":       "float32",
        "close":     "float32",
        "volume":    "float32"
    }

    print("Loading raw files...\n")
    
    file_list = [f for f in input_dir.iterdir()]
    
    partial_func = partial(compute_features, csv_dtypes=csv_dtypes, training_dir=training_dir, validation_dir=validation_dir)
    with ProcessPoolExecutor() as executor:
        list(executor.map(partial_func, file_list))
    print("Done!")

if __name__ == "__main__":
    run()
