import requests
import pandas as pd
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def fetch_ticker_data(ticker_name, dir_path, multiplier, timespan):
    api_key = "tuelfP7BndxIOjsQ3FeBz6bqyHpRXuFJ"
    rows = []

    start_date = "2015-08-01"
    end_date = "2025-07-01"
    dates = pd.date_range(start=start_date, end=end_date, freq="2MS")

    print(f"Requesting data from polygon: {ticker_name}, {multiplier}-{timespan}")

    for i in range(len(dates) - 1):
        current_date = dates[i].date()
        next_date = dates[i + 1].date()

        # Retry logic
        for attempt in range(3):
            try:
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker_name}/range/{multiplier}/{timespan}/{current_date}/{next_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
                response = requests.get(url)
                if response.status_code == 429:
                    print(f"[{ticker_name}] Rate limit (HTTP 429) on {current_date} → retrying...")
                    time.sleep(2 * (attempt + 1))
                    continue

                data = response.json()
                if "error" in data or data.get("status") == "ERROR":
                    print(f"[{ticker_name}] Polygon error on {current_date}: {data.get('error', 'Unknown')} → retrying...")
                    time.sleep(2 * (attempt + 1))
                    continue

                # Success
                break

            except requests.exceptions.RequestException as e:
                print(f"[{ticker_name}] Exception on {current_date}: {e} → retrying...")
                time.sleep(2 * (attempt + 1))
        else:
            print(f"[{ticker_name}] Failed after 3 retries: {current_date} → {next_date}")
            continue

        if not data.get("results"):
            continue

        for candle in data["results"]:
            rows.append({
                "t": candle["t"],
                "o": candle["o"],
                "h": candle["h"],
                "l": candle["l"],
                "c": candle["c"],
                "v": candle["v"]
            })

    print(f"Finished processing: {ticker_name}, Total: {len(rows)} candles")

    df = pd.DataFrame(rows)
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
    df = df.set_index("timestamp").between_time('09:30', '15:55').reset_index()
    file_path = dir_path / Path(f'{ticker_name}.csv')
    df.to_csv(file_path, index=False)

def run(tickers):
    multiplier = '5'
    timespan = 'minute'
    output_dir = Path(f'data/raw_tickers_{multiplier}-{timespan}')
    
    output_dir.mkdir(parents=True, exist_ok=True)

    partial_function = partial(fetch_ticker_data, dir_path=output_dir, multiplier=multiplier, timespan=timespan)
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(executor.map(partial_function, tickers))
    
if __name__ == '__main__':
    run()


