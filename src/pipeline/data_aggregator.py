import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import freeze_support
from functools import partial

# ------------------ Worker Function ------------------

def aggregate_file(file_path: Path, output_dir: Path, csv_dtypes: dict):
    # 1. Read CSV with explicit dtypes
    df = pd.read_csv(file_path, dtype=csv_dtypes)
    df['window_start'] = pd.to_datetime(df['window_start'], format='%H:%M:%S', errors='coerce')

    # 2. Build full 1-min trading timeline (dummy date)
    dummy_date = "2000-01-01"
    start = f"{dummy_date} 09:30:00"
    end   = f"{dummy_date} 16:00:00"
    full_minutes = pd.date_range(start, end, freq="1min")

    results = []
    ohlc_cols = ['open', 'high', 'low', 'close']

    # 3. Process each ticker
    for ticker, group in df.groupby('ticker'):
        group = group.copy()
        group = group.set_index('window_start').sort_index()

        # Align to dummy datetime index
        times = group.index.strftime('%H:%M:%S')
        group.index = pd.to_datetime(dummy_date + ' ' + times)

        # Fill in missing minutes
        group = group.reindex(full_minutes)

        # 4. Drop tickers with >14 missing OHLC in a row
        mask = group[ohlc_cols].isna().any(axis=1)
        max_run = 0
        run = 0
        for m in mask:
            if m:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run > 14:
            continue

        # 5. Fill missing candles from last known OHLC with volume set to 0
        group[ohlc_cols] = group[ohlc_cols].ffill()
        group['volume'] = group['volume'].fillna(0)

        # 6. Resample into 5-minute candles
        agg_dict = {
            'open':  'first',
            'high':  'max',
            'low':   'min',
            'close': 'last',
            'volume':'sum'
        }
        five_min = group.resample('5min').agg(agg_dict)
        five_min['ticker'] = ticker

        # 8. Reset index to time-only
        five_min = five_min.reset_index()
        five_min.rename(columns={'window_start': 'index'}, inplace=True)
        five_min['window_start'] = five_min['index'].dt.time
        five_min = five_min.drop(columns=['index'])

        five_min = five_min[cols_order]

        results.append(five_min)

    # 9. Combine final DataFrame
    cols_order = ['ticker', 'window_start'] + ohlc_cols + ['volume']
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df[cols_order]

    # Only write files that are not empty
    output_file = output_dir / file_path.name
    if not final_df.empty:
        final_df.to_csv(output_file, index=False)

    # Return row count for this file
    return len(final_df)

# ------------------ Multiprocessing ------------------

def run():
    project_root = Path(__file__).resolve().parents[2]
    parsed_dir   = project_root / 'data' / 'parsed'
    aggregated_dir = project_root / 'data' / 'aggregated'

    # Explicit dtypes including window_start
    csv_dtypes = {
        'ticker':       'category',
        'window_start': 'string',
        'open':         'float32',
        'high':         'float32',
        'low':          'float32',
        'close':        'float32',
        'volume':       'float32'
    }

    aggregated_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in parsed_dir.iterdir() if f.suffix == '.csv']

    process = partial(aggregate_file, output_dir=aggregated_dir, csv_dtypes=csv_dtypes)
    with ProcessPoolExecutor(max_workers=8) as executor:
        counts = list(tqdm(executor.map(process, files), total=len(files)))

    total = sum(counts)
    print(f"Total candles across all files: {total}")

if __name__ == '__main__':
    freeze_support()
    run()
