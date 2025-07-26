import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import freeze_support
from functools import partial

# ------------------ Aggregation Main Logic ------------------

# This helper prevents using an excessive amount of python for loops which are slower
# The pandas methods used here are running C loops underneath
def vectorizedinput(ticker_df, index_range, price_cols, threshold, date):
    ticker_name = ticker_df['ticker'].iloc[0]
    ticker_df = ticker_df.drop(columns=['ticker'])  # Dropping temporarily prevents errors during aggregation

    # Set index to window_start and reindex to create rows for missing minutes
    ticker_df = ticker_df.set_index('window_start').reindex(index_range)

    # Check tickers only past 10:30 AM for long runs of NaN values
    check_df = ticker_df.loc[f'{date} 10:31:00' : f'{date} 16:00:00']

    # Using a mask and then summing boolean values for the runs
    mask = check_df[price_cols].isna().any(axis=1)
    flipped = ~mask
    run_ids = flipped.cumsum()
    group_sizes = mask.groupby(run_ids).transform('sum')
    
    if (group_sizes > threshold).any():
        return pd.DataFrame()  # Skip this group
    
    # Forward fill missing price data and set volume to 0
    ticker_df[price_cols] = ticker_df[price_cols].ffill()
    ticker_df['volume'] = ticker_df['volume'].fillna(0)

    # Resample the clean run of 1-minute candles to 5-minute
    resampled = ticker_df.resample('5min').agg({
        price_cols[0]: 'first',   # open
        price_cols[1]: 'max',     # high
        price_cols[2]: 'min',     # low
        price_cols[3]: 'last',    # close
        'volume': 'sum'
    })
    resampled = resampled.loc[f'{date} 09:30:00' : f'{date} 15:55:00'].reset_index()    # Final candle begins at 3:55 PM

    # Add the ticker column again and reorder columns
    resampled['ticker'] = ticker_name
    output = resampled[['ticker', 'window_start', *price_cols, 'volume']]

    return output


# ------------------ Worker Function ------------------

def aggregate_file(file_path, output_dir, csv_dtypes):
    # Read CSV with explicit dtypes
    df = pd.read_csv(file_path, dtype=csv_dtypes)
    df['window_start'] = pd.to_datetime(df['window_start'], format='%Y-%m-%d %H:%M:%S')

    price_cols = ['open', 'high', 'low', 'close']
    threshold = 9
    
    # Build full 1-min trading timeline (format date is required for full datetime object)
    trading_date = df['window_start'].dt.date.iloc[0]
    start = pd.Timestamp(trading_date).replace(hour=8, minute=0, second=0)
    end = pd.Timestamp(trading_date).replace(hour=16, minute=0, second=0)
    full_minute_range = pd.date_range(start, end, freq='1min', name='window_start')

    # Clean and aggregate the data before saving as CSV
    final_df = (
        df.groupby('ticker', group_keys=False, observed=True)
        .apply(vectorizedinput, index_range=full_minute_range, price_cols=price_cols, threshold=threshold,
               date=trading_date)
    )

    if not df.empty and len(final_df) > 0:  # Check for empty CSVs, empty alone won't work due to columns
        final_df.to_csv(output_dir / file_path.name, index=False)

    return len(final_df)

# ------------------ Multiprocessing ------------------
def run():
    project_root = Path(__file__).resolve().parents[2]
    parsed_dir   = project_root / 'data' / 'parsed'
    aggregated_dir = project_root / 'data' / 'aggregated'

    # Explicit dtypes
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
        counts = list(tqdm(
            executor.map(process, files, chunksize=1), 
            total=len(files)
        ))
        
    total_rows = sum(counts)
    print(f"Total candles across all files: {(total_rows)}")

if __name__ == '__main__':
    freeze_support()
    run()