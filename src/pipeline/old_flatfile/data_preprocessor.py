import os
import pandas as pd
from pathlib import Path
from utils.technical_indicators import TechnicalIndicators

def vectorizedinput(ticker_df, processed_dir):
    ticker_name = ticker_df['ticker'].iloc[0]
    print(f"Applying technical indicators to: {ticker_name}")
    ticker_df = ticker_df.set_index('window_start')
    ti = TechnicalIndicators(ticker_df)
    ticker_df = ti.generate_all_indicators().reset_index()

    ticker_df.insert(0, 'date', ticker_df['window_start'].dt.date)
    ticker_df['window_start'] = ticker_df['window_start'].dt.time

    ticker_df.to_csv(processed_dir / f'{ticker_name}.csv', index=False)

    return None

# ------------------ Preprocessing Logic ------------------

def run():
    project_root = Path(__file__).resolve().parents[2]
    aggregated_dir = project_root / 'data' / 'aggregated'
    processed_dir  = project_root / 'data' / 'processed'
    
    processed_dir.mkdir(parents=True, exist_ok=True)

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

    print('\nLoading aggregated files...')

    # Set up files for aggregation, skip files that are empty
    df_list = []
    for file in aggregated_dir.iterdir():
        if os.path.getsize(file) < 2048:
            continue
        df_list.append(pd.read_csv(file, dtype=csv_dtypes))

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df['window_start'] = pd.to_datetime(combined_df['window_start'], format='%Y-%m-%d %H:%M:%S')
    
    combined_df.groupby('ticker', group_keys=False, observed=True).apply(vectorizedinput, processed_dir)
    
    print('Done!')

if __name__ == '__main__':
    run()
