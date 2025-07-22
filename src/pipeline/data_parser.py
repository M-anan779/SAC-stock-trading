import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import freeze_support
from functools import partial

# ------------------ Top-Level Constants ------------------

TICKERS = [
    'NVDA', 'TSLA', 'INTC', 'AAPL', 'AMZN', 'GOOGL', 'AMD', 'AVGO', 'MSFT',
    'MRVL', 'CSCO', 'MU', 'META', 'SHOP', 'QCOM', 'TXN', 'MCHP', 'ON',
    'PYPL', 'CRM', 'ADBE', 'NFLX'
]   # Nasdaq 100 highest volume tech sector stock picks

CSV_COLS = ['ticker', 'window_start', 'open', 'high', 'low', 'close', 'volume']
CSV_DTYPES = {
    'ticker': pd.CategoricalDtype(categories=TICKERS),
    'window_start': 'int64',  # UNIX timestamp (nanoseconds)
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'float32'
}

# ------------------ Worker Function ------------------

def parse_file(file_path, output_dir):
    # Read compressed CSV directly with optimized engine
    df = pd.read_csv(
        file_path,
        usecols=CSV_COLS,
        dtype=CSV_DTYPES,
        compression='gzip',
        engine='pyarrow' 
    )

    # Filter by ticker
    df = df[df['ticker'].isin(TICKERS)]

    # Convert unix timestamp to datetime
    df['window_start'] = pd.to_datetime(df['window_start'], unit='ns')

    # Filter by session hours
    df = df.set_index('window_start').between_time('08:00', '16:00').reset_index()

    # Keep only desired columns (remove transactions column)
    df = df[CSV_COLS]

    # Save uncompressed as csv
    output_path = output_dir / (file_path.stem)
    df.to_csv(output_path, index=False)

    return len(df)

# ------------------ Multiprocessing ------------------

def run():
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / 'data' / 'raw'
    parsed_dir = project_root / 'data' / 'parsed'

    parsed_dir.mkdir(parents=True, exist_ok=True)
    file_list = [f for f in raw_dir.iterdir() if f.suffix == '.gz']

    process = partial(parse_file, output_dir=parsed_dir)

    with ProcessPoolExecutor(max_workers=8) as executor:
        counts = list(tqdm(
            executor.map(process, file_list),  
            total=len(file_list)
        ))

    print(f"Total candles across all files: {sum(counts)}")

if __name__ == '__main__':
    freeze_support()
    run()
