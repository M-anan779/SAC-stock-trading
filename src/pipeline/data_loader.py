import boto3
from botocore.config import Config
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime, timedelta
from functools import partial

# ------------------ Download Function ------------------

def download_key(key, raw_dir, s3_client, bucket_name):
    local_file_name = key.split('/')[-1]
    local_file_path = raw_dir / local_file_name

    if local_file_path.exists():
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        s3_client.download_file(bucket_name, key, str(local_file_path))
        print(f"✅ Downloaded: {key}")
    except Exception as e:
        print(f"❌ Failed to download {key}: {e}")


# ------------------ Main ------------------

def run():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / 'data'
    raw_dir = data_dir / 'raw'

    # AWS Session & S3 Client Setup
    session = boto3.Session(
        aws_access_key_id='0d9d0747-9942-49bb-af6d-d63d0d4e91f5',
        aws_secret_access_key='Cub04jV6iJzWd9ekfO3UYlwV0Wnotz6u',
    )
    s3 = session.client(
        's3',
        endpoint_url='https://files.polygon.io',
        config=Config(signature_version='s3v4'),
    )

    bucket_name = 'flatfiles'

    # Generate S3 object keys
    start_date = datetime(2020, 6, 4)
    end_date = datetime(2025, 6, 1)
    object_keys = [
        f"us_stocks_sip/minute_aggs_v1/{start_date.year}/{start_date.month:02d}/{start_date.strftime('%Y-%m-%d')}.csv.gz"
        for _ in range((end_date - start_date).days + 1)
    ]
    while start_date <= end_date:
        start_date += timedelta(days=1)

    download_fn = partial(download_key, raw_dir=raw_dir, s3_client=s3, bucket_name=bucket_name)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(download_fn, object_keys), total=len(object_keys)))


if __name__ == '__main__':
    run()
