"""Download the two public datasets into data/raw/.

  news.parquet            benstaf/FNSPID-nasdaq-100-post2019-1newsperrow (about 204 MB)
  prices_2019_2023.csv    benstaf/nasdaq_2013_2023, trade_data_2019_2023.csv (about 31 MB)

Both come from Hugging Face over plain HTTPS, no account needed. Files that
already exist with the expected size are skipped.

Usage:
  python scripts/download_data.py
"""
import sys
import time
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "news.parquet": "https://huggingface.co/datasets/benstaf/FNSPID-nasdaq-100-post2019-1newsperrow/"
    "resolve/main/data/train-00000-of-00001.parquet",
    "prices_2019_2023.csv": "https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/"
    "resolve/main/trade_data_2019_2023.csv",
}


def download(name: str, url: str, retries: int = 4) -> Path:
    dest = RAW / name
    for attempt in range(retries):
        try:
            head = requests.head(url, allow_redirects=True, timeout=30)
            size = int(head.headers.get("content-length", 0))
            if dest.exists() and size and dest.stat().st_size == size:
                print(f"skip {name} (already {size:,} bytes)")
                return dest
            print(f"downloading {name} ({size:,} bytes)")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.rename(dest)
            return dest
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"{name}: {e}, retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"could not download {name}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        download(name, url)


if __name__ == "__main__":
    main()
