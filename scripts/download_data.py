"""Download the public datasets into data/raw/.

  news.parquet               benstaf/FNSPID-nasdaq-100-post2019-1newsperrow (about 204 MB)
  prices_2019_2023.csv       benstaf/nasdaq_2013_2023, trade_data_2019_2023.csv (about 31 MB)
  lm_master_dictionary.csv   Loughran-McDonald master dictionary (about 9 MB, optional)

The first two come from Hugging Face over plain HTTPS, no account needed.
The LM dictionary is a Google Drive link from https://sraf.nd.edu. It is free
for academic use only, so it is never committed. If that link fails, the
pipeline still runs and only skips the LM comparison.

Files that already exist with the expected size are skipped.

Usage:
  python scripts/download_data.py
"""
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger("download_data")

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "news.parquet": "https://huggingface.co/datasets/benstaf/FNSPID-nasdaq-100-post2019-1newsperrow/"
    "resolve/main/data/train-00000-of-00001.parquet",
    "prices_2019_2023.csv": "https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/"
    "resolve/main/trade_data_2019_2023.csv",
}
OPTIONAL = {
    # Loughran-McDonald_MasterDictionary_1993-2025.csv, linked from
    # https://sraf.nd.edu/loughranmcdonald-master-dictionary/
    "lm_master_dictionary.csv": "https://drive.usercontent.google.com/download"
    "?id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF&export=download&confirm=t",
}


def download(name: str, url: str, retries: int = 4) -> Path:
    dest = RAW / name
    for attempt in range(retries):
        try:
            head = requests.head(url, allow_redirects=True, timeout=30)
            size = int(head.headers.get("content-length", 0))
            if dest.exists() and (dest.stat().st_size == size or not size):
                log.info("skip %s (already %s bytes)", name, f"{dest.stat().st_size:,}")
                return dest
            log.info("downloading %s (%s bytes)", name, f"{size:,}" if size else "unknown")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.rename(dest)
            return dest
        except requests.RequestException as e:
            wait = 2**attempt
            log.warning("%s: %s, retrying in %ss", name, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"could not download {name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        download(name, url)
    for name, url in OPTIONAL.items():
        try:
            dest = download(name, url)
            if not dest.read_text(encoding="utf-8", errors="ignore")[:200].startswith("Word,"):
                dest.unlink()
                raise RuntimeError("the link did not return the CSV")
        except RuntimeError as e:
            log.warning("%s: %s. The LM comparison will be skipped.", name, e)


if __name__ == "__main__":
    main()
