import json

import pandas as pd

from pipeline.producer import replay_to_files


def test_replay_to_files_writes_all_messages_in_batches(tmp_path):
    df = pd.DataFrame(
        {
            "headline_id": range(1, 8),
            "ticker": ["AAPL"] * 7,
            "published_at": [f"2023-01-0{i} 00:00:00" for i in range(1, 8)],
            "headline": [f"headline {i}" for i in range(1, 8)],
            "publisher": [""] * 7,
        }
    )
    landing = tmp_path / "landing"
    n = replay_to_files(df, landing, batch_size=3)
    files = sorted(landing.glob("part-*.json"))
    assert n == 7
    assert len(files) == 3  # 3 + 3 + 1
    lines = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    assert [m["headline_id"] for m in lines] == list(range(1, 8))
    assert not list(landing.glob(".*.tmp"))
