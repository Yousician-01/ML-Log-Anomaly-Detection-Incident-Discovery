import sys
from pathlib import Path
import pandas as pd
from datetime import timedelta

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

INPUT_PATH = ROOT_DIR / "data" / "processed" / "unified_logs.parquet"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "unified_logs_with_incidents.parquet"

INJECTION_TIME = pd.Timestamp("2008-11-09 21:00:00")
NUM_EVENTS = 6
TIME_SPACING_SEC = 20

df = pd.read_parquet(INPUT_PATH)

synthetic_events = []

for i in range(NUM_EVENTS):
    synthetic_events.append({
        "timestamp": INJECTION_TIME + timedelta(seconds=i * TIME_SPACING_SEC),
        "source": "hdfs",
        "level": "ERROR",
        "component": "dfs.DataNode",
        "message": "Disk failure detected on DataNode",
        "event_id": "SYNTH_DISK_FAIL",
        "template": "Disk failure detected on DataNode",
        "params": None
    })

synthetic_df = pd.DataFrame(synthetic_events)

df_augmented = pd.concat([df, synthetic_df], ignore_index=True)
df_augmented = df_augmented.sort_values("timestamp").reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_augmented.to_parquet(OUTPUT_PATH, index=False)

print("Synthetic incident injected")
print(f"Injected events: {len(synthetic_df)}")
print(f"Saved augmented logs to: {OUTPUT_PATH}")
