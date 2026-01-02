import sys
from pathlib import Path
import pandas as pd

# root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# config
INPUT_PATH = ROOT_DIR / "data" / "processed" / "logs_with_clusters.parquet"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "incidents.parquet"

TIME_WINDOW = "10min"   # incident window
MIN_EVENTS = 2          # minimum anomalies to form an incident

df = pd.read_parquet(INPUT_PATH)

# Only anomalous events
anomalies = df[df["is_anomaly"]].copy()

if anomalies.empty:
    print("No anomalies found. No incidents generated.")
    exit()

anomalies = anomalies.sort_values("timestamp")
anomalies["window"] = anomalies["timestamp"].dt.floor(TIME_WINDOW)

# aggregate anomalies into incidents
incidents = (
    anomalies
    .groupby("window")
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        event_count=("timestamp", "count"),
        sources=("source", lambda x: list(set(x))),
        components=("component", lambda x: list(set(x))),
        levels=("level", lambda x: list(set(x))),
        messages=("message", lambda x: list(x)[:5])  # sample messages
    )
    .reset_index(drop=True)
)

# filter weak incidents
incidents = incidents[incidents["event_count"] >= MIN_EVENTS]

# scoring
def severity_score(levels, count):
    score = count
    if "ERROR" in levels:
        score += 5
    elif "WARN" in levels:
        score += 3
    return score

incidents["severity_score"] = incidents.apply(
    lambda row: severity_score(row["levels"], row["event_count"]),
    axis=1
)

# labeling
def label_incident(score):
    if score >= 10:
        return "HIGH"
    elif score >= 5:
        return "MEDIUM"
    return "LOW"

incidents["severity"] = incidents["severity_score"].apply(label_incident)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
incidents.to_parquet(OUTPUT_PATH, index=False)

print("Incident detection complete")
print(f"Total incidents: {len(incidents)}")
print(incidents[["start_time", "end_time", "event_count", "severity"]])
