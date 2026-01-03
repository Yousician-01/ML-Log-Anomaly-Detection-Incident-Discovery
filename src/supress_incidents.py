import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

INCIDENT_PATH = ROOT_DIR / "data" / "processed" / "incidents.parquet"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "final_incidents.parquet"

incidents = pd.read_parquet(INCIDENT_PATH)

if incidents.empty:
    print("⚠️ No incidents to suppress.")
    incidents.to_parquet(OUTPUT_PATH, index=False)
    exit()

# Suppression rules
def should_suppress(row):
    """
    Returns True if incident should be suppressed
    """
    # Rule 1: Low severity INFO-only incidents
    if (
        row["severity"] == "LOW"
        and all(level == "INFO" for level in row["levels"])
    ):
        return True

    # Rule 2: Known benign HDFS maintenance
    if (
        "dfs.FSNamesystem" in row["components"]
        and any(
            kw in " ".join(row["messages"]).lower()
            for kw in ["delete", "replicate", "block"]
        )
        and row["severity"] == "LOW"
    ):
        return True

    # Rule 3: Very small incidents
    if row["event_count"] < 3:
        return True

    return False


# add suppression flag
incidents["is_suppressed"] = incidents.apply(should_suppress, axis=1)

# Final incidents after suppression
final_incidents = incidents[~incidents["is_suppressed"]].copy()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
final_incidents.to_parquet(OUTPUT_PATH, index=False)

# report
print("Incident suppression complete")
print(f"Total incidents (before): {len(incidents)}")
print(f"Suppressed incidents: {incidents['is_suppressed'].sum()}")
print(f"Final actionable incidents: {len(final_incidents)}")

if not final_incidents.empty:
    print(
        final_incidents[
            ["start_time", "end_time", "event_count", "severity"]
        ]
    )
else:
    print("No actionable incidents — system stable.")
