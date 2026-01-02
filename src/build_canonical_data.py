import pandas as pd
from pathlib import Path

# APACHE LOADER
def load_apache_structured(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(columns={
        "Time": "timestamp",
        "Level": "level",
        "Content": "message",
        "EventId": "event_id",
        "EventTemplate": "template"
    })

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%a %b %d %H:%M:%S %Y",
        errors="coerce"
    )

    df["level"] = df["level"].str.upper()
    df["component"] = "apache"
    df["source"] = "apache"
    df["params"] = None

    return df[
        [
            "timestamp",
            "source",
            "level",
            "component",
            "message",
            "event_id",
            "template",
            "params",
        ]
    ]

# HDFS LOADER
def load_hdfs_structured(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Combine Date + Time = timestamp
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%y%m%d %H%M%S",
        errors="coerce"
    )

    df = df.rename(columns={
        "Level": "level",
        "Component": "component",
        "Content": "message",
        "EventId": "event_id",
        "EventTemplate": "template"
    })

    df["level"] = df["level"].str.upper()
    df["source"] = "hdfs"
    df["params"] = None

    return df[
        [
            "timestamp",
            "source",
            "level",
            "component",
            "message",
            "event_id",
            "template",
            "params",
        ]
    ]



# Main Pipeline
def main():
    apache_path = "data/raw/loghub/apache/apache_structured.csv"
    hdfs_path = "data/raw/loghub/hdfs/hdfs_structured.csv"

    apache_df = load_apache_structured(apache_path)
    hdfs_df = load_hdfs_structured(hdfs_path)

    logs_df = pd.concat([apache_df, hdfs_df], ignore_index=True)
    logs_df = logs_df.sort_values("timestamp").reset_index(drop=True)

    output_path = Path("data/processed/unified_logs.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logs_df.to_parquet(output_path, index=False)

    print("Canonical log dataset built successfully")
    print(f"Total log events: {len(logs_df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
