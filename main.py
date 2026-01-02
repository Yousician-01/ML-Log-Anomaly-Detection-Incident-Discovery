import pandas as pd

df = pd.read_parquet("data/processed/logs_with_clusters.parquet")

anomalies = df[df["is_anomaly"]]

print(anomalies[[
    "source",
    "level",
    "component",
    "message"
]].head(10))