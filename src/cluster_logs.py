import pandas as pd
import joblib
import sys
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# config
DATA_PATH = "data/processed/unified_logs.parquet"
EMBED_PATH = "data/processed/embeddings/tfidf_embeddings.joblib"
OUTPUT_PATH = "data/processed/logs_with_clusters.parquet"

EPS = 0.7
MIN_SAMPLES = 10

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

print("Loading data...")
df = pd.read_parquet(DATA_PATH)
X = joblib.load(EMBED_PATH)

# scaling 
X_scaled = StandardScaler(with_mean=False).fit_transform(X)

# clustering
print("Running DBSCAN clustering...")
dbscan = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES,
    metric="cosine"
)

cluster_labels = dbscan.fit_predict(X_scaled)

df["cluster_id"] = cluster_labels
df["is_anomaly"] = df["cluster_id"] == -1

# save results
output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path, index=False)

# report
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_anomalies = (cluster_labels == -1).sum()

print("DBSCAN clustering complete")
print(f"Clusters found: {n_clusters}")
print(f"Anomalies detected: {n_anomalies}")
print(f"Saved to: {OUTPUT_PATH}")


df = pd.read_parquet("data/processed/logs_with_clusters.parquet")

anomalies = df[df["is_anomaly"]]

print(anomalies[[
    "source",
    "level",
    "component",
    "message"
]].head(10))
