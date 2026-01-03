import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_distances

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# --------------------------------------------------
# PATHS
# --------------------------------------------------
VECTORIZER_PATH = ROOT_DIR / "data" / "processed" / "embeddings" / "tfidf_vectorizer.joblib"
EMBEDDINGS_PATH = ROOT_DIR / "data" / "processed" / "embeddings" / "tfidf_embeddings.joblib"
HIST_LOGS_PATH = ROOT_DIR / "data" / "processed" / "logs_with_clusters.parquet"
LIVE_LOGS_PATH = ROOT_DIR / "data" / "processed" / "live_logs.parquet"

# --------------------------------------------------
# LOAD ARTIFACTS (ONCE)
# --------------------------------------------------
vectorizer = joblib.load(VECTORIZER_PATH)
historical_embeddings = joblib.load(EMBEDDINGS_PATH)
historical_logs = pd.read_parquet(HIST_LOGS_PATH)

# Only NORMAL logs define "normal behaviour"
normal_mask = (historical_logs["cluster_id"] != -1).values
NORMAL_CENTROID = historical_embeddings[normal_mask].mean(axis=0)
NORMAL_CENTROID = np.asarray(NORMAL_CENTROID).reshape(1, -1)

# Cache known templates
KNOWN_TEMPLATES = set(historical_logs["template"].astype(str).unique())

ANOMALY_DISTANCE_THRESHOLD = 0.35  # intentionally conservative

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(title="Log Intelligence Inference API")

# --------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------
class LogEvent(BaseModel):
    timestamp: datetime | None = None
    source: str
    level: str
    component: str
    message: str

# --------------------------------------------------
# ANOMALY LOGIC (IMPORTANT)
# --------------------------------------------------
def infer_anomaly(event: LogEvent) -> dict:
    """
    Multi-rule anomaly detection.
    Returns anomaly decision + reason.
    """
    text = event.message
    level = event.level.upper()

    # Rule 1: ERROR override
    if level == "ERROR":
        return {"is_anomaly": True, "reason": "ERROR_level"}

    # Rule 2: Unseen template
    if text not in KNOWN_TEMPLATES:
        return {"is_anomaly": True, "reason": "unseen_template"}

    # Rule 3: Distance-based novelty
    X_new = vectorizer.transform([text])
    dist = cosine_distances(X_new, NORMAL_CENTROID)[0][0]

    if dist > ANOMALY_DISTANCE_THRESHOLD:
        return {"is_anomaly": True, "reason": f"semantic_distance={dist:.3f}"}

    return {"is_anomaly": False, "reason": f"normal_distance={dist:.3f}"}

# --------------------------------------------------
# API ENDPOINT
# --------------------------------------------------
@app.post("/logs")
def ingest_log(event: LogEvent):
    ts = event.timestamp or datetime.utcnow()

    anomaly_result = infer_anomaly(event)

    row = {
        "timestamp": ts,
        "source": event.source,
        "level": event.level.upper(),
        "component": event.component,
        "message": event.message,
        "template": event.message,
        "is_anomaly": anomaly_result["is_anomaly"],
        "anomaly_reason": anomaly_result["reason"]
    }

    if LIVE_LOGS_PATH.exists():
        df = pd.read_parquet(LIVE_LOGS_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_parquet(LIVE_LOGS_PATH, index=False)

    return {
        "status": "ok",
        **anomaly_result
    }
