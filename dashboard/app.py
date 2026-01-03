import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

LOG_PATH = ROOT_DIR / "data" / "processed" / "logs_with_clusters.parquet"
INCIDENTS_PATH = ROOT_DIR / "data" / "processed" / "final_incidents.parquet"

# Config
st.set_page_config(
    page_title="Log Intelligence & Incident Discovery", 
    layout="wide"
    )

st.title("Log Intelligence & Incident Discovery Dashboard")
st.caption("Unsupervised log clustering, anomaly detection, and incident aggregation")

# Load data
@st.cache_data
def load_data():
    logs = pd.read_parquet(LOG_PATH)
    incidents = pd.read_parquet(INCIDENTS_PATH)
    return logs, incidents

logs_df, incidents_df = load_data()

# System Status

st.subheader("System Status")

if incidents_df.empty:
    st.success("No actionable incidents detected. System operating normally.")
else:
    st.error(f"{len(incidents_df)} actionable incidents detected!")

# Matrics

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Logs Processed", len(logs_df))
col2.metric("Clusters Identified", logs_df["cluster_id"].nunique() - 1)
col3.metric("Anomalous Events", logs_df["is_anomaly"].sum())
col4.metric("Actionable Incidents", len(incidents_df))

# Incident Table
st.subheader("Actionable Incidents")

if incidents_df.empty:
    st.info("No actionable incidents to display.")
else:
    st.dataframe(
        incidents_df[
            ["start_time", "end_time", "event_count", "severity", "components"]
        ],
        use_container_width=True
    )

# Anomaly Explorer
st.subheader("Anomalous Log Events")

anomalies = logs_df[logs_df["is_anomaly"]].copy()

if anomalies.empty:
    st.info("No anomalous events detected.")
else:
    st.dataframe(
        anomalies[
            ["timestamp", "source", "level", "component", "message", "cluster_id"]
        ].sort_values("timestamp"),
        use_container_width=True
    )

# Cluster Distribution
st.subheader("Log Cluster Distribution")

cluster_counts = (
    logs_df[logs_df["cluster_id"] != -1].groupby("cluster_id").size().sort_values(ascending=False)
)

st.bar_chart(cluster_counts)

# Timeline View
st.subheader("Incident Timeline (Anomalies Highlighted)")

timeline_df = logs_df.copy()
timeline_df["is _anomaly"] = timeline_df["is_anomaly"].map({True: "Anomaly", False: "Normal"})

st.dataframe(
    timeline_df[
        ["timestamp", "source", "level", "component", "message", "cluster_id", "is _anomaly"]
    ].sort_values("timestamp"),
    use_container_width=True,
    height=300
)

# Footer
st.markdown("---")
st.caption(
    "Built with unsupervised learning: TF-IDF embeddings, density-based clustering, "
    "temporal aggregation, and incident suppression logic."
    "© 2024 Log Intelligence Dashboard"
    )