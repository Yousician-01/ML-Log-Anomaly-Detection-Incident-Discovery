import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# --------------------------------------------------
# PATHS
# --------------------------------------------------
LOG_PATH = ROOT_DIR / "data" / "processed" / "logs_with_clusters.parquet"
INCIDENTS_PATH = ROOT_DIR / "data" / "processed" / "final_incidents.parquet"
LIVE_LOGS_PATH = ROOT_DIR / "data" / "processed" / "live_logs.parquet"

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Log Intelligence & Incident Discovery",
    layout="wide"
)

st.title("🚨 Log Intelligence & Incident Discovery Dashboard")
st.caption("Unsupervised log clustering, anomaly detection, and incident aggregation")

# --------------------------------------------------
# LOAD BATCH DATA
# --------------------------------------------------
@st.cache_data
def load_batch_data():
    logs = pd.read_parquet(LOG_PATH)
    incidents = pd.read_parquet(INCIDENTS_PATH)
    return logs, incidents

logs_df, incidents_df = load_batch_data()

# --------------------------------------------------
# LOAD LIVE DATA (AUTO REFRESH)
# --------------------------------------------------
@st.cache_data(ttl=5)
def load_live_data():
    if LIVE_LOGS_PATH.exists():
        return pd.read_parquet(LIVE_LOGS_PATH)
    return pd.DataFrame()

live_df = load_live_data()

# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------
st.subheader("🟢 System Status")

if not live_df.empty and "is_anomaly" in live_df.columns and live_df["is_anomaly"].any():
    st.error("🚨 Live anomalies detected!")
elif incidents_df.empty:
    st.success("No actionable incidents detected. System operating normally.")
else:
    st.warning("Historical incidents detected. Review below.")

# --------------------------------------------------
# METRICS
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Logs Processed (Batch)", len(logs_df))
col2.metric("Clusters Identified", logs_df["cluster_id"].nunique() - 1)
col3.metric("Anomalous Events (Batch)", logs_df["is_anomaly"].sum())
col4.metric("Actionable Incidents", len(incidents_df))

# --------------------------------------------------
# LIVE LOG FEED
# --------------------------------------------------
st.subheader("📡 Live Logs (via API)")

if live_df.empty:
    st.info("No live logs received yet.")
else:
    st.dataframe(
        live_df.sort_values("timestamp", ascending=False).head(50),
        use_container_width=True
    )

# --------------------------------------------------
# LIVE ANOMALIES
# --------------------------------------------------
st.subheader("🚨 Live Anomalies")

if not live_df.empty and "is_anomaly" in live_df.columns:
    live_anomalies = live_df[live_df["is_anomaly"]]
    if live_anomalies.empty:
        st.success("No live anomalies detected.")
    else:
        st.error(f"{len(live_anomalies)} live anomaly/anomalies detected!")
        st.dataframe(
            live_anomalies.sort_values("timestamp", ascending=False),
            use_container_width=True
        )

# --------------------------------------------------
# ACTIONABLE INCIDENTS (BATCH)
# --------------------------------------------------
st.subheader("🚨 Actionable Incidents (Batch)")

if incidents_df.empty:
    st.info("No actionable incidents to display.")
else:
    st.dataframe(
        incidents_df[
            ["start_time", "end_time", "event_count", "severity", "components"]
        ],
        use_container_width=True
    )

# --------------------------------------------------
# ANOMALY EXPLORER (BATCH)
# --------------------------------------------------
st.subheader("🧪 Anomalous Log Events (Batch)")

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

# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------
st.subheader("🧩 Log Cluster Distribution")

cluster_counts = (
    logs_df[logs_df["cluster_id"] != -1]
    .groupby("cluster_id")
    .size()
    .sort_values(ascending=False)
)

st.bar_chart(cluster_counts)

# --------------------------------------------------
# TIMELINE VIEW
# --------------------------------------------------
st.subheader("⏱ Log Timeline (Batch)")

timeline_df = logs_df.copy()
timeline_df["anomaly_flag"] = timeline_df["is_anomaly"].map(
    {True: "Anomaly", False: "Normal"}
)

st.dataframe(
    timeline_df[
        ["timestamp", "source", "level", "component", "message", "cluster_id", "anomaly_flag"]
    ].sort_values("timestamp"),
    use_container_width=True,
    height=300
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Built with unsupervised learning: TF-IDF embeddings, density-based clustering, "
    "temporal aggregation, incident suppression, and live inference via FastAPI. "
    "© 2024 Log Intelligence System"
)
