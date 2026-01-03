import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

LOG_PATH = ROOT_DIR / "data" / "processed" / "logs_with_clusters.parquet"
INCIDENTS_PATH = ROOT_DIR / "data" / "processed" / "final_incidents.parquet"
LIVE_LOGS_PATH = ROOT_DIR / "data" / "processed" / "live_logs.parquet"
LIVE_SESSION_FLAG = ROOT_DIR / ".live_session_initialized"

# Reset live data
# RESET LIVE DATA ON STREAMLIT SERVER START (ONCE)
if not LIVE_SESSION_FLAG.exists():
    # First time Streamlit server starts
    if LIVE_LOGS_PATH.exists():
        LIVE_LOGS_PATH.unlink()

    # Create flag so this runs ONLY ONCE
    LIVE_SESSION_FLAG.touch()
    


# Config
st.set_page_config(
    page_title="Log Intelligence & Incident Discovery",
    layout="wide"
)

st.title("Log Intelligence & Incident Discovery Dashboard")
st.caption(
    "Unsupervised log clustering, anomaly detection, "
    "incident aggregation, and live inference"
)

@st.cache_data
def load_batch_data():
    logs = pd.read_parquet(LOG_PATH)
    incidents = pd.read_parquet(INCIDENTS_PATH)
    return logs, incidents

def load_live_data():
    if LIVE_LOGS_PATH.exists():
        return pd.read_parquet(LIVE_LOGS_PATH)
    return pd.DataFrame()

logs_df, incidents_df = load_batch_data()
live_df = load_live_data()

history_tab, live_tab = st.tabs(
    ["Historical Analysis", "Live Monitoring"]
)

# Live Monitoring Tab
with live_tab:
    st.subheader("Live System Status")
    if st.button("clear logs"):
        LIVE_SESSION_FLAG.unlink()
        st_autorefresh(key="live_refresh")

    if not live_df.empty and "is_anomaly" in live_df.columns and live_df["is_anomaly"].any():
        st.error("Live anomalies detected!")
    else:
        st.success("System stable. No live anomalies.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Live Logs Received", len(live_df))
    col2.metric(
        "Live Anomalies",
        int(live_df["is_anomaly"].sum()) if "is_anomaly" in live_df.columns else 0
    )
    col3.metric("Live Sources", live_df["source"].nunique() if not live_df.empty else 0)

    st.subheader("Live Log Stream")

    if live_df.empty:
        st.info("No live logs received yet.")
    else:
        st.dataframe(
            live_df.sort_values("timestamp", ascending=False).head(50),
            use_container_width=True
        )

    st.subheader("Live Anomalies")
    if not live_df.empty and "is_anomaly" in live_df.columns:
        live_anomalies = live_df[live_df["is_anomaly"]]
        if live_anomalies.empty:
            st.success("No live anomalies detected.")
        else:
            st.dataframe(
                live_anomalies.sort_values("timestamp", ascending=False),
                use_container_width=True
            )

# Historical Analysis Tab
with history_tab:
    st.subheader("Historical System Overview")

    st.caption(
    "This view summarizes historical system behavior by clustering log patterns, "
    "identifying rare anomalies, and surfacing actionable incidents from past data."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Logs Processed", len(logs_df))
    col2.metric("Clusters Identified", logs_df["cluster_id"].nunique() - 1)
    col3.metric("Historical Anomalies", logs_df["is_anomaly"].sum())
    col4.metric("Actionable Incidents", len(incidents_df))

    st.subheader("Actionable Incidents")

    if incidents_df.empty:
        st.info("No actionable incidents detected.")
    else:
        st.dataframe(
            incidents_df[
                ["start_time", "end_time", "event_count", "severity", "components"]
            ],
            use_container_width=True
        )

    st.subheader("Historical Anomalous Events")

    anomalies = logs_df[logs_df["is_anomaly"]].copy()

    if anomalies.empty:
        st.info("No historical anomalies detected.")
    else:
        st.dataframe(
            anomalies[
                ["timestamp", "source", "level", "component", "message", "cluster_id"]
            ].sort_values("timestamp"),
            use_container_width=True
        )

    st.subheader("Log Cluster Distribution")

    cluster_counts = (
        logs_df[logs_df["cluster_id"] != -1]
        .groupby("cluster_id")
        .size()
        .sort_values(ascending=False)
    )

    st.bar_chart(cluster_counts)

    st.subheader("⏱ Historical Log Timeline")

    timeline_df = logs_df.copy()
    timeline_df["anomaly_flag"] = timeline_df["is_anomaly"].map(
        {True: "Anomaly", False: "Normal"}
    )

    st.dataframe(
        timeline_df[
            [
                "timestamp",
                "source",
                "level",
                "component",
                "message",
                "cluster_id",
                "anomaly_flag",
            ]
        ].sort_values("timestamp"),
        use_container_width=True,
        height=300,
    )

# Footer
st.markdown("---")
st.caption(
    "Log Intelligence System — Batch & Live Unsupervised Anomaly Detection "
    "with API-driven inference and real-time visualization."
)
