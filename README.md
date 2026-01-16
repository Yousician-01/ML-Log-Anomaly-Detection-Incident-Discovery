# Log Intelligence & Incident Discovery System
![CI](https://github.com/Yousician-01/ML-Log-Anomaly-Detection-Incident-Discovery/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/docker/v/atharvaraut01/log-intelligence-system/v1.0.0)

An end-to-end **unsupervised log intelligence system** for detecting anomalous log events and surfacing incidents from both historical and live streaming logs.

This project demonstrates how unsupervised NLP techniques, statistical baselines, and system design principles can be combined to build a **real-world observability and anomaly detection pipeline**.

## Key Features
- Unsupervised anomaly detection on system logs
- Semantic log representation using NLP (TF-IDF)
- Density-based clustering for historical analysis
- Live log ingestion via FastAPI
- Real-time anomaly visualization with Streamlit
- Clear separation of **batch intelligence and live inference**
- Session-scoped live monitoring (no stale data)

---

## Problem Statement
Modern systems generate massive volumes of logs.
Rule-based monitoring struggles to detect  **unknown failure modes**, while supervised ML requires labeled incidents that rarely exist.

This project addresses:

- Detecting rare and novel log patterns
- Distinguishing noise from meaningful anomalies
- Aggregating anomalous events into incidents
- Monitoring logs in near real-time
- All without labeled data.

---

## Datasets Used
The system is trained and evaluated using **public real-world log datasets**:

1. Apache Logs
- Server and application lifecycle events
- Includes INFO, NOTICE, ERROR level logs
- Used to model application-level behavior

2. HDFS Logs
- Distributed filesystem operational logs
- Includes block replication, deletion, heartbeat, and failure events
- Widely used benchmark for log mining research

**Why these datasets?**
- Realistic log formats
- High variability
- Frequently used in log anomaly research
- No manual labels (ideal for unsupervised learning)

---

## Model & Learning Approach

### NLP Technique Used
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Converts log messages into sparse numerical vectors
- Captures semantic importance of words
- Scales well for large log corpora
- Interpretable and production-friendly

Artifacts saved:
- ```tfidf_vectorizer.joblib```
- ```tfidf_embeddings.joblib```

### Historical Anomaly Detection (Batch Mode)
- Log messages are vectorized using TF-IDF
- Density-based clustering (DBSCAN / HDBSCAN style logic)
- Logs not belonging to any dense cluster are marked as anomalies
- Temporal aggregation groups anomalies into incidents
- Incident suppression rules reduce noise

Output artifacts:
- ```logs_with_clusters.parquet```
- ```final_incidents.parquet```

### Live Anomaly Detection (Streaming Mode)

Live inference is **not retraining**.

Instead:
- New logs are embedded using the pre-trained TF-IDF vectorizer
- Distance from **historical normal centroid** is computed
- Severity-aware rules are applied (e.g., ERROR override)
- Decisions are explainable (distance-based reasoning)

This mirrors how **real observability systems** operate.

---
## Tech Stack

**Core Languages & Frameworks**
- **Python 3.11** – Stable runtime for ML & backend services
- **FastAPI** – High-performance API for live log ingestion and inference
- **Streamlit** – Interactive dashboard for live monitoring and historical analysis

**Machine Learning & NLP**
- **scikit-learn** – TF-IDF vectorization, clustering logic, similarity metrics
- **NumPy / SciPy** – Numerical computation and sparse matrix operations
- **Pandas** – Log preprocessing, aggregation, and storage

**Storage & Artifacts**
- **Parquet** – Efficient columnar storage for logs and incidents
- **Joblib** – Serialization of vectorizers and embeddings

**Development & Ops**
- **Uvicorn** – ASGI server for FastAPI
- **Virtualenv** – Environment isolation

---

## System Architecture (Conceptual)

The system is intentionally split into three independent layers:

1. **Batch Intelligence (Offline)**
- Learns normal behavior from historical logs
- Builds semantic representations and clusters
- Produces artifacts used by live inference

2. **Live Inference Service (Online)**
- Accepts logs via HTTP API
- Performs anomaly detection using frozen representations
- Writes live results to a session-scoped store

3. **Visualization Layer**
- Reads processed outputs only
- Never performs ML or inference
- Auto-refreshes to reflect live state

This separation mirrors real-world observability platforms and avoids tight coupling between UI and ML logic.

---

## Repository Structure
```
.
├── api/
│   └── main.py              # FastAPI inference service
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── data/
│   └── processed/
│       ├── logs_with_clusters.parquet
│       ├── final_incidents.parquet
│       ├── live_logs.parquet
│       └── embeddings/
├── src/
│   ├── preprocessing/
│   ├── clustering/
│   └── incident_detection/
├── requirements.txt
└── README.md

```

---

## How to Run the Project

1. Clone the Repository
```bash
git clone <your-repo-url>
cd log-intelligence-system
```

2. Create Virtual Environment (Python 3.11)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Start FastAPI (for Live Inference)
```bash
uvicorn api.main:app --reload
```
API docs available at:
```
http://127.0.0.1:8000/docs
```

5. Start Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

### Using the API
**Example Request**
```json
POST /logs
{
  "source": "hdfs",
  "level": "ERROR",
  "component": "dfs.DataNode",
  "message": "Disk failure detected on DataNode"
}
```

**Response**
```json
{
  "status": "ok",
  "is_anomaly": true,
  "reason": "ERROR_level"
}
```
Logs appear automatically in the **Live Monitoring** tab.

---

### Docker Support
This project can be containerized to ensure environment reproducibility and easy deployment.
Docker is recommended for:
- Avoiding Python version conflicts
- Ensuring consistent dependency resolution
- Running the system without local setup complexity

Dockerized Components
The Docker setup supports:
- FastAPI inference service
- Streamlit dashboard

Both services run using a fixed Python runtime and pinned dependencies.

**Build the Docker Image**
From the project root:
```bash
docker build -t log-intelligence-system .
```
**Run the Container**
```bash
docker run -p 8000:8000 -p 8501:8501 log-intelligence-system
```
- FastApi -> ```http://localhost:8000/docs```
- Streamlit -> ```http://localhost:8501```

---

## Design Decisions & Trade-offs

**Why Unsupervised Learning?**
- Labeled incident data is rarely available
- Systems evolve over time
- Unknown failure modes matter more than known ones

**Why TF-IDF Instead of Deep Embeddings?**
- Interpretable features
- Lightweight and fast
- No GPU dependency
- Sufficient for structured log text

**Why No Online Retraining?**
- Prevents feedback loops
- Ensures stability during live monitoring
- Mirrors production practices where retraining is scheduled, not continuous

**Why Session-Scoped Live Logs?**
- Prevents stale data leakage
- Makes live monitoring semantics clear
- Allows clean restarts without manual cleanup

---

## Current Limitations
- Static historical baseline (no rolling adaptation)
- No component-specific normal profiles
- No alerting or notification integration
- No authentication on API endpoints
These are deliberate scope choices, not oversights.

---

## Possible Future Improvements
- Rolling or windowed baselines for adaptive normal behavior
- Component-level or service-level anomaly models
- Alerting via Slack / Email / Webhooks
- Online template parsing (e.g., Drain)
- Metrics-based anomaly correlation
- Multi-node deployment

---

## Conclusion
This project focuses on **realistic system design**, not algorithmic novelty.

It demonstrates how unsupervised NLP techniques can be operationalized into a practical log intelligence system with:
- Clear architectural boundaries
- Explainable anomaly detection
- Production-aware trade-offs

The goal is not perfection, but **correctness, clarity, and extensibility.**