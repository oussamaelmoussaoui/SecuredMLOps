# SecuredMLOps — Model Monitoring Stack

Drift monitoring using **Evidently AI** (drift detection) + **Prometheus** (metrics) + **Grafana** (dashboards).

## Architecture

```
Your Data
    |
    v
drift_metrics_exporter.py  :8001/metrics   (Evidently + Prometheus client)
    |
    v
Prometheus                 :9090            (scrapes every 30s)
    |
    v
Grafana                    :3000            (provisioned dashboards)
```

---

## Quick Start

### 1. Configure environment

Copy `.env.example` at the project root to `.env`, then fill in the paths:

```cmd
cd ..
copy .env.example .env
```

`.env` example:
```
GRAFANA_ADMIN_PASSWORD=your-secure-password

# Paths seen by the exporter container (mounted from DATA_ROOT)
REFERENCE_DATA_PATH=/data/reference.csv
CURRENT_DATA_PATH=/data/current.csv
TARGET_COLUMN=Label
DRIFT_INTERVAL_SECONDS=300

# Host-side data folder mounted as /data inside the container
DATA_ROOT=../Model/data
```

> The container mounts `DATA_ROOT` (host) as `/data` (container).
> Set `REFERENCE_DATA_PATH=/data/<filename>` accordingly.

### 2. Start the stack

```cmd
cd monitoring
docker-compose up -d
```

### 3. Access services

| Service    | URL                          | Credentials              |
|------------|------------------------------|--------------------------|
| Grafana    | http://localhost:3000        | admin / GRAFANA_ADMIN_PASSWORD |
| Prometheus | http://localhost:9090        | —                        |
| Exporter   | http://localhost:8001/metrics | —                       |

### 4. Generate a standalone HTML report (no Docker required)

```cmd
set REFERENCE_DATA_PATH=C:\path\to\reference.csv
set CURRENT_DATA_PATH=C:\path\to\current.csv
python monitoring/evidently/drift_report.py
```

The HTML report and a JSON metrics file are written to `monitoring/reports/`.

---

## Smoke test

After `docker-compose up -d`, run from the project root:

```cmd
curl http://localhost:8001/metrics
```

Expected output includes lines such as:
```
data_drift_score 0.12
n_drifted_features 3.0
dataset_drift_detected 0.0
prediction_drift_score 0.04
feature_drift_score{feature="Flow Duration"} 0.08
```

---

## Updating current data

To point the exporter at a new production snapshot, update `CURRENT_DATA_PATH` in `.env`, then:

```cmd
cd monitoring
docker-compose restart evidently_exporter
```

---

## Stop the stack

```cmd
cd monitoring
docker-compose down
```

---

## Dashboard panels explained

| Panel                       | Prometheus metric         | How to interpret                                            |
|-----------------------------|---------------------------|-------------------------------------------------------------|
| Overall Drift Score (gauge) | `data_drift_score`        | Share of features drifted. >0.15 → investigate input data.  |
| Drifted Features (stat)     | `n_drifted_features`      | Raw count of drifted columns.                               |
| Dataset Drift Detected      | `dataset_drift_detected`  | 1 = Evidently confirmed overall drift. 0 = stable.          |
| Prediction Drift Score      | `prediction_drift_score`  | Drift on Label column. >0.10 may mean model degradation.    |
| Drift Over Time (timeseries)| both drift scores         | Trend view — spot gradual degradation over hours/days.       |
| Per-Feature Drift (bars)    | `feature_drift_score`     | Which specific features drifted most (red = >0.30).         |

---

## Troubleshooting

- **All metrics are 0**: `REFERENCE_DATA_PATH` / `CURRENT_DATA_PATH` are not set or not accessible inside the container. Check your `.env` and the `DATA_ROOT` volume mount.
- **Dashboard not visible**: wait ~30s after startup for Grafana provisioning, then hard-refresh the browser.
- **ColumnDriftMetric error**: reference and current CSV must share the same column names.
- **Grafana login fails**: verify `GRAFANA_ADMIN_PASSWORD` in `.env` — it must be set before first startup.
