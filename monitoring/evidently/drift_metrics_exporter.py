"""
Prometheus exporter for Evidently drift metrics.
Exposes metrics on port 8001 and recomputes drift on a configurable schedule.

Metrics exposed:
    data_drift_score              overall share of drifted features (0-1)
    n_drifted_features            count of drifted feature columns
    prediction_drift_score        drift score of the target/prediction column
    dataset_drift_detected        binary flag (1 = drift confirmed, 0 = stable)
    feature_drift_score           per-feature drift score with label {feature="..."}
    attack_ratio                  proportion of attack traffic in current data (Label=1)
    n_attack_samples              count of attack samples in current data
    n_benign_samples              count of benign samples in current data
    n_total_samples               total samples in current data
    feature_mean                  mean of key network features {feature="..."}
    feature_std                   std deviation of key network features {feature="..."}
    last_drift_compute_timestamp  unix timestamp of last successful computation

Env vars:
    REFERENCE_DATA_PATH      path to reference CSV or Parquet
    CURRENT_DATA_PATH        path to current CSV or Parquet
    TARGET_COLUMN            target column name         (default: Label)
    EXPORTER_PORT            HTTP port                  (default: 8001)
    DRIFT_INTERVAL_SECONDS   recompute interval in s    (default: 300)
"""

import logging
import os
import threading
import time

import pandas as pd
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.legacy.report import Report
from prometheus_client import Gauge, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_DATA_PATH    = os.environ.get("REFERENCE_DATA_PATH", "")
CURRENT_DATA_PATH      = os.environ.get("CURRENT_DATA_PATH", "")
TARGET_COLUMN          = os.environ.get("TARGET_COLUMN", "Label")
EXPORTER_PORT          = int(os.environ.get("EXPORTER_PORT", "8001"))
DRIFT_INTERVAL_SECONDS = int(os.environ.get("DRIFT_INTERVAL_SECONDS", "300"))

KEY_FEATURES = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

# ── Prometheus Gauges ─────────────────────────────────────────────────────────
g_data_drift_score       = Gauge("data_drift_score",       "Share of drifted features (0-1)")
g_n_drifted_features     = Gauge("n_drifted_features",     "Number of drifted feature columns")
g_prediction_drift_score = Gauge("prediction_drift_score", "Drift score of target/prediction column")
g_dataset_drift_detected = Gauge("dataset_drift_detected", "Dataset drift flag (1=drifted, 0=stable)")
g_feature_drift          = Gauge("feature_drift_score",    "Per-feature drift score", ["feature"])

g_attack_ratio           = Gauge("attack_ratio",           "Proportion of attack traffic in current data (0-1)")
g_n_attack_samples       = Gauge("n_attack_samples",       "Number of attack samples (Label=1) in current data")
g_n_benign_samples       = Gauge("n_benign_samples",       "Number of benign samples (Label=0) in current data")
g_n_total_samples        = Gauge("n_total_samples",        "Total number of samples in current data")

g_feature_mean           = Gauge("feature_mean",           "Mean value of key network features", ["feature"])
g_feature_std            = Gauge("feature_std",            "Std deviation of key network features", ["feature"])

g_last_compute_ts        = Gauge("last_drift_compute_timestamp", "Unix timestamp of last successful drift computation")


def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _update_traffic_stats(current: pd.DataFrame) -> None:
    total = len(current)
    g_n_total_samples.set(total)

    if TARGET_COLUMN in current.columns:
        n_attack = int((current[TARGET_COLUMN] != 0).sum())
        n_benign = total - n_attack
        g_n_attack_samples.set(n_attack)
        g_n_benign_samples.set(n_benign)
        g_attack_ratio.set(n_attack / total if total > 0 else 0.0)
        logger.info("Traffic stats — total: %d, attack: %d, benign: %d", total, n_attack, n_benign)


def _update_feature_stats(current: pd.DataFrame) -> None:
    for feat in KEY_FEATURES:
        if feat in current.columns:
            col = pd.to_numeric(current[feat], errors="coerce").dropna()
            if len(col) > 0:
                g_feature_mean.labels(feature=feat).set(float(col.mean()))
                g_feature_std.labels(feature=feat).set(float(col.std()))


def _compute_and_update() -> None:
    if not REFERENCE_DATA_PATH or not CURRENT_DATA_PATH:
        logger.warning(
            "REFERENCE_DATA_PATH or CURRENT_DATA_PATH not set — skipping drift computation."
        )
        return

    try:
        reference    = _load(REFERENCE_DATA_PATH)
        current      = _load(CURRENT_DATA_PATH)
        feature_cols = [c for c in reference.columns if c != TARGET_COLUMN]
        has_target   = TARGET_COLUMN in reference.columns and TARGET_COLUMN in current.columns

        column_mapping = ColumnMapping(target=TARGET_COLUMN) if has_target else ColumnMapping()

        metrics = [DatasetDriftMetric()]
        for col in feature_cols:
            if col in current.columns:
                metrics.append(ColumnDriftMetric(column_name=col))
        if has_target:
            metrics.append(ColumnDriftMetric(column_name=TARGET_COLUMN))

        report = Report(metrics=metrics)
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping,
        )
        result_dict = report.as_dict()

        for item in result_dict.get("metrics", []):
            metric_name = item.get("metric", "")
            result      = item.get("result", {})

            if metric_name == "DatasetDriftMetric":
                g_data_drift_score.set(result.get("drift_share", 0.0))
                g_n_drifted_features.set(result.get("number_of_drifted_columns", 0))
                g_dataset_drift_detected.set(int(result.get("dataset_drift", False)))

            elif metric_name == "ColumnDriftMetric":
                col   = result.get("column_name", "")
                score = result.get("drift_score", 0.0)
                if col == TARGET_COLUMN:
                    g_prediction_drift_score.set(score)
                elif col:
                    g_feature_drift.labels(feature=col).set(score)

        _update_traffic_stats(current)
        _update_feature_stats(current)

        g_last_compute_ts.set(time.time())
        logger.info("Drift metrics updated successfully.")

    except Exception as exc:
        logger.error("Drift computation failed: %s", exc, exc_info=True)


def _scheduler_loop() -> None:
    while True:
        time.sleep(DRIFT_INTERVAL_SECONDS)
        _compute_and_update()


if __name__ == "__main__":
    start_http_server(EXPORTER_PORT)
    logger.info("Prometheus exporter running on :%d/metrics", EXPORTER_PORT)
    logger.info("Drift recompute interval: %ds", DRIFT_INTERVAL_SECONDS)

    _compute_and_update()

    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()

    while True:
        time.sleep(60)
