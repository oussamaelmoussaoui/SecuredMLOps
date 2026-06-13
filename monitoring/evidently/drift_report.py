"""
Generates Evidently data drift reports comparing reference vs current data.
Exports an HTML report to monitoring/reports/ and metrics as JSON.

Required env vars:
    REFERENCE_DATA_PATH  path to reference (training baseline) CSV or Parquet
    CURRENT_DATA_PATH    path to current (production) CSV or Parquet

Optional env vars:
    TARGET_COLUMN        target column name           (default: Label)
    REPORTS_DIR          output directory             (default: monitoring/reports)

Usage:
    python monitoring/evidently/drift_report.py
"""

import datetime
import json
import os
import pathlib

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

REFERENCE_DATA_PATH = os.environ.get("REFERENCE_DATA_PATH", "")
CURRENT_DATA_PATH   = os.environ.get("CURRENT_DATA_PATH", "")
TARGET_COLUMN       = os.environ.get("TARGET_COLUMN", "Label")
REPORTS_DIR         = os.environ.get(
    "REPORTS_DIR",
    str(pathlib.Path(__file__).parents[2] / "monitoring" / "reports"),
)


def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def run_report() -> dict:
    if not REFERENCE_DATA_PATH or not CURRENT_DATA_PATH:
        raise ValueError(
            "Set REFERENCE_DATA_PATH and CURRENT_DATA_PATH environment variables."
        )

    reference = _load(REFERENCE_DATA_PATH)
    current   = _load(CURRENT_DATA_PATH)

    has_target     = TARGET_COLUMN in reference.columns and TARGET_COLUMN in current.columns
    column_mapping = ColumnMapping(target=TARGET_COLUMN) if has_target else ColumnMapping()

    presets = [DataDriftPreset()]
    if has_target:
        presets.append(TargetDriftPreset())

    report = Report(metrics=presets)
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    ts        = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = pathlib.Path(REPORTS_DIR) / f"drift_report_{ts}.html"
    json_path = pathlib.Path(REPORTS_DIR) / f"drift_metrics_{ts}.json"

    report.save_html(str(html_path))

    report_dict = report.as_dict()
    with open(json_path, "w") as fh:
        json.dump(report_dict, fh, indent=2, default=str)

    print(f"HTML  -> {html_path}")
    print(f"JSON  -> {json_path}")
    return report_dict


if __name__ == "__main__":
    run_report()
