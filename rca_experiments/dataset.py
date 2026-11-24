"""Prepare incidents once and persist the exact split used by every model."""
from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config
from .features import extract_metric_features


@dataclass
class ExperimentDataset:
    X: pd.DataFrame
    y: pd.Series
    systems: pd.Series
    incidents: list
    train_idx: np.ndarray
    test_idx: np.ndarray
    candidates: dict


def prepare_dataset(data_dir=config.DATA_DIR, download=True):
    from RCAEval.utility import read_metrics
    data_dir = Path(data_dir).resolve()
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="phamquiluan/RCAEval", repo_type="dataset",
            allow_patterns=["re2*/**", "cases.parquet"], local_dir=str(data_dir),
        )
    cases = pd.read_parquet(data_dir / "cases.parquet")
    cases = cases[cases["dataset"].isin(config.DATASETS)].copy()
    print("Incidents per dataset:")
    print(cases.groupby("dataset").size())
    rows = []
    labels = []
    systems = []
    incident_info = []

    for _, case in cases.iterrows():
        case_dir = data_dir / str(case["case"])
        system = str(case["dataset"])

        try:
            metrics = read_metrics(str(case_dir))
            injection_time = float(case["inject_time"])

            features = extract_metric_features(
                metrics, injection_time
            )

            features[f"system_{system}"] = 1

            root_service = str(case["root_cause_service"])
            label = f"{system}::{root_service}"

            rows.append(features)
            labels.append(label)
            systems.append(system)

            incident_info.append({
                "case_dir": str(case_dir),
                "case": str(case["case"]),
                "system": system,
                "injection_time": injection_time,
                "root_cause_service": root_service,
                "label": label,
                "metric_features": features.copy(),
            })

        except Exception as exc:
            print(f"Skipping {case_dir}: {exc}")

    if not rows:
        raise RuntimeError("No RE2 incidents loaded")

    X = pd.DataFrame(rows)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    system_cols = [
        col for col in X.columns
        if col.startswith("system_RE2-")
    ]
    X[system_cols] = X[system_cols].fillna(0)

    y = pd.Series(labels, name="root_cause")
    systems = pd.Series(systems, name="system")

    print("\nFeature matrix:", X.shape)
    print("Classes:", y.nunique())



    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )
    # Preserve the original benchmark label inventory, not a per-case answer.
    candidates = {
        system: sorted({label.split("::", 1)[1] for label in labels
                        if label.startswith(f"{system}::")})
        for system in config.DATASETS
    }
    return ExperimentDataset(X, y, systems, incident_info, train_idx, test_idx, candidates)


def save_dataset(dataset, path=config.PREPARED_DATA_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dataset, path)
    manifest = {
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "datasets": config.DATASETS,
        "metric_window_seconds": config.METRIC_WINDOW_SECONDS,
        "train": [dataset.incidents[int(i)]["case"] for i in dataset.train_idx],
        "test": [dataset.incidents[int(i)]["case"] for i in dataset.test_idx],
    }
    path.with_suffix(".split.json").write_text(json.dumps(manifest, indent=2))


def load_dataset(path=config.PREPARED_DATA_PATH):
    # Only load trusted, locally generated joblib files.
    return joblib.load(path)
