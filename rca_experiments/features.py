"""Metric features shared by all experiment models."""
import numpy as np
import pandas as pd
from .config import (
    METRIC_WINDOW_SECONDS,
)

def extract_metric_features(df, injection_time):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    df["time"] = pd.to_numeric(
        df["time"], errors="coerce"
    )

    before = df[df["time"] < injection_time]
    after = df[df["time"] >= injection_time]

    if METRIC_WINDOW_SECONDS is not None:
        before = before[
            before["time"] >=
            injection_time - METRIC_WINDOW_SECONDS
        ]
        after = after[
            after["time"] <
            injection_time + METRIC_WINDOW_SECONDS
        ]

    if before.empty or after.empty:
        raise ValueError("Missing pre/post metric samples")

    features = {}

    for col in df.columns:
        if col == "time":
            continue

        a = pd.to_numeric(before[col], errors="coerce")
        b = pd.to_numeric(after[col], errors="coerce")

        features[f"{col}_mean"] = b.mean()
        features[f"{col}_std"] = b.std()
        features[f"{col}_max"] = b.max()
        features[f"{col}_delta"] = b.mean() - a.mean()

        baseline_std = a.std()

        features[f"{col}_zshift"] = (
            (b.mean() - a.mean()) / baseline_std
            if pd.notna(baseline_std)
            and baseline_std > 1e-8
            else np.nan
        )

    return features


