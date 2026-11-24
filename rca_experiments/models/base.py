"""Contract: fit(dataset), then predict(features, incidents) in input order.

fit must train using dataset.train_idx only. Predictions use system::service
labels. Return one Prediction per incident, including failures.
"""
from dataclasses import dataclass, field


@dataclass
class Prediction:
    top1: str | None
    top3: list[str] = field(default_factory=list)
    error: str | None = None
