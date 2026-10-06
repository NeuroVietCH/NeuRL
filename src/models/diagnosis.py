from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class EvidenceDiagnosisHead:
    """Simple AD/CN head over pooled evidence embeddings and anomaly summaries."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    def featurize(self, embeddings: np.ndarray, anomaly_scores: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        pooled = x.mean(axis=0)
        if anomaly_scores is None or len(anomaly_scores) == 0:
            stats = np.zeros(4, dtype=np.float32)
        else:
            s = np.asarray(anomaly_scores, dtype=np.float32)
            stats = np.asarray([s.mean(), s.max(), s.min(), len(s)], dtype=np.float32)
        return np.concatenate([pooled, stats], axis=0)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "EvidenceDiagnosisHead":
        self.model.fit(features, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)
