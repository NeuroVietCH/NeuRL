from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class EvidenceDiagnosisHead:
    """Simple AD/CN head over mean-pooled evidence embeddings."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    def featurize(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(embeddings, dtype=np.float32).mean(axis=0)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "EvidenceDiagnosisHead":
        self.model.fit(features, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)
