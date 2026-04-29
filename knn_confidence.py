"""
KNN-based confidence score: inverse distance to k nearest past feature rows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

HistoryRow = Tuple[str, np.ndarray]  # (date_str, features)

# Hyperparameter defaults — overridable via settings dict
MIN_TRAIN_SAMPLES_DEFAULT = 25   # don't score until we have this many past rows
MAX_TRAIN_SAMPLES_DEFAULT = 300  # rolling window cap on training data
K_DEFAULT = 8                    # number of nearest neighbors to average
NEUTRAL_SCORE = 0.5              # fallback when scoring isn't possible


def score_confidence_knn(
    history: List[HistoryRow],
    features: np.ndarray,
    settings: Dict[str, Any] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Parameters mirror NobecStrategy._score_confidence_knn.

    history: past (date_str, feature_vector) rows, oldest → newest as appended in engine.
    features: current 6-D vector (same order as CONFIDENCE_FEATURE_NAMES).
    settings: knn block from confidence_model_settings, e.g.
        min_train_samples, max_train_samples, k
    """
    settings = settings or {}
    min_train = int(settings.get("min_train_samples", MIN_TRAIN_SAMPLES_DEFAULT))
    max_train = int(settings.get("max_train_samples", MAX_TRAIN_SAMPLES_DEFAULT))
    k = int(settings.get("k", K_DEFAULT))

    # 1. Guard: not enough history yet, return neutral
    if len(history) < min_train:
        return NEUTRAL_SCORE, {
            "status": "insufficient_history",
            "train_samples": len(history),
            "min_train_samples": min_train,
        }

    # 2. Build training matrix from the most recent max_train rows
    train = np.stack([v for _, v in history[-max_train:]], axis=0)

    # 3. Compute Euclidean distance from current vector to every training row
    dists = np.linalg.norm(train - features.reshape(1, -1), axis=1)

    # 4. Find the k nearest neighbors and average their distances
    k_eff = max(1, min(k, len(dists)))
    nn = np.partition(dists, k_eff - 1)[:k_eff]
    dist_mean = float(np.mean(nn))

    # 5. Convert mean distance to confidence: closer neighbors → higher score
    score = float(1.0 / (1.0 + dist_mean))

    return float(np.clip(score, 0.0, 1.0)), {
        "status": "ok",
        "train_samples": len(train),
        "k": k_eff,
        "mean_neighbor_distance": dist_mean,
    }