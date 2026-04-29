"""
GMM confidence: fit GaussianMixture on past feature rows; score current point via log-likelihood
normalized to training min/max (course: EM / mixture model).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

HistoryRow = Tuple[str, np.ndarray]

# Hyperparameter defaults — overridable via settings dict
MIN_TRAIN_SAMPLES_DEFAULT = 30   # don't score until we have this many past rows
MAX_TRAIN_SAMPLES_DEFAULT = 250  # rolling window cap on training data
N_COMPONENTS_DEFAULT = 2         # number of Gaussian components in the mixture
RANDOM_STATE_DEFAULT = 42        # reproducibility seed
NEUTRAL_SCORE = 0.5              # fallback when scoring isn't possible


def score_confidence_gmm(
    history: List[HistoryRow],
    features: np.ndarray,
    settings: Dict[str, Any] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Parameters mirror NobecStrategy._score_confidence_gmm.

    settings: gmm block from confidence_model_settings, e.g.
        min_train_samples, max_train_samples, n_components
    """
    settings = settings or {}
    min_train = int(settings.get("min_train_samples", MIN_TRAIN_SAMPLES_DEFAULT))
    max_train = int(settings.get("max_train_samples", MAX_TRAIN_SAMPLES_DEFAULT))
    n_components = int(settings.get("n_components", N_COMPONENTS_DEFAULT))

    # 1. Guard: not enough history yet, return neutral
    if len(history) < min_train:
        return NEUTRAL_SCORE, {
            "status": "insufficient_history",
            "train_samples": len(history),
            "min_train_samples": min_train,
        }

    # 2. Build training matrix from the most recent max_train rows
    train = np.stack([v for _, v in history[-max_train:]], axis=0)
    try:
        # 3. Fit GMM via EM on the training window
        n_comp = max(1, min(n_components, len(train)))
        gmm = GaussianMixture(
            n_components=n_comp,
            random_state=RANDOM_STATE_DEFAULT,
        )
        gmm.fit(train)

        # 4. Score: get log-likelihood of every training point and the current point
        ll_train = gmm.score_samples(train)
        ll_curr = float(gmm.score_samples(features.reshape(1, -1))[0])

        # 5. Normalize current log-likelihood to [0, 1] using training min/max
        lo = float(np.min(ll_train))
        hi = float(np.max(ll_train))
        score = NEUTRAL_SCORE if hi <= lo else float((ll_curr - lo) / (hi - lo))
        score = float(np.clip(score, 0.0, 1.0))

        return score, {
            "status": "ok",
            "train_samples": len(train),
            "ll_curr": ll_curr,
            "ll_lo": lo,
            "ll_hi": hi,
            "n_components": n_comp,
        }
    except Exception as exc:
        return NEUTRAL_SCORE, {
            "status": "gmm_error",
            "error": str(exc),
            "train_samples": len(train),
        }
