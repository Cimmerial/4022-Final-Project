"""
6-D confidence feature vector used by KNN and GMM confidence in NobecStrategy.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# The 6 features fed into both KNN and GMM scorers, in order
CONFIDENCE_FEATURE_NAMES: Tuple[str, ...] = (
    "gap_abs",           # absolute gap magnitude at open
    "gap_pct",           # signed gap as a percentage
    "volume_ratio",      # current volume relative to session baseline
    "risk_reward_ratio", # reward-to-risk at proposed entry levels
    "rel_strength",      # relative strength vs. benchmark proxy
    "vix",               # volatility index level
)


def build_confidence_features(context: Dict[str, Any]) -> np.ndarray:
    """Build feature vector from a context dict (atoms from NOBEC evaluation)."""
    ordered = [
        float(context.get("gap_abs", 0.0)),
        float(context.get("gap_pct", 0.0)),
        float(context.get("volume_ratio", 0.0)),
        float(context.get("risk_reward_ratio", 0.0)),
        float(context.get("rel_strength", 0.0)),
        float(context.get("vix", 0.0)),
    ]
    # Sanitize non-finite values to 0 so models don't receive NaN/inf
    clean = [0.0 if (pd.isna(v) or np.isinf(v)) else float(v) for v in ordered]
    return np.asarray(clean, dtype=float)


def feature_map_from_vector(features: np.ndarray) -> Dict[str, float]:
    """Utility: convert a raw feature array back to a named dict for inspection."""
    vals = [float(x) for x in features.tolist()]
    return {
        name: (vals[i] if i < len(vals) else 0.0)
        for i, name in enumerate(CONFIDENCE_FEATURE_NAMES)
    }