"""Helpers that saved P3 models depend on. Must stay importable as src.p3_utils
(load the .pkl files from the repo root)."""
import numpy as np


def signed_log(X):
    """sign(x) * log(1 + |x|): tames heavy-tailed dollar amounts and ratios, keeps sign."""
    X = np.where(np.isfinite(X), X, np.nan)
    return np.sign(X) * np.log1p(np.abs(X))
