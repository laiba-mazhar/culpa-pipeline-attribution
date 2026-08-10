"""The utility oracle u(S).

Training-time semantics: fit the downstream model on whatever training table the
hybrid replay produced, then score it on a *fixed* clean probe set drawn from
the true data-generating process. The probe is frozen across every coalition --
otherwise u(S) compares different things and Shapley efficiency is meaningless.

The probe standing in for reality is what makes this measure the thing we care
about: not "did the numbers move" but "does the model this pipeline built still
work on the world".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LABEL = "label"


def build_probe() -> pd.DataFrame:
    """A clean feature table from a held-out day, built by the unfaulted
    pipeline in reference state."""
    from .workload import build_pipeline

    return build_pipeline(probe=True).replay(frozenset())


def make_utility(probe: pd.DataFrame, seed: int = 0):
    """Return u(sink_df) -> ROC AUC on the frozen probe.

    Common random numbers: the learner is deterministic and the probe is fixed,
    so two coalitions differing by one operator differ only because of the data,
    not because of training noise. Proposal section 4.4 -- this is the variance
    control that makes small marginal contributions readable.
    """
    probe_y = probe[LABEL].to_numpy()

    def utility(sink: pd.DataFrame) -> float:
        if LABEL not in sink.columns or len(sink) < 50:
            return 0.5
        y = sink[LABEL].to_numpy()
        if len(np.unique(y)) < 2:
            return 0.5

        feature_cols = [c for c in sink.columns if c != LABEL]
        if not feature_cols:
            return 0.5

        X = sink[feature_cols].astype(float).to_numpy()
        # Align the probe to whatever schema this pipeline actually emitted.
        # A column the pipeline dropped is a column the model never learned.
        Xp = (
            probe.reindex(columns=feature_cols, fill_value=0.0)
            .astype(float)
            .to_numpy()
        )

        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        Xps = (Xp - mu) / sd  # train-time statistics, deliberately: this is
                              # exactly how training/serving skew manifests

        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(Xs, y)
        scores = model.predict_proba(Xps)[:, 1]
        return float(roc_auc_score(probe_y, scores))

    return utility
