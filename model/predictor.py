"""Walk-forward models for 1/5/20-event mid-price movement.

Per horizon we fit:
  * multinomial logistic regression on the direction class c_h (down/flat/up);
  * ridge regression on the continuous signed move y_h (ticks), giving an
    expected-magnitude estimate E[delta mid] for execution decisions.

`signal_mix` in [0,1] blends predictions toward the unconditional prior,
emulating weaker alpha; it drives the prediction-accuracy sweep that answers
"how much accuracy is needed to offset queueing and latency".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             log_loss)
from sklearn.preprocessing import StandardScaler

import config
from features.features import FEATURE_COLS, FeatureStore


@dataclass
class HorizonModel:
    horizon: int
    clf: LogisticRegression
    ridge: Ridge
    scaler: StandardScaler
    prior: np.ndarray
    classes: np.ndarray
    signal_mix: float


class MidPredictor:
    def __init__(self, store: FeatureStore, train_frac: float = 0.5,
                 signal_mix: float = 0.0):
        self.store = store
        df = store.df
        self.n = len(df)
        self.split = int(self.n * train_frac)
        self.signal_mix = signal_mix
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        self.scaler = StandardScaler().fit(X[: self.split])
        Xs = self.scaler.transform(X)
        self.models: dict[int, HorizonModel] = {}
        self._e_move: dict[int, np.ndarray] = {}
        for h in config.HORIZONS:
            yc = df[f"c{h}"].to_numpy()
            yr = df[f"y{h}"].to_numpy()
            # last h rows have no label
            fit_idx = np.arange(self.split)
            clf = LogisticRegression(max_iter=500,
                                     class_weight="balanced").fit(
                Xs[fit_idx], yc[fit_idx])
            ridge = Ridge(alpha=1.0).fit(Xs[fit_idx], yr[fit_idx])
            classes = clf.classes_
            prior = np.array([(yc[fit_idx] == c).mean() for c in classes])
            self.models[h] = HorizonModel(h, clf, ridge, self.scaler, prior,
                                          classes, signal_mix)
            self._e_move[h] = ridge.predict(Xs)

    # -- inference ---------------------------------------------------------
    def _x(self, seq: int) -> np.ndarray:
        return self.store.features_at(seq).to_numpy(dtype=float).reshape(1, -1)

    def probs(self, seq: int, horizon: int) -> np.ndarray:
        m = self.models[horizon]
        x = m.scaler.transform(self._x(seq))
        p = m.clf.predict_proba(x)[0]
        if self.signal_mix > 0:
            p = (1 - self.signal_mix) * p + self.signal_mix * m.prior
        return p

    def expected_move(self, seq: int, horizon: int) -> float:
        """Expected signed mid move in ticks."""
        seq = min(max(seq, 0), self.n - 1)
        return float(self._e_move[horizon][seq]) * (1 - self.signal_mix)

    def direction(self, seq: int, horizon: int) -> int:
        p = self.probs(seq, horizon)
        return int(self.models[horizon].classes[int(np.argmax(p))])

    # -- evaluation --------------------------------------------------------
    def evaluation(self) -> pd.DataFrame:
        rows = []
        df = self.store.df
        test_idx = np.arange(self.split, self.n)
        for h in config.HORIZONS:
            m = self.models[h]
            Xs = m.scaler.transform(df[FEATURE_COLS].to_numpy(dtype=float))
            idx = test_idx[test_idx < self.n - h]
            p_raw = m.clf.predict_proba(Xs[idx])
            if self.signal_mix > 0:
                p = (1 - self.signal_mix) * p_raw + self.signal_mix * m.prior
            else:
                p = p_raw
            pred = m.classes[np.argmax(p, axis=1)]
            yc = df[f"c{h}"].to_numpy()[idx]
            yr = df[f"y{h}"].to_numpy()[idx]
            e_move = m.ridge.predict(Xs[idx]) * (1 - self.signal_mix)
            # directional accuracy excluding flat
            directional = yc != 0
            rows.append({
                "horizon": h,
                "accuracy": accuracy_score(yc, pred),
                "balanced_accuracy": balanced_accuracy_score(yc, pred),
                "directional_accuracy":
                    accuracy_score(yc[directional], pred[directional])
                    if directional.any() else np.nan,
                "log_loss": log_loss(yc, p, labels=m.classes),
                "move_corr": np.corrcoef(e_move, yr)[0, 1],
                "signal_mix": self.signal_mix,
            })
        return pd.DataFrame(rows)
