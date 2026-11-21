from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor


@dataclass(frozen=True)
class ParadigmAConfig:
    n_estimators: int = 200
    ocsvm_nu: float = 0.10
    lof_neighbors: int = 20
    ae_hidden: int = 8
    ae_max_iter: int = 300
    vote_fraction: float = 0.5


def build_features(x_round: np.ndarray) -> np.ndarray:
    x = np.asarray(x_round, dtype=float)
    med = np.median(x)
    dev = x - med
    mad = np.median(np.abs(dev))
    mad_z = np.abs(dev) / max(1.4826 * mad, 1e-9)
    return np.column_stack([x, dev, mad_z])


def copod_like_score(X: np.ndarray, train_sorted: list[np.ndarray]) -> np.ndarray:
    """Deterministic COPOD-style empirical tail score; larger = more anomalous."""
    n, d = X.shape
    cols = []
    for j in range(d):
        ref = train_sorted[j]
        left = np.searchsorted(ref, X[:, j], side="right") / (len(ref) + 1.0)
        right = (len(ref) - np.searchsorted(ref, X[:, j], side="left")) / (len(ref) + 1.0)
        p = np.minimum(left, right)
        cols.append(-np.log(np.clip(p, 1e-9, 1.0)))
    return np.max(np.column_stack(cols), axis=1)


class ParadigmA:
    """Honest-only report anomaly ensemble.

    All model scores are normalized so larger means more anomalous.
    Thresholds are frozen from honest training scores using the upper Tukey/IQR fence.
    """

    def __init__(self, seed: int, cfg: ParadigmAConfig = ParadigmAConfig()):
        self.seed = int(seed)
        self.cfg = cfg
        self.models = {
            "IsolationForest": IsolationForest(
                n_estimators=cfg.n_estimators,
                random_state=seed,
                n_jobs=1,
            ),
            "OneClassSVM": OneClassSVM(kernel="rbf", nu=cfg.ocsvm_nu),
            "LOF": LocalOutlierFactor(
                n_neighbors=cfg.lof_neighbors,
                novelty=True,
            ),
            "Autoencoder": MLPRegressor(
                hidden_layer_sizes=(cfg.ae_hidden,),
                activation="relu",
                solver="adam",
                max_iter=cfg.ae_max_iter,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=30,
                tol=1e-5,
            ),
        }
        self.thresholds: dict[str, float] = {}
        self.train_sorted: list[np.ndarray] | None = None
        self.fitted = False

    def fit(self, honest_rounds: list[np.ndarray]) -> "ParadigmA":
        X_train = np.vstack([build_features(r) for r in honest_rounds])
        self.train_sorted = [np.sort(X_train[:, j]) for j in range(X_train.shape[1])]
        scores = {}

        self.models["IsolationForest"].fit(X_train)
        scores["IsolationForest"] = -self.models["IsolationForest"].decision_function(X_train)

        self.models["OneClassSVM"].fit(X_train)
        scores["OneClassSVM"] = -self.models["OneClassSVM"].decision_function(X_train)

        self.models["LOF"].fit(X_train)
        scores["LOF"] = -self.models["LOF"].decision_function(X_train)

        self.models["Autoencoder"].fit(X_train, X_train)
        recon = self.models["Autoencoder"].predict(X_train)
        scores["Autoencoder"] = np.mean((recon - X_train) ** 2, axis=1)

        scores["COPOD"] = copod_like_score(X_train, self.train_sorted)

        for name, s in scores.items():
            q1, q3 = np.percentile(s, [25, 75])
            self.thresholds[name] = float(q3 + 1.5 * (q3 - q1))
        self.fitted = True
        return self

    def _scores(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not self.fitted or self.train_sorted is None:
            raise RuntimeError("Paradigm A must be fitted before detect")
        out = {
            "IsolationForest": -self.models["IsolationForest"].decision_function(X),
            "OneClassSVM": -self.models["OneClassSVM"].decision_function(X),
            "LOF": -self.models["LOF"].decision_function(X),
        }
        recon = self.models["Autoencoder"].predict(X)
        out["Autoencoder"] = np.mean((recon - X) ** 2, axis=1)
        out["COPOD"] = copod_like_score(X, self.train_sorted)
        return out

    def detect(self, x_round: np.ndarray) -> np.ndarray:
        X = build_features(x_round)
        scores = self._scores(X)
        votes = np.zeros(len(X), dtype=int)
        for name, s in scores.items():
            votes += (s > self.thresholds[name]).astype(int)
        required = self.cfg.vote_fraction * len(scores)
        return np.where(votes >= required)[0].astype(int)

    def describe(self) -> dict:
        return {
            "models": list(self.models.keys()) + ["COPOD"],
            "threshold_rule": "Q3 + 1.5*IQR on training anomaly score",
            "features": ["raw_report", "deviation_from_round_median", "absolute_deviation/(1.4826*MAD)"],
            "score_direction": "larger = more anomalous",
            "vote_fraction": self.cfg.vote_fraction,
            "hyperparameters": {
                "IsolationForest.n_estimators": self.cfg.n_estimators,
                "OneClassSVM.nu": self.cfg.ocsvm_nu,
                "LOF.n_neighbors": self.cfg.lof_neighbors,
                "Autoencoder.hidden": self.cfg.ae_hidden,
                "Autoencoder.max_iter": self.cfg.ae_max_iter,
            },
        }
