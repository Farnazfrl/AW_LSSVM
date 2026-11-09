"""
Multi-view Least-Squares SVM classifiers.

This module implements:

- ``LSSVM``: single-view binary LS-SVM (dual formulation, RBF kernel).
- ``MultiClassLSSVM`` / ``PrecomputedMultiClassLSSVM``: single-view
  one-vs-all multiclass LS-SVM, from raw features or a precomputed kernel.
- ``MultiViewLSSVM`` / ``PrecomputedMultiViewLSSVM``: coupled multi-view
  LS-SVM (Houthuys et al., "Multi-view least squares support vector
  machines classification", Neurocomputing 2018).
- ``AW_LSSVM`` / ``PrecomputedAW_LSSVM``: Adaptive Weighted LS-SVM, an
  iterative multi-view classifier that reweights samples across views
  based on cross-view misclassification (see Algorithm 1 and Eqs. 1-7 of
  "Adaptive Weighted LS-SVM for Multi-View Classification"). Both variants
  share their iterative training/prediction logic via
  ``_AdaptiveWeightedLSSVMBase`` so that fixes only need to be made once.
- ``MidlateFusion`` / ``MidlateFusion_precomputed``: simple mid/late-fusion
  baselines (committee-style soft-score aggregation, or majority voting).
- ``cross_val_acc`` / ``late_fusion_committee``: cross-validation helpers.
- ``compute_gamma``: heuristic RBF bandwidth selection.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, solve
from scipy.stats import mode
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

__all__ = [
    "LSSVM",
    "MultiClassLSSVM",
    "MultiViewLSSVM",
    "MidlateFusion",
    "MidlateFusion_precomputed",
    "PrecomputedMultiViewLSSVM",
    "PrecomputedMultiClassLSSVM",
    "AW_LSSVM",
    "PrecomputedAW_LSSVM",
    "cross_val_acc",
    "late_fusion_committee",
    "compute_gamma",
]


# --------------------------------------------------------------------------- #
# Shared kernel helper
# --------------------------------------------------------------------------- #

def rbf_kernel_matrix(X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """RBF kernel matrix K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)."""
    sq_dist = (
        np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        + np.sum(X2 ** 2, axis=1)
        - 2 * X1 @ X2.T
    )
    return np.exp(-gamma * sq_dist)


# --------------------------------------------------------------------------- #
# Single-view models
# --------------------------------------------------------------------------- #

class LSSVM:
    """Binary LS-SVM, dual formulation, RBF kernel."""

    def __init__(self, gamma: float = 1, gamma_rbf: float = 1, kernel: str = "rbf", ro: float = 1):
        self.kernel = kernel
        self.gamma_rbf = gamma_rbf
        self.gamma = gamma
        self.ro = ro

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray | int = 0) -> "LSSVM":
        n_samples = X.shape[0]
        Y = y.reshape(-1, 1)
        self.X = X
        self.Y = Y

        if self.kernel == "rbf":
            K = rbf_kernel_matrix(X, X, self.gamma_rbf)

        Omega = (Y * Y.T * K) + (np.eye(n_samples) / (self.gamma + self.ro * weights))
        P = np.block([[0, Y.T], [Y, Omega]])
        q = np.vstack([0, np.ones((n_samples, 1))])

        solution = np.linalg.solve(P, q)
        self.b = solution[0, 0]
        self.alpha = solution[1:].flatten()
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.kernel == "rbf":
            K_test = rbf_kernel_matrix(X_test, self.X, self.gamma_rbf)
            self.K_test = K_test
            self.y_pred = np.sum((K_test * self.alpha) * self.Y.T, axis=1) + self.b
        return np.sign(self.y_pred)


class MultiClassLSSVM:
    """One-vs-all multiclass LS-SVM from raw features (RBF kernel)."""

    def __init__(self, gamma: float = 1.0, gamma_rbf: float = 1.0, ro: float = 1.0):
        self.gamma = gamma
        self.gamma_rbf = gamma_rbf
        self.ro = ro

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray | int = 0) -> "MultiClassLSSVM":
        self.X_train = X
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        self.n_samples = n_samples
        self.n_classes = n_classes

        K = rbf_kernel_matrix(X, X, self.gamma_rbf)

        self.alphas_, self.biases_, self.y_binaries_ = [], [], []

        for idx, cls in enumerate(self.classes_):
            y_binary = np.where(y == cls, 1.0, -1.0)
            self.y_binaries_.append(y_binary)

            if isinstance(weights, int):
                D = np.eye(n_samples) / self.gamma
            else:
                D = np.diag(1.0 / (self.gamma + self.ro * weights[:, idx]))

            Y_outer = y_binary.reshape(-1, 1) @ y_binary.reshape(1, -1)
            Omega = Y_outer * K
            K_reg = Omega + D

            A = np.zeros((n_samples + 1, n_samples + 1))
            A[0, 1:] = y_binary
            A[1:, 0] = y_binary
            A[1:, 1:] = K_reg
            A += 1e-9 * np.eye(A.shape[0])

            b = np.zeros(n_samples + 1)
            b[1:] = 1.0

            solution = np.linalg.solve(A, b)
            self.biases_.append(solution[0])
            self.alphas_.append(solution[1:])

        self.biases_ = np.array(self.biases_)
        self.alphas_ = np.array(self.alphas_)
        self.y_binaries_ = np.array(self.y_binaries_)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        K_test = rbf_kernel_matrix(X_test, self.X_train, self.gamma_rbf)
        n_samples_test = X_test.shape[0]
        y_pred = np.zeros((n_samples_test, self.n_classes))
        for i in range(self.n_classes):
            y_pred[:, i] = (K_test @ (self.alphas_[i] * self.y_binaries_[i])) + self.biases_[i]
        self.y_pred = y_pred
        return np.argmax(y_pred, axis=1)


class PrecomputedMultiClassLSSVM:
    """One-vs-all multiclass LS-SVM from a precomputed kernel matrix."""

    def __init__(self, gamma: float = 1.0, ro: float = 1.0):
        self.gamma = gamma
        self.ro = ro

    def fit(self, K_train: np.ndarray, y: np.ndarray, weights: np.ndarray | int = 0) -> "PrecomputedMultiClassLSSVM":
        self.K = K_train
        self.classes_ = np.unique(y)
        n_samples = K_train.shape[0]
        n_classes = len(self.classes_)
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.alphas_, self.biases_, self.y_binaries_ = [], [], []

        for idx, cls in enumerate(self.classes_):
            y_binary = np.where(y == cls, 1.0, -1.0)
            self.y_binaries_.append(y_binary)

            if isinstance(weights, int):
                D = np.eye(n_samples) / self.gamma
            else:
                D = np.diag(1.0 / (self.gamma + self.ro * weights[:, idx]))

            Y_outer = y_binary.reshape(-1, 1) @ y_binary.reshape(1, -1)
            K_reg = Y_outer * K_train + D

            A = np.zeros((n_samples + 1, n_samples + 1))
            A[0, 1:] = y_binary
            A[1:, 0] = y_binary
            A[1:, 1:] = K_reg
            A += 1e-9 * np.eye(A.shape[0])

            b = np.zeros(n_samples + 1)
            b[1:] = 1.0

            solution = np.linalg.solve(A, b)
            self.biases_.append(solution[0])
            self.alphas_.append(solution[1:])

        self.biases_ = np.array(self.biases_)
        self.alphas_ = np.array(self.alphas_)
        self.y_binaries_ = np.array(self.y_binaries_)
        return self

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        n_samples_test = K_test.shape[0]
        y_pred = np.zeros((n_samples_test, self.n_classes))
        for i in range(self.n_classes):
            y_pred[:, i] = (K_test @ (self.alphas_[i] * self.y_binaries_[i])) + self.biases_[i]
        self.y_pred = y_pred
        return np.argmax(y_pred, axis=1)


# --------------------------------------------------------------------------- #
# Coupled multi-view LS-SVM (Houthuys et al. 2018)
# --------------------------------------------------------------------------- #

class MultiViewLSSVM:
    """Multi-view LS-SVM with pairwise error coupling across views (raw features)."""

    def __init__(self, gamma, gamma_rbf: float = 1.0, ro: float = 10, ro_adaptive: float = 1):
        self.gamma = gamma          # per-view error regularization
        self.gamma_rbf = gamma_rbf  # RBF kernel width
        self.ro = ro                # coupling regularization
        self.ro_adaptive = ro_adaptive

    def get_params(self, deep: bool = True) -> dict:
        return {"gamma": self.gamma, "gamma_rbf": self.gamma_rbf, "ro": self.ro}

    def set_params(self, **params) -> "MultiViewLSSVM":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y, weights=0) -> None:
        self.X_train = X
        self.classes = np.unique(y)
        self.n_classes = int(len(self.classes))
        self.n_samples = int(X[0].shape[0])
        self.n_views = len(self.X_train)

        self.K = [rbf_kernel_matrix(X[v], X[v], self.gamma_rbf) for v in range(self.n_views)]

        if self.n_classes > 2:
            Y = np.zeros((self.n_samples, self.n_classes))
            for i, cls in enumerate(self.classes):
                Y[:, i] = np.where(y == cls, 1, -1)
            self.Y = Y
        else:
            Y = y
            self.Y = y

        block_gamma = np.array([])
        for v in range(self.n_views):
            diag_gamma = np.diag(np.array([self.gamma[v] + self.ro_adaptive * weights] * self.n_samples))
            block_gamma = block_diag(block_gamma, diag_gamma)
        block_gamma = block_gamma[1:, :]
        self.block_gamma = block_gamma

        I = np.zeros((self.n_samples * self.n_views, self.n_samples * self.n_views))
        for v1 in range(self.n_views):
            for v2 in range(self.n_views):
                if v1 != v2:
                    sl1 = slice(v1 * self.n_samples, (v1 + 1) * self.n_samples)
                    sl2 = slice(v2 * self.n_samples, (v2 + 1) * self.n_samples)
                    I[sl1, sl2] = np.eye(self.n_samples)

        if self.n_classes == 2:
            self.biases_multiclass, self.alphas_multiclass = self._solve_subproblem(Y, block_gamma, I)
        else:
            biases_multiclass = np.empty((self.n_views, 1))
            alphas_multiclass = np.empty((self.n_samples * self.n_views, 1))
            for i in range(self.n_classes):
                b_i, a_i = self._solve_subproblem(Y[:, i], block_gamma, I)
                biases_multiclass = np.hstack((biases_multiclass, b_i))
                alphas_multiclass = np.hstack((alphas_multiclass, a_i))
            self.biases_multiclass = biases_multiclass[:, 1:]
            self.alphas_multiclass = alphas_multiclass[:, 1:]

    def _solve_subproblem(self, y_binary, block_gamma, I):
        block_Y = block_diag(*[y_binary.reshape(-1, 1) for _ in range(self.n_views)])
        omegas = [y_binary.reshape(-1, 1) * y_binary.reshape(-1, 1).T * self.K[v] for v in range(self.n_views)]
        block_omega = block_diag(*omegas)

        p_bottom_left = (block_gamma @ block_Y) + (self.ro * I @ block_Y)
        p_bottom_right = (block_gamma @ block_omega) + np.eye(self.n_samples * self.n_views) + (self.ro * I @ block_omega)
        p = np.block([[np.zeros((self.n_views, self.n_views)), block_Y.T], [p_bottom_left, p_bottom_right]])
        p += 1e-4 * np.eye(p.shape[0])

        q_bottom = (block_gamma @ np.ones((self.n_samples * self.n_views, 1))) + \
                   ((self.n_views - 1) * self.ro * np.ones((self.n_samples * self.n_views, 1)))
        q = np.vstack([np.zeros((self.n_views, 1)), q_bottom])

        solution = solve(p, q)
        return solution[:self.n_views], solution[self.n_views:]

    def predict(self, X_test):
        K_test = [rbf_kernel_matrix(X_test[v], self.X_train[v], self.gamma_rbf) for v in range(self.n_views)]
        self.K_test = K_test
        n_samples_test = X_test[0].shape[0]
        self.n_samples_test = n_samples_test
        block_K_test = block_diag(*K_test)
        self.block_K_test = block_K_test

        if self.n_classes == 2:
            return self._predict_binary(self.Y, self.alphas_multiclass, self.biases_multiclass, n_samples_test)

        y_pred_multiclass = np.empty((n_samples_test, 1))
        for cls in range(self.n_classes):
            if self.n_views > 1:
                alphas_cls = self.alphas_multiclass[:, cls]
                biases_cls = self.biases_multiclass[:, cls]
            else:
                alphas_cls = self.alphas_multiclass[:, cls]
                biases_cls = self.biases_multiclass.flatten()[cls]
            y_pred = self._aggregate_views(self.Y[:, cls], alphas_cls, biases_cls, n_samples_test)
            y_pred_multiclass = np.hstack((y_pred_multiclass, y_pred.reshape(-1, 1)))

        self.y_pred_multiclass = y_pred_multiclass[:, 1:]
        return np.argmax(self.y_pred_multiclass, axis=1)

    def _predict_binary(self, y_binary, alphas, biases, n_samples_test):
        if self.n_views > 1:
            block_alphas = block_diag(*[alphas[i * self.n_samples:(i + 1) * self.n_samples].reshape(-1, 1) for i in range(self.n_views)])
            block_biases = block_diag(*[np.array([biases[i]] * n_samples_test).reshape(-1, 1) for i in range(self.n_views)])
        else:
            block_alphas = alphas.reshape(-1, 1)
            block_biases = np.array([biases.flatten()] * n_samples_test).reshape(-1, 1)

        block_Y = block_diag(*[y_binary.reshape(-1, 1) for _ in range(self.n_views)])
        y_pred_block = self.block_K_test @ (block_alphas * block_Y) + block_biases

        y_pred = np.zeros((n_samples_test, self.n_views))
        j = 0
        for i in range(self.n_views):
            y_pred[:, i] = y_pred_block[j:j + n_samples_test, i]
            j += n_samples_test

        self.y_pred_multiclass = np.mean(y_pred, axis=1)
        return np.sign(self.y_pred_multiclass)

    def _aggregate_views(self, y_binary, alphas_cls, biases_cls, n_samples_test):
        if self.n_views > 1:
            block_alphas = block_diag(*[alphas_cls[i * self.n_samples:(i + 1) * self.n_samples].reshape(-1, 1) for i in range(self.n_views)])
            block_biases = block_diag(*[np.array([biases_cls[i]] * n_samples_test).reshape(-1, 1) for i in range(self.n_views)])
        else:
            block_alphas = alphas_cls.reshape(-1, 1)
            block_biases = np.array([biases_cls] * n_samples_test).reshape(-1, 1)

        block_Y = block_diag(*[y_binary.reshape(-1, 1) for _ in range(self.n_views)])
        y_pred_block = self.block_K_test @ (block_alphas * block_Y) + block_biases

        y_pred = np.zeros((n_samples_test, self.n_views))
        j = 0
        for i in range(self.n_views):
            y_pred[:, i] = y_pred_block[j:j + n_samples_test, i]
            j += n_samples_test
        return np.mean(y_pred, axis=1)


class PrecomputedMultiViewLSSVM(MultiViewLSSVM):
    """Multi-view LS-SVM with pairwise error coupling, from precomputed per-view kernels."""

    def __init__(self, gamma, ro: float = 10, ro_adaptive: float = 1):
        self.gamma = gamma
        self.ro = ro
        self.ro_adaptive = ro_adaptive

    def get_params(self, deep: bool = True) -> dict:
        return {"gamma": self.gamma, "ro": self.ro}

    def fit(self, K, y, weights=0) -> None:
        self.K = K
        self.classes = np.unique(y)
        self.n_classes = int(len(self.classes))
        self.n_samples = int(K[0].shape[0])
        self.n_views = len(self.K)
        # `X_train` is unused for prediction here since kernels are precomputed,
        # but kept for API symmetry with MultiViewLSSVM.
        self.X_train = None

        if self.n_classes > 2:
            Y = np.zeros((self.n_samples, self.n_classes))
            for i, cls in enumerate(self.classes):
                Y[:, i] = np.where(y == cls, 1, -1)
            self.Y = Y
        else:
            Y = y
            self.Y = y

        block_gamma = np.array([])
        for v in range(self.n_views):
            diag_gamma = np.diag(np.array([self.gamma[v] + self.ro_adaptive * weights] * self.n_samples))
            block_gamma = block_diag(block_gamma, diag_gamma)
        block_gamma = block_gamma[1:, :]
        self.block_gamma = block_gamma

        I = np.zeros((self.n_samples * self.n_views, self.n_samples * self.n_views))
        for v1 in range(self.n_views):
            for v2 in range(self.n_views):
                if v1 != v2:
                    sl1 = slice(v1 * self.n_samples, (v1 + 1) * self.n_samples)
                    sl2 = slice(v2 * self.n_samples, (v2 + 1) * self.n_samples)
                    I[sl1, sl2] = np.eye(self.n_samples)

        if self.n_classes == 2:
            self.biases_multiclass, self.alphas_multiclass = self._solve_subproblem(Y, block_gamma, I)
        else:
            biases_multiclass = np.empty((self.n_views, 1))
            alphas_multiclass = np.empty((self.n_samples * self.n_views, 1))
            for i in range(self.n_classes):
                b_i, a_i = self._solve_subproblem(Y[:, i], block_gamma, I)
                biases_multiclass = np.hstack((biases_multiclass, b_i))
                alphas_multiclass = np.hstack((alphas_multiclass, a_i))
            self.biases_multiclass = biases_multiclass[:, 1:]
            self.alphas_multiclass = alphas_multiclass[:, 1:]

    def predict(self, K_test):
        self.K_test = K_test
        n_samples_test = K_test[0].shape[0]
        self.n_samples_test = n_samples_test
        self.block_K_test = block_diag(*K_test)

        if self.n_classes == 2:
            return self._predict_binary(self.Y, self.alphas_multiclass, self.biases_multiclass, n_samples_test)

        y_pred_multiclass = np.empty((n_samples_test, 1))
        for cls in range(self.n_classes):
            alphas_cls = self.alphas_multiclass[:, cls]
            biases_cls = self.biases_multiclass[:, cls] if self.n_views > 1 else self.biases_multiclass.flatten()[cls]
            y_pred = self._aggregate_views(self.Y[:, cls], alphas_cls, biases_cls, n_samples_test)
            y_pred_multiclass = np.hstack((y_pred_multiclass, y_pred.reshape(-1, 1)))

        self.y_pred_multiclass = y_pred_multiclass[:, 1:]
        return np.argmax(self.y_pred_multiclass, axis=1)


# --------------------------------------------------------------------------- #
# Mid-late fusion baselines
# --------------------------------------------------------------------------- #

class MidlateFusion:
    """Committee-style mid/late fusion: sums per-view soft decision scores
    (or, alternatively, majority-votes per-view hard predictions)."""

    def __init__(self, gamma: float = 1, gamma_rbf: float = 1):
        self.gamma = gamma
        self.gamma_rbf = gamma_rbf

    def get_params(self, deep: bool = True) -> dict:
        return {"gamma": self.gamma, "gamma_rbf": self.gamma_rbf}

    def set_params(self, **params) -> "MidlateFusion":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X_train, X_test, y_train, y_test):
        n_clients = len(X_train)
        n_samples = X_train[0].shape[0]
        n_samples_test = X_test[0].shape[0]
        n_classes = len(np.unique(y_train))

        y_sum = np.zeros((n_samples, n_classes))
        y_sum_test = np.zeros((n_samples_test, n_classes))

        for c in range(n_clients):
            model = MultiClassLSSVM(self.gamma[c], self.gamma_rbf[c])
            model.fit(X_train[c], y_train)
            model.predict(X_train[c])
            y_sum += model.y_pred
            model.predict(X_test[c])
            y_sum_test += model.y_pred

        # Committee-style soft-score aggregation (mid/late fusion).
        # For majority voting on hard predictions instead, aggregate
        # per-view np.argmax predictions with scipy.stats.mode.
        y_pred_com = np.argmax(y_sum, axis=1)
        y_pred_test_com = np.argmax(y_sum_test, axis=1)
        self.y_pred_com = y_pred_com

        sc_train = [accuracy_score(y_train, y_pred_com)]
        sc_test = [accuracy_score(y_test, y_pred_test_com)]

        self.test_accuracy = sc_test[-1]
        self.test_blacc = balanced_accuracy_score(y_test, y_pred_test_com)
        self.test_f1 = f1_score(y_test, y_pred_test_com, average="macro")
        return sc_train, sc_test


class MidlateFusion_precomputed:
    """Late-fusion baseline via majority voting over per-view predictions,
    for precomputed per-view kernels."""

    def __init__(self, gamma: float = 1):
        self.gamma = gamma

    def get_params(self, deep: bool = True) -> dict:
        return {"gamma": self.gamma}

    def set_params(self, **params) -> "MidlateFusion_precomputed":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X_train, X_test, y_train, y_test):
        n_clients = len(X_train)
        n_samples = X_train[0].shape[0]
        n_samples_test = X_test[0].shape[0]

        y_pred = np.zeros((n_samples, n_clients))
        y_pred_test = np.zeros((n_samples_test, n_clients))

        for c in range(n_clients):
            model = MultiClassLSSVM(self.gamma[c])
            model.fit(X_train[c], y_train)
            y_pred[:, c] = model.predict(X_train[c])
            y_pred_test[:, c] = model.predict(X_test[c])

        # Majority-voting late fusion over per-view hard predictions.
        y_pred_com = mode(y_pred, axis=1, keepdims=False).mode
        y_pred_test_com = mode(y_pred_test, axis=1, keepdims=False).mode
        self.y_pred_com = y_pred_com

        sc_train = [accuracy_score(y_train, y_pred_com)]
        sc_test = [accuracy_score(y_test, y_pred_test_com)]

        self.test_accuracy = sc_test[-1]
        self.test_blacc = balanced_accuracy_score(y_test, y_pred_test_com)
        self.test_f1 = f1_score(y_test, y_pred_test_com, average="macro")
        return sc_train, sc_test


# --------------------------------------------------------------------------- #
# Cross-validation helpers
# --------------------------------------------------------------------------- #

def _score(y_true, y_pred, metric: str) -> float:
    if metric == "acc":
        return accuracy_score(y_true, y_pred)
    if metric == "blacc":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1":
        return f1_score(y_true, y_pred, average="macro")
    raise ValueError(f"Unknown metric: {metric!r}")


def cross_val_acc(clf, X, y, cv: int = 5, mv: int = 0, random_state: int = 0, metric: str = "acc") -> np.ndarray:
    """Stratified K-fold CV score for a single-view (`mv=0`) or
    multi-view (`mv=1`, `X` is a list of per-view arrays) classifier."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []

    if mv:
        for train_idx, test_idx in skf.split(X[0], y):
            X_train_mv = [Xv[train_idx] for Xv in X]
            X_test_mv = [Xv[test_idx] for Xv in X]
            y_train, y_test = y[train_idx], y[test_idx]
            clf.fit(X_train_mv, y_train)
            y_pred = clf.predict(X_test_mv)
            scores.append(_score(y_test, y_pred, metric))
    else:
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores.append(_score(y_test, y_pred, metric))

    return np.array(scores)


def late_fusion_committee(clf, X, y, cv: int = 5, mv: int = 0, random_state: int = 42):
    """Collect per-fold (y_test, y_pred) pairs for a committee-style late-fusion analysis."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    y_pred, y_test_cv = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        y_pred.append(clf.predict(X_test))
        y_test_cv.append(y_test)

    return y_test_cv, y_pred


def compute_gamma(X_mv: np.ndarray, type: str = "scale") -> float:
    """Heuristic RBF bandwidth, similar to sklearn's 'scale'/'auto' presets."""
    n_features = X_mv.shape[1]
    mean_var = np.var(X_mv, axis=0).mean()
    if type == "scale":
        return 1.0 / (n_features * mean_var)
    if type == "auto":
        return 1.0 / n_features
    return type


# --------------------------------------------------------------------------- #
# Adaptive Weighted LS-SVM (AW-LSSVM)
# --------------------------------------------------------------------------- #

class _AdaptiveWeightedLSSVMBase:
    """Shared training/prediction logic for the AW-LSSVM family.

    Implements Algorithm 1 of "Adaptive Weighted LS-SVM for Multi-View
    Classification": at each iteration, every view is retrained with
    sample weights that push it to compensate for the other views'
    misclassified samples, using either uniform ('avg', Eq. 2) or
    dissimilarity-aware ('dissim', Eq. 3) aggregation of the other views'
    squared errors.

    Subclasses (``AW_LSSVM`` for raw features, ``PrecomputedAW_LSSVM`` for
    precomputed kernels) only need to implement ``_build_view_model`` and
    thin ``fit``/``predict`` wrappers around ``_fit_impl``/``_predict_impl``
    — every other algorithmic detail (including the safe-division and
    weight-aggregation fixes) lives here once, so the two variants can't
    drift out of sync with each other.
    """

    def __init__(self, ro=10, decay: float = 0.7, coupling: str = "hard", method: str = "dissim"):
        # coupling='all': every sample's error is coupled through weights.
        # coupling='hard': only misclassified samples' errors are coupled.
        # method='dissim': dissimilarity-aware error aggregation (Eq. 3).
        # method='avg': uniform average-based error aggregation (Eq. 2).
        self.ro = ro
        self.decay = decay
        self.coupling = coupling
        self.method = method

    # ---- to be implemented by subclasses ----------------------------------

    def _build_view_model(self, view_idx: int):
        """Return a fresh per-view LSSVM instance for client `view_idx`."""
        raise NotImplementedError

    # ---- algorithm building blocks -----------------------------------------

    @staticmethod
    def _sample_errors(y_train, y_pred_labels_c, decision_scores_c, coupling: str):
        """Per-view squared errors used for cross-view coupling (see Eq. 5
        and the paragraph following Eq. 3: ``ẽ_k = 0`` whenever the sample
        is not misclassified under `coupling='hard'`).
        """
        n_samples, n_classes = decision_scores_c.shape
        label_matrix = -np.ones((n_samples, n_classes))
        label_matrix[np.arange(n_samples), y_train] = 1.0

        errors_orig = 1.0 - label_matrix * decision_scores_c
        squared = errors_orig ** 2

        if coupling == "hard":
            misclassified = (y_pred_labels_c != y_train)[:, None]
            errors = np.where(misclassified, squared, 0.0)
        else:
            errors = squared

        return errors, errors_orig

    @staticmethod
    def _common_hard_matrix(errors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Diagnostic: pairwise Dice overlap of which samples each pair of
        views finds "hard" (nonzero error)."""
        hard_mask = np.mean(errors, axis=2) != 0
        n_clients = hard_mask.shape[0]
        hard_counts = hard_mask.sum(axis=1).astype(float)

        common_hard = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(n_clients):
                intersection = np.sum(hard_mask[i] & hard_mask[j])
                common_hard[i, j] = 2.0 * intersection / (hard_counts[i] + hard_counts[j] + eps)
        return common_hard

    @staticmethod
    def _error_covariance(errors_orig: np.ndarray) -> np.ndarray:
        """Diagnostic: class-averaged cross-view error covariance."""
        n_clients, n_samples, n_classes = errors_orig.shape
        cov = np.zeros((n_classes, n_clients, n_clients))
        for cls in range(n_classes):
            E = errors_orig[:, :, cls]
            cov[cls] = (E @ E.T) / n_samples
        return cov.mean(axis=0)

    @staticmethod
    def _scaled_dissimilarity(errors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Row-normalized pairwise dissimilarity of squared errors across
        views, per class (Eq. 3's normalization term). Rows that sum to
        zero (identical error patterns) are safely handled instead of
        dividing by zero.
        """
        n_clients, _, n_classes = errors.shape
        scaled = np.zeros((n_classes, n_clients, n_clients))
        for cls in range(n_classes):
            E = errors[:, :, cls]
            diff = E[:, None, :] - E[None, :, :]
            dissimilarity = np.sqrt(np.sum(diff ** 2, axis=2))
            row_sums = dissimilarity.sum(axis=1)
            safe_row_sums = np.where(row_sums == 0, eps, row_sums)
            scaled[cls] = dissimilarity / safe_row_sums[:, None]
        return scaled

    def _update_weights(self, weights: np.ndarray, errors: np.ndarray,
                         scaled_dissimilarity: np.ndarray, t: int) -> np.ndarray:
        """Accumulate the cross-view coupling term into `weights` (Eqs. 2-3).

        For `method='avg'`, `scaled_dissimilarity` is all-ones and the
        excluded-views' errors are averaged (the explicit 1/(V-1) factor
        in Eq. 2). For `method='dissim'`, `scaled_dissimilarity` is already
        row-normalized to sum to 1 across the other views, so the
        contributions are summed rather than averaged again (Eq. 3).
        """
        n_clients, n_samples, n_classes = errors.shape
        for c in range(n_clients):
            errors_excluded = np.delete(errors, c, axis=0)
            dissim_excluded = np.delete(scaled_dissimilarity, c, axis=2)
            for cls in range(n_classes):
                w = dissim_excluded[cls, c, :].reshape(n_clients - 1, 1, 1)
                product = errors_excluded[:, :, cls].reshape(n_clients - 1, n_samples, 1) * w
                agg = product.mean(axis=0) if self.method == "avg" else product.sum(axis=0)
                weights[c, :, cls] += (self.decay ** t) * agg.flatten()
        return weights

    # ---- main training / inference loops -----------------------------------

    def _fit_impl(self, data, y_train, range_steps: int):
        n_clients = len(data)
        self.n_clients = n_clients
        n_samples = data[0].shape[0]
        n_classes = len(np.unique(y_train))
        self.n_classes = n_classes
        self.range_steps = range_steps

        weights = np.zeros((n_clients, n_samples, n_classes))

        sc_train = []
        dissimilarity_over_time = []
        cov_over_time = []
        common_hard_over_time = []
        all_models = []

        for t in range(range_steps):
            errors = np.zeros((n_clients, n_samples, n_classes))
            errors_orig = np.zeros((n_clients, n_samples, n_classes))
            y_sum = np.zeros((n_samples, n_classes))
            y_pred_labels = np.zeros((n_samples, n_clients))
            models_per_client = []

            for c in range(n_clients):
                model = self._build_view_model(c)
                model.fit(data[c], y_train, weights[c, :, :])
                models_per_client.append(model)

                y_pred_labels[:, c] = model.predict(data[c])
                y_sum += model.y_pred

                errors[c], errors_orig[c] = self._sample_errors(
                    y_train, y_pred_labels[:, c], model.y_pred, self.coupling
                )

            common_hard_over_time.append(self._common_hard_matrix(errors))
            cov_over_time.append(self._error_covariance(errors_orig))
            all_models.append(models_per_client)

            y_pred_com = np.argmax(y_sum, axis=1)
            self.y_pred_com = y_pred_com
            sc_train.append(balanced_accuracy_score(y_train, y_pred_com))

            if np.sum(errors) == 0:
                break

            scaled_dissimilarity = self._scaled_dissimilarity(errors)
            dissimilarity_over_time.append(scaled_dissimilarity.copy())

            if self.method == "avg":
                scaled_dissimilarity = np.ones((n_classes, n_clients, n_clients))
            elif self.method == "dissim" and np.sum(scaled_dissimilarity) == 0:
                break

            weights = self._update_weights(weights, errors, scaled_dissimilarity, t)
            self.weights = weights
            self.errors = errors

        self.all_models = all_models
        self.common_hard_OT = common_hard_over_time
        self.t = t

        return sc_train, dissimilarity_over_time, cov_over_time

    def _predict_impl(self, data_test, y_test):
        n_samples_test = data_test[0].shape[0]
        range_steps = self.range_steps
        steps_trained = len(self.all_models)
        if steps_trained != self.range_steps:
            range_steps = steps_trained

        client_performances_test = np.zeros((self.n_clients, self.range_steps))
        y_pred_test = np.zeros((n_samples_test, self.n_clients))
        sc_test = []

        for t in range(range_steps):
            y_sum_test = np.zeros((n_samples_test, self.n_classes))
            for c in range(self.n_clients):
                y_pred_test[:, c] = self.all_models[t][c].predict(data_test[c])
                client_performances_test[c, t] = balanced_accuracy_score(y_test, y_pred_test[:, c])
                y_sum_test += self.all_models[t][c].y_pred
            y_pred_test_com = np.argmax(y_sum_test, axis=1)
            sc_test.append(balanced_accuracy_score(y_test, y_pred_test_com))

        self.test_accuracy = accuracy_score(y_test, y_pred_test_com)
        self.test_blacc = balanced_accuracy_score(y_test, y_pred_test_com)
        self.test_f1 = f1_score(y_test, y_pred_test_com, average="macro")

        return sc_test, client_performances_test


class AW_LSSVM(_AdaptiveWeightedLSSVMBase):
    """Adaptive Weighted LS-SVM for multi-view classification, raw features."""

    def __init__(self, gamma=1, gamma_rbf=1, ro=10, decay: float = 0.7,
                 coupling: str = "hard", method: str = "dissim"):
        super().__init__(ro=ro, decay=decay, coupling=coupling, method=method)
        self.gamma = gamma
        self.gamma_rbf = gamma_rbf

    def get_params(self, deep: bool = True) -> dict:
        return {"gamma": self.gamma, "gamma_rbf": self.gamma_rbf, "ro": self.ro,
                "decay": self.decay, "coupling": self.coupling, "method": self.method}

    def set_params(self, **params) -> "AW_LSSVM":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _build_view_model(self, view_idx: int) -> MultiClassLSSVM:
        return MultiClassLSSVM(self.gamma[view_idx], self.gamma_rbf[view_idx], ro=self.ro[view_idx])

    def fit(self, X_train, y_train, range_steps: int):
        return self._fit_impl(X_train, y_train, range_steps)

    def predict(self, X_test, y_test):
        return self._predict_impl(X_test, y_test)


class PrecomputedAW_LSSVM(_AdaptiveWeightedLSSVMBase):
    """Adaptive Weighted LS-SVM for multi-view classification, precomputed
    per-view kernel matrices. Shares its algorithm with ``AW_LSSVM`` via
    ``_AdaptiveWeightedLSSVMBase``."""

    def __init__(self, gamma=1, ro=10, decay: float = 0.7,
                 coupling: str = "hard", method: str = "dissim"):
        super().__init__(ro=ro, decay=decay, coupling=coupling, method=method)
        self.gamma = gamma

    def get_params(self, deep: bool = True) -> dict:
        return {"gamma": self.gamma, "ro": self.ro, "decay": self.decay,
                "coupling": self.coupling, "method": self.method}

    def set_params(self, **params) -> "PrecomputedAW_LSSVM":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _build_view_model(self, view_idx: int) -> PrecomputedMultiClassLSSVM:
        return PrecomputedMultiClassLSSVM(self.gamma[view_idx], ro=self.ro[view_idx])

    def fit(self, K_train, y_train, range_steps: int):
        return self._fit_impl(K_train, y_train, range_steps)

    def predict(self, K_test, y_test):
        return self._predict_impl(K_test, y_test)