"""
Dual-Annealing hyperparameter search for (Precomputed) Adaptive-Weight LSSVM.

Runs a dual-annealing search over the LSSVM hyperparameters (gamma, gamma_rbf,
ro, decay, and number of range steps) for a set of random seeds, evaluates the
best configuration on a held-out test split, and writes per-seed results plus
a summary to a CSV file.

Two data modes are supported, chosen automatically from the dataset name:
    * Raw multi-view features (default)   -> AW_LSSVM
    * Precomputed kernels (e.g. "Flower") -> PrecomputedAW_LSSVM

Example
-------
    python3 da_awlssvm_search.py --dataset 3Sources  --coupling hard --method dissim --maxiter 1 --results_folder results
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import scipy.io
from scipy.optimize import dual_annealing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mv_lssvm import AW_LSSVM, PrecomputedAW_LSSVM

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATASET_CHOICES = [
    "3Sources", "ACM_subsampled", "Caltech101-7_subsampled", "Cora_subsampled",
    "Flower_subsampled", "MSRC-v5", "NUSWIDE_subsampled", "Prokaryotic",
    "ProteinFold_subsampled", "WebKB", "YouTube3views_subsampled_500", "Yale",
    "OutdoorScene_subsampled", "BBC4views",
]

COUPLING_CHOICES = ["all", "hard"]
METHOD_CHOICES = ["dissim", "avg"]

CV_SPLITS = 3
TEST_SIZE = 0.2
SEEDS = [0, 42, 123]
METRIC = "blacc"  # one of {"blacc", "acc", "f1"}

# Search bounds (log10-scale for all but the range-steps bound).
GAMMA_BOUNDS = (1e-5, 1e4)
GAMMA_RBF_BOUNDS = (1e-13, 1e-1)
RO_BOUNDS = (1e-5, 1e12)
DECAY_BOUNDS = (1e-5, 1)
STEPS_BOUNDS = (2, 6)

INIT_GAMMA = 500
INIT_GAMMA_RBF = 1e-4
INIT_RO = 500


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ctor: Callable


MODEL_SPECS = {"AW_LSSVM": ModelSpec("AW_LSSVM", AW_LSSVM)}
MODEL_SPECS_PRECOMPUTED = {
    "PrecomputedAW_LSSVM": ModelSpec("PrecomputedAW_LSSVM", PrecomputedAW_LSSVM)
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-annealing hyperparameter search for AW-LSSVM."
    )
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--coupling", required=True, choices=COUPLING_CHOICES)
    parser.add_argument("--method", required=True, choices=METHOD_CHOICES)
    parser.add_argument("--maxiter", type=int, required=True,
                         help="Max iterations for dual_annealing (e.g. 100, 1000).")
    parser.add_argument("--results_folder", required=True,
                         help="Directory the output CSV will be written to.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_dataset(dataset: str):
    """Load a .mat dataset, returning either raw multi-view features or
    precomputed kernels depending on the dataset name.

    Returns
    -------
    precomputed : bool
    X_or_K      : list of arrays (feature views, or kernel matrices)
    y           : label-encoded target array
    n_clients   : number of views / kernels
    """
    data = scipy.io.loadmat(f"datasets/{dataset}.mat")

    if dataset == "Flower":
        n_views = data["K"].shape[0]
        list_K = [data["K"][i][0] for i in range(n_views)]
        y = LabelEncoder().fit_transform(data["y"].flatten())
        return True, list_K, y, n_views

    n_views = data["X"].shape[0]
    X = [data["X"][i][0] for i in range(n_views)]
    X = [StandardScaler().fit_transform(view) for view in X]
    y = LabelEncoder().fit_transform(data["y"].flatten())
    return False, X, y, n_views


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def subset_views(views: Sequence[np.ndarray], idx: np.ndarray) -> list[np.ndarray]:
    return [view[idx] for view in views]


def subset_kernels(kernels: Sequence[np.ndarray], idx_tr: np.ndarray, idx_te: np.ndarray):
    """Return train/test kernel blocks for each view."""
    k_train = [K[idx_tr][:, idx_tr] for K in kernels]
    k_test = [K[idx_tr][:, idx_te].T for K in kernels]
    return k_train, k_test


def score_from_model(model, metric_name: str) -> float:
    if metric_name == "acc":
        return model.test_accuracy
    if metric_name == "f1":
        return model.test_f1
    return model.test_blacc


def clip_round_steps(value: float) -> int:
    return int(np.clip(round(value), STEPS_BOUNDS[0], STEPS_BOUNDS[1]))


# --------------------------------------------------------------------------- #
# Cross-validation scoring
# --------------------------------------------------------------------------- #

def cv_score_vfl(ctor, params: dict, X_train, y_train, *, cv: int = CV_SPLITS,
                  steps: int | None = None, inner_seed: int = 0,
                  metric_name: str = METRIC) -> float:
    """K-fold CV score for the raw multi-view (vertical federated learning) model."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=inner_seed)
    scores = []
    for tr_idx, va_idx in skf.split(X_train[0], y_train):
        X_tr, X_va = subset_views(X_train, tr_idx), subset_views(X_train, va_idx)
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        model = ctor(**params)
        model.fit(X_tr, y_tr, range_steps=steps)
        model.predict(X_va, y_va)
        scores.append(score_from_model(model, metric_name))
    return float(np.mean(scores))


def cv_score_kernel(ctor, params: dict, K_train: Sequence[np.ndarray], y_train,
                     *, cv: int = CV_SPLITS, steps: int | None = None,
                     inner_seed: int = 0, metric_name: str = METRIC,
                     adaptive: bool = False) -> float:
    """K-fold CV score for the precomputed-kernel model."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=inner_seed)
    scores = []
    for tr_idx, va_idx in skf.split(np.zeros(len(y_train)), y_train):
        K_tr, K_va = subset_kernels(K_train, tr_idx, va_idx)
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        model = ctor(**params)

        if adaptive:
            model.fit(K_tr, y_tr, range_steps=steps)
            model.predict(K_va, y_va)
            scores.append(score_from_model(model, metric_name))
            continue

        model.fit(K_tr, y_tr)
        y_pred = model.predict(K_va)
        if metric_name == "f1":
            scores.append(f1_score(y_va, y_pred, average="macro"))
        elif metric_name == "acc":
            scores.append(accuracy_score(y_va, y_pred))
        else:
            scores.append(balanced_accuracy_score(y_va, y_pred))

    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Train / test evaluation
# --------------------------------------------------------------------------- #

def train_and_test(train_X, train_y, test_X, test_y, spec: ModelSpec,
                    params: tuple, metric_name: str, *, n_clients: int,
                    coupling: str, method: str) -> float:
    gamma, gamma_rbf, ro, decay, steps = params
    model = spec.ctor(
        gamma=np.ones(n_clients) * gamma,
        gamma_rbf=np.ones(n_clients) * gamma_rbf,
        ro=ro, decay=decay, coupling=coupling, method=method,
    )
    model.fit(train_X, train_y, range_steps=steps)
    model.predict(test_X, test_y)
    return score_from_model(model, metric_name)


def train_and_test_precomputed(train_K, train_y, test_K, test_y, spec: ModelSpec,
                                params: tuple, metric_name: str, *, n_clients: int,
                                coupling: str, method: str) -> float:
    gamma, ro, decay, steps = params
    model = spec.ctor(
        np.ones(n_clients) * gamma, ro,
        decay=decay, coupling=coupling, method=method,
    )
    model.fit(train_K, train_y, range_steps=steps)
    model.predict(test_K, test_y)
    return score_from_model(model, metric_name)


# --------------------------------------------------------------------------- #
# Dual-annealing search
# --------------------------------------------------------------------------- #

def search_raw(train_X, train_y, spec: ModelSpec, metric_name: str, seed: int,
                *, n_clients: int, coupling: str, method: str, cv: int,
                maxiter: int) -> tuple:
    """Dual-annealing search over (gamma, gamma_rbf, ro[per client], decay, steps)."""
    bounds = (
        [(np.log10(GAMMA_BOUNDS[0]), np.log10(GAMMA_BOUNDS[1]))]
        + [(np.log10(GAMMA_RBF_BOUNDS[0]), np.log10(GAMMA_RBF_BOUNDS[1]))]
        + [(np.log10(RO_BOUNDS[0]), np.log10(RO_BOUNDS[1]))] * n_clients
        + [(np.log10(DECAY_BOUNDS[0]), np.log10(DECAY_BOUNDS[1]))]
        + [(STEPS_BOUNDS[0], STEPS_BOUNDS[1])]
    )

    def objective(x_log: np.ndarray) -> float:
        try:
            gamma = 10 ** x_log[0]
            gamma_rbf = 10 ** x_log[1]
            ro = np.array(10 ** x_log[2:-2])
            decay = 10 ** x_log[-2]
            steps = clip_round_steps(x_log[-1])

            params = {
                "gamma": np.ones(n_clients) * gamma,
                "gamma_rbf": np.ones(n_clients) * gamma_rbf,
                "ro": ro,
                "decay": decay,
                "method": method,
                "coupling": coupling,
            }
            score = cv_score_vfl(spec.ctor, params, train_X, train_y, cv=cv,
                                  steps=steps, inner_seed=seed, metric_name=metric_name)
            return -score
        except Exception:
            return 1e6

    result = dual_annealing(objective, bounds=bounds, maxiter=maxiter,
                             seed=seed, no_local_search=True)

    gamma_star = 10 ** result.x[0]
    gamma_rbf_star = 10 ** result.x[1]
    ro_star = np.array(10 ** result.x[2:-2])
    decay_star = 10 ** result.x[-2]
    steps_star = int(round(result.x[-1]))
    return gamma_star, gamma_rbf_star, ro_star, decay_star, steps_star, -result.fun


def search_precomputed(spec: ModelSpec, seed: int, K_train: Sequence[np.ndarray],
                        y_train, *, n_clients: int, metric_name: str,
                        cv: int, maxiter: int) -> tuple:
    """Dual-annealing search over (gamma, ro[per client], decay, steps)."""

    def objective(x_log: np.ndarray) -> float:
        try:
            gamma = 10.0 ** x_log[0]
            ro = 10.0 ** x_log[1]
            params = {"gamma": np.ones(n_clients) * gamma, "ro": ro}
            score = cv_score_kernel(spec.ctor, params, K_train, y_train, cv=cv,
                                     inner_seed=seed, metric_name=metric_name,
                                     adaptive=False)
            return -score
        except Exception:
            return 1e6

    bounds = (
        [(np.log10(GAMMA_BOUNDS[0]), np.log10(GAMMA_BOUNDS[1]))]
        + [(np.log10(RO_BOUNDS[0]), np.log10(RO_BOUNDS[1]))] * n_clients
        + [(np.log10(DECAY_BOUNDS[0]), np.log10(DECAY_BOUNDS[1]))]
        + [(STEPS_BOUNDS[0], STEPS_BOUNDS[1])]
    )

    result = dual_annealing(objective, bounds=bounds, maxfun=maxiter, seed=seed)

    gamma_star = 10.0 ** result.x[0]
    ro_star = np.array(10.0 ** result.x[1:-2])
    decay_star = 10.0 ** result.x[-2]
    steps_star = int(round(result.x[-1]))
    return gamma_star, ro_star, decay_star, steps_star, -result.fun


# --------------------------------------------------------------------------- #
# Experiment loops
# --------------------------------------------------------------------------- #

def run_raw_experiment(X, y, *, n_clients: int, coupling: str, method: str,
                        maxiter: int, results_path: Path, metric_name: str) -> None:
    records = []

    for seed in SEEDS:
        for model_name, spec in MODEL_SPECS.items():
            idx = np.arange(len(y))
            idx_tr, idx_te = train_test_split(
                idx, test_size=TEST_SIZE, random_state=int(seed), stratify=y
            )
            X_train, X_test = subset_views(X, idx_tr), subset_views(X, idx_te)
            y_train, y_test = y[idx_tr], y[idx_te]

            gamma, gamma_rbf, ro, decay, steps, cv_score = search_raw(
                X_train, y_train, spec, metric_name, seed=int(seed),
                n_clients=n_clients, coupling=coupling, method=method,
                cv=CV_SPLITS, maxiter=maxiter,
            )
            test_score = train_and_test(
                X_train, y_train, X_test, y_test, spec,
                (gamma, gamma_rbf, ro, decay, steps), metric_name,
                n_clients=n_clients, coupling=coupling, method=method,
            )

            records.append({
                "seed": seed,
                "model": model_name,
                "da_gamma": gamma,
                "da_gamma_rbf": gamma_rbf,
                "da_ro": ro,
                "da_decay": decay,
                "da_steps": steps,
                "cv_da": cv_score,
                f"test_{metric_name}": test_score,
            })

    save_results(records, results_path, metric_name, test_col_prefix="test")


def run_precomputed_experiment(list_K, y, *, n_clients: int, coupling: str,
                                method: str, maxiter: int, results_path: Path,
                                metric_name: str) -> None:
    records = []
    idx_all = np.arange(len(y))

    for seed in SEEDS:
        for model_name, spec in MODEL_SPECS_PRECOMPUTED.items():
            idx_tr, idx_te, y_tr, y_te = train_test_split(
                idx_all, y, test_size=TEST_SIZE, random_state=seed, stratify=y
            )
            K_train, K_test = subset_kernels(list_K, idx_tr, idx_te)

            gamma, ro, decay, steps, cv_score = search_precomputed(
                spec, seed, K_train, y_tr, n_clients=n_clients,
                metric_name=metric_name, cv=CV_SPLITS, maxiter=maxiter,
            )
            test_score = train_and_test_precomputed(
                K_train, y_tr, K_test, y_te, spec,
                (gamma, ro, decay, steps), metric_name,
                n_clients=n_clients, coupling=coupling, method=method,
            )

            records.append({
                "seed": seed,
                "model": model_name,
                "da_gamma": gamma,
                "da_ro": ro,
                "da_decay": decay,
                "da_steps": steps,
                "da_cv": cv_score,
                f"da_test_{metric_name}": test_score,
            })

    save_results(records, results_path, metric_name, test_col_prefix="da_test")


def save_results(records: list[dict], results_path: Path, metric_name: str,
                  *, test_col_prefix: str) -> None:
    df = pd.DataFrame.from_records(records).sort_values(["model", "seed"]).reset_index(drop=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    logger.info("\nSaved: %s", results_path)
    logger.info(df.head(12).to_string(index=False))

    test_col = f"{test_col_prefix}_{metric_name}"
    summary = df.groupby("model")[test_col].agg(["mean", "std"]).reset_index()
    logger.info("\nSummary (mean +/- std across seeds):")
    for _, row in summary.iterrows():
        logger.info("%-30s | %.4f +/- %.4f", row["model"], row["mean"], row["std"])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    csv_name = f"DA_{args.dataset}_{args.coupling}_{args.method}_{args.maxiter}.csv"
    results_path = Path(args.results_folder) / csv_name

    logger.info(
        "DA %s | coupling=%s | method=%s | maxiter=%s | results_folder=%s",
        args.dataset, args.coupling, args.method, args.maxiter, args.results_folder,
    )

    precomputed, data, y, n_clients = load_dataset(args.dataset)

    start = time.time()
    if precomputed:
        run_precomputed_experiment(
            data, y, n_clients=n_clients, coupling=args.coupling,
            method=args.method, maxiter=args.maxiter,
            results_path=results_path, metric_name=METRIC,
        )
    else:
        run_raw_experiment(
            data, y, n_clients=n_clients, coupling=args.coupling,
            method=args.method, maxiter=args.maxiter,
            results_path=results_path, metric_name=METRIC,
        )

    logger.info("Runtime: %.2f seconds", time.time() - start)


if __name__ == "__main__":
    main()