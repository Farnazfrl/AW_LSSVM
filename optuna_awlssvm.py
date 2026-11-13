"""
Optuna hyperparameter search for (Precomputed) Adaptive-Weight LSSVM.

Same protocol as ``da_awlssvm.py`` (seeds, train/test split, 3-fold CV,
search ranges, metric), but the search uses Optuna's TPE sampler instead of
dual annealing:

    * Raw multi-view features (default)   -> AW_LSSVM, tunes
      (gamma, gamma_rbf, ro per view, decay, steps)
    * Precomputed kernels (e.g. "Flower") -> PrecomputedAW_LSSVM, tunes
      (gamma, ro per view, decay, steps)

Example
-------
    python3 optuna_awlssvm.py --dataset 3Sources --coupling hard --method dissim --n_trials 200 --results_folder results
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

from da_awlssvm import (
    CV_SPLITS, DECAY_BOUNDS, GAMMA_BOUNDS, GAMMA_RBF_BOUNDS, METRIC, RO_BOUNDS,
    SEEDS, STEPS_BOUNDS, TEST_SIZE, COUPLING_CHOICES, METHOD_CHOICES,
    cv_score_kernel, cv_score_vfl, load_dataset, score_from_model,
    subset_kernels, split_views, subset_views,
)
from kernel_MVL import AW_LSSVM, PrecomputedAW_LSSVM

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# The ten benchmark datasets of the AW-LSSVM paper (Table 1), full versions
# (only YT is subsampled, to 500 samples, as described in the paper).
DATASET_CHOICES = [
    "3Sources", "YouTube3views_subsampled_500", "NUSWIDE", "Prokaryotic",
    "ProteinFold", "Caltech101-7", "Flower", "ACM", "Cora", "OutdoorScene",
]

FAILED_SCORE = 0.0  # CV score assigned when a configuration fails to train


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for AW-LSSVM."
    )
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--coupling", required=True, choices=COUPLING_CHOICES)
    parser.add_argument("--method", required=True, choices=METHOD_CHOICES)
    parser.add_argument("--n_trials", type=int, required=True,
                        help="Number of Optuna trials per seed (e.g. 200).")
    parser.add_argument("--results_folder", required=True,
                        help="Directory the output CSVs will be written to.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Seeds to run (default: all). Use one seed per job to parallelize.")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Optuna trials run in parallel (threads). >1 is faster but "
                             "not exactly reproducible.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #

def suggest_params(trial: optuna.Trial, n_views: int, *, precomputed: bool) -> tuple[dict, int]:
    """Sample one configuration. gamma (and gamma_rbf) are shared across views,
    ro is tuned per view, as in the dual-annealing search."""
    gamma = trial.suggest_float("gamma", *GAMMA_BOUNDS, log=True)
    params = {"gamma": np.ones(n_views) * gamma}
    if not precomputed:
        gamma_rbf = trial.suggest_float("gamma_rbf", *GAMMA_RBF_BOUNDS, log=True)
        params["gamma_rbf"] = np.ones(n_views) * gamma_rbf
    params["ro"] = np.array([trial.suggest_float(f"ro_{v}", *RO_BOUNDS, log=True)
                             for v in range(n_views)])
    params["decay"] = trial.suggest_float("decay", *DECAY_BOUNDS, log=True)
    steps = trial.suggest_int("steps", *STEPS_BOUNDS)
    return params, steps


def search(train, y_train, *, precomputed: bool, n_views: int, coupling: str,
           method: str, seed: int, metric_name: str, n_trials: int, n_jobs: int = 1):
    ctor = PrecomputedAW_LSSVM if precomputed else AW_LSSVM

    def objective(trial: optuna.Trial) -> float:
        params, steps = suggest_params(trial, n_views, precomputed=precomputed)
        params.update(coupling=coupling, method=method)
        try:
            if precomputed:
                return cv_score_kernel(ctor, params, train, y_train, cv=CV_SPLITS,
                                       steps=steps, inner_seed=seed,
                                       metric_name=metric_name, adaptive=True)
            return cv_score_vfl(ctor, params, train, y_train, cv=CV_SPLITS,
                                steps=steps, inner_seed=seed, metric_name=metric_name)
        except Exception:
            return FAILED_SCORE

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_params, best_steps = suggest_params(
        optuna.trial.FixedTrial(study.best_params), n_views, precomputed=precomputed
    )
    best_params.update(coupling=coupling, method=method)
    return best_params, best_steps, study


# --------------------------------------------------------------------------- #
# Experiment loop
# --------------------------------------------------------------------------- #

def run_experiment(data, y, *, precomputed: bool, n_views: int, coupling: str,
                   method: str, n_trials: int, seeds: list[int], n_jobs: int, results_path: Path,
                   metric_name: str) -> None:
    model_name = "PrecomputedAW_LSSVM" if precomputed else "AW_LSSVM"
    ctor = PrecomputedAW_LSSVM if precomputed else AW_LSSVM
    idx_all = np.arange(len(y))
    records = []

    for seed in seeds:
        idx_tr, idx_te = train_test_split(
            idx_all, test_size=TEST_SIZE, random_state=int(seed), stratify=y
        )
        y_train, y_test = y[idx_tr], y[idx_te]
        if precomputed:
            train, test = subset_kernels(data, idx_tr, idx_te)
            cv_train = train
        else:
            train, test = split_views(data, idx_tr, idx_te)
            cv_train = subset_views(data, idx_tr)   # raw rows: every CV fold is standardized with its own training rows

        params, steps, study = search(
            cv_train, y_train, precomputed=precomputed, n_views=n_views,
            coupling=coupling, method=method, seed=int(seed),
            metric_name=metric_name, n_trials=n_trials, n_jobs=n_jobs,
        )

        model = ctor(**params)
        model.fit(train, y_train, range_steps=steps)
        model.predict(test, y_test)
        test_score = score_from_model(model, metric_name)
        logger.info("Seed %s: %s | cv=%.4f | test=%.4f",
                    seed, study.best_params, study.best_value, test_score)

        save_trials(study, results_path, seed)
        records.append({
            "seed": seed,
            "model": model_name,
            **{f"opt_{k}": v for k, v in study.best_params.items()},
            "cv_opt": study.best_value,
            f"test_{metric_name}": test_score,
        })

    save_results(records, results_path, metric_name)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def save_trials(study: optuna.Study, results_path: Path, seed: int) -> None:
    """Save every trial of one seed's study (for convergence plots etc.)."""
    stem = results_path.stem.split("_seeds")[0]
    path = results_path.with_name(f"{stem}_trials_seed{seed}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(path, index=False)


def save_results(records: list[dict], results_path: Path, metric_name: str) -> None:
    df = pd.DataFrame.from_records(records).sort_values(["model", "seed"]).reset_index(drop=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    logger.info("\nSaved: %s", results_path)
    logger.info(df.to_string(index=False))

    test_col = f"test_{metric_name}"
    summary = df.groupby("model")[test_col].agg(["mean", "std"]).reset_index()
    logger.info("\nSummary (mean +/- std across seeds):")
    for _, row in summary.iterrows():
        logger.info("%-30s | %.4f +/- %.4f", row["model"], row["mean"], row["std"])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def seed_tag(seeds: list[int]) -> str:
    """'' for the default seed set, else e.g. '_seeds0' / '_seeds0-42'."""
    return "" if list(seeds) == list(SEEDS) else "_seeds" + "-".join(map(str, seeds))


def main() -> None:
    args = parse_args()
    csv_name = f"OPT_awlssvm_{args.dataset}_{args.coupling}_{args.method}_{args.n_trials}{seed_tag(args.seeds)}.csv"
    results_path = Path(args.results_folder) / csv_name

    logger.info("Optuna AW-LSSVM %s | coupling=%s | method=%s | n_trials=%s | results_folder=%s",
                args.dataset, args.coupling, args.method, args.n_trials, args.results_folder)

    precomputed, data, y, n_views = load_dataset(args.dataset)

    start = time.time()
    run_experiment(data, y, precomputed=precomputed, n_views=n_views,
                   coupling=args.coupling, method=args.method,
                   n_trials=args.n_trials, seeds=args.seeds, n_jobs=args.n_jobs, results_path=results_path,
                   metric_name=METRIC)
    logger.info("Runtime: %.2f seconds", time.time() - start)


if __name__ == "__main__":
    main()
