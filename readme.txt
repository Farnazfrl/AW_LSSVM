# AW-LSSVM Dual-Annealing Hyperparameter Search

Hyperparameter search (via `scipy.optimize.dual_annealing`) for the
Adaptive Weighted Least-Squares SVM (`AW_LSSVM` / `PrecomputedAW_LSSVM`) on
multi-view datasets.

For each dataset the script:

1. Loads either raw multi-view features or precomputed kernels from a
   `.mat` file in `datasets/`.
2. For each of 3 fixed seeds, splits the data into train/test, runs dual
   annealing over the model's hyperparameters using 3-fold CV on the
   training split, then evaluates the best configuration on the held-out
   test split.
3. Saves per-seed results and a mean/std summary to a CSV file.

## Requirements

```bash
pip install -r requirements.txt
```

This script also depends on a local `mv_lssvm_hard` module (not included
here) that implements `AW_LSSVM`, `PrecomputedAW_LSSVM`, and
`cross_val_acc`.

## Usage

```bash
python da_lssvm_search.py \
    --dataset 3Sources \
    --coupling hard \
    --method dissim \
    --maxiter 1 \
    --results_folder results_DA/results_1
```

### Arguments

| Flag | Description |
|---|---|
| `--dataset` | One of the supported dataset names (see `DATASET_CHOICES` in the script). |
| `--coupling` | `all` or `hard`. default for AW-LSSVM paper is `hard`|
| `--method` | `dissim` or `avg`. |
| `--maxiter` | Max iterations passed to `dual_annealing`. |
| `--results_folder` | Output directory for the results CSV. |

## Data layout

Datasets are expected at `datasets/<dataset_name>.mat`. 
The `Flower` dataset is treated as precomputed-kernel data (`data['K']`); all other
datasets are treated as raw multi-view feature data (`data['X']`).

## Output

A CSV is written to `<results_folder>/DA_<dataset>_<coupling>_<method>_<maxiter>.csv`
containing, per seed and model: the tuned hyperparameters, the CV score
found during search, and the test-set score.