# AW-LSSVM: Adaptive Weighted LS-SVM for Multi-View Classification

Code for the paper:

> Faramarzi Lighvan, F., Asadi, M., Houthuys, L. (2027). **Adaptive Weighted LS-SVM for Multi-view Classification.** In: Pasa, L., et al. *Artificial Neural Networks and Machine Learning – ICANN 2026*. Lecture Notes in Computer Science, vol 17089. Springer, Cham. https://doi.org/10.1007/978-3-032-38398-3_34

AW-LSSVM is an iterative multi-view classifier built on LS-SVM. At each iteration, every view assigns higher weights to the samples that the other views misclassify. This explicitly enforces complementary learning across all views, rather than only pairwise interactions between them.

## Repository contents

| File | Description |
|---|---|
| `kernel_MVL.py` | Model implementations: `AW_LSSVM` and `PrecomputedAW_LSSVM` (the proposed method), plus single-view LS-SVM, coupled multi-view LS-SVM (`MultiViewLSSVM`), late fusion and cross-validation helpers. |
| `da_awlssvm.py` | Hyperparameter search with dual annealing (`scipy.optimize.dual_annealing`) and train/test evaluation of AW-LSSVM. |
| `requirements.txt` | Python dependencies. |

## Installation

```bash
git clone https://github.com/Farnazfrl/AW_LSSVM.git
cd AW_LSSVM
pip install -r requirements.txt
```

## Data

The benchmark datasets are available at https://github.com/ChuanbinZhang/Multi-view-datasets.

Place each dataset as a `.mat` file in a `datasets/` folder at the repository root, i.e. `datasets/<dataset_name>.mat`:

- **Raw multi-view features:** `X` holds one feature matrix per view, and `y` holds the labels.
- **Precomputed kernels** (`Flower`): `K` holds one kernel matrix per view, and `y` holds the labels.

Features are standardized per view, and labels are label-encoded, when loaded.

## Usage

```bash
python3 da_awlssvm.py \
    --dataset 3Sources \
    --coupling hard \
    --method dissim \
    --maxiter 100 \
    --results_folder results
```

| Flag | Description |
|---|---|
| `--dataset` | Dataset name (see `DATASET_CHOICES` in `da_awlssvm.py`). |
| `--coupling` | `hard` (only misclassified samples are coupled; used in the paper) or `all`. |
| `--method` | Aggregation strategy: `dissim` (AW-LSSVM<sub>dissim</sub>, Eq. 3) or `avg` (AW-LSSVM<sub>avg</sub>, Eq. 2). |
| `--maxiter` | Maximum number of dual-annealing iterations. |
| `--results_folder` | Output directory for the results CSV. |

## Experimental protocol

For each of three seeds (`0`, `42`, `123`), the script:

1. Makes a stratified 80/20 train/test split.
2. Tunes the hyperparameters with dual annealing, using 3-fold stratified cross-validation on the training split. The hyperparameters are the regularization γ, the RBF bandwidth, the coupling strengths ρ<sup>(v)</sup>, the decay and the number of iterations T.
3. Retrains on the full training split with the best configuration and evaluates on the test split.

The evaluation metric is balanced accuracy.

## Output

The results are saved to `<results_folder>/DA_<dataset>_<coupling>_<method>_<maxiter>.csv`. For each seed, the file contains the tuned hyperparameters, the best cross-validation score and the test score. The mean ± standard deviation across seeds is also printed.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{faramarzi2026adaptive,
  title={Adaptive Weighted LS-SVM for Multi-view Classification},
  author={Faramarzi Lighvan, Farnaz and Asadi, Mehrdad and Houthuys, Lynn},
  booktitle={International Conference on Artificial Neural Networks},
  pages={408--419},
  year={2026},
  organization={Springer}
}
```
