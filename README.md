# TCSDG — Task-Conditioned Synthetic Data Generation for Tabular Agricultural Data

A framework for **tabular synthetic data generation (SDG)** and a reproducible
**benchmarking suite** that evaluates it on two use cases (crop type
classification and crop yield prediction) and compares it against six
established SDG algorithms.


## Repository structure

```
TCSDG/
├── README.md                  # this file
├── LICENSE                    # Apache license 2.0.
├── requirements.txt           # Python dependencies
├── .gitignore
├── sdg_functions.py           # core library: teacher, generators, evaluation
├── main.py                    # command-line entry point / experiment runner
├── demo.ipynb                 # quick-start Jupyter notebook
├── example_data/              # bundled public-domain example datasets
│   ├── README.md
│   ├── classification_grape_cultivar.csv
│   └── regression_diabetes.csv
└── results/                   # output CSVs are written here
```

All functions live in `sdg_functions.py`. `main.py` and `demo.ipynb` only
*import and call* them.

---

## Installation

Python **3.10 or newer** is recommended.

```bash
git clone https://github.com/HamidEbrahimy/TCSDG.git
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Notes on dependencies**

* `synthcity` and `tabicl` are **required** as they provide, respectively, the
  Bayesian-network generator (and the six baseline plugins) and the TabICL
  teacher model.
* On its **first run**, TabICL downloads pretrained model weights, so an
  internet connection is needed the first time.

---

## Quick start

### Option A — the demo notebook

```bash
jupyter notebook demo.ipynb
```

`demo.ipynb` loads the bundled example datasets and runs a deliberately small
TCSDG benchmark (a few seeds, low ratios).

### Option B — the command line, smoke test

Running the entry point with no arguments executes the same small smoke test on
the example data:

```bash
python main.py
```

Results are written to `results/`.

### Option C — the command line, full benchmark

```bash
python main.py \
    --task both \
    --data_classification example_data/classification_grape_cultivar.csv \
    --target_classification cultivar \
    --data_regression example_data/regression_diabetes.csv \
    --target_regression target \
    --output_dir results
```

With no `--seeds` / `--ratios` / `--train_fractions` / `--synthcity_plugins`
flags, the **full default grid** applies: 10 seeds, ratios 1× / 2× / 4× / 8×,
training fractions 0.70 and 0.30, and all six SynthCity baseline plugins.

---

## Using your own data

The pipeline accepts any flat CSV with one target column and any number of
feature columns. For a single task:

```bash
python main.py \
    --task classification \
    --data path/to/your_data.csv \
    --target your_target_column \
    --output_dir results
```

Use `--task regression` for a continuous target. Numeric and categorical
columns are detected automatically; override with
`--categorical_cols col_a col_b ...` if needed.

---

## The experiment grid

| Component             | Default                                   | CLI flag                |
|-----------------------|-------------------------------------------|-------------------------|
| Random seeds          | first 10 primes (2, 3, 5, …, 29)          | `--seeds`               |
| Multiplication ratios | 1, 2, 4, 8                                | `--ratios`              |
| Training fractions    | 0.70 (full) and 0.30 (reduced)            | `--train_fractions`     |
| Baseline SDG plugins  | BN, ctgan, adsgan, ddpm, rtvae, tvae | `--synthcity_plugins` / `--skip_synthcity` |
| Downstream learners   | RF, MLP, SVM                   | —                       |

For every (seed, training fraction) pair the dataset is split 70/30; when the
training fraction is below 0.70 the training portion is sub-sampled while the
hold-out set stays fixed.

---

## Outputs

Each run writes a tidy (long-format) CSV to `--output_dir`. One row per
evaluated combination, with the columns:

| Column       | Meaning                                                          |
|--------------|------------------------------------------------------------------|
| `task`       | `classification` or `regression`                                 |
| `seed`       | random seed of the repetition                                    |
| `ratio`      | multiplication ratio (1, 2, 4, 8)                             |
| `train_frac` | training fraction used                                           |
| `model`      | downstream learner (`RF`, `MLP`, `SVM`)                |
| `regime`     | training data used — e.g. `Baseline`, `TCSDG_Merged`, `TCSDG_Synthetic`, `ctgan_merged`, … |
| `metric`     | evaluation metric                                                |
| `value`      | metric value                                                     |

Classification metrics: balanced accuracy.
Regression metrics: RMSE.

`mean ± std` over all seeds.


---

## Citation
If you use this code, please cite the associated paper:

```bibtex
@misc{ebrahimy2026taskconditionedsyntheticdatageneration,
      title={Task-Conditioned Synthetic Data Generation for Improving Machine Learning Performance in Agricultural Prediction Tasks}, 
      author={Hamid Ebrahimy and Moritz Lucas and Martin Atzmueller},
      year={2026},
      eprint={2607.09751},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.09751},
}
```

---

## License

Released under the Apache license 2.0. — see the [`LICENSE`](LICENSE) file for details.

