# Example datasets

These two small, **publicly available** datasets are bundled so that the
benchmarking pipeline can be run end-to-end without any private data. They are
**stand-ins** for the proprietary crop datasets used in the paper, they are
intended only to verify that the code runs and to illustrate the expected CSV
format.

Both files were exported from datasets shipped with
[scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html);
no download is required to regenerate them.

## `classification_grape_cultivar.csv`

A multi-class **classification** task — a thematic analogue of the paper's crop
type classification.

| Property        | Value                                                        |
|-----------------|--------------------------------------------------------------|
| Source          | UCI Wine dataset (via `sklearn.datasets.load_wine`)          |
| Rows            | 178                                                          |
| Feature columns | 13 (all numeric chemical measurements of grape samples)      |
| Target column   | `cultivar` — 3 classes (`cultivar_1`, `cultivar_2`, `cultivar_3`) |
| Task            | Identify the grape cultivar from chemical analysis           |

`classification_grape_cultivar.csv` is derived from the UCI Wine dataset:
Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J
The UCI Wine dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). This repository redistributes a reformatted CSV version with renamed target labels.


## `regression_diabetes.csv`

A continuous-target **regression** task — a generic analogue of the paper's
crop yield prediction.

| Property        | Value                                                        |
|-----------------|--------------------------------------------------------------|
| Source          | Diabetes dataset (via `sklearn.datasets.load_diabetes`)      |
| Rows            | 442                                                          |
| Feature columns | 10 (all numeric)                                             |
| Target column   | `target` — continuous disease-progression measure           |
| Task            | Predict a continuous outcome from numeric predictors         |

`regression_diabetes.csv` is derived from the Diabetes dataset bundled with scikit-learn and loaded using sklearn.datasets.load_diabetes. Users should cite scikit-learn when using this example dataset.

These two datasets are included for reproducibility and demonstration purposes only.


## CSV format expected by the pipeline

Any tabular dataset works, provided it is a flat CSV with:

* one row per observation;
* one **target column** (its name is passed via `--target`,
  `--target_classification`, or `--target_regression`);
* any number of feature columns.

Numeric and categorical feature columns are detected automatically; categorical
columns can also be specified explicitly with `--categorical_cols`.
