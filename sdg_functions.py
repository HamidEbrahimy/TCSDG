
from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# ═════════════════════════════════════════════════════════════════════════════
# §1  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_SEEDS: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
DEFAULT_RATIOS: List[int] = [1, 2, 4, 8]
DEFAULT_TRAIN_FRACTIONS: List[float] = [0.70, 0.30]

INITIAL_TRAIN_SIZE: float = 0.70
INITIAL_TEST_SIZE: float = 0.30

SYNTHCITY_PLUGINS: List[str] = [
    "bayesian_network",
    "ctgan",
    "adsgan",
    "ddpm",
    "arfpy",
    "rtvae",
    "tvae",
]

RESULT_COLUMNS: List[str] = [
    "task", "seed", "ratio", "train_frac",
    "model", "regime", "metric", "value",
]


# ═════════════════════════════════════════════════════════════════════════════
# §2  REPRODUCIBILITY
# ═════════════════════════════════════════════════════════════════════════════
def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# §3  OPTIONAL IMPORTS
# ═════════════════════════════════════════════════════════════════════════════
def try_import_tabicl():
    try:
        from tabicl import TabICLClassifier, TabICLRegressor  # type: ignore
        return TabICLClassifier, TabICLRegressor
    except Exception:
        return None, None


    
def try_import_synthcity():
    try:
        from synthcity.plugins import Plugins
        return Plugins
    except Exception as e:
        import traceback
        warnings.warn(f"synthcity import failed: {e!r}\n{traceback.format_exc()}")
        return None

# ═════════════════════════════════════════════════════════════════════════════
# §4  TEACHER MODELS
# ═════════════════════════════════════════════════════════════════════════════
class TeacherBase:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TeacherBase":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_mean_variance(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class TabICLTeacherClassifier(TeacherBase):
    def __init__(self, seed: int = 42, device: str = "cpu"):
        Cls, _ = try_import_tabicl()
        if Cls is None:
            raise RuntimeError("TabICL not installed")
        self.model = Cls(
            n_estimators=8, allow_auto_download=True,
            device=device, random_state=seed, verbose=False,
        )

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_mean_variance(self, X):
        raise NotImplementedError


class TabICLTeacherRegressor(TeacherBase):
    def __init__(self, seed: int = 42, device: str = "cpu"):
        _, Reg = try_import_tabicl()
        if Reg is None:
            raise RuntimeError("TabICL not installed")
        self.model = Reg(
            n_estimators=8, allow_auto_download=True,
            device=device, random_state=seed, verbose=False,
        )

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict_proba(self, X):
        raise NotImplementedError

    def predict_mean_variance(self, X):
        out = self.model.predict(X, output_type=["mean", "variance"])
        return (np.asarray(out["mean"]).reshape(-1),
                np.asarray(out["variance"]).reshape(-1))


def get_teacher(task: str, seed: int) -> TeacherBase:
    if task == "classification":
        return TabICLTeacherClassifier(seed=seed, device="cpu")
    if task == "regression":
        return TabICLTeacherRegressor(seed=seed, device="cpu")
    raise ValueError(task)


# ═════════════════════════════════════════════════════════════════════════════
# §5  GENERATORS
# ═════════════════════════════════════════════════════════════════════════════
class GeneratorBase:
    def fit(self, df_train: pd.DataFrame) -> "GeneratorBase":
        raise NotImplementedError

    def sample(self, n: int) -> pd.DataFrame:
        raise NotImplementedError


class SynthCityBNGenerator(GeneratorBase):
    def __init__(self, seed: int = 42, plugin_kwargs: Optional[dict] = None):
        Plugins = try_import_synthcity()
        if Plugins is None:
            raise RuntimeError("SynthCity not installed")
        self.Plugins = Plugins
        self.seed = seed
        self.plugin_kwargs = plugin_kwargs or {}
        self.model = None

    def fit(self, df_train):
        self.model = self.Plugins().get(
            "bayesian_network", random_state=self.seed,
            **self.plugin_kwargs,
        )
        self.model.fit(df_train)
        return self

    def sample(self, n):
        if self.model is None:
            raise RuntimeError("Generator not fit")
        out = self.model.generate(count=n)
        if isinstance(out, pd.DataFrame):
            return out
        if hasattr(out, "dataframe"):
            return out.dataframe()
        if hasattr(out, "numpy"):
            return pd.DataFrame(out.numpy())
        return pd.DataFrame(out)


def build_feature_generator(seed: int, numeric_cols: List[str]) -> GeneratorBase:
    return SynthCityBNGenerator(seed=seed)


def build_joint_generator(seed: int, target_col: str,
                          numeric_cols: List[str]) -> GeneratorBase:
    return SynthCityBNGenerator(seed=seed)


# ═════════════════════════════════════════════════════════════════════════════
# §6  PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class PreprocessArtifacts:
    transformer: ColumnTransformer
    feature_names: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]


def infer_column_types(
    df: pd.DataFrame, target_col: str,
    categorical_cols_override: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c != target_col]
    if categorical_cols_override:
        cat = [c for c in categorical_cols_override if c in cols]
        num = [c for c in cols if c not in cat]
        return num, cat
    cat, num = [], []
    for c in cols:
        s = df[c]
        is_cat = isinstance(s.dtype, pd.CategoricalDtype)
        if (pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_object_dtype(s) or is_cat):
            cat.append(c)
        else:
            num.append(c)
    return num, cat


def compute_numeric_medians(
    df: pd.DataFrame, numeric_cols: List[str]
) -> Dict[str, float]:
    medians: Dict[str, float] = {}
    for c in numeric_cols:
        val = float(pd.to_numeric(df[c], errors="coerce").median())
        if not np.isfinite(val):
            val = 0.0
        medians[c] = val
    return medians


def impute_with_stats(
    df: pd.DataFrame, numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_medians: Dict[str, float],
    categorical_fill: str = "MISSING",
) -> pd.DataFrame:
    out = df.copy()
    for c in numeric_cols:
        if c not in out.columns:
            continue
        fill_val = numeric_medians.get(c, 0.0)
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(fill_val)
    for c in categorical_cols:
        if c not in out.columns:
            continue
        if out[c].isna().any():
            out[c] = out[c].fillna(categorical_fill)
    return out


def cast_categoricals(
    df: pd.DataFrame, categorical_cols: List[str]
) -> pd.DataFrame:
    out = df.copy()
    for c in categorical_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def fit_preprocessor(
    X_train: pd.DataFrame, numeric_cols: List[str],
    categorical_cols: List[str],
) -> PreprocessArtifacts:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", ohe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.fit(X_train)
    try:
        names = transformer.get_feature_names_out().tolist()
    except Exception:
        n_extra = (transformer.transform(X_train).shape[1]
                   - len(numeric_cols))
        names = numeric_cols + [f"cat_{i}" for i in range(n_extra)]
    return PreprocessArtifacts(
        transformer=transformer, feature_names=names,
        numeric_cols=numeric_cols, categorical_cols=categorical_cols,
    )


def transform_to_df(
    X: pd.DataFrame, art: PreprocessArtifacts
) -> pd.DataFrame:
    arr = art.transformer.transform(X)
    return pd.DataFrame(arr, columns=art.feature_names, index=X.index)


# ═════════════════════════════════════════════════════════════════════════════
# §7  UTILITY HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def compute_fill_values(
    X_train_raw: pd.DataFrame, numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Any]:
    fill: Dict[str, Any] = {}
    for c in X_train_raw.columns:
        if c in numeric_cols:
            fill[c] = float(
                pd.to_numeric(X_train_raw[c], errors="coerce").median()
            )
        else:
            mode = X_train_raw[c].mode(dropna=True)
            fill[c] = mode.iloc[0] if len(mode) > 0 else "MISSING"
    return fill


def harmonize_columns(
    df: pd.DataFrame, feature_cols: List[str],
    fill_values: Dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = fill_values.get(c, 0.0)
    return out[feature_cols].copy()


def sample_exact_clean(
    gen: GeneratorBase, n: int, seed: int, max_tries: int = 10
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks, remaining = [], n
    for _ in range(max_tries):
        if remaining <= 0:
            break
        df = gen.sample(remaining).dropna()
        if len(df) == 0:
            continue
        chunks.append(df)
        remaining = n - sum(len(c) for c in chunks)
    if not chunks:
        raise RuntimeError("Generator produced no usable samples")
    out = pd.concat(chunks, axis=0).reset_index(drop=True)
    if len(out) < n:
        idx = rng.integers(0, len(out), size=(n - len(out)))
        out = pd.concat([out, out.iloc[idx].copy()], axis=0).reset_index(
            drop=True
        )
    return out.iloc[:n].reset_index(drop=True)


def entropy_from_proba(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def margin_top1_top2(p: np.ndarray) -> np.ndarray:
    if p.shape[1] < 2:
        return np.ones(p.shape[0], dtype=float)
    part = np.partition(p, kth=-2, axis=1)
    return part[:, -1] - part[:, -2]


def diverse_select_indices(
    X: np.ndarray, k: int, seed: int
) -> np.ndarray:
    n = X.shape[0]
    if k >= n:
        return np.arange(n)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = MiniBatchKMeans(
        n_clusters=k, random_state=seed,
        batch_size=min(1024, n),
    )
    labels = km.fit_predict(Xs)
    centers = km.cluster_centers_
    selected = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        d2 = np.sum((Xs[idx] - centers[c]) ** 2, axis=1)
        selected.append(idx[int(np.argmin(d2))])
    if len(selected) < k:
        rng = np.random.default_rng(seed)
        remaining = np.setdiff1d(
            np.arange(n), np.array(selected, dtype=int)
        )
        if len(remaining) > 0:
            extra = rng.choice(
                remaining,
                size=min(len(remaining), k - len(selected)),
                replace=False,
            )
            selected.extend(extra.tolist())
    if len(selected) < k:
        rng = np.random.default_rng(seed + 1)
        selected.extend(
            rng.integers(0, n, size=(k - len(selected))).tolist()
        )
    return np.array(selected[:k], dtype=int)


def robust_qcut_codes(values: np.ndarray, q: int) -> np.ndarray:
    s = pd.Series(values)
    bins = pd.qcut(
        s, q=min(q, max(2, len(np.unique(values)))),
        duplicates="drop",
    )
    return bins.cat.codes.to_numpy()


# ═════════════════════════════════════════════════════════════════════════════
# §8  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class TcsdgConfig:
    ratio: int = 2
    candidate_multiplier: int = 6
    w_syn: float = 0.3
    alpha_weight: float = 1.0

    # classification
    class_prior_mix: float = 0.25
    filtered_fraction_clf: float = 0.60
    min_teacher_prob: float = 0.50
    agree_margin_min: float = 0.05
    ent_low_q: float = 0.20
    ent_high_q: float = 0.95
    score_margin_weight: float = 0.20
    score_entropy_weight: float = 0.15

    # regression
    n_bins_regression: int = 5
    filtered_fraction_reg: float = 0.60
    reg_resid_q: float = 0.80
    reg_var_low_q: float = 0.20
    reg_var_high_q: float = 0.80


# ═════════════════════════════════════════════════════════════════════════════
# §9  BUDGET ALLOCATION
# ═════════════════════════════════════════════════════════════════════════════
def allocate_mixed_class_budgets(
    y_train: np.ndarray, n_synth_total: int, prior_mix: float,
) -> Dict[int, int]:
    classes, counts = np.unique(y_train.astype(int), return_counts=True)
    emp = counts.astype(float) / counts.sum()
    inv = 1.0 / np.maximum(counts.astype(float), 1.0)
    inv = inv / inv.sum()
    w = (1.0 - prior_mix) * emp + prior_mix * inv
    alloc = np.maximum(1, np.round(w * n_synth_total).astype(int))
    while alloc.sum() > n_synth_total:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < n_synth_total:
        alloc[np.argmin(alloc)] += 1
    return {int(c): int(a) for c, a in zip(classes, alloc)}


def allocate_empirical_bin_budgets(
    bin_ids_train: np.ndarray, n_synth_total: int,
) -> Dict[int, int]:
    uniq, counts = np.unique(bin_ids_train, return_counts=True)
    w = counts.astype(float) / counts.sum()
    alloc = np.maximum(1, np.round(w * n_synth_total).astype(int))
    while alloc.sum() > n_synth_total:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < n_synth_total:
        alloc[np.argmin(alloc)] += 1
    return {int(b): int(a) for b, a in zip(uniq, alloc)}


# ═════════════════════════════════════════════════════════════════════════════
# §10  CLASSIFICATION GENERATION
# ═════════════════════════════════════════════════════════════════════════════
def fit_class_feature_generator(
    X_train_raw, y_train, class_id, cfg, seed, numeric_cols,
):
    mask = y_train.astype(int) == int(class_id)
    Xc = X_train_raw.loc[mask].copy().reset_index(drop=True)
    gen = build_feature_generator(
        seed=seed + int(class_id), numeric_cols=numeric_cols
    )
    gen.fit(Xc)
    return gen


def tcsdg_generate_classification(
    X_train_raw, y_train, X_train_enc, teacher, budgets, cfg, seed,
    feature_cols, fill_values, numeric_cols, preproc,
):
    classes = sorted(budgets.keys())
    proba_train = teacher.predict_proba(X_train_enc)
    ent_train = entropy_from_proba(proba_train)
    mar_train = margin_top1_top2(proba_train)

    class_real_stats: Dict[int, Dict[str, float]] = {}
    for c in classes:
        m = y_train.astype(int) == int(c)
        if np.sum(m) == 0:
            class_real_stats[c] = {
                "tau": cfg.min_teacher_prob,
                "ent_med": float(np.median(ent_train)),
                "ent_low": float(np.quantile(ent_train, cfg.ent_low_q)),
                "ent_high": float(np.quantile(ent_train, cfg.ent_high_q)),
            }
            continue
        p_true = proba_train[m, int(c)]
        ent_c = ent_train[m]
        class_real_stats[c] = {
            "tau": float(max(cfg.min_teacher_prob,
                             np.quantile(p_true, 0.20))),
            "ent_med": float(np.median(ent_c)),
            "ent_low": float(np.quantile(ent_c, cfg.ent_low_q)),
            "ent_high": float(np.quantile(ent_c, cfg.ent_high_q)),
            "mar_med": float(np.median(mar_train[m])),
        }

    X_parts, y_parts, w_parts = [], [], []

    for c in classes:
        budget = budgets[c]
        if budget <= 0:
            continue
        gen_c = fit_class_feature_generator(
            X_train_raw, y_train, c, cfg,
            seed=seed + 20000, numeric_cols=numeric_cols,
        )
        n_cand = max(50, int(cfg.candidate_multiplier * budget))
        X_cand_raw = sample_exact_clean(
            gen_c, n_cand, seed=seed + 20000 + int(c)
        )
        X_cand_raw = harmonize_columns(X_cand_raw, feature_cols, fill_values)
        X_cand_enc = transform_to_df(X_cand_raw, preproc)

        proba = teacher.predict_proba(X_cand_enc)
        ent = entropy_from_proba(proba)
        mar = margin_top1_top2(proba)
        p_c = proba[:, int(c)]

        stats = class_real_stats[c]
        ent_span = max(stats["ent_high"] - stats["ent_low"], 1e-8)
        ent_dev = np.abs(ent - stats["ent_med"]) / ent_span
        score = (p_c
                 + cfg.score_margin_weight * mar
                 - cfg.score_entropy_weight * ent_dev)

        keep = ((p_c >= stats["tau"])
                & (mar >= cfg.agree_margin_min)
                & (ent >= stats["ent_low"])
                & (ent <= stats["ent_high"]))
        idx_keep = np.where(keep)[0]
        filtered_target = min(
            int(round(cfg.filtered_fraction_clf * budget)), budget
        )

        selected_filtered: List[int] = []
        if len(idx_keep) > 0 and filtered_target > 0:
            idx_sorted = idx_keep[np.argsort(-score[idx_keep])]
            take = min(len(idx_sorted),
                       max(filtered_target * 3, filtered_target))
            pre_idx = idx_sorted[:take]
            sel_local = diverse_select_indices(
                X_cand_enc.iloc[pre_idx].to_numpy(dtype=float),
                min(filtered_target, len(pre_idx)),
                seed=seed + 30000 + int(c),
            )
            selected_filtered = pre_idx[sel_local].tolist()

        remaining = budget - len(selected_filtered)
        all_idx = np.arange(len(X_cand_enc))
        leftover_idx = np.setdiff1d(
            all_idx, np.array(selected_filtered, dtype=int)
        )

        if remaining > 0:
            if len(leftover_idx) == 0:
                X_more_raw = sample_exact_clean(
                    gen_c, remaining, seed=seed + 40000 + int(c)
                )
                X_more_raw = harmonize_columns(
                    X_more_raw, feature_cols, fill_values
                )
                X_more_enc = transform_to_df(X_more_raw, preproc)
                X_sel = pd.concat([
                    X_cand_enc.iloc[selected_filtered].reset_index(drop=True),
                    X_more_enc.reset_index(drop=True),
                ], axis=0)
                p_sel = np.concatenate([
                    p_c[selected_filtered],
                    np.full(remaining,
                            float(np.mean(p_c)) if len(p_c) else 0.5),
                ])
                score_sel = np.concatenate([
                    score[selected_filtered],
                    np.full(remaining,
                            float(np.mean(score)) if len(score) else 0.0),
                ])
            else:
                sel_naive_local = diverse_select_indices(
                    X_cand_enc.iloc[leftover_idx].to_numpy(dtype=float),
                    min(remaining, len(leftover_idx)),
                    seed=seed + 50000 + int(c),
                )
                selected_naive = leftover_idx[sel_naive_local].tolist()
                chosen = selected_filtered + selected_naive
                X_sel = X_cand_enc.iloc[chosen].reset_index(drop=True)
                p_sel = p_c[chosen]
                score_sel = score[chosen]

                if len(X_sel) < budget:
                    need = budget - len(X_sel)
                    X_more_raw = sample_exact_clean(
                        gen_c, need, seed=seed + 60000 + int(c)
                    )
                    X_more_raw = harmonize_columns(
                        X_more_raw, feature_cols, fill_values
                    )
                    X_more_enc = transform_to_df(X_more_raw, preproc)
                    X_sel = pd.concat(
                        [X_sel, X_more_enc], axis=0
                    ).reset_index(drop=True)
                    p_sel = np.concatenate([
                        p_sel,
                        np.full(need,
                                float(np.mean(p_c)) if len(p_c) else 0.5),
                    ])
                    score_sel = np.concatenate([
                        score_sel,
                        np.full(need,
                                float(np.mean(score)) if len(score) else 0.0),
                    ])
        else:
            chosen = selected_filtered
            X_sel = X_cand_enc.iloc[chosen].reset_index(drop=True)
            p_sel = p_c[chosen]
            score_sel = score[chosen]

        if len(score_sel) > 0:
            rng_s = np.max(score_sel) - np.min(score_sel) + 1e-12
            score_norm = (score_sel - np.min(score_sel)) / rng_s
        else:
            score_norm = np.array([], dtype=float)
        w_sel = cfg.w_syn * (1.0 + cfg.alpha_weight * score_norm)

        X_parts.append(X_sel)
        y_parts.append(np.full(len(X_sel), int(c), dtype=int))
        w_parts.append(w_sel.astype(float))

    X_syn = pd.concat(X_parts, axis=0).reset_index(drop=True)
    y_syn = np.concatenate(y_parts)
    w_syn = np.concatenate(w_parts)
    return X_syn, y_syn, w_syn


# ═════════════════════════════════════════════════════════════════════════════
# §11  REGRESSION GENERATION
# ═════════════════════════════════════════════════════════════════════════════
def tcsdg_generate_regression(
    X_train_raw, y_train, X_train_enc, teacher,
    n_synth_total, target_col, numeric_cols, feature_cols,
    fill_values, preproc, seed, cfg,
):
    gen = build_joint_generator(
        seed=seed, target_col=target_col, numeric_cols=numeric_cols
    )
    df_joint = X_train_raw.copy()
    df_joint[target_col] = y_train.astype(float)
    gen.fit(df_joint)
    n_cand = max(100, int(cfg.candidate_multiplier * n_synth_total))
    df_cand = sample_exact_clean(gen, n_cand, seed=seed + 2000)
    X_cand_raw = harmonize_columns(
        df_cand.drop(columns=[target_col], errors="ignore"),
        feature_cols, fill_values,
    )
    X_cand_enc = transform_to_df(X_cand_raw, preproc)
    y_cand = (pd.to_numeric(df_cand[target_col], errors="coerce")
              .astype(float).to_numpy())

    mean_train, var_train = teacher.predict_mean_variance(X_train_enc)
    mean_cand, var_cand = teacher.predict_mean_variance(X_cand_enc)
    resid_train = np.abs(mean_train - y_train.astype(float))
    resid_cand = np.abs(mean_cand - y_cand)

    delta = float(np.quantile(resid_train, cfg.reg_resid_q))
    var_low = float(np.quantile(var_train, cfg.reg_var_low_q))
    var_high = float(np.quantile(var_train, cfg.reg_var_high_q))

    keep = ((resid_cand <= delta)
            & (var_cand >= var_low) & (var_cand <= var_high))
    keep_idx = np.where(keep)[0]

    bin_ids_train = robust_qcut_codes(
        y_train.astype(float), cfg.n_bins_regression
    )
    budgets = allocate_empirical_bin_budgets(bin_ids_train, n_synth_total)

    y_keep = y_cand[keep_idx]
    X_keep = X_cand_enc.iloc[keep_idx].reset_index(drop=True)
    resid_keep = resid_cand[keep_idx]
    var_keep = var_cand[keep_idx]
    if len(keep_idx) > 0:
        bin_ids_keep = robust_qcut_codes(y_keep, cfg.n_bins_regression)
    else:
        bin_ids_keep = np.array([], dtype=int)
    bin_ids_cand = robust_qcut_codes(y_cand, cfg.n_bins_regression)

    X_parts, y_parts, w_parts = [], [], []

    for b, budget in budgets.items():
        if budget <= 0:
            continue
        filtered_target = min(
            int(round(cfg.filtered_fraction_reg * budget)), budget
        )
        idx_b = np.where(bin_ids_keep == int(b))[0]
        selected_filtered: List[int] = []
        if len(idx_b) > 0 and filtered_target > 0:
            var_mid = 0.5 * (var_low + var_high)
            var_dev = (np.abs(var_keep[idx_b] - var_mid)
                       / max(var_high - var_low, 1e-12))
            score_b = -resid_keep[idx_b] - 0.25 * var_dev
            idx_sorted_local = idx_b[np.argsort(-score_b)]
            take = min(len(idx_sorted_local),
                       max(filtered_target * 3, filtered_target))
            pre_idx = idx_sorted_local[:take]
            sl = diverse_select_indices(
                X_keep.iloc[pre_idx].to_numpy(dtype=float),
                min(filtered_target, len(pre_idx)),
                seed=seed + 3000 + int(b),
            )
            selected_filtered = pre_idx[sl].tolist()

        remaining = budget - len(selected_filtered)
        chosen_y: List[float] = []
        chosen_scores: List[float] = []
        X_sel_list = []

        if selected_filtered:
            X_sel_list.append(
                X_keep.iloc[selected_filtered].reset_index(drop=True)
            )
            chosen_y.extend(y_keep[selected_filtered].tolist())
            chosen_scores.extend(
                (-resid_keep[selected_filtered]).tolist()
            )

        if remaining > 0:
            idx_bin_all = np.where(bin_ids_cand == int(b))[0]
            if len(idx_bin_all) > 0:
                sl2 = diverse_select_indices(
                    X_cand_enc.iloc[idx_bin_all].to_numpy(dtype=float),
                    min(remaining, len(idx_bin_all)),
                    seed=seed + 4000 + int(b),
                )
                idx_naive = idx_bin_all[sl2]
                X_sel_list.append(
                    X_cand_enc.iloc[idx_naive].reset_index(drop=True)
                )
                chosen_y.extend(y_cand[idx_naive].tolist())
                chosen_scores.extend(
                    (-resid_cand[idx_naive]).tolist()
                )
                remaining = budget - len(chosen_y)

        if remaining > 0:
            df_more = sample_exact_clean(
                gen, remaining, seed=seed + 5000 + int(b)
            )
            X_more_raw = harmonize_columns(
                df_more.drop(columns=[target_col], errors="ignore"),
                feature_cols, fill_values,
            )
            X_more_enc = transform_to_df(X_more_raw, preproc)
            X_sel_list.append(X_more_enc.reset_index(drop=True))
            chosen_y.extend(
                pd.to_numeric(df_more[target_col], errors="coerce")
                .astype(float).tolist()
            )
            med_neg_resid = (float(np.median(-resid_cand))
                             if len(resid_cand) else 0.0)
            chosen_scores.extend([med_neg_resid] * remaining)

        X_sel = pd.concat(X_sel_list, axis=0).reset_index(drop=True)
        y_sel = np.asarray(chosen_y, dtype=float)
        score_sel = np.asarray(chosen_scores, dtype=float)
        if len(score_sel) > 0:
            rng_s = np.max(score_sel) - np.min(score_sel) + 1e-12
            score_norm = (score_sel - np.min(score_sel)) / rng_s
        else:
            score_norm = np.array([], dtype=float)
        w_sel = cfg.w_syn * (1.0 + cfg.alpha_weight * score_norm)

        X_parts.append(X_sel)
        y_parts.append(y_sel)
        w_parts.append(w_sel.astype(float))

    X_hybrid = pd.concat(X_parts, axis=0).reset_index(drop=True)
    y_hybrid = np.concatenate(y_parts)
    w_hybrid = np.concatenate(w_parts)
    return X_hybrid, y_hybrid, w_hybrid


# ═════════════════════════════════════════════════════════════════════════════
# §12  SYNTHCITY BENCHMARK HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def fit_synthcity_plugin(
    df_train: pd.DataFrame, plugin_name: str,
    target_col: str, seed: int,
) -> Any:
    """Fit one SynthCity plugin on the combined (X, y) training frame."""
    Plugins = try_import_synthcity()
    if Plugins is None:
        raise RuntimeError("synthcity is not installed")
    try:
        plugin = Plugins().get(plugin_name, random_state=seed)
    except (TypeError, Exception):
        plugin = Plugins().get(plugin_name)
    plugin.fit(df_train)
    return plugin


def generate_synthcity_data(
    fitted_plugin: Any,
    n_synth: int,
    target_col: str,
    task: str,
    all_classes: Optional[np.ndarray],
    feature_cols: List[str],
    fill_values: Dict[str, Any],
    numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_medians: Dict[str, float],
    preproc: PreprocessArtifacts,
) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """Generate *n_synth* rows from an already-fitted SynthCity plugin."""
    try:
        n_request = max(n_synth, int(n_synth * 1.2))
        raw = fitted_plugin.generate(count=n_request)
        if isinstance(raw, pd.DataFrame):
            df_syn = raw
        elif hasattr(raw, "dataframe"):
            df_syn = raw.dataframe()
        elif hasattr(raw, "numpy"):
            df_syn = pd.DataFrame(raw.numpy())
        else:
            df_syn = pd.DataFrame(np.asarray(raw))

        if target_col not in df_syn.columns:
            warnings.warn(
                f"SynthCity output missing target column '{target_col}'."
            )
            return None

        y_raw = df_syn[target_col].copy()
        X_raw = df_syn.drop(columns=[target_col]).copy()

        if task == "classification":
            y_num = pd.to_numeric(y_raw, errors="coerce")
            y_rounded = np.round(y_num).astype(float)
            valid = (np.isfinite(y_rounded)
                     & np.isin(y_rounded.astype(int), all_classes))
            if valid.sum() == 0:
                warnings.warn("No valid clf labels in SynthCity output.")
                return None
            y_syn = y_rounded[valid].astype(int).values
            X_raw = X_raw.loc[valid].reset_index(drop=True)
        else:
            y_num = pd.to_numeric(y_raw, errors="coerce").astype(float)
            valid = np.isfinite(y_num)
            if valid.sum() == 0:
                warnings.warn("No valid reg targets in SynthCity output.")
                return None
            y_syn = y_num[valid].values
            X_raw = X_raw.loc[valid].reset_index(drop=True)

        if len(y_syn) < max(10, n_synth // 4):
            warnings.warn(
                f"Too few valid SynthCity samples "
                f"({len(y_syn)}/{n_synth})."
            )
            return None

        # Truncate to budget
        if len(y_syn) > n_synth:
            X_raw = X_raw.iloc[:n_synth].reset_index(drop=True)
            y_syn = y_syn[:n_synth]

        X_raw = harmonize_columns(X_raw, feature_cols, fill_values)
        X_raw = impute_with_stats(
            X_raw, numeric_cols, categorical_cols, numeric_medians
        )
        X_raw = cast_categoricals(X_raw, categorical_cols)
        X_enc = transform_to_df(X_raw, preproc)
        return X_enc, y_syn

    except Exception as exc:
        warnings.warn(f"SynthCity generation error: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# §13  DOWNSTREAM MODELS & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
def build_models(task: str, seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {}

    from sklearn.ensemble import (RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR

    if task == "classification":
        models["RF"] = RandomForestClassifier(random_state=seed)
        models["MLP"] = MLPClassifier(random_state=seed, max_iter=300)
        models["SVM"] = SVC(probability=True, random_state=seed)
    else:
        models["RF"] = RandomForestRegressor(random_state=seed)
        models["MLP"] = MLPRegressor(random_state=seed, max_iter=300)
        models["SVM"] = SVR()
    return models


def model_needs_scaling(name: str) -> bool:
    return name in {"MLP", "SVM"}


def try_fit_with_weight(model, X, y, sample_weight):
    if sample_weight is None:
        model.fit(X, y)
        return
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)


def align_predicted_proba(model, proba, all_classes):
    if not hasattr(model, "classes_"):
        return proba
    mc = np.asarray(model.classes_)
    out = np.full((proba.shape[0], len(all_classes)), 1e-15, dtype=float)
    c2i = {int(c): i for i, c in enumerate(all_classes.astype(int))}
    for j, c in enumerate(mc.astype(int)):
        if int(c) in c2i:
            out[:, c2i[int(c)]] = proba[:, j]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-15, None)
    return out


def eval_classification(model, X_test, y_test, all_classes):
    y_pred = model.predict(X_test)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }
    return out


def eval_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


# ═════════════════════════════════════════════════════════════════════════════
# §14  CORE EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════
def _evaluate_regime(
    task, model_name, proto, regime_name,
    Xtr, ytr, sw, Xte, yte, all_classes,
    needs_scale, scaler,
    seed, ratio, train_frac,
):
    """Train one model on one regime, return list of result tuples."""
    m = copy.deepcopy(proto)
    Xtr_use = scaler.transform(Xtr) if needs_scale else Xtr
    Xte_use = scaler.transform(Xte) if needs_scale else Xte
    try:
        try_fit_with_weight(m, Xtr_use, ytr, sw)
        if task == "classification":
            metrics = eval_classification(m, Xte_use, yte, all_classes)
        else:
            metrics = eval_regression(m, Xte_use, yte)
        return [(task, seed, ratio, train_frac,
                 model_name, regime_name, k, float(v))
                for k, v in metrics.items()]
    except Exception as exc:
        warnings.warn(f"{model_name}/{regime_name} failed: {exc}")
        return []


def run_one_seed_trainfrac(
    task: str,
    df: pd.DataFrame,
    target_col: str,
    seed: int,
    ratios: List[int],
    train_frac: float,
    cfg: TcsdgConfig,
    synthcity_plugins: List[str],
    categorical_cols_override: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run every *ratio* for one (seed, train_frac) pair.

    The initial 70 / 30 split is done once; when ``train_frac < 0.70`` the
    training portion is further sub-sampled while the test set stays fixed.
    """
    set_global_seed(seed)
    print(f"\n{'─'*72}")
    print(f"  seed={seed}  train_frac={train_frac:.0%}  task={task}")
    print(f"{'─'*72}")

    # ── 1. column types ──────────────────────────────────────────────────
    num_cols, cat_cols = infer_column_types(
        df, target_col, categorical_cols_override
    )
    X_all = df.drop(columns=[target_col]).reset_index(drop=True)
    y_all_raw = df[target_col].reset_index(drop=True)

    # ── 2. initial 70/30 split ───────────────────────────────────────────
    strat = y_all_raw.values if task == "classification" else None
    X_train_full, X_test_raw, y_train_full_raw, y_test_raw = train_test_split(
        X_all, y_all_raw,
        train_size=INITIAL_TRAIN_SIZE,
        test_size=INITIAL_TEST_SIZE,
        random_state=seed, stratify=strat,
    )
    X_train_full = X_train_full.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)
    y_train_full_raw = y_train_full_raw.reset_index(drop=True)
    y_test_raw = y_test_raw.reset_index(drop=True)

    # ── 3. sub-sample training set when train_frac < initial ─────────────
    if train_frac < INITIAL_TRAIN_SIZE - 1e-9:
        sub_ratio = train_frac / INITIAL_TRAIN_SIZE
        strat_sub = (y_train_full_raw.values
                     if task == "classification" else None)
        X_train_raw, _, y_train_raw, _ = train_test_split(
            X_train_full, y_train_full_raw,
            train_size=sub_ratio,
            random_state=seed + 7919,
            stratify=strat_sub,
        )
        X_train_raw = X_train_raw.reset_index(drop=True)
        y_train_raw = y_train_raw.reset_index(drop=True)
    else:
        X_train_raw = X_train_full
        y_train_raw = y_train_full_raw

    # ── 4. label encoding (classification) ───────────────────────────────
    all_classes = None
    if task == "classification":
        le = LabelEncoder()
        y_train = le.fit_transform(
            y_train_raw.astype(str).values
        ).astype(int)
        known = y_test_raw.astype(str).isin(
            le.classes_.astype(str)
        ).values
        if not known.all():
            n_drop = int((~known).sum())
            warnings.warn(
                f"Dropping {n_drop} test rows with unseen classes "
                f"(seed={seed}, frac={train_frac:.0%})."
            )
        X_test_raw = X_test_raw.loc[known].reset_index(drop=True)
        y_test_raw = y_test_raw.loc[known].reset_index(drop=True)
        y_test = le.transform(
            y_test_raw.astype(str).values
        ).astype(int)
        all_classes = np.unique(y_train).astype(int)
    else:
        y_train = (pd.to_numeric(y_train_raw, errors="coerce")
                   .astype(float).values)
        y_test = (pd.to_numeric(y_test_raw, errors="coerce")
                  .astype(float).values)

    # ── 5. imputation (train stats only) ─────────────────────────────────
    numeric_medians = compute_numeric_medians(X_train_raw, num_cols)
    X_train_raw = impute_with_stats(
        X_train_raw, num_cols, cat_cols, numeric_medians
    )
    X_test_raw = impute_with_stats(
        X_test_raw, num_cols, cat_cols, numeric_medians
    )
    X_train_raw = cast_categoricals(X_train_raw, cat_cols)
    X_test_raw = cast_categoricals(X_test_raw, cat_cols)

    if task == "regression":
        tr_ok = ~pd.isna(y_train)
        te_ok = ~pd.isna(y_test)
        if not tr_ok.all() or not te_ok.all():
            warnings.warn("Dropping NaN-target rows (regression).")
        X_train_raw = X_train_raw.loc[tr_ok].reset_index(drop=True)
        y_train = y_train[tr_ok]
        X_test_raw = X_test_raw.loc[te_ok].reset_index(drop=True)
        y_test = y_test[te_ok]

    feature_cols = list(X_train_raw.columns)
    fill_values = compute_fill_values(X_train_raw, num_cols, cat_cols)

    # ── 6. preprocessing ─────────────────────────────────────────────────
    preproc = fit_preprocessor(X_train_raw, num_cols, cat_cols)
    X_train_enc_df = transform_to_df(X_train_raw, preproc)
    X_test_enc_df = transform_to_df(X_test_raw, preproc)

    # ── 7. teacher ───────────────────────────────────────────────────────
    teacher = get_teacher(task, seed)
    teacher.fit(X_train_enc_df, y_train)

    # ── 8. fit SynthCity benchmark plugins (once) ────────────────────────
    df_for_sc = X_train_raw.copy()
    df_for_sc[target_col] = y_train
    fitted_sc: Dict[str, Any] = {}
    for pname in synthcity_plugins:
        t0 = time.time()
        try:
            fitted_sc[pname] = fit_synthcity_plugin(
                df_for_sc, pname, target_col, seed
            )
            print(f"    ✓ SynthCity/{pname} fitted "
                  f"({time.time()-t0:.1f}s)")
        except Exception as exc:
            print(f"    ✗ SynthCity/{pname} fit failed: {exc}")

    # ── 9. common arrays ─────────────────────────────────────────────────
    n_train = len(X_train_raw)
    X_train_enc = X_train_enc_df.to_numpy(dtype=float)
    X_test_enc = X_test_enc_df.to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_train_enc)
    models = build_models(task, seed)

    all_rows: List[tuple] = []

    # ── 10. per-ratio loop ───────────────────────────────────────────────
    for ratio in ratios:
        n_synth = int(ratio * n_train)
        print(f"  ratio={ratio}  n_train={n_train}  "
              f"n_synth={n_synth}")

        # ---- TCSDG generation ----
        if task == "classification":
            budgets = allocate_mixed_class_budgets(
                y_train.astype(int), n_synth, cfg.class_prior_mix
            )
            X_hyb_df, y_hyb, w_hyb = tcsdg_generate_classification(
                X_train_raw, y_train.astype(int), X_train_enc_df,
                teacher, budgets, cfg, seed, feature_cols,
                fill_values, num_cols, preproc,
            )
        else:
            X_hyb_df, y_hyb, w_hyb = tcsdg_generate_regression(
                X_train_raw, y_train.astype(float), X_train_enc_df,
                teacher, n_synth, target_col, num_cols,
                feature_cols, fill_values, preproc, seed, cfg,
            )

        X_hyb = X_hyb_df.to_numpy(dtype=float)

        # ---- SynthCity benchmark data ----
        sc_data: Dict[str, Dict[str, np.ndarray]] = {}
        for pname, plugin in fitted_sc.items():
            res = generate_synthcity_data(
                plugin, n_synth, target_col, task, all_classes,
                feature_cols, fill_values, num_cols, cat_cols,
                numeric_medians, preproc,
            )
            if res is not None:
                Xe, ys = res
                sc_data[pname] = {
                    "X": Xe.to_numpy(dtype=float), "y": ys,
                }

        # ---- evaluate downstream models ----
        for mname, proto in models.items():
            ns = model_needs_scaling(mname)

            # Baseline
            all_rows.extend(_evaluate_regime(
                task, mname, proto, "Baseline",
                X_train_enc, y_train, None,
                X_test_enc, y_test, all_classes,
                ns, scaler, seed, ratio, train_frac,
            ))

            # TCSDG_merged
            Xa = np.vstack([X_train_enc, X_hyb])
            ya = np.concatenate([y_train, y_hyb])
            wa = np.concatenate([np.ones(n_train), w_hyb])
            all_rows.extend(_evaluate_regime(
                task, mname, proto, "TCSDG_Merged",
                Xa, ya, wa,
                X_test_enc, y_test, all_classes,
                ns, scaler, seed, ratio, train_frac,
            ))

            # TCSDG_synthetic
            all_rows.extend(_evaluate_regime(
                task, mname, proto, "TCSDG_Synthetic",
                X_hyb, y_hyb, None,
                X_test_enc, y_test, all_classes,
                ns, scaler, seed, ratio, train_frac,
            ))

            # ---- SynthCity regimes ----
            for pname, sd in sc_data.items():
                Xs, ys = sd["X"], sd["y"]

                # {plugin}_aug
                Xa = np.vstack([X_train_enc, Xs])
                ya = np.concatenate([y_train, ys])
                wa = np.concatenate([
                    np.ones(n_train), np.full(len(ys), cfg.w_syn)
                ])
                all_rows.extend(_evaluate_regime(
                    task, mname, proto, f"{pname}_aug",
                    Xa, ya, wa,
                    X_test_enc, y_test, all_classes,
                    ns, scaler, seed, ratio, train_frac,
                ))

                # {plugin}_syn_only
                all_rows.extend(_evaluate_regime(
                    task, mname, proto, f"{pname}_syn_only",
                    Xs, ys, None,
                    X_test_enc, y_test, all_classes,
                    ns, scaler, seed, ratio, train_frac,
                ))

    return pd.DataFrame(all_rows, columns=RESULT_COLUMNS)


# backward-compatible thin wrapper
def run_one_seed_ratio(
    task, df, target_col, seed, ratio, cfg,
    categorical_cols_override=None,
):
    """Legacy wrapper – single (seed, ratio) at the default 70 % split."""
    return run_one_seed_trainfrac(
        task=task, df=df, target_col=target_col,
        seed=seed, ratios=[ratio],
        train_frac=INITIAL_TRAIN_SIZE, cfg=cfg,
        synthcity_plugins=[],
        categorical_cols_override=categorical_cols_override,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §15  TASK RUNNER
# ═════════════════════════════════════════════════════════════════════════════
def run_task(
    task: str,
    df: pd.DataFrame,
    target_col: str,
    cfg: TcsdgConfig,
    seeds: List[int],
    ratios: List[int],
    train_fractions: List[float],
    synthcity_plugins: List[str],
    categorical_cols_override: Optional[List[str]] = None,
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        for tf in train_fractions:
            frames.append(
                run_one_seed_trainfrac(
                    task=task, df=df, target_col=target_col,
                    seed=seed, ratios=ratios, train_frac=tf,
                    cfg=cfg, synthcity_plugins=synthcity_plugins,
                    categorical_cols_override=categorical_cols_override,
                )
            )
    return pd.concat(frames, axis=0).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# §16  SUMMARY & REPORTING
# ═════════════════════════════════════════════════════════════════════════════
def summarize(df: pd.DataFrame, seeds: List[int]) -> None:
    grp = ["task", "train_frac", "ratio", "model", "regime", "metric"]
    summary = (df.groupby(grp)["value"]
               .agg(["mean", "std"])
               .reset_index())

    for task_val in sorted(summary["task"].unique()):
        for tf_val in sorted(summary["train_frac"].unique()):
            header = (f"  {task_val.upper()} | "
                      f"train = {tf_val:.0%} of data | "
                      f"mean ± std over {len(seeds)} seeds")
            print(f"\n{'═'*120}")
            print(header)
            print(f"{'═'*120}")
            sub = summary[
                (summary["task"] == task_val)
                & (summary["train_frac"] == tf_val)
            ]
            for met in sorted(sub["metric"].unique()):
                sm = sub[sub["metric"] == met].copy()
                sm["mean_std"] = (
                    sm["mean"].map(lambda x: f"{x:.4f}")
                    + " ± "
                    + sm["std"].fillna(0).map(lambda x: f"{x:.4f}")
                )
                piv = sm.pivot_table(
                    index=["ratio", "model"],
                    columns="regime",
                    values="mean_std",
                    aggfunc="first",
                )
                print(f"\n  Metric: {met}")
                with pd.option_context(
                    "display.max_columns", None,
                    "display.width", 220,
                ):
                    print(piv.sort_index())