
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd

from sdg_functions import (
    DEFAULT_RATIOS,
    DEFAULT_SEEDS,
    DEFAULT_TRAIN_FRACTIONS,
    SYNTHCITY_PLUGINS,
    TcsdgConfig,
    run_task,
    summarize,
)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook detection
# ─────────────────────────────────────────────────────────────────────────────
def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TCSDG — benchmarking suite",
    )

    # ── data / task ──────────────────────────────────────────────────────
    p.add_argument("--task",
                   choices=["classification", "regression", "both"],
                   default="classification")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--data_classification", type=str, default=None)
    p.add_argument("--target_classification", type=str, default=None)
    p.add_argument("--data_regression", type=str, default=None)
    p.add_argument("--target_regression", type=str, default=None)
    p.add_argument("--sep", type=str, default=",")
    p.add_argument("--categorical_cols", nargs="*", default=None)

    # ── experiment grid ──────────────────────────────────────────────────
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="Random seeds (default: first 10 primes)")
    p.add_argument("--ratios", nargs="*", type=int, default=None,
                   help="Synthetic / real ratios (default: 1 2 4 8)")
    p.add_argument("--train_fractions", nargs="*", type=float,
                   default=None,
                   help="Fractions of original data used for training "
                        "(default: 0.70 0.30)")

    # ── SynthCity benchmarks ─────────────────────────────────────────────
    p.add_argument("--synthcity_plugins", nargs="*", type=str,
                   default=None,
                   help="SynthCity plugin names to benchmark "
                        "(default: all seven)")
    p.add_argument("--skip_synthcity", action="store_true",
                   help="Disable all SynthCity benchmarks")

    # ── output ───────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default=".")

    # ── TCSDG hyper-parameters ──────────────────────────────────────────
    p.add_argument("--candidate_multiplier", type=int, default=6)
    p.add_argument("--w_syn", type=float, default=0.3)
    p.add_argument("--alpha_weight", type=float, default=1.0)

    p.add_argument("--class_prior_mix", type=float, default=0.25)
    p.add_argument("--filtered_fraction_clf", type=float, default=0.60)
    p.add_argument("--min_teacher_prob", type=float, default=0.50)
    p.add_argument("--agree_margin_min", type=float, default=0.05)
    p.add_argument("--ent_low_q", type=float, default=0.20)
    p.add_argument("--ent_high_q", type=float, default=0.95)
    p.add_argument("--score_margin_weight", type=float, default=0.20)
    p.add_argument("--score_entropy_weight", type=float, default=0.15)

    p.add_argument("--n_bins_regression", type=int, default=5)
    p.add_argument("--filtered_fraction_reg", type=float, default=0.60)
    p.add_argument("--reg_resid_q", type=float, default=0.80)
    p.add_argument("--reg_var_low_q", type=float, default=0.20)
    p.add_argument("--reg_var_high_q", type=float, default=0.80)

    if argv is None and _running_in_notebook():
        argv = []
    args, _ = p.parse_known_args(args=argv)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.task in {"classification", "regression"}:
        if not args.data or not args.target:
            raise ValueError(
                "For classification / regression supply --data and --target."
            )
    if args.task == "both":
        need = []
        if not args.data_classification:
            need.append("--data_classification")
        if not args.target_classification:
            need.append("--target_classification")
        if not args.data_regression:
            need.append("--data_regression")
        if not args.target_regression:
            need.append("--target_regression")
        if need:
            raise ValueError(
                "For --task both, still need: " + ", ".join(need)
            )


def load_csv(path: str, sep: str = ",") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    # ── resolve experiment grid ──────────────────────────────────────────
    seeds = (args.seeds if args.seeds and len(args.seeds) > 0
             else DEFAULT_SEEDS)
    ratios = (args.ratios if args.ratios and len(args.ratios) > 0
              else DEFAULT_RATIOS)
    train_fracs = (
        args.train_fractions
        if args.train_fractions and len(args.train_fractions) > 0
        else DEFAULT_TRAIN_FRACTIONS
    )

    if args.skip_synthcity:
        sc_plugins: List[str] = []
    elif args.synthcity_plugins and len(args.synthcity_plugins) > 0:
        sc_plugins = args.synthcity_plugins
    else:
        sc_plugins = list(SYNTHCITY_PLUGINS)

    cat_override = (
        args.categorical_cols
        if args.categorical_cols and len(args.categorical_cols) > 0
        else None
    )

    # ── build config ─────────────────────────────────────────────────────
    cfg = TcsdgConfig(
        candidate_multiplier=int(args.candidate_multiplier),
        w_syn=float(args.w_syn),
        alpha_weight=float(args.alpha_weight),
        class_prior_mix=float(args.class_prior_mix),
        filtered_fraction_clf=float(args.filtered_fraction_clf),
        min_teacher_prob=float(args.min_teacher_prob),
        agree_margin_min=float(args.agree_margin_min),
        ent_low_q=float(args.ent_low_q),
        ent_high_q=float(args.ent_high_q),
        score_margin_weight=float(args.score_margin_weight),
        score_entropy_weight=float(args.score_entropy_weight),
        n_bins_regression=int(args.n_bins_regression),
        filtered_fraction_reg=float(args.filtered_fraction_reg),
        reg_resid_q=float(args.reg_resid_q),
        reg_var_low_q=float(args.reg_var_low_q),
        reg_var_high_q=float(args.reg_var_high_q),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── single-task mode ─────────────────────────────────────────────────
    if args.task in {"classification", "regression"}:
        df = load_csv(args.data, sep=args.sep)
        if args.target not in df.columns:
            raise ValueError(
                f"Target column '{args.target}' not found in data."
            )
        res = run_task(
            args.task, df, args.target, cfg,
            seeds, ratios, train_fracs, sc_plugins, cat_override,
        )
        summarize(res, seeds)
        out_csv = os.path.join(
            args.output_dir,
            f"results_TCSDG_{args.task}.csv",
        )
        res.to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}")
        return

    # ── both-task mode ───────────────────────────────────────────────────
    df_c = load_csv(args.data_classification, sep=args.sep)
    df_r = load_csv(args.data_regression, sep=args.sep)
    if args.target_classification not in df_c.columns:
        raise ValueError(
            f"Classification target '{args.target_classification}' "
            f"not found."
        )
    if args.target_regression not in df_r.columns:
        raise ValueError(
            f"Regression target '{args.target_regression}' not found."
        )

    res_c = run_task(
        "classification", df_c, args.target_classification,
        cfg, seeds, ratios, train_fracs, sc_plugins, cat_override,
    )
    res_r = run_task(
        "regression", df_r, args.target_regression,
        cfg, seeds, ratios, train_fracs, sc_plugins, cat_override,
    )
    res = pd.concat([res_c, res_r], axis=0).reset_index(drop=True)
    summarize(res, seeds)

    out_csv = os.path.join(args.output_dir, "Bench_TCSDG.csv")
    res.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Default invocation: a small *smoke test* on the two bundled
    # public-domain example datasets, so that `python main.py` runs
    # out of the box with no extra arguments.
    #
    # It uses a reduced experiment grid (2 seeds, ratios 1x/2x, a single
    # training fraction, SynthCity baselines skipped) so it finishes
    # quickly while still exercising the full TCSDG pipeline.
    #
    # To reproduce the full benchmark reported in the paper, run from the
    # command line and override the grid, e.g.:
    #
    #   python main.py --task both \
    #       --data_classification example_data/classification_grape_cultivar.csv \
    #       --target_classification cultivar \
    #       --data_regression example_data/regression_diabetes.csv \
    #       --target_regression target \
    #       --output_dir results
    #
    # With no --seeds / --ratios / --train_fractions / --synthcity_plugins
    # flags, the full defaults apply: 10 seeds, ratios 1 2 4 8, training
    # fractions 0.70 and 0.30, and all seven SynthCity baseline plugins.
    # ------------------------------------------------------------------
    main([
        "--task", "both",
        "--data_classification",
        "example_data/classification_grape_cultivar.csv",
        "--target_classification", "cultivar",
        "--data_regression", "example_data/regression_diabetes.csv",
        "--target_regression", "target",
        "--sep", ",",
        "--output_dir", "results",
        "--seeds", "2", "3",
        "--ratios", "1", "2",
        "--train_fractions", "0.70",
        "--skip_synthcity",
    ])