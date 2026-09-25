#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
permutation_test_srim_residuals.py

Paired permutation/sign-flip test for two SRIM-comparison residual CSVs.

This is meant to be run after compare_two_datasets_same_srim_fit_exclude_events.py,
which writes files like:

  <prefix>_dataset1_events.csv
  <prefix>_dataset2_events.csv

Those files contain:
  event_id
  distance_data_mm
  Ecm_data_MeV
  distance_same_SRIM_fit_mm
  distance_residual_mm
  used_in_metric

The test uses common event_id values and compares residual quality between
Dataset 1 and Dataset 2. By default it keeps only rows where used_in_metric is
true in both datasets, so the test is paired and uses the same event list.

Positive differences mean Dataset 1 is better than Dataset 2 when using the
default labels:
  diff = metric(Dataset 2) - metric(Dataset 1)

Examples:

  python3 permutation_test_srim_residuals.py \
    --prefix srim_pressure_combined_ml_vs_hough \
    --outdir srim_pressure_combined_ml_vs_hough_permutation_test \
    --n-permutations 100000 \
    --seed 12345

Python 3.6 compatible.
"""
from __future__ import print_function

import argparse
import os
import sys
import math

import numpy as np
import pandas as pd


def str_to_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def parse_args():
    p = argparse.ArgumentParser(
        description="Paired permutation test for two SRIM residual event CSVs."
    )

    p.add_argument("--prefix", default=None,
                   help=("Output prefix used by compare_two_datasets_same_srim_fit_exclude_events.py. "
                         "The script will read <prefix>_dataset1_events.csv and <prefix>_dataset2_events.csv."))

    p.add_argument("--data1-events", default=None,
                   help="Dataset 1 event residual CSV. Overrides --prefix for dataset 1.")
    p.add_argument("--data2-events", default=None,
                   help="Dataset 2 event residual CSV. Overrides --prefix for dataset 2.")

    p.add_argument("--label1", default="Dataset 1",
                   help="Label for dataset 1. Default: Dataset 1")
    p.add_argument("--label2", default="Dataset 2",
                   help="Label for dataset 2. Default: Dataset 2")

    p.add_argument("--outdir", required=True,
                   help="Output directory.")

    p.add_argument("--event-col", default="event_id")
    p.add_argument("--residual-col", default="distance_residual_mm")
    p.add_argument("--used-col", default="used_in_metric")

    p.add_argument("--use-all-finite", action="store_true",
                   help=("Use all common finite residuals, ignoring used_in_metric. "
                         "Default requires used_in_metric true in both files."))

    p.add_argument("--n-permutations", type=int, default=100000)
    p.add_argument("--seed", type=int, default=12345)

    p.add_argument("--alternative", choices=["two-sided", "greater", "less"],
                   default="greater",
                   help=("Alternative for diff = metric(label2) - metric(label1). "
                         "'greater' tests whether label1 is better than label2. Default: greater."))

    p.add_argument("--make-plots", action="store_true",
                   help="Write null-distribution histograms if matplotlib is available.")

    return p.parse_args()


def resolve_paths(a):
    data1 = a.data1_events
    data2 = a.data2_events

    if a.prefix:
        if data1 is None:
            data1 = a.prefix + "_dataset1_events.csv"
        if data2 is None:
            data2 = a.prefix + "_dataset2_events.csv"

    if data1 is None or data2 is None:
        sys.exit("[ERROR] Provide --prefix or both --data1-events and --data2-events")

    if not os.path.exists(data1):
        sys.exit("[ERROR] Missing dataset 1 event CSV: {0}".format(data1))
    if not os.path.exists(data2):
        sys.exit("[ERROR] Missing dataset 2 event CSV: {0}".format(data2))

    return data1, data2


def load_paired(a, path1, path2):
    d1 = pd.read_csv(path1)
    d2 = pd.read_csv(path2)

    for name, d, path in [("dataset1", d1, path1), ("dataset2", d2, path2)]:
        for col in [a.event_col, a.residual_col]:
            if col not in d.columns:
                sys.exit("[ERROR] Missing column {0} in {1}".format(col, path))
        if (not a.use_all_finite) and a.used_col not in d.columns:
            sys.exit("[ERROR] Missing column {0} in {1}. Use --use-all-finite to ignore it.".format(a.used_col, path))

    d1 = d1[[a.event_col, a.residual_col] + ([] if a.use_all_finite else [a.used_col])].copy()
    d2 = d2[[a.event_col, a.residual_col] + ([] if a.use_all_finite else [a.used_col])].copy()

    d1 = d1.rename(columns={
        a.event_col: "event_id",
        a.residual_col: "residual1",
        a.used_col: "used1",
    })
    d2 = d2.rename(columns={
        a.event_col: "event_id",
        a.residual_col: "residual2",
        a.used_col: "used2",
    })

    d1["event_id"] = pd.to_numeric(d1["event_id"], errors="coerce")
    d2["event_id"] = pd.to_numeric(d2["event_id"], errors="coerce")
    d1["residual1"] = pd.to_numeric(d1["residual1"], errors="coerce")
    d2["residual2"] = pd.to_numeric(d2["residual2"], errors="coerce")

    d1 = d1.dropna(subset=["event_id"]).copy()
    d2 = d2.dropna(subset=["event_id"]).copy()
    d1["event_id"] = d1["event_id"].astype(int)
    d2["event_id"] = d2["event_id"].astype(int)

    # In case there are duplicate event ids, keep the first row. Usually there should not be duplicates.
    d1 = d1.drop_duplicates(subset=["event_id"], keep="first")
    d2 = d2.drop_duplicates(subset=["event_id"], keep="first")

    m = pd.merge(d1, d2, on="event_id", how="inner")

    finite = np.isfinite(m["residual1"].values) & np.isfinite(m["residual2"].values)

    if a.use_all_finite:
        used = finite
    else:
        used1 = str_to_bool_series(m["used1"])
        used2 = str_to_bool_series(m["used2"])
        used = finite & used1.values & used2.values

    paired = m[used].copy()
    return d1, d2, m, paired


def rmse(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x * x)))


def mae(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.abs(x)))


def median_abs(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    return float(np.median(np.abs(x)))


def score_pair(r1, r2):
    return {
        "rmse1": rmse(r1),
        "rmse2": rmse(r2),
        "rmse_diff_2_minus_1": rmse(r2) - rmse(r1),
        "mae1": mae(r1),
        "mae2": mae(r2),
        "mae_diff_2_minus_1": mae(r2) - mae(r1),
        "median_abs1": median_abs(r1),
        "median_abs2": median_abs(r2),
        "median_abs_diff_2_minus_1": median_abs(r2) - median_abs(r1),
        "mean_abs_pair_diff_2_minus_1": float(np.mean(np.abs(r2) - np.abs(r1))) if len(r1) else np.nan,
        "mean_sq_pair_diff_2_minus_1": float(np.mean(r2 * r2 - r1 * r1)) if len(r1) else np.nan,
        "frac_events_abs1_lt_abs2": float(np.mean(np.abs(r1) < np.abs(r2))) if len(r1) else np.nan,
        "frac_events_abs1_eq_abs2": float(np.mean(np.abs(r1) == np.abs(r2))) if len(r1) else np.nan,
    }


def p_value(null, observed, alternative):
    null = np.asarray(null, dtype=float)
    if alternative == "greater":
        # H1: observed is unusually positive, i.e. dataset 1 has smaller metric than dataset 2.
        return float((np.sum(null >= observed) + 1.0) / (len(null) + 1.0))
    if alternative == "less":
        return float((np.sum(null <= observed) + 1.0) / (len(null) + 1.0))
    return float((np.sum(np.abs(null) >= abs(observed)) + 1.0) / (len(null) + 1.0))


def permutation_null(r1, r2, nperm, seed):
    rng = np.random.RandomState(int(seed))
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    n = len(r1)

    null_rmse = np.empty(int(nperm), dtype=float)
    null_mae = np.empty(int(nperm), dtype=float)
    null_mean_abs_pair = np.empty(int(nperm), dtype=float)
    null_mean_sq_pair = np.empty(int(nperm), dtype=float)

    for i in range(int(nperm)):
        swap = rng.rand(n) < 0.5
        a = r1.copy()
        b = r2.copy()
        tmp = a[swap].copy()
        a[swap] = b[swap]
        b[swap] = tmp

        null_rmse[i] = rmse(b) - rmse(a)
        null_mae[i] = mae(b) - mae(a)
        null_mean_abs_pair[i] = float(np.mean(np.abs(b) - np.abs(a)))
        null_mean_sq_pair[i] = float(np.mean(b * b - a * a))

    return {
        "rmse_diff_2_minus_1": null_rmse,
        "mae_diff_2_minus_1": null_mae,
        "mean_abs_pair_diff_2_minus_1": null_mean_abs_pair,
        "mean_sq_pair_diff_2_minus_1": null_mean_sq_pair,
    }


def write_hist(values, observed, title, xlabel, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(values, bins=80)
    ax.axvline(observed, linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Permutations")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    a = parse_args()
    path1, path2 = resolve_paths(a)

    if not os.path.isdir(a.outdir):
        os.makedirs(a.outdir)

    d1, d2, common, paired = load_paired(a, path1, path2)

    if len(paired) < 2:
        sys.exit("[ERROR] Fewer than two paired events after filtering; cannot run permutation test.")

    r1 = paired["residual1"].values.astype(float)
    r2 = paired["residual2"].values.astype(float)

    obs = score_pair(r1, r2)
    nulls = permutation_null(r1, r2, int(a.n_permutations), int(a.seed))

    rows = []
    for key in ["rmse_diff_2_minus_1", "mae_diff_2_minus_1",
                "mean_abs_pair_diff_2_minus_1", "mean_sq_pair_diff_2_minus_1"]:
        vals = nulls[key]
        rows.append({
            "statistic": key,
            "observed": float(obs[key]),
            "null_mean": float(np.mean(vals)),
            "null_std": float(np.std(vals, ddof=1)),
            "p_value": p_value(vals, obs[key], a.alternative),
            "alternative": a.alternative,
            "n_permutations": int(a.n_permutations),
        })

    summary = {
        "label1": a.label1,
        "label2": a.label2,
        "path1": path1,
        "path2": path2,
        "n_dataset1_events": int(len(d1)),
        "n_dataset2_events": int(len(d2)),
        "n_common_events_before_metric_filter": int(len(common)),
        "n_paired_events_used": int(len(paired)),
        "use_all_finite": bool(a.use_all_finite),
        "alternative": a.alternative,
    }
    summary.update(obs)

    summary_csv = os.path.join(a.outdir, "permutation_test_summary.csv")
    pvals_csv = os.path.join(a.outdir, "permutation_test_pvalues.csv")
    paired_csv = os.path.join(a.outdir, "paired_residuals_used.csv")
    null_csv = os.path.join(a.outdir, "permutation_null_samples.csv")
    txt_path = os.path.join(a.outdir, "permutation_test_results.txt")

    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    pd.DataFrame(rows).to_csv(pvals_csv, index=False)
    paired.to_csv(paired_csv, index=False)

    # Save all nulls. This can be a large file but is still manageable for 100k permutations.
    pd.DataFrame(nulls).to_csv(null_csv, index=False)

    with open(txt_path, "w") as f:
        f.write("Paired SRIM residual permutation test\n")
        f.write("====================================\n\n")
        f.write("Dataset 1: {0}\n".format(a.label1))
        f.write("Dataset 2: {0}\n".format(a.label2))
        f.write("Path 1: {0}\n".format(path1))
        f.write("Path 2: {0}\n\n".format(path2))
        f.write("Events in dataset 1 file: {0}\n".format(len(d1)))
        f.write("Events in dataset 2 file: {0}\n".format(len(d2)))
        f.write("Common events before metric filtering: {0}\n".format(len(common)))
        f.write("Paired events used: {0}\n".format(len(paired)))
        f.write("Filtering: {0}\n\n".format("all common finite residuals" if a.use_all_finite else "used_in_metric true in both datasets"))
        f.write("Positive difference means Dataset 1 is better:\n")
        f.write("  diff = metric(Dataset 2) - metric(Dataset 1)\n\n")

        f.write("Observed metrics\n")
        f.write("----------------\n")
        f.write("{0} RMSE: {1:.9g} mm\n".format(a.label1, obs["rmse1"]))
        f.write("{0} RMSE: {1:.9g} mm\n".format(a.label2, obs["rmse2"]))
        f.write("RMSE diff 2-1: {0:.9g} mm\n\n".format(obs["rmse_diff_2_minus_1"]))

        f.write("{0} MAE: {1:.9g} mm\n".format(a.label1, obs["mae1"]))
        f.write("{0} MAE: {1:.9g} mm\n".format(a.label2, obs["mae2"]))
        f.write("MAE diff 2-1: {0:.9g} mm\n\n".format(obs["mae_diff_2_minus_1"]))

        f.write("{0} median |residual|: {1:.9g} mm\n".format(a.label1, obs["median_abs1"]))
        f.write("{0} median |residual|: {1:.9g} mm\n".format(a.label2, obs["median_abs2"]))
        f.write("Median-|residual| diff 2-1: {0:.9g} mm\n\n".format(obs["median_abs_diff_2_minus_1"]))

        f.write("Fraction of paired events with |Dataset 1 residual| < |Dataset 2 residual|: {0:.9g}\n".format(obs["frac_events_abs1_lt_abs2"]))
        f.write("Fraction of paired events with equal absolute residuals: {0:.9g}\n\n".format(obs["frac_events_abs1_eq_abs2"]))

        f.write("Permutation p-values\n")
        f.write("--------------------\n")
        for row in rows:
            f.write("{0}: observed={1:.9g}, p={2:.9g}, alternative={3}\n".format(
                row["statistic"], row["observed"], row["p_value"], row["alternative"]
            ))

        f.write("\nOutputs\n")
        f.write("-------\n")
        f.write(summary_csv + "\n")
        f.write(pvals_csv + "\n")
        f.write(paired_csv + "\n")
        f.write(null_csv + "\n")

    if a.make_plots:
        for key in ["rmse_diff_2_minus_1", "mae_diff_2_minus_1",
                    "mean_abs_pair_diff_2_minus_1", "mean_sq_pair_diff_2_minus_1"]:
            out_png = os.path.join(a.outdir, key + "_null_hist.png")
            write_hist(
                nulls[key],
                obs[key],
                "Permutation null: " + key,
                key + " [Dataset 2 - Dataset 1]",
                out_png,
            )

    print("[INFO] wrote", txt_path)
    print("[INFO] wrote", summary_csv)
    print("[INFO] wrote", pvals_csv)
    print("[INFO] wrote", paired_csv)
    print("[INFO] wrote", null_csv)
    print("")
    print(open(txt_path).read())


if __name__ == "__main__":
    main()
