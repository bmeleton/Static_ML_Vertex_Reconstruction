#!/usr/bin/env python3
"""
Plot x/y/z vertex residuals for the static ML-selected vertex candidates.

This is intended for outputs from apply_static_candidate_model.py, especially:

  static_candidate_model_v1_apply_traincheck/static_candidate_selected_by_model.csv

Residual sign convention:

  residual = ML-selected vertex coordinate - truth coordinate

The script first tries to use existing residual columns, for example:
  dx_candidate_minus_truth, dy_candidate_minus_truth, dz_candidate_minus_truth

If those are not present, it tries to compute residuals from candidate and truth
coordinate columns.

Outputs:
  residual_summary.csv
  residual_values.csv
  residual_hist_xyz.png
  residual_hist_xyz.pdf      if --save-pdf is supplied
  residual_hist_radius.png   if radial residual can be computed
"""

from __future__ import print_function

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-08-07-static-ml-vertex-residuals-v1"


DX_CANDIDATES = [
    "dx_candidate_minus_truth",
    "dx_selected_minus_truth",
    "dx_ml_minus_truth",
    "dx_final_minus_truth",
    "dx_seed_minus_truth",
    "residual_x",
    "x_residual",
    "dx",
]
DY_CANDIDATES = [
    "dy_candidate_minus_truth",
    "dy_selected_minus_truth",
    "dy_ml_minus_truth",
    "dy_final_minus_truth",
    "dy_seed_minus_truth",
    "residual_y",
    "y_residual",
    "dy",
]
DZ_CANDIDATES = [
    "dz_candidate_minus_truth",
    "dz_selected_minus_truth",
    "dz_ml_minus_truth",
    "dz_final_minus_truth",
    "dz_seed_minus_truth",
    "residual_z",
    "z_residual",
    "dz",
]

X_CANDIDATES = [
    "cand_x", "candidate_x", "candidate_x_mm", "static_candidate_x_mm",
    "selected_x", "selected_x_mm", "ml_x", "ml_x_mm", "vertex_x", "x",
]
Y_CANDIDATES = [
    "cand_y", "candidate_y", "candidate_y_mm", "static_candidate_y_mm",
    "selected_y", "selected_y_mm", "ml_y", "ml_y_mm", "vertex_y", "y",
]
Z_CANDIDATES = [
    "cand_z", "candidate_z", "candidate_z_mm", "static_candidate_z_mm",
    "selected_z", "selected_z_mm", "ml_z", "ml_z_mm", "vertex_z", "z",
]

TRUTH_X_CANDIDATES = [
    "truth_x", "truth_x_mm", "true_x", "true_x_mm", "truth_vertex_x", "vertexX",
]
TRUTH_Y_CANDIDATES = [
    "truth_y", "truth_y_mm", "true_y", "true_y_mm", "truth_vertex_y", "vertexY",
]
TRUTH_Z_CANDIDATES = [
    "truth_z", "truth_z_mm", "true_z", "true_z_mm", "truth_vertex_z", "vertexZ",
]

DIST_CANDIDATES = [
    "dist_candidate_minus_truth",
    "dist_selected_minus_truth",
    "dist_ml_minus_truth",
    "dist_final_minus_truth",
    "distance_candidate_minus_truth",
    "dr",
]


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def as_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def get_residuals(df):
    """Return residual DataFrame and a string describing how residuals were made."""
    dx_col = first_existing_column(df, DX_CANDIDATES)
    dy_col = first_existing_column(df, DY_CANDIDATES)
    dz_col = first_existing_column(df, DZ_CANDIDATES)

    out = pd.DataFrame(index=df.index)

    if dx_col and dy_col and dz_col:
        out["dx_mm"] = as_numeric(df[dx_col])
        out["dy_mm"] = as_numeric(df[dy_col])
        out["dz_mm"] = as_numeric(df[dz_col])
        method = "existing residual columns: {0}, {1}, {2}".format(dx_col, dy_col, dz_col)
    else:
        x_col = first_existing_column(df, X_CANDIDATES)
        y_col = first_existing_column(df, Y_CANDIDATES)
        z_col = first_existing_column(df, Z_CANDIDATES)
        tx_col = first_existing_column(df, TRUTH_X_CANDIDATES)
        ty_col = first_existing_column(df, TRUTH_Y_CANDIDATES)
        tz_col = first_existing_column(df, TRUTH_Z_CANDIDATES)
        missing = []
        for label, col in [("candidate x", x_col), ("candidate y", y_col), ("candidate z", z_col),
                           ("truth x", tx_col), ("truth y", ty_col), ("truth z", tz_col)]:
            if not col:
                missing.append(label)
        if missing:
            raise RuntimeError(
                "Could not find residual columns, and could not compute residuals. "
                "Missing: {0}\nAvailable columns:\n{1}".format(
                    ", ".join(missing), ", ".join(list(df.columns))
                )
            )
        out["dx_mm"] = as_numeric(df[x_col]) - as_numeric(df[tx_col])
        out["dy_mm"] = as_numeric(df[y_col]) - as_numeric(df[ty_col])
        out["dz_mm"] = as_numeric(df[z_col]) - as_numeric(df[tz_col])
        method = "computed as candidate - truth from: {0},{1},{2} minus {3},{4},{5}".format(
            x_col, y_col, z_col, tx_col, ty_col, tz_col
        )

    dist_col = first_existing_column(df, DIST_CANDIDATES)
    if dist_col:
        out["dr_mm"] = as_numeric(df[dist_col])
    else:
        out["dr_mm"] = np.sqrt(out["dx_mm"]**2 + out["dy_mm"]**2 + out["dz_mm"]**2)

    # carry useful bookkeeping columns if they exist
    for col in ["event", "candidate_id", "candidate_source", "predicted_dist_mm", "selection_rule"]:
        if col in df.columns:
            out[col] = df[col]

    return out, method


def summarize_array(name, values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    row = {
        "quantity": name,
        "n": int(len(v)),
        "mean_mm": np.nan,
        "median_mm": np.nan,
        "std_mm": np.nan,
        "rms_mm": np.nan,
        "q05_mm": np.nan,
        "q16_mm": np.nan,
        "q84_mm": np.nan,
        "q95_mm": np.nan,
        "width68_mm": np.nan,
        "sigma68_mm": np.nan,
        "frac_abs_lt_0p25": np.nan,
        "frac_abs_lt_0p5": np.nan,
        "frac_abs_lt_1p0": np.nan,
        "frac_abs_lt_2p0": np.nan,
        "frac_abs_lt_5p0": np.nan,
        "min_mm": np.nan,
        "max_mm": np.nan,
    }
    if len(v) == 0:
        return row
    q05, q16, q84, q95 = np.percentile(v, [5, 16, 84, 95])
    row.update({
        "mean_mm": float(np.mean(v)),
        "median_mm": float(np.median(v)),
        "std_mm": float(np.std(v)),
        "rms_mm": float(np.sqrt(np.mean(v * v))),
        "q05_mm": float(q05),
        "q16_mm": float(q16),
        "q84_mm": float(q84),
        "q95_mm": float(q95),
        "width68_mm": float(q84 - q16),
        "sigma68_mm": float(0.5 * (q84 - q16)),
        "frac_abs_lt_0p25": float(np.mean(np.abs(v) < 0.25)),
        "frac_abs_lt_0p5": float(np.mean(np.abs(v) < 0.5)),
        "frac_abs_lt_1p0": float(np.mean(np.abs(v) < 1.0)),
        "frac_abs_lt_2p0": float(np.mean(np.abs(v) < 2.0)),
        "frac_abs_lt_5p0": float(np.mean(np.abs(v) < 5.0)),
        "min_mm": float(np.min(v)),
        "max_mm": float(np.max(v)),
    })
    return row


def normal_pdf(x, mu, sigma):
    if sigma <= 0.0 or not np.isfinite(sigma):
        return np.zeros_like(x)
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * math.sqrt(2.0 * math.pi))


def choose_display_limits(residual_df, fixed_range_mm=None, auto_percentile=99.0):
    if fixed_range_mm is not None and fixed_range_mm > 0:
        return -float(fixed_range_mm), float(fixed_range_mm), "fixed +/- {0:g} mm".format(fixed_range_mm)
    vals = []
    for col in ["dx_mm", "dy_mm", "dz_mm"]:
        v = residual_df[col].values.astype(float)
        v = v[np.isfinite(v)]
        if len(v):
            vals.append(np.abs(v))
    if not vals:
        return -5.0, 5.0, "default +/- 5 mm"
    all_abs = np.concatenate(vals)
    lim = float(np.percentile(all_abs, auto_percentile))
    if lim <= 0.0 or not np.isfinite(lim):
        lim = 5.0
    # round up a little for nicer axis
    lim = max(0.5, math.ceil(lim * 10.0) / 10.0)
    return -lim, lim, "auto +/- {0:g} mm from {1:g} percentile".format(lim, auto_percentile)


def plot_xyz_histograms(residual_df, out_png, out_pdf, bins, xlim, title, label):
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True)
    quantities = [
        ("dx_mm", "x residual: ML - truth"),
        ("dy_mm", "y residual: ML - truth"),
        ("dz_mm", "z residual: ML - truth"),
    ]
    xmin, xmax = xlim
    for ax, (col, axis_title) in zip(axes, quantities):
        v_all = residual_df[col].values.astype(float)
        v_all = v_all[np.isfinite(v_all)]
        v_plot = v_all[(v_all >= xmin) & (v_all <= xmax)]
        n_out = len(v_all) - len(v_plot)
        ax.hist(v_plot, bins=bins, range=(xmin, xmax), alpha=0.85)
        if len(v_all):
            med = np.median(v_all)
            q16, q84 = np.percentile(v_all, [16, 84])
            sigma68 = 0.5 * (q84 - q16)
            mean = np.mean(v_all)
            std = np.std(v_all)
            ax.axvline(med, linestyle="--", linewidth=1.5, label="median={0:.3g} mm".format(med))
            # Overlay a robust Gaussian using median and sigma68, normalized to the displayed histogram.
            if sigma68 > 0 and len(v_plot) > 1:
                xs = np.linspace(xmin, xmax, 400)
                bin_width = float(xmax - xmin) / float(bins)
                ys = normal_pdf(xs, med, sigma68) * len(v_plot) * bin_width
                ax.plot(xs, ys, linewidth=1.5, label="robust sigma={0:.3g} mm".format(sigma68))
            txt = "n={0}\nmean={1:.3g}\nstd={2:.3g}\noutside range={3}".format(len(v_all), mean, std, n_out)
            ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_ylabel("Counts")
        ax.set_title(axis_title)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Residual [mm]")
    fig.suptitle("{0}\n{1}".format(title, label), fontsize=13)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(out_png, dpi=160)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)


def plot_radius_histogram(residual_df, out_png, bins, max_radius_mm, title, label):
    if "dr_mm" not in residual_df.columns:
        return
    v_all = residual_df["dr_mm"].values.astype(float)
    v_all = v_all[np.isfinite(v_all)]
    if len(v_all) == 0:
        return
    if max_radius_mm is None or max_radius_mm <= 0:
        max_radius_mm = float(np.percentile(v_all, 99.0))
        if max_radius_mm <= 0.0 or not np.isfinite(max_radius_mm):
            max_radius_mm = 5.0
        max_radius_mm = max(0.5, math.ceil(max_radius_mm * 10.0) / 10.0)
    v_plot = v_all[(v_all >= 0.0) & (v_all <= max_radius_mm)]
    n_out = len(v_all) - len(v_plot)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.hist(v_plot, bins=bins, range=(0.0, max_radius_mm), alpha=0.85)
    med = np.median(v_all)
    q16, q84 = np.percentile(v_all, [16, 84])
    ax.axvline(med, linestyle="--", linewidth=1.5, label="median={0:.3g} mm".format(med))
    txt = "n={0}\nq16={1:.3g}\nq84={2:.3g}\noutside range={3}".format(len(v_all), q16, q84, n_out)
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("3D distance |ML - truth| [mm]")
    ax.set_ylabel("Counts")
    ax.set_title("3D vertex-distance residual")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.suptitle("{0}\n{1}".format(title, label), fontsize=13)
    fig.tight_layout(rect=[0, 0.0, 1, 0.90])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Plot ML-selected vertex residuals against truth.")
    p.add_argument("--selected-csv", default="static_candidate_model_v1_apply_traincheck/static_candidate_selected_by_model.csv",
                   help="CSV containing one ML-selected candidate per event. Default: %(default)s")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Default: <selected-csv directory>/residual_plots")
    p.add_argument("--label", default="static ML selected vertex",
                   help="Label used in plot titles. Default: %(default)s")
    p.add_argument("--bins", type=int, default=100, help="Histogram bins. Default: %(default)s")
    p.add_argument("--fixed-range-mm", type=float, default=5.0,
                   help="Use a symmetric +/- range for x/y/z plots. Default: %(default)s. Use <=0 with --auto-range for auto.")
    p.add_argument("--auto-range", action="store_true",
                   help="Use an automatic robust x/y/z plot range instead of --fixed-range-mm.")
    p.add_argument("--auto-percentile", type=float, default=99.0,
                   help="Percentile used by --auto-range. Default: %(default)s")
    p.add_argument("--radius-max-mm", type=float, default=10.0,
                   help="Maximum displayed radius for radial histogram. Default: %(default)s. Use <=0 for auto.")
    p.add_argument("--save-pdf", action="store_true", help="Also save residual_hist_xyz.pdf")
    p.add_argument("--print-summary", action="store_true", help="Print summary table to terminal.")
    p.add_argument("--version", action="store_true", help="Print script version and exit.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.version:
        print(SCRIPT_VERSION)
        return 0

    if not os.path.isfile(args.selected_csv):
        print("[ERROR] selected CSV not found: {0}".format(args.selected_csv), file=sys.stderr)
        return 2

    if args.outdir is None:
        base_dir = os.path.dirname(os.path.abspath(args.selected_csv))
        args.outdir = os.path.join(base_dir, "residual_plots")
    ensure_dir(args.outdir)

    df = pd.read_csv(args.selected_csv)
    residual_df, method = get_residuals(df)

    before = len(residual_df)
    residual_df = residual_df.replace([np.inf, -np.inf], np.nan)
    residual_df = residual_df.dropna(subset=["dx_mm", "dy_mm", "dz_mm"])
    after = len(residual_df)
    if after == 0:
        print("[ERROR] no rows with finite dx/dy/dz residuals", file=sys.stderr)
        return 3

    # Recompute dr from dx/dy/dz so radial distance is self-consistent.
    residual_df["dr_from_xyz_mm"] = np.sqrt(
        residual_df["dx_mm"].astype(float)**2 +
        residual_df["dy_mm"].astype(float)**2 +
        residual_df["dz_mm"].astype(float)**2
    )

    values_csv = os.path.join(args.outdir, "residual_values.csv")
    residual_df.to_csv(values_csv, index=False)

    rows = []
    rows.append(summarize_array("dx_mm", residual_df["dx_mm"].values))
    rows.append(summarize_array("dy_mm", residual_df["dy_mm"].values))
    rows.append(summarize_array("dz_mm", residual_df["dz_mm"].values))
    rows.append(summarize_array("dr_from_xyz_mm", residual_df["dr_from_xyz_mm"].values))
    if "dr_mm" in residual_df.columns:
        rows.append(summarize_array("dr_mm_input", residual_df["dr_mm"].values))
    summary = pd.DataFrame(rows)
    summary_csv = os.path.join(args.outdir, "residual_summary.csv")
    summary.to_csv(summary_csv, index=False)

    if args.auto_range:
        fixed = None
    else:
        fixed = args.fixed_range_mm
    xmin, xmax, range_label = choose_display_limits(residual_df, fixed, args.auto_percentile)
    title = "Vertex residuals: ML-selected candidate minus truth"
    full_label = "{0}; {1}; rows={2}, finite={3}; residuals from {4}".format(
        args.label, range_label, before, after, method
    )

    xyz_png = os.path.join(args.outdir, "residual_hist_xyz.png")
    xyz_pdf = os.path.join(args.outdir, "residual_hist_xyz.pdf") if args.save_pdf else None
    plot_xyz_histograms(residual_df, xyz_png, xyz_pdf, args.bins, (xmin, xmax), title, full_label)

    radius_png = os.path.join(args.outdir, "residual_hist_radius.png")
    plot_radius_histogram(residual_df, radius_png, args.bins, args.radius_max_mm, title, args.label)

    print("[INFO] script version: {0}".format(SCRIPT_VERSION))
    print("[INFO] read {0} rows from {1}".format(len(df), args.selected_csv))
    print("[INFO] finite residual rows: {0}".format(after))
    print("[INFO] residual method: {0}".format(method))
    print("[INFO] wrote {0}".format(summary_csv))
    print("[INFO] wrote {0}".format(values_csv))
    print("[INFO] wrote {0}".format(xyz_png))
    if xyz_pdf:
        print("[INFO] wrote {0}".format(xyz_pdf))
    print("[INFO] wrote {0}".format(radius_png))

    if args.print_summary:
        print("")
        print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
