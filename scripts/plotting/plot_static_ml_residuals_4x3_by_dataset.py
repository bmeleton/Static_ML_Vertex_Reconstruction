#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_static_ml_residuals_4x3_by_dataset.py

Make a 4 x 3 panel residual figure for static-ML selected vertices:

    rows    = alpha_300, alpha_400, alpha_580, elastic_new
    columns = dx, dy, dz residuals

Each panel shows:
    * residual histogram
    * robust central Gaussian fit
    * zero-residual vertical line
    * legend containing only Gaussian mean, FWHM, and STD

Python 3.6 compatible.
"""
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def safe_float_series(s):
    return pd.to_numeric(s, errors="coerce")


def gaussian_zero_baseline(x, amplitude, mean, sigma):
    sigma = abs(float(sigma))
    if sigma <= 0:
        sigma = 1.0
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def robust_location_scale(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, 1.0

    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sig = 1.4826 * mad

    if (not np.isfinite(sig)) or sig <= 0.0:
        sig = float(np.std(v))
    if (not np.isfinite(sig)) or sig <= 0.0:
        sig = 1.0

    return med, sig


def robust_gaussian_fit(data, hist_edges, fit_window_sigma=3.0):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    counts, _ = np.histogram(data, bins=hist_edges)

    if len(data) < 3 or len(centers) < 3:
        med, sig = robust_location_scale(data)
        amp = float(np.max(counts)) if len(counts) else 1.0
        fwhm = 2.354820045 * abs(sig)
        return amp, med, abs(sig), fwhm, centers, gaussian_zero_baseline(centers, amp, med, sig), False

    med, sig = robust_location_scale(data)

    window = float(fit_window_sigma) * sig
    if (not np.isfinite(window)) or window <= 0.0:
        window = 3.0 * sig

    fit_mask = (
        np.isfinite(centers)
        & np.isfinite(counts)
        & (centers >= med - window)
        & (centers <= med + window)
        & (counts > 0)
    )

    if fit_mask.sum() < 3:
        fit_mask = np.isfinite(centers) & np.isfinite(counts) & (counts > 0)

    if fit_mask.sum() < 3 or not HAVE_SCIPY:
        amp = float(np.max(counts)) if len(counts) else 1.0
        fwhm = 2.354820045 * abs(sig)
        return amp, med, abs(sig), fwhm, centers, gaussian_zero_baseline(centers, amp, med, sig), False

    xfit = centers[fit_mask]
    yfit = counts[fit_mask].astype(float)

    amp0 = float(np.max(yfit))
    mean0 = float(med)
    sigma0 = float(sig)

    xmin = float(hist_edges[0])
    xmax = float(hist_edges[-1])
    width = max(1.0e-9, xmax - xmin)

    lower = [0.0, xmin, max(1.0e-6, 0.01 * sigma0)]
    upper = [max(1.0, 5.0 * amp0), xmax, max(width, 10.0 * sigma0)]

    sigma_y = np.sqrt(np.maximum(yfit, 1.0))

    try:
        popt, _ = curve_fit(
            gaussian_zero_baseline,
            xfit,
            yfit,
            p0=[amp0, mean0, sigma0],
            bounds=(lower, upper),
            sigma=sigma_y,
            absolute_sigma=False,
            maxfev=50000,
        )
        amp = float(popt[0])
        mean = float(popt[1])
        sigma = abs(float(popt[2]))
        ok = True
    except Exception:
        amp = amp0
        mean = mean0
        sigma = abs(sigma0)
        ok = False

    if (not np.isfinite(sigma)) or sigma <= 0.0:
        sigma = abs(sigma0)
    if not np.isfinite(mean):
        mean = mean0
    if (not np.isfinite(amp)) or amp <= 0.0:
        amp = amp0

    fwhm = 2.354820045 * sigma
    xfine = np.linspace(hist_edges[0], hist_edges[-1], 600)
    yfine = gaussian_zero_baseline(xfine, amp, mean, sigma)

    return amp, mean, sigma, fwhm, xfine, yfine, ok


def find_column(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError("Could not find {0}; tried {1}".format(label, ", ".join(candidates)))


def infer_residual_columns(df):
    dx = find_column(df, [
        "dx_candidate_minus_truth",
        "dx_seed_minus_truth",
        "dx_final_minus_truth",
        "dx_found_minus_truth",
        "dx",
    ], "dx residual column")

    dy = find_column(df, [
        "dy_candidate_minus_truth",
        "dy_seed_minus_truth",
        "dy_final_minus_truth",
        "dy_found_minus_truth",
        "dy",
    ], "dy residual column")

    dz = find_column(df, [
        "dz_candidate_minus_truth",
        "dz_seed_minus_truth",
        "dz_final_minus_truth",
        "dz_found_minus_truth",
        "dz",
    ], "dz residual column")

    return dx, dy, dz


def infer_dist_column(df):
    for c in [
        "dist_candidate_minus_truth",
        "dist_seed_minus_truth",
        "dist_final_minus_truth",
        "dist_found_minus_truth",
        "dist",
    ]:
        if c in df.columns:
            return c
    return None


def parse_input_spec(spec):
    parts = str(spec).split(":", 1)
    if len(parts) != 2:
        raise ValueError("--input must be label:path, got {0}".format(spec))
    return parts[0], parts[1]

def default_inputs():
    base = "static_candidate_combined_4datasets"
    return [
        (r"$^{14}\mathrm{O}(\alpha,\alpha)^{14}\mathrm{O}$ 300 Torr",
         os.path.join(base, "apply_train70_to_test30_alpha_300", "static_candidate_selected_by_model.csv")),

        (r"$^{14}\mathrm{O}(\alpha,\alpha)^{14}\mathrm{O}$ 400 Torr",
         os.path.join(base, "apply_train70_to_test30_alpha_400", "static_candidate_selected_by_model.csv")),

        (r"$^{14}\mathrm{O}(\alpha,\alpha)^{14}\mathrm{O}$ 580 Torr",
         os.path.join(base, "apply_train70_to_test30_alpha_580", "static_candidate_selected_by_model.csv")),

        (r"$^{14}\mathrm{O}(p,p)^{14}\mathrm{O}$",
         os.path.join(base, "apply_train70_to_test30_elastic_new", "static_candidate_selected_by_model.csv")),
    ]

def read_dataset(label, path, far_param_mm=None):
    if not os.path.exists(path):
        raise RuntimeError("Missing input for {0}: {1}".format(label, path))

    df = pd.read_csv(path)
    dxcol, dycol, dzcol = infer_residual_columns(df)
    distcol = infer_dist_column(df)

    out = pd.DataFrame()
    if "event" in df.columns:
        out["event"] = pd.to_numeric(df["event"], errors="coerce")
    elif "event_id" in df.columns:
        out["event"] = pd.to_numeric(df["event_id"], errors="coerce")

    out["dx"] = safe_float_series(df[dxcol])
    out["dy"] = safe_float_series(df[dycol])
    out["dz"] = safe_float_series(df[dzcol])

    if distcol is not None:
        out["dist"] = safe_float_series(df[distcol])
    else:
        out["dist"] = np.sqrt(out["dx"] ** 2 + out["dy"] ** 2 + out["dz"] ** 2)

    finite = np.isfinite(out[["dx", "dy", "dz", "dist"]]).all(axis=1)
    before = len(out)
    out = out[finite].copy()

    if far_param_mm is not None:
        out = out[out["dist"] <= float(far_param_mm)].copy()

    print("[INFO] {0}: read {1} rows, finite kept {2}, after far filter {3}".format(
        label, before, int(finite.sum()), len(out)
    ))
    print("[INFO] {0}: residual columns dx={1}, dy={2}, dz={3}, dist={4}".format(
        label, dxcol, dycol, dzcol, distcol if distcol is not None else "computed"
    ))

    return out


def make_figure(datasets, args):
    quantities = ["dx", "dy", "dz"]
    col_labels = [
        r"Vertex Residual $dx$",
        r"Vertex Residual $dy$",
        r"Vertex Residual $dz$",
    ]

    nrows = len(datasets)
    ncols = 3

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.3 * ncols, 2.9 * nrows),
        sharex="col",
        sharey=False,
    )

    if nrows == 1:
        axes = np.asarray(axes).reshape(1, ncols)

    hist_edges = np.arange(
        -float(args.range_mm),
        float(args.range_mm) + float(args.binwidth),
        float(args.binwidth),
    )
    if len(hist_edges) < 4:
        hist_edges = np.linspace(-float(args.range_mm), float(args.range_mm), 50)

    summary_rows = []

    for row, item in enumerate(datasets):
        label, df = item
        for col, q in enumerate(quantities):
            ax = axes[row, col]

            data = safe_float_series(df[q]).dropna().values
            data = data[np.isfinite(data)]
            in_view = data[(data >= -float(args.range_mm)) & (data <= float(args.range_mm))]

            ax.hist(
                in_view,
                bins=hist_edges,
                alpha=0.78,
                edgecolor="black",
                linewidth=0.45,
            )
            ax.axvline(0.0, linestyle="--", linewidth=1.2)

            if len(in_view) >= 3:
                amp, mean, sigma, fwhm, xfit, yfit, fit_ok = robust_gaussian_fit(
                    in_view,
                    hist_edges,
                    fit_window_sigma=float(args.fit_window_sigma),
                )

                legend_label = (
                    "Gaussian Fit:\n"
                    "Mean = {0:.3f} mm\n"
                    "FWHM = {1:.3f} mm\n"
                    "STD = {2:.3f} mm"
                ).format(mean, fwhm, sigma)

                ax.plot(
                    xfit,
                    yfit,
                    linestyle="-",
                    linewidth=2.0,
                    label=legend_label,
                )
                ax.legend(loc="upper right", fontsize=float(args.legend_fontsize), framealpha=0.5)

                summary_rows.append({
                    "dataset": label,
                    "quantity": q,
                    "n_total_after_far_filter": int(len(data)),
                    "n_in_view": int(len(in_view)),
                    "gauss_mean_mm": float(mean),
                    "gauss_fwhm_mm": float(fwhm),
                    "gauss_std_mm": float(sigma),
                    "fit_ok": bool(fit_ok),
                    "hist_range_mm": float(args.range_mm),
                    "binwidth_mm": float(args.binwidth),
                })
            else:
                ax.text(
                    0.5, 0.5,
                    "Too few points",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                summary_rows.append({
                    "dataset": label,
                    "quantity": q,
                    "n_total_after_far_filter": int(len(data)),
                    "n_in_view": int(len(in_view)),
                    "gauss_mean_mm": np.nan,
                    "gauss_fwhm_mm": np.nan,
                    "gauss_std_mm": np.nan,
                    "fit_ok": False,
                    "hist_range_mm": float(args.range_mm),
                    "binwidth_mm": float(args.binwidth),
                })

            ax.grid(True, alpha=0.25)
            ax.set_xlim(-float(args.range_mm), float(args.range_mm))

            if row == 0:
                ax.set_title(col_labels[col], fontsize=14)
            if col == 0:
                ax.set_ylabel(label + "\nEvents", fontsize=12)
            if row == nrows - 1:
                ax.set_xlabel("Found - True [mm]", fontsize=11)

    fig.suptitle(args.title, fontsize=16)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    os.makedirs(args.outdir, exist_ok=True)
    out_png = os.path.join(args.outdir, args.outfile)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    summary_csv = os.path.join(args.outdir, os.path.splitext(args.outfile)[0] + "_gaussian_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print("[INFO] wrote {0}".format(out_png))
    print("[INFO] wrote {0}".format(summary_csv))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Make 4 x 3 panel dx/dy/dz residual histogram figure by dataset."
    )
    p.add_argument(
        "--input",
        action="append",
        default=None,
        help="Dataset input as label:path. Repeat 4 times. If omitted, uses current 4-dataset default paths.",
    )
    p.add_argument("--outdir", required=True)
    p.add_argument("--outfile", default="static_ml_residuals_4x3_by_dataset.png")
    p.add_argument("--title", default="Static ML Vertex Residuals by Dataset")
    p.add_argument("--binwidth", type=float, default=0.075)
    p.add_argument("--range-mm", type=float, default=5.0)
    p.add_argument("--fit-window-sigma", type=float, default=3.0)
    p.add_argument("--legend-fontsize", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument(
        "--far-param-mm",
        type=float,
        default=100.0,
        help="Drop rows with 3D distance residual above this value. Default 100 mm.",
    )
    p.add_argument("--no-far-filter", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.no_far_filter:
        args.far_param_mm = None

    if args.input:
        specs = [parse_input_spec(x) for x in args.input]
    else:
        specs = default_inputs()

    if len(specs) != 4:
        raise RuntimeError("This script is designed for exactly 4 inputs. Got {0}.".format(len(specs)))

    datasets = []
    for label, path in specs:
        datasets.append((label, read_dataset(label, path, args.far_param_mm)))

    make_figure(datasets, args)


if __name__ == "__main__":
    main()

