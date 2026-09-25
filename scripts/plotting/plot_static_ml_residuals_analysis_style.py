#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_static_ml_residuals_analysis_style_landau_dr.py

Make vertex-residual histograms for static ML-selected candidate output.

Compared with the older Analysis-style plotting script:
  - fixed-width bins over the visible plotting range;
  - robust central-core Gaussian fits for dx, dy, dz;
  - no constant baseline in the plotted residual curves, so the dashed curve
    falls to zero rather than to a horizontal pedestal;
  - Landau-like/Moyal fit for the positive-definite distance dr panel;
  - compact text boxes containing only fit mean/MPV, FWHM, and width;
  - panel titles use mathtext labels:
      Vertex Residual $dx$, Vertex Residual $dy$, Vertex Residual $dz$,
      Vertex Distance $dr$.

Note: scipy does not reliably provide a Landau distribution in older Python 3.6
installations, so the distance panel uses the Moyal distribution, which is the
standard Landau-like approximation often used for asymmetric energy-loss shapes.

Residual convention:
  found - truth = ML-selected candidate - truth

Python 3.6 compatible.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


SQRT_8LN2 = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian_no_baseline(x, amplitude, mean, stddev):
    stddev = np.maximum(np.asarray(stddev), 1.0e-12)
    return amplitude * np.exp(-((x - mean) ** 2) / (2.0 * stddev ** 2))


def gaussian_with_baseline(x, amplitude, mean, stddev, baseline):
    stddev = np.maximum(np.asarray(stddev), 1.0e-12)
    return baseline + gaussian_no_baseline(x, amplitude, mean, stddev)


def landau_like_moyal_no_baseline(x, amplitude, mpv, scale):
    """Landau-like Moyal curve with peak amplitude at x=mpv.

    y = A exp[-0.5(z + exp(-z))], z=(x-mpv)/scale

    This is not a true ROOT TMath::Landau, but it is a stable Landau-like
    approximation available without requiring newer scipy.stats.landau.
    """
    scale = np.maximum(np.asarray(scale), 1.0e-12)
    z = (np.asarray(x) - mpv) / scale
    # Clip to avoid floating overflow in exp(-z) for very negative z.
    z = np.clip(z, -80.0, 80.0)
    return amplitude * np.exp(-0.5 * (z + np.exp(-z)))


def safe_float_series(s):
    return pd.to_numeric(s, errors='coerce')


def find_column(df, candidates, required=True, label='column'):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError('Could not find {}. Tried: {}'.format(label, ', '.join(candidates)))
    return None


def infer_residual_columns(df, args):
    if args.dxcol and args.dycol and args.dzcol:
        dxcol, dycol, dzcol = args.dxcol, args.dycol, args.dzcol
        for c in [dxcol, dycol, dzcol]:
            if c not in df.columns:
                raise RuntimeError('Requested residual column not found: {}'.format(c))
        return dxcol, dycol, dzcol, None

    dxcol = find_column(df, [
        'dx_candidate_minus_truth',
        'dx_seed_minus_truth',
        'dx_final_minus_truth',
        'dx_found_minus_truth',
        'dx',
    ], required=False, label='dx residual column')
    dycol = find_column(df, [
        'dy_candidate_minus_truth',
        'dy_seed_minus_truth',
        'dy_final_minus_truth',
        'dy_found_minus_truth',
        'dy',
    ], required=False, label='dy residual column')
    dzcol = find_column(df, [
        'dz_candidate_minus_truth',
        'dz_seed_minus_truth',
        'dz_final_minus_truth',
        'dz_found_minus_truth',
        'dz',
    ], required=False, label='dz residual column')

    if dxcol and dycol and dzcol:
        return dxcol, dycol, dzcol, None

    found_sets = [
        ('cand_x', 'cand_y', 'cand_z'),
        ('candidate_x', 'candidate_y', 'candidate_z'),
        ('selected_x', 'selected_y', 'selected_z'),
        ('found_x', 'found_y', 'found_z'),
        ('seed_x', 'seed_y', 'seed_z'),
        ('final_x', 'final_y', 'final_z'),
    ]
    truth_sets = [
        ('truth_x', 'truth_y', 'truth_z'),
        ('true_x', 'true_y', 'true_z'),
        ('vertexX', 'vertexY', 'vertexZ'),
    ]

    found = None
    for cols in found_sets:
        if all(c in df.columns for c in cols):
            found = cols
            break
    truth = None
    for cols in truth_sets:
        if all(c in df.columns for c in cols):
            truth = cols
            break

    if found is None or truth is None:
        raise RuntimeError(
            'Could not infer residuals. Provide --dxcol --dycol --dzcol, or include candidate/truth coordinate columns.'
        )

    df['_analysis_style_dx'] = safe_float_series(df[found[0]]) - safe_float_series(df[truth[0]])
    df['_analysis_style_dy'] = safe_float_series(df[found[1]]) - safe_float_series(df[truth[1]])
    df['_analysis_style_dz'] = safe_float_series(df[found[2]]) - safe_float_series(df[truth[2]])
    return '_analysis_style_dx', '_analysis_style_dy', '_analysis_style_dz', 'computed_from_xyz'


def choose_dist_column(df, args, dx, dy, dz):
    if args.distcol:
        if args.distcol not in df.columns:
            raise RuntimeError('Requested distance column not found: {}'.format(args.distcol))
        return args.distcol
    for c in [
        'dist_candidate_minus_truth',
        'dist_seed_minus_truth',
        'dist_final_minus_truth',
        'dist_found_minus_truth',
        'dist',
    ]:
        if c in df.columns:
            return c
    df['_analysis_style_dist'] = np.sqrt(df[dx] ** 2 + df[dy] ** 2 + df[dz] ** 2)
    return '_analysis_style_dist'


def robust_median_sigma(data):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return 0.0, 1.0

    med = float(np.median(data))
    mad = float(np.median(np.abs(data - med)))
    sig = 1.4826 * mad

    if (not np.isfinite(sig)) or sig <= 0.0:
        sig = float(np.std(data))
    if (not np.isfinite(sig)) or sig <= 0.0:
        sig = 1.0

    return med, sig


def histogram_mode_estimate(data, bin_edges):
    counts, edges = np.histogram(data, bins=bin_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(counts) == 0 or np.max(counts) <= 0:
        return robust_median_sigma(data)[0]
    return float(centers[int(np.argmax(counts))])


def make_fixed_bins(lo, hi, binwidth):
    lo = float(lo)
    hi = float(hi)
    binwidth = float(binwidth)
    if binwidth <= 0.0:
        raise RuntimeError('--binwidth must be positive')
    if hi <= lo:
        hi = lo + binwidth

    nbins = int(np.ceil((hi - lo) / binwidth))
    if nbins < 1:
        nbins = 1
    return lo + np.arange(nbins + 1, dtype=float) * binwidth


def numeric_fwhm_from_curve(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 3 or np.max(y) <= 0.0:
        return np.nan

    imax = int(np.argmax(y))
    half = 0.5 * float(y[imax])

    left_x = np.nan
    for i in range(imax, 0, -1):
        y1 = y[i - 1]
        y2 = y[i]
        if (y1 <= half <= y2) or (y2 <= half <= y1):
            if y2 == y1:
                left_x = x[i]
            else:
                left_x = x[i - 1] + (half - y1) * (x[i] - x[i - 1]) / (y2 - y1)
            break

    right_x = np.nan
    for i in range(imax, len(x) - 1):
        y1 = y[i]
        y2 = y[i + 1]
        if (y1 >= half >= y2) or (y2 >= half >= y1):
            if y2 == y1:
                right_x = x[i]
            else:
                right_x = x[i] + (half - y1) * (x[i + 1] - x[i]) / (y2 - y1)
            break

    if np.isfinite(left_x) and np.isfinite(right_x) and right_x > left_x:
        return float(right_x - left_x)
    return np.nan


def robust_gaussian_fit(data, bin_edges, fit_window_sigma=3.0,
                        min_fit_points=30, fit_baseline=False, fit_core_only=True):
    """Robustly fit the central histogram peak with a Gaussian."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    visible_lo = float(bin_edges[0])
    visible_hi = float(bin_edges[-1])
    data_visible = data[(data >= visible_lo) & (data <= visible_hi)]
    if len(data_visible) == 0:
        raise RuntimeError('No finite data inside visible plotting range')

    center0, sig0 = robust_median_sigma(data_visible)
    binwidth = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
    sig0 = max(float(abs(sig0)), binwidth)

    if fit_core_only:
        fit_lo = max(center0 - float(fit_window_sigma) * sig0, visible_lo)
        fit_hi = min(center0 + float(fit_window_sigma) * sig0, visible_hi)
        fit_data = data_visible[(data_visible >= fit_lo) & (data_visible <= fit_hi)]
        if len(fit_data) < int(min_fit_points):
            fit_data = data_visible
            fit_lo = visible_lo
            fit_hi = visible_hi
    else:
        fit_data = data_visible
        fit_lo = visible_lo
        fit_hi = visible_hi

    fit_edges = bin_edges[(bin_edges >= fit_lo) & (bin_edges <= fit_hi)]
    if len(fit_edges) < 4:
        fit_edges = make_fixed_bins(fit_lo, fit_hi, binwidth)
        if len(fit_edges) < 4:
            fit_edges = bin_edges

    counts, edges = np.histogram(fit_data, bins=fit_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    good = np.isfinite(centers) & np.isfinite(counts)
    centers = centers[good]
    counts = counts[good].astype(float)

    if len(centers) < 3:
        amp = float(max(np.max(counts), 1.0)) if len(counts) else 1.0
        mean = float(center0)
        sigma = float(sig0)
        baseline = 0.0
        status = 'fallback_too_few_bins'
    else:
        baseline0 = 0.0
        if fit_baseline and len(counts) >= 10:
            baseline0 = float(np.percentile(counts, 10.0))
        amp0 = max(float(np.max(counts)) - baseline0, 1.0)
        mean0 = float(centers[int(np.argmax(counts))]) if len(counts) and np.max(counts) > 0 else float(center0)
        sigma0 = float(sig0)

        sigma_min = max(0.5 * binwidth, 1.0e-6)
        sigma_max = max(float(fit_hi - fit_lo), 2.0 * binwidth)
        if sigma_max <= sigma_min:
            sigma_max = sigma_min * 100.0

        if fit_baseline:
            p0 = [amp0, mean0, sigma0, baseline0]
            lower = [0.0, fit_lo, sigma_min, 0.0]
            upper = [np.inf, fit_hi, sigma_max, np.inf]
            func = gaussian_with_baseline
        else:
            p0 = [amp0, mean0, sigma0]
            lower = [0.0, fit_lo, sigma_min]
            upper = [np.inf, fit_hi, sigma_max]
            func = gaussian_no_baseline

        if HAVE_SCIPY and np.sum(counts > 0.0) >= 3:
            try:
                yerr = np.sqrt(np.maximum(counts, 1.0))
                popt, pcov = curve_fit(
                    func,
                    centers,
                    counts,
                    p0=p0,
                    bounds=(lower, upper),
                    sigma=yerr,
                    absolute_sigma=False,
                    maxfev=50000,
                )
                if fit_baseline:
                    amp, mean, sigma, baseline = popt
                else:
                    amp, mean, sigma = popt
                    baseline = 0.0
                status = 'curve_fit_robust_core'
            except Exception as exc:
                print('  [WARN] robust Gaussian fit failed; using robust estimate: {}'.format(exc))
                amp = amp0
                mean = mean0
                sigma = sigma0
                baseline = baseline0 if fit_baseline else 0.0
                status = 'fallback_fit_failed'
        else:
            if not HAVE_SCIPY:
                print('  [WARN] scipy not available; using robust estimate for Gaussian overlay')
            amp = amp0
            mean = mean0
            sigma = sigma0
            baseline = baseline0 if fit_baseline else 0.0
            status = 'fallback_no_scipy_or_empty_counts'

    sigma = abs(float(sigma))
    if (not np.isfinite(sigma)) or sigma <= 0.0:
        sigma = float(sig0)
    mean = float(mean)
    amp = float(amp)
    baseline = float(baseline)
    fwhm = SQRT_8LN2 * sigma

    x_fine = np.linspace(visible_lo, visible_hi, 800)
    # By default, fit_baseline is False, so the curve sits on y=0.
    if fit_baseline:
        y_fine = gaussian_with_baseline(x_fine, amp, mean, sigma, baseline)
    else:
        y_fine = gaussian_no_baseline(x_fine, amp, mean, sigma)

    print('  fit type:        Gaussian')
    print('  fit status:      {}'.format(status))
    print('  fit window:      {:.4f} to {:.4f} mm'.format(float(fit_lo), float(fit_hi)))
    print('  fit data rows:   {}'.format(int(len(fit_data))))
    print('  gaussian mean:   {:.4f}'.format(mean))
    print('  gaussian STD:    {:.4f}'.format(sigma))
    print('  gaussian FWHM:   {:.4f}'.format(fwhm))

    return {
        'fit_type': 'gaussian',
        'amp': amp,
        'mean': mean,
        'sigma': sigma,
        'fwhm': fwhm,
        'baseline': baseline,
        'x': x_fine,
        'y': y_fine,
        'status': status,
        'n_fit_data': int(len(fit_data)),
        'fit_lo': float(fit_lo),
        'fit_hi': float(fit_hi),
    }


def robust_landau_like_fit(data, bin_edges, fit_window_sigma=4.0,
                           min_fit_points=30, fit_core_only=True):
    """Fit the positive-definite distance distribution with a Landau-like Moyal."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    visible_lo = float(bin_edges[0])
    visible_hi = float(bin_edges[-1])
    data_visible = data[(data >= visible_lo) & (data <= visible_hi)]
    if len(data_visible) == 0:
        raise RuntimeError('No finite distance data inside visible plotting range')

    binwidth = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
    mode0 = histogram_mode_estimate(data_visible, bin_edges)
    med0, sig0 = robust_median_sigma(data_visible)
    scale0 = max(0.40 * float(sig0), binwidth)

    if fit_core_only:
        # Moyal/Landau has a right tail, so use more right-side range than left.
        fit_lo = max(visible_lo, mode0 - 1.5 * float(fit_window_sigma) * scale0)
        fit_hi = min(visible_hi, mode0 + float(fit_window_sigma) * max(sig0, scale0))
        fit_data = data_visible[(data_visible >= fit_lo) & (data_visible <= fit_hi)]
        if len(fit_data) < int(min_fit_points):
            fit_data = data_visible
            fit_lo = visible_lo
            fit_hi = visible_hi
    else:
        fit_data = data_visible
        fit_lo = visible_lo
        fit_hi = visible_hi

    fit_edges = bin_edges[(bin_edges >= fit_lo) & (bin_edges <= fit_hi)]
    if len(fit_edges) < 4:
        fit_edges = make_fixed_bins(fit_lo, fit_hi, binwidth)
        if len(fit_edges) < 4:
            fit_edges = bin_edges

    counts, edges = np.histogram(fit_data, bins=fit_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    good = np.isfinite(centers) & np.isfinite(counts)
    centers = centers[good]
    counts = counts[good].astype(float)

    if len(centers) < 3:
        amp = float(max(np.max(counts), 1.0)) if len(counts) else 1.0
        mpv = float(mode0)
        scale = float(scale0)
        status = 'fallback_too_few_bins'
    else:
        amp0 = max(float(np.max(counts)), 1.0)
        mpv0 = float(centers[int(np.argmax(counts))]) if len(counts) and np.max(counts) > 0 else float(mode0)
        scale_min = max(0.25 * binwidth, 1.0e-6)
        scale_max = max(float(fit_hi - fit_lo), 2.0 * binwidth)
        if scale_max <= scale_min:
            scale_max = scale_min * 100.0

        p0 = [amp0, mpv0, scale0]
        lower = [0.0, visible_lo, scale_min]
        upper = [np.inf, visible_hi, scale_max]

        if HAVE_SCIPY and np.sum(counts > 0.0) >= 3:
            try:
                yerr = np.sqrt(np.maximum(counts, 1.0))
                popt, pcov = curve_fit(
                    landau_like_moyal_no_baseline,
                    centers,
                    counts,
                    p0=p0,
                    bounds=(lower, upper),
                    sigma=yerr,
                    absolute_sigma=False,
                    maxfev=50000,
                )
                amp, mpv, scale = popt
                status = 'curve_fit_landau_like_moyal'
            except Exception as exc:
                print('  [WARN] Landau-like fit failed; using robust estimate: {}'.format(exc))
                amp = amp0
                mpv = mpv0
                scale = scale0
                status = 'fallback_fit_failed'
        else:
            if not HAVE_SCIPY:
                print('  [WARN] scipy not available; using robust estimate for Landau-like overlay')
            amp = amp0
            mpv = mpv0
            scale = scale0
            status = 'fallback_no_scipy_or_empty_counts'

    amp = float(amp)
    mpv = float(mpv)
    scale = abs(float(scale))
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = max(float(scale0), binwidth)

    x_fine = np.linspace(visible_lo, visible_hi, 1200)
    y_fine = landau_like_moyal_no_baseline(x_fine, amp, mpv, scale)
    fwhm = numeric_fwhm_from_curve(x_fine, y_fine)

    print('  fit type:        Landau-like Moyal')
    print('  fit status:      {}'.format(status))
    print('  fit window:      {:.4f} to {:.4f} mm'.format(float(fit_lo), float(fit_hi)))
    print('  fit data rows:   {}'.format(int(len(fit_data))))
    print('  Landau MPV:      {:.4f}'.format(mpv))
    print('  Landau scale:    {:.4f}'.format(scale))
    print('  Landau FWHM:     {:.4f}'.format(fwhm))

    return {
        'fit_type': 'landau_like_moyal',
        'amp': amp,
        'mean': mpv,
        'sigma': scale,
        'fwhm': float(fwhm) if np.isfinite(fwhm) else np.nan,
        'baseline': 0.0,
        'x': x_fine,
        'y': y_fine,
        'status': status,
        'n_fit_data': int(len(fit_data)),
        'fit_lo': float(fit_lo),
        'fit_hi': float(fit_hi),
    }


def panel_info(idx):
    if idx == 'dx':
        return (0, 0, 'Vertex Residual $dx$', 'Found - Truth [mm]')
    if idx == 'dy':
        return (0, 1, 'Vertex Residual $dy$', 'Found - Truth [mm]')
    if idx == 'dz':
        return (1, 0, 'Vertex Residual $dz$', 'Found - Truth [mm]')
    return (1, 1, 'Vertex Distance $dr$', r'$dr = |Found - Truth|$ [mm]')


def make_histograms(truth, binwidth, rnge, histdir, histname='vertex_residuals_hist.png',
                    dist_xmax=20.0, dpi=220, max_bins=5000, fit_window_sigma=3.0,
                    landau_fit_window_sigma=4.0, min_fit_points=30, fit_baseline=False,
                    fit_full_visible=False, textbox_fontsize=13):
    fig, axes = plt.subplots(2, 2, figsize=(17, 9))

    rows = []
    things = ['dx', 'dy', 'dz', 'dist']
    for idx in things:
        data = safe_float_series(truth[idx]).dropna()
        data = data[np.isfinite(data)]
        if len(data) == 0:
            print('\n  == {} info ==\n'.format(idx))
            print('  No finite data; skipping panel')
            continue

        is_distance = (idx == 'dist')
        if is_distance:
            xlo, xhi = 0.0, float(dist_xmax)
        else:
            xlo, xhi = -float(rnge), float(rnge)

        bins = make_fixed_bins(xlo, xhi, float(binwidth))
        if max_bins is not None and len(bins) - 1 > int(max_bins):
            print('  [WARN] {} requested {} bins; limiting to {}'.format(idx, len(bins) - 1, int(max_bins)))
            bins = np.linspace(xlo, xhi, int(max_bins) + 1)

        plot_data = data[(data >= xlo) & (data <= xhi)]
        if len(plot_data) == 0:
            print('\n  == {} info ==\n'.format(idx))
            print('  No finite data inside plotting range; skipping panel')
            continue

        print('\n  == {} info ==\n'.format(idx))
        if is_distance:
            fit = robust_landau_like_fit(
                data=plot_data,
                bin_edges=bins,
                fit_window_sigma=landau_fit_window_sigma,
                min_fit_points=min_fit_points,
                fit_core_only=(not fit_full_visible),
            )
        else:
            fit = robust_gaussian_fit(
                data=plot_data,
                bin_edges=bins,
                fit_window_sigma=fit_window_sigma,
                min_fit_points=min_fit_points,
                fit_baseline=fit_baseline,
                fit_core_only=(not fit_full_visible),
            )

        rr, cc, title, xlabel = panel_info(idx)
        ax = axes[rr, cc]

        ax.hist(plot_data, bins=bins)
        ax.plot(fit['x'], fit['y'], color='red', linestyle='--', linewidth=2)

        if is_distance:
            ax.set_xlim(-0.1, dist_xmax)
        else:
            ax.set_xlim(-rnge, rnge)

        ax.set_ylim(bottom=0.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Events')
        ax.grid(True, alpha=0.3)

        if is_distance:
            text = 'Landau MPV: {0:.4f} mm\nLandau FWHM: {1:.4f} mm\nLandau scale: {2:.4f} mm'.format(
                float(fit['mean']), float(fit['fwhm']), float(fit['sigma'])
            )
        else:
            text = 'Gauss mean: {0:.4f} mm\nGauss FWHM: {1:.4f} mm\nGauss STD: {2:.4f} mm'.format(
                float(fit['mean']), float(fit['fwhm']), float(fit['sigma'])
            )
        ax.text(
            0.95, 0.95,
            text,
            transform=ax.transAxes,
            fontsize=int(textbox_fontsize),
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', alpha=0.3),
        )

        rows.append({
            'quantity': idx,
            'fit_type': fit['fit_type'],
            'count_all_after_filters': int(len(data)),
            'count_plotted': int(len(plot_data)),
            'mean_all_after_filters': float(data.mean()),
            'std_all_after_filters': float(data.std()),
            'median_all_after_filters': float(data.quantile(0.50)),
            'fit_amp': float(fit['amp']),
            'fit_mean_or_mpv': float(fit['mean']),
            'fit_fwhm': float(fit['fwhm']),
            'fit_std_or_scale': float(fit['sigma']),
            'fit_baseline': float(fit['baseline']),
            'fit_status': fit['status'],
            'fit_window_lo_mm': float(fit['fit_lo']),
            'fit_window_hi_mm': float(fit['fit_hi']),
            'n_fit_data': int(fit['n_fit_data']),
            'binwidth_mm': float(binwidth),
            'bincount': int(len(bins) - 1),
        })

    plt.tight_layout()
    if not os.path.isdir(histdir):
        os.makedirs(histdir)
    out_png = os.path.join(histdir, histname)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    summary_csv = os.path.join(histdir, os.path.splitext(histname)[0] + '_summary.csv')
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print('\n[INFO] wrote {}'.format(out_png))
    print('[INFO] wrote {}'.format(summary_csv))
    return out_png, summary_csv


def build_truth_like_dataframe(df, args):
    dxcol, dycol, dzcol, how = infer_residual_columns(df, args)
    distcol = choose_dist_column(df, args, dxcol, dycol, dzcol)

    out = pd.DataFrame()
    if 'event' in df.columns:
        out['event'] = pd.to_numeric(df['event'], errors='coerce')
    elif 'event_id' in df.columns:
        out['event'] = pd.to_numeric(df['event_id'], errors='coerce')

    out['dx'] = safe_float_series(df[dxcol])
    out['dy'] = safe_float_series(df[dycol])
    out['dz'] = safe_float_series(df[dzcol])
    out['dist'] = safe_float_series(df[distcol])

    recomputed = np.sqrt(out['dx'] ** 2 + out['dy'] ** 2 + out['dz'] ** 2)
    if args.recompute_dist:
        out['dist'] = recomputed
    else:
        out['dist'] = out['dist'].where(np.isfinite(out['dist']), recomputed)

    for c in ['candidate_source', 'selection_rule', 'predicted_dist_mm', 'candidate_id']:
        if c in df.columns:
            out[c] = df[c]

    finite = np.isfinite(out[['dx', 'dy', 'dz', 'dist']]).all(axis=1)
    before = len(out)
    out = out[finite].copy()
    print('[INFO] finite residual rows: {} / {}'.format(len(out), before))
    print('[INFO] residual columns: dx={}, dy={}, dz={}, dist={}'.format(dxcol, dycol, dzcol, distcol))
    if how:
        print('[INFO] residuals were {}'.format(how))

    if args.far_param_mm is not None:
        far_param = float(args.far_param_mm)
        num_far = int((out['dist'] > far_param).sum())
        print(' number of events {}'.format(before))
        print(' number missed {}'.format(before - len(out)))
        print(' number farther than {}mm {}'.format(far_param, num_far))
        out = out[out['dist'] <= far_param].copy()
        print('[INFO] rows after far-distance filter: {}'.format(len(out)))

    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Plot static ML vertex residuals with robust Gaussian fits and Landau-like dr fit.')
    ap.add_argument('--selected-csv', required=True,
                    help='CSV containing one selected ML vertex per event, usually static_candidate_selected_by_model.csv')
    ap.add_argument('--outdir', required=True,
                    help='Output histogram directory.')
    ap.add_argument('--histname', default='vertex_residuals_hist.png',
                    help='Output PNG filename.')
    ap.add_argument('--binwidth', type=float, default=0.075,
                    help='Histogram bin width in mm.')
    ap.add_argument('--range-mm', type=float, default=10.0,
                    help='Residual x-axis range +/- this value in mm.')
    ap.add_argument('--dist-xmax-mm', type=float, default=20.0,
                    help='Distance panel x-axis maximum.')
    ap.add_argument('--far-param-mm', type=float, default=100.0,
                    help='Drop rows with dist above this value. Use --no-far-filter to disable.')
    ap.add_argument('--no-far-filter', action='store_true',
                    help='Disable the far-distance filter.')
    ap.add_argument('--dpi', type=int, default=220,
                    help='Output DPI.')
    ap.add_argument('--max-bins', type=int, default=5000,
                    help='Safety cap on bin count. Set <=0 to disable.')
    ap.add_argument('--fit-window-sigma', type=float, default=3.0,
                    help='Gaussian dx/dy/dz fit window in robust sigmas. Default 3.0.')
    ap.add_argument('--landau-fit-window-sigma', type=float, default=4.0,
                    help='Landau-like dr fit window size. Default 4.0.')
    ap.add_argument('--fit-full-visible', action='store_true',
                    help='Fit the full visible histogram range instead of the robust central/core range.')
    ap.add_argument('--min-fit-points', type=int, default=30,
                    help='Minimum data points in fit window before falling back to full visible data.')
    ap.add_argument('--fit-baseline', action='store_true',
                    help='Allow a Gaussian constant baseline for dx/dy/dz. Default is no baseline, so curve bottom is y=0.')
    ap.add_argument('--textbox-fontsize', type=int, default=13,
                    help='Font size for fit text boxes.')
    ap.add_argument('--dxcol', default=None)
    ap.add_argument('--dycol', default=None)
    ap.add_argument('--dzcol', default=None)
    ap.add_argument('--distcol', default=None)
    ap.add_argument('--recompute-dist', action='store_true',
                    help='Recompute dist from dx/dy/dz even if a distance column exists.')
    ap.add_argument('--write-truth-like-csv', default=None,
                    help='Optional CSV path to write the simplified dx/dy/dz/dist dataframe used for plotting.')
    args = ap.parse_args(argv)

    if args.no_far_filter:
        args.far_param_mm = None
    if args.max_bins is not None and args.max_bins <= 0:
        args.max_bins = None

    df = pd.read_csv(args.selected_csv)
    print('[INFO] read {} rows={} cols={}'.format(args.selected_csv, len(df), len(df.columns)))
    if 'candidate_source' in df.columns:
        print('[INFO] candidate_source counts:')
        print(df['candidate_source'].value_counts().to_string())

    truth_like = build_truth_like_dataframe(df, args)

    if args.write_truth_like_csv is None:
        args.write_truth_like_csv = os.path.join(args.outdir, 'analysis_style_residual_values.csv')
    out_parent = os.path.dirname(args.write_truth_like_csv) if os.path.dirname(args.write_truth_like_csv) else '.'
    if not os.path.isdir(out_parent):
        os.makedirs(out_parent)
    truth_like.to_csv(args.write_truth_like_csv, index=False)
    print('[INFO] wrote {}'.format(args.write_truth_like_csv))

    print('Furthest guesses:')
    print(truth_like[['dist', 'dx', 'dy', 'dz']].nlargest(10, 'dist').to_string(index=False))
    print('Smallest dy:')
    print(truth_like[['dist', 'dx', 'dy', 'dz']].nsmallest(10, 'dy').to_string(index=False))

    make_histograms(
        truth=truth_like,
        binwidth=args.binwidth,
        rnge=args.range_mm,
        histdir=args.outdir,
        histname=args.histname,
        dist_xmax=args.dist_xmax_mm,
        dpi=args.dpi,
        max_bins=args.max_bins,
        fit_window_sigma=args.fit_window_sigma,
        landau_fit_window_sigma=args.landau_fit_window_sigma,
        min_fit_points=args.min_fit_points,
        fit_baseline=args.fit_baseline,
        fit_full_visible=args.fit_full_visible,
        textbox_fontsize=args.textbox_fontsize,
    )


if __name__ == '__main__':
    main()
