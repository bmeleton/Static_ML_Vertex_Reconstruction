#!/usr/bin/env python3
"""
static_candidate_confusion_matrices.py

Make validation diagnostics for the static vertex-candidate selector.

This script treats the static candidate-quality regressor as a scorer/ranker.
It creates:

  1. Event-level oracle-vs-ML table
     - oracle candidate = candidate with minimum true distance in each event
     - ML candidate     = candidate with minimum predicted distance in each event

  2. Candidate-source confusion matrix
     - rows    = source of truth-oracle candidate
     - columns = source of ML-selected candidate

  3. Good/bad confusion matrices at one or more distance thresholds
     - candidate-level: all candidate rows
     - event-selected: only the ML-selected candidate per event

The honest/default use is on the validation split:

  python3 static_candidate_confusion_matrices.py \
    --scored static_candidate_model_v1/static_candidate_training_scored_rows.csv \
    --outdir static_candidate_model_v1/confusion_matrices \
    --split val \
    --thresholds 0.5,1.0,2.0 \
    --make-plots

Python 3.6 compatible.
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def find_column(df, explicit, exact_candidates, contains_all, friendly_name, required=True):
    if explicit:
        if explicit not in df.columns:
            raise RuntimeError("Requested {0} column not found: {1}".format(friendly_name, explicit))
        return explicit

    for c in exact_candidates:
        if c in df.columns:
            return c

    matches = []
    for c in df.columns:
        cl = c.lower()
        ok = True
        for token in contains_all:
            if token not in cl:
                ok = False
                break
        if ok:
            matches.append(c)

    if matches:
        return matches[0]

    if required:
        raise RuntimeError(
            "Could not auto-detect {0} column. Available columns:\n{1}".format(
                friendly_name, "\n".join(["  " + str(c) for c in df.columns])
            )
        )
    return None


def parse_thresholds(s):
    vals = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise RuntimeError("No thresholds parsed from: {0}".format(s))
    return vals


def safe_name(x):
    return str(x).replace('.', 'p').replace('-', 'm').replace('/', '_')


def compute_binary_metrics(cm):
    # cm is a crosstab with string labels. Missing cells are treated as zero.
    def get(row, col):
        try:
            return float(cm.loc[row, col])
        except Exception:
            return 0.0

    tp = get('actual_good', 'pred_good')
    tn = get('actual_bad', 'pred_bad')
    fp = get('actual_bad', 'pred_good')
    fn = get('actual_good', 'pred_bad')
    total = tp + tn + fp + fn

    acc = (tp + tn) / total if total > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = 2.0 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else np.nan

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total': total,
        'accuracy': acc,
        'precision_pred_good': prec,
        'recall_actual_good': rec,
        'specificity_actual_bad': spec,
        'f1_good': f1,
    }


def plot_matrix(matrix_df, title, out_png):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('[WARN] matplotlib unavailable; skipping plot {0}: {1}'.format(out_png, e))
        return

    # Drop margins for plotting, if present.
    df = matrix_df.copy()
    if 'All' in df.index:
        df = df.drop(index='All')
    if 'All' in df.columns:
        df = df.drop(columns=['All'])

    if df.shape[0] == 0 or df.shape[1] == 0:
        print('[WARN] empty matrix; skipping plot:', out_png)
        return

    vals = df.values.astype(float)
    fig_w = max(6.0, 0.9 * df.shape[1] + 3.0)
    fig_h = max(4.5, 0.7 * df.shape[0] + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(vals)
    ax.set_title(title)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=45, ha='right')
    ax.set_yticklabels([str(i) for i in df.index])
    ax.set_xlabel('Predicted / ML-selected')
    ax.set_ylabel('Actual / oracle')

    max_val = np.nanmax(vals) if vals.size else 0.0
    threshold = 0.5 * max_val
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = vals[i, j]
            txt = str(int(v)) if abs(v - int(v)) < 1e-9 else '{0:.3g}'.format(v)
            # Let matplotlib choose ordinary text colors where possible; switch only for readability.
            color = 'white' if v > threshold and max_val > 0 else 'black'
            ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def crosstab_good_bad(df, truth_col, pred_col, threshold):
    tmp = df.copy()
    tmp['actual'] = np.where(tmp[truth_col].astype(float) < threshold, 'actual_good', 'actual_bad')
    tmp['predicted'] = np.where(tmp[pred_col].astype(float) < threshold, 'pred_good', 'pred_bad')
    cm = pd.crosstab(tmp['actual'], tmp['predicted'], rownames=['actual'], colnames=['predicted'], margins=True)

    # Force consistent ordering when categories are missing.
    for row in ['actual_good', 'actual_bad']:
        if row not in cm.index:
            cm.loc[row] = 0
    for col in ['pred_good', 'pred_bad']:
        if col not in cm.columns:
            cm[col] = 0
    ordered_rows = ['actual_good', 'actual_bad']
    ordered_cols = ['pred_good', 'pred_bad']
    if 'All' in cm.index:
        ordered_rows.append('All')
    if 'All' in cm.columns:
        ordered_cols.append('All')
    cm = cm.loc[ordered_rows, ordered_cols]
    return cm


def summarize_event_selection(events_df):
    n = len(events_df)
    same_source = np.nan
    same_candidate = np.nan
    if n > 0:
        if 'oracle_candidate_source' in events_df.columns and 'ml_candidate_source' in events_df.columns:
            same_source = float((events_df['oracle_candidate_source'].astype(str) == events_df['ml_candidate_source'].astype(str)).mean())
        if 'oracle_candidate_id' in events_df.columns and 'ml_candidate_id' in events_df.columns:
            same_candidate = float((events_df['oracle_candidate_id'].astype(str) == events_df['ml_candidate_id'].astype(str)).mean())

    out = {
        'n_events': n,
        'same_candidate_source_fraction': same_source,
        'same_candidate_id_fraction': same_candidate,
    }

    for col in ['oracle_true_dist_mm', 'ml_true_dist_mm', 'ml_pred_dist_mm', 'ml_minus_oracle_true_dist_mm']:
        if col in events_df.columns and n > 0:
            vals = pd.to_numeric(events_df[col], errors='coerce')
            out[col + '_median'] = float(np.nanmedian(vals))
            out[col + '_mean'] = float(np.nanmean(vals))
            out[col + '_max'] = float(np.nanmax(vals))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Make confusion-matrix diagnostics for static candidate selector.')
    ap.add_argument('--scored', default='static_candidate_model_v1/static_candidate_training_scored_rows.csv',
                    help='CSV with candidate rows, truth distance, predicted distance, and split column.')
    ap.add_argument('--outdir', default='static_candidate_model_v1/confusion_matrices',
                    help='Output directory.')
    ap.add_argument('--split', default='val',
                    help='Split to use, usually val. Use all to disable split filtering.')
    ap.add_argument('--thresholds', default='0.5,1.0,2.0',
                    help='Comma-separated mm thresholds for good/bad confusion matrices.')
    ap.add_argument('--event-col', default='event')
    ap.add_argument('--candidate-col', default='candidate_id')
    ap.add_argument('--source-col', default='candidate_source')
    ap.add_argument('--truth-col', default=None,
                    help='True distance column. Auto-detected by default.')
    ap.add_argument('--pred-col', default=None,
                    help='Predicted distance column. Auto-detected by default.')
    ap.add_argument('--split-col', default='split')
    ap.add_argument('--make-plots', action='store_true', help='Write PNG heatmaps with matplotlib.')
    ap.add_argument('--no-plots', action='store_true', help='Disable plots even if --make-plots is supplied.')
    args = ap.parse_args(argv)

    if not os.path.isfile(args.scored):
        raise IOError('input scored CSV does not exist: {0}'.format(args.scored))

    ensure_dir(args.outdir)

    d0 = pd.read_csv(args.scored)
    print('[INFO] read {0} rows from {1}'.format(len(d0), args.scored))

    for c in [args.event_col, args.candidate_col, args.source_col]:
        if c not in d0.columns:
            if c == args.source_col:
                print('[WARN] source column not found; source confusion will be skipped:', c)
            else:
                raise RuntimeError('Required column not found: {0}'.format(c))

    truth_col = find_column(
        d0,
        args.truth_col,
        ['dist_candidate_minus_truth', 'true_dist_mm', 'truth_dist_mm', 'candidate_truth_dist_mm',
         'dist_to_truth_mm', 'candidate_dist_to_truth_mm'],
        ['truth', 'dist'],
        'truth distance'
    )

    pred_col = find_column(
        d0,
        args.pred_col,
        ['pred_dist_mm', 'predicted_dist_mm', 'model_pred_dist_mm', 'static_model_pred_dist_mm',
         'predicted_distance_mm', 'pred_candidate_dist_mm'],
        ['pred', 'dist'],
        'predicted distance'
    )

    print('[INFO] truth distance column:', truth_col)
    print('[INFO] predicted distance column:', pred_col)

    d = d0.copy()
    if args.split.lower() != 'all':
        if args.split_col not in d.columns:
            raise RuntimeError('Requested split={0}, but split column not found: {1}'.format(args.split, args.split_col))
        d = d[d[args.split_col].astype(str) == str(args.split)].copy()
        print('[INFO] using split {0}: {1} rows'.format(args.split, len(d)))
    else:
        print('[INFO] using all rows')

    # Clean numeric columns.
    d[truth_col] = pd.to_numeric(d[truth_col], errors='coerce')
    d[pred_col] = pd.to_numeric(d[pred_col], errors='coerce')
    before = len(d)
    d = d[np.isfinite(d[truth_col].values) & np.isfinite(d[pred_col].values)].copy()
    print('[INFO] finite truth/pred rows: {0} / {1}'.format(len(d), before))

    if len(d) == 0:
        raise RuntimeError('No usable rows after filtering.')

    # Event-level oracle and ML-selected candidates.
    idx_oracle = d.groupby(args.event_col)[truth_col].idxmin()
    idx_ml = d.groupby(args.event_col)[pred_col].idxmin()

    oracle_cols = [args.event_col, args.candidate_col, truth_col]
    ml_cols = [args.event_col, args.candidate_col, pred_col, truth_col]
    source_ok = args.source_col in d.columns
    if source_ok:
        oracle_cols.insert(2, args.source_col)
        ml_cols.insert(2, args.source_col)

    oracle = d.loc[idx_oracle, oracle_cols].copy()
    ml = d.loc[idx_ml, ml_cols].copy()

    rename_oracle = {
        args.event_col: 'event',
        args.candidate_col: 'oracle_candidate_id',
        truth_col: 'oracle_true_dist_mm',
    }
    rename_ml = {
        args.event_col: 'event',
        args.candidate_col: 'ml_candidate_id',
        pred_col: 'ml_pred_dist_mm',
        truth_col: 'ml_true_dist_mm',
    }
    if source_ok:
        rename_oracle[args.source_col] = 'oracle_candidate_source'
        rename_ml[args.source_col] = 'ml_candidate_source'

    oracle = oracle.rename(columns=rename_oracle)
    ml = ml.rename(columns=rename_ml)

    event_table = oracle.merge(ml, on='event', how='inner')
    event_table['ml_minus_oracle_true_dist_mm'] = event_table['ml_true_dist_mm'] - event_table['oracle_true_dist_mm']

    event_out = os.path.join(args.outdir, 'oracle_vs_ml_selected_events.csv')
    event_table.to_csv(event_out, index=False)
    print('[INFO] wrote:', event_out)

    # Event summary.
    event_summary = summarize_event_selection(event_table)
    event_summary_df = pd.DataFrame([event_summary])
    event_summary_out = os.path.join(args.outdir, 'event_selection_summary.csv')
    event_summary_df.to_csv(event_summary_out, index=False)
    print('[INFO] wrote:', event_summary_out)

    print('\n[EVENT SELECTION SUMMARY]')
    for k in sorted(event_summary.keys()):
        print('  {0}: {1}'.format(k, event_summary[k]))

    make_plots = args.make_plots and not args.no_plots

    # Candidate-source confusion matrix.
    if source_ok:
        source_cm = pd.crosstab(
            event_table['oracle_candidate_source'],
            event_table['ml_candidate_source'],
            rownames=['truth_oracle_source'],
            colnames=['ml_selected_source'],
            margins=True
        )
        source_out = os.path.join(args.outdir, 'candidate_source_confusion_matrix_counts.csv')
        source_cm.to_csv(source_out)
        print('[INFO] wrote:', source_out)

        source_nomar = source_cm.copy()
        if 'All' in source_nomar.index:
            source_nomar = source_nomar.drop(index='All')
        if 'All' in source_nomar.columns:
            source_nomar = source_nomar.drop(columns=['All'])
        row_sums = source_nomar.sum(axis=1).replace(0, np.nan)
        source_frac = source_nomar.div(row_sums, axis=0)
        source_frac_out = os.path.join(args.outdir, 'candidate_source_confusion_matrix_row_fraction.csv')
        source_frac.to_csv(source_frac_out)
        print('[INFO] wrote:', source_frac_out)

        print('\n[CANDIDATE SOURCE CONFUSION MATRIX: counts]')
        print(source_cm.to_string())

        if make_plots:
            plot_matrix(source_cm, 'Oracle source vs ML-selected source',
                        os.path.join(args.outdir, 'candidate_source_confusion_matrix_counts.png'))
            plot_matrix(source_frac, 'Oracle source vs ML-selected source, row fraction',
                        os.path.join(args.outdir, 'candidate_source_confusion_matrix_row_fraction.png'))

    thresholds = parse_thresholds(args.thresholds)
    all_metrics_rows = []

    for thr in thresholds:
        name = safe_name(thr)

        cand_cm = crosstab_good_bad(d, truth_col, pred_col, thr)
        cand_out = os.path.join(args.outdir, 'candidate_level_good_bad_confusion_matrix_{0}mm.csv'.format(name))
        cand_cm.to_csv(cand_out)
        print('[INFO] wrote:', cand_out)

        cand_metrics = compute_binary_metrics(cand_cm)
        cand_metrics['threshold_mm'] = thr
        cand_metrics['level'] = 'candidate_level_all_rows'
        all_metrics_rows.append(cand_metrics)

        # Event-selected: only the candidate the ML would choose in each event.
        selected = event_table.copy()
        selected_tmp = pd.DataFrame({
            'actual_truth_dist_mm': selected['ml_true_dist_mm'].astype(float),
            'predicted_dist_mm': selected['ml_pred_dist_mm'].astype(float),
        })
        selected_tmp['actual'] = np.where(selected_tmp['actual_truth_dist_mm'] < thr, 'actual_good', 'actual_bad')
        selected_tmp['predicted'] = np.where(selected_tmp['predicted_dist_mm'] < thr, 'pred_good', 'pred_bad')
        selected_cm = pd.crosstab(selected_tmp['actual'], selected_tmp['predicted'],
                                  rownames=['actual_selected'], colnames=['predicted_selected'], margins=True)
        for row in ['actual_good', 'actual_bad']:
            if row not in selected_cm.index:
                selected_cm.loc[row] = 0
        for col in ['pred_good', 'pred_bad']:
            if col not in selected_cm.columns:
                selected_cm[col] = 0
        ordered_rows = ['actual_good', 'actual_bad']
        ordered_cols = ['pred_good', 'pred_bad']
        if 'All' in selected_cm.index:
            ordered_rows.append('All')
        if 'All' in selected_cm.columns:
            ordered_cols.append('All')
        selected_cm = selected_cm.loc[ordered_rows, ordered_cols]

        selected_out = os.path.join(args.outdir, 'event_selected_good_bad_confusion_matrix_{0}mm.csv'.format(name))
        selected_cm.to_csv(selected_out)
        print('[INFO] wrote:', selected_out)

        selected_metrics = compute_binary_metrics(selected_cm)
        selected_metrics['threshold_mm'] = thr
        selected_metrics['level'] = 'event_level_ml_selected_rows'
        all_metrics_rows.append(selected_metrics)

        print('\n[GOOD/BAD CONFUSION MATRIX: candidate level, threshold {0} mm]'.format(thr))
        print(cand_cm.to_string())
        print('\n[GOOD/BAD CONFUSION MATRIX: ML-selected events only, threshold {0} mm]'.format(thr))
        print(selected_cm.to_string())

        if make_plots:
            plot_matrix(cand_cm, 'Candidate-level good/bad, threshold {0} mm'.format(thr),
                        os.path.join(args.outdir, 'candidate_level_good_bad_confusion_matrix_{0}mm.png'.format(name)))
            plot_matrix(selected_cm, 'ML-selected good/bad, threshold {0} mm'.format(thr),
                        os.path.join(args.outdir, 'event_selected_good_bad_confusion_matrix_{0}mm.png'.format(name)))

    metrics_df = pd.DataFrame(all_metrics_rows)
    # Put the most readable columns first.
    preferred = ['level', 'threshold_mm', 'total', 'tp', 'tn', 'fp', 'fn',
                 'accuracy', 'precision_pred_good', 'recall_actual_good',
                 'specificity_actual_bad', 'f1_good']
    cols = [c for c in preferred if c in metrics_df.columns] + [c for c in metrics_df.columns if c not in preferred]
    metrics_df = metrics_df[cols]
    metrics_out = os.path.join(args.outdir, 'good_bad_confusion_metrics_summary.csv')
    metrics_df.to_csv(metrics_out, index=False)
    print('[INFO] wrote:', metrics_out)

    print('\n[DONE] outputs in:', args.outdir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
