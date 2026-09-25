#!/usr/bin/env python3
"""
summarize_static_candidate_oracle_misses.py

Event-level diagnostic for the static candidate model.

For each event:
  oracle = candidate with smallest true distance to truth
  ML     = candidate with smallest predicted distance

It reports:
  - how often the ML selected the exact oracle candidate
  - how often the selected candidate is within a tolerance of the oracle
  - how much true-distance "regret" the ML paid when it did not choose oracle
  - candidate-source agreement
  - true-rank of the ML-selected candidate within each event

Python 3.6 compatible.
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd


def _find_col(cols, preferred, must_contain_all=None, must_contain_any=None):
    cols = list(cols)
    for c in preferred:
        if c in cols:
            return c

    lower_map = {c: c.lower() for c in cols}
    if must_contain_all is None:
        must_contain_all = []
    if must_contain_any is None:
        must_contain_any = []

    matches = []
    for c in cols:
        lc = lower_map[c]
        ok = True
        for token in must_contain_all:
            if token.lower() not in lc:
                ok = False
                break
        if not ok:
            continue
        if must_contain_any:
            ok_any = False
            for token in must_contain_any:
                if token.lower() in lc:
                    ok_any = True
                    break
            if not ok_any:
                continue
        matches.append(c)

    if len(matches) == 0:
        return None
    return matches[0]


def _parse_float_list(s):
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if part == "":
            continue
        vals.append(float(part))
    return vals


def _ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def _safe_quantile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, q))


def _summary_rows(name, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [
            {"metric": name + "_n", "value": 0},
        ]
    rows = []
    rows.append({"metric": name + "_n", "value": int(len(values))})
    rows.append({"metric": name + "_mean", "value": float(np.mean(values))})
    rows.append({"metric": name + "_median", "value": float(np.median(values))})
    rows.append({"metric": name + "_rms", "value": float(np.sqrt(np.mean(values * values)))})
    rows.append({"metric": name + "_min", "value": float(np.min(values))})
    rows.append({"metric": name + "_q16", "value": _safe_quantile(values, 16)})
    rows.append({"metric": name + "_q68", "value": _safe_quantile(values, 68)})
    rows.append({"metric": name + "_q84", "value": _safe_quantile(values, 84)})
    rows.append({"metric": name + "_q90", "value": _safe_quantile(values, 90)})
    rows.append({"metric": name + "_q95", "value": _safe_quantile(values, 95)})
    rows.append({"metric": name + "_q99", "value": _safe_quantile(values, 99)})
    rows.append({"metric": name + "_max", "value": float(np.max(values))})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare ML-selected static candidate to truth-oracle closest candidate."
    )
    ap.add_argument(
        "--scored",
        required=True,
        help="Scored candidate CSV, usually static_candidate_model_v1/static_candidate_training_scored_rows.csv",
    )
    ap.add_argument(
        "--outdir",
        default="static_candidate_model_v1/oracle_miss_summary",
        help="Output directory.",
    )
    ap.add_argument(
        "--split",
        default="val",
        help="Split to use if split column exists. Use all for all rows. Default: val",
    )
    ap.add_argument(
        "--pred-col",
        default="auto",
        help="Predicted-distance column name, or auto.",
    )
    ap.add_argument(
        "--truth-col",
        default="auto",
        help="True-distance column name, or auto.",
    )
    ap.add_argument(
        "--event-col",
        default="event",
        help="Event column name.",
    )
    ap.add_argument(
        "--candidate-col",
        default="candidate_id",
        help="Candidate id column name.",
    )
    ap.add_argument(
        "--source-col",
        default="candidate_source",
        help="Candidate source column name.",
    )
    ap.add_argument(
        "--tolerances-mm",
        default="0,0.05,0.1,0.25,0.5,1,2,5",
        help="Comma-separated regret tolerances in mm. Regret = ML true distance - oracle true distance.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of worst missed events to print.",
    )
    ap.add_argument(
        "--make-plots",
        action="store_true",
        help="Also write regret/rank plots if matplotlib is available.",
    )
    ap.add_argument("--regret-xmax", type=float, default=2.0,
                    help="Maximum x value shown in regret histogram [mm]. Default: 2.0")
    ap.add_argument("--regret-bin-width", type=float, default=0.05,
                    help="Regret histogram bin width [mm]. Default: 0.05")
    ap.add_argument("--rank-collapse-after", type=int, default=4,
                    help="Collapse true ranks above this value into one overflow bin. Default: 4")
    ap.add_argument("--rank-regret-xmax", type=float, default=2.0,
                    help="Maximum non-oracle regret shown in rank/regret category plot [mm]. Default: 2.0")
    ap.add_argument("--rank-regret-bin-width", type=float, default=0.25,
                    help="Bin width for non-oracle regret categories [mm]. Default: 0.25")
    args = ap.parse_args(argv)

    _ensure_dir(args.outdir)

    if not os.path.exists(args.scored):
        raise IOError("input CSV does not exist: {0}".format(args.scored))

    d = pd.read_csv(args.scored)
    original_rows = len(d)

    if args.split.lower() != "all" and "split" in d.columns:
        d = d[d["split"].astype(str) == args.split].copy()

    if len(d) == 0:
        raise RuntimeError("No rows left after split filter: {0}".format(args.split))

    event_col = args.event_col
    cand_col = args.candidate_col
    source_col = args.source_col

    if event_col not in d.columns:
        raise RuntimeError("Could not find event column: {0}".format(event_col))

    if cand_col not in d.columns:
        # Fall back to a row-within-event id if candidate_id is missing.
        d[cand_col] = d.groupby(event_col).cumcount()

    if source_col not in d.columns:
        d[source_col] = "unknown"

    if args.pred_col != "auto":
        pred_col = args.pred_col
    else:
        pred_col = _find_col(
            d.columns,
            preferred=[
                "pred_dist_mm",
                "predicted_dist_mm",
                "model_pred_dist_mm",
                "pred_candidate_dist_mm",
                "pred_distance_mm",
            ],
            must_contain_all=["pred", "dist"],
        )
    if pred_col is None or pred_col not in d.columns:
        raise RuntimeError(
            "Could not find predicted-distance column. Use --pred-col. Columns are: {0}".format(
                list(d.columns)
            )
        )

    if args.truth_col != "auto":
        truth_col = args.truth_col
    else:
        truth_col = _find_col(
            d.columns,
            preferred=[
                "dist_candidate_minus_truth",
                "candidate_dist_to_truth_mm",
                "true_dist_mm",
                "truth_dist_mm",
                "dist_to_truth_mm",
            ],
            must_contain_all=["dist", "truth"],
        )
    if truth_col is None or truth_col not in d.columns:
        raise RuntimeError(
            "Could not find truth-distance column. Use --truth-col. Columns are: {0}".format(
                list(d.columns)
            )
        )

    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce")
    d[truth_col] = pd.to_numeric(d[truth_col], errors="coerce")
    d = d[np.isfinite(d[pred_col].values) & np.isfinite(d[truth_col].values)].copy()

    # Oracle = candidate with true minimum distance per event.
    idx_oracle = d.groupby(event_col)[truth_col].idxmin()
    oracle = d.loc[idx_oracle, [event_col, cand_col, source_col, truth_col, pred_col]].copy()
    oracle = oracle.rename(columns={
        cand_col: "oracle_candidate_id",
        source_col: "oracle_candidate_source",
        truth_col: "oracle_true_dist_mm",
        pred_col: "oracle_pred_dist_mm",
    })

    # ML = candidate with predicted minimum distance per event.
    idx_ml = d.groupby(event_col)[pred_col].idxmin()
    ml = d.loc[idx_ml, [event_col, cand_col, source_col, truth_col, pred_col]].copy()
    ml = ml.rename(columns={
        cand_col: "ml_candidate_id",
        source_col: "ml_candidate_source",
        truth_col: "ml_true_dist_mm",
        pred_col: "ml_pred_dist_mm",
    })

    # Compute true-rank of every candidate within event. Rank 1 means closest to truth.
    d_rank = d[[event_col, cand_col, truth_col, pred_col, source_col]].copy()
    d_rank["true_rank_in_event"] = d_rank.groupby(event_col)[truth_col].rank(
        method="min", ascending=True
    )
    d_rank["pred_rank_in_event"] = d_rank.groupby(event_col)[pred_col].rank(
        method="min", ascending=True
    )
    ranks = d_rank[[event_col, cand_col, "true_rank_in_event", "pred_rank_in_event"]].copy()
    ml = ml.merge(
        ranks.rename(columns={
            cand_col: "ml_candidate_id",
            "true_rank_in_event": "ml_true_rank_in_event",
            "pred_rank_in_event": "ml_pred_rank_in_event",
        }),
        on=[event_col, "ml_candidate_id"],
        how="left",
    )

    m = oracle.merge(ml, on=event_col, how="inner")

    # Add event-level counts and oracle tie count.
    g = d.groupby(event_col)
    event_n_candidates = g.size().rename("n_candidates").reset_index()
    m = m.merge(event_n_candidates, on=event_col, how="left")

    m["regret_mm"] = m["ml_true_dist_mm"] - m["oracle_true_dist_mm"]
    m["abs_regret_mm"] = np.abs(m["regret_mm"])
    m["same_candidate_id"] = m["ml_candidate_id"].astype(str) == m["oracle_candidate_id"].astype(str)
    m["same_candidate_source"] = m["ml_candidate_source"].astype(str) == m["oracle_candidate_source"].astype(str)

    # How many candidates are tied with oracle within tiny epsilon?
    eps_tie = 1.0e-9
    oracle_dist_map = dict(zip(m[event_col], m["oracle_true_dist_mm"]))
    d_tmp = d[[event_col, truth_col]].copy()
    d_tmp["oracle_true_dist_mm"] = d_tmp[event_col].map(oracle_dist_map)
    d_tmp["is_oracle_tie"] = np.abs(d_tmp[truth_col] - d_tmp["oracle_true_dist_mm"]) <= eps_tie
    tie_counts = d_tmp.groupby(event_col)["is_oracle_tie"].sum().rename("n_exact_oracle_ties").reset_index()
    m = m.merge(tie_counts, on=event_col, how="left")

    tolerances = _parse_float_list(args.tolerances_mm)
    for tol in tolerances:
        key = "within_{0:g}mm_of_oracle".format(tol).replace(".", "p")
        m[key] = m["regret_mm"] <= tol + 1.0e-12

    # Write event table sorted by worst regret.
    out_events = os.path.join(args.outdir, "ml_vs_oracle_by_event.csv")
    m.sort_values("regret_mm", ascending=False).to_csv(out_events, index=False)

    # Source crosstab.
    source_cm = pd.crosstab(
        m["oracle_candidate_source"],
        m["ml_candidate_source"],
        rownames=["oracle_source"],
        colnames=["ml_source"],
        margins=True,
    )
    source_cm.to_csv(os.path.join(args.outdir, "oracle_vs_ml_source_counts.csv"))

    source_rowfrac = source_cm.copy().astype(float)
    if "All" in source_rowfrac.index:
        source_rowfrac_noall = source_rowfrac.drop(index=["All"])
    else:
        source_rowfrac_noall = source_rowfrac
    if "All" in source_rowfrac_noall.columns:
        source_rowfrac_noall = source_rowfrac_noall.drop(columns=["All"])
    denom = source_rowfrac_noall.sum(axis=1).replace(0, np.nan)
    source_rowfrac_noall = source_rowfrac_noall.div(denom, axis=0)
    source_rowfrac_noall.to_csv(os.path.join(args.outdir, "oracle_vs_ml_source_row_fraction.csv"))

    # Rank counts.
    rank_counts = m["ml_true_rank_in_event"].value_counts().sort_index()
    rank_counts.to_csv(os.path.join(args.outdir, "ml_selected_true_rank_counts.csv"), header=["count"])

    # Summary metrics.
    rows = []
    n_events = len(m)
    rows.append({"metric": "input_rows_before_split_filter", "value": int(original_rows)})
    rows.append({"metric": "candidate_rows_used", "value": int(len(d))})
    rows.append({"metric": "events_used", "value": int(n_events)})
    rows.append({"metric": "split", "value": str(args.split)})
    rows.append({"metric": "pred_col", "value": str(pred_col)})
    rows.append({"metric": "truth_col", "value": str(truth_col)})

    rows.append({"metric": "same_candidate_id_count", "value": int(m["same_candidate_id"].sum())})
    rows.append({"metric": "same_candidate_id_fraction", "value": float(m["same_candidate_id"].mean())})
    rows.append({"metric": "same_candidate_source_count", "value": int(m["same_candidate_source"].sum())})
    rows.append({"metric": "same_candidate_source_fraction", "value": float(m["same_candidate_source"].mean())})

    rows.append({"metric": "ml_true_rank_1_count", "value": int((m["ml_true_rank_in_event"] <= 1).sum())})
    rows.append({"metric": "ml_true_rank_1_fraction", "value": float((m["ml_true_rank_in_event"] <= 1).mean())})
    rows.append({"metric": "ml_true_rank_le2_count", "value": int((m["ml_true_rank_in_event"] <= 2).sum())})
    rows.append({"metric": "ml_true_rank_le2_fraction", "value": float((m["ml_true_rank_in_event"] <= 2).mean())})
    rows.append({"metric": "ml_true_rank_le3_count", "value": int((m["ml_true_rank_in_event"] <= 3).sum())})
    rows.append({"metric": "ml_true_rank_le3_fraction", "value": float((m["ml_true_rank_in_event"] <= 3).mean())})

    for tol in tolerances:
        key = "within_{0:g}mm_of_oracle".format(tol).replace(".", "p")
        rows.append({"metric": key + "_count", "value": int(m[key].sum())})
        rows.append({"metric": key + "_fraction", "value": float(m[key].mean())})

    for tol in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        rows.append({"metric": "regret_gt_{0:g}mm_count".format(tol).replace(".", "p"), "value": int((m["regret_mm"] > tol).sum())})
        rows.append({"metric": "regret_gt_{0:g}mm_fraction".format(tol).replace(".", "p"), "value": float((m["regret_mm"] > tol).mean())})

    rows.extend(_summary_rows("oracle_true_dist_mm", m["oracle_true_dist_mm"].values))
    rows.extend(_summary_rows("ml_true_dist_mm", m["ml_true_dist_mm"].values))
    rows.extend(_summary_rows("regret_mm", m["regret_mm"].values))
    rows.extend(_summary_rows("ml_pred_dist_mm", m["ml_pred_dist_mm"].values))
    rows.extend(_summary_rows("n_candidates_per_event", m["n_candidates"].values))

    summary = pd.DataFrame(rows)
    out_summary = os.path.join(args.outdir, "ml_vs_oracle_summary.csv")
    summary.to_csv(out_summary, index=False)

    worst_cols = [
        event_col,
        "n_candidates",
        "oracle_candidate_id",
        "ml_candidate_id",
        "oracle_candidate_source",
        "ml_candidate_source",
        "oracle_true_dist_mm",
        "ml_true_dist_mm",
        "regret_mm",
        "ml_pred_dist_mm",
        "oracle_pred_dist_mm",
        "ml_true_rank_in_event",
    ]
    worst = m.sort_values("regret_mm", ascending=False).head(args.top_n)
    out_worst = os.path.join(args.outdir, "worst_ml_oracle_misses_top{0}.csv".format(args.top_n))
    worst[worst_cols].to_csv(out_worst, index=False)

    # Optional plots.
    if args.make_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Use one clean event table for all plots so ranks/regrets line up.
            plot_m = m[["regret_mm", "ml_true_rank_in_event"]].copy()
            plot_m = plot_m[np.isfinite(plot_m["regret_mm"].values)]
            plot_m = plot_m[np.isfinite(plot_m["ml_true_rank_in_event"].values)]
            regret = plot_m["regret_mm"].values.astype(float)
            ranks = plot_m["ml_true_rank_in_event"].values.astype(int)
            n_plot = len(plot_m)

            # ------------------------------------------------------------
            # 1. Regret histogram: zoomed, fixed-width bins, overflow label
            # ------------------------------------------------------------
            regret_xmax = float(args.regret_xmax)
            regret_bin_width = float(args.regret_bin_width)
            if regret_xmax <= 0.0:
                regret_xmax = 2.0
            if regret_bin_width <= 0.0:
                regret_bin_width = 0.05

            in_view = regret[(regret >= 0.0) & (regret <= regret_xmax)]
            n_in_view = len(in_view)
            n_overflow = int(np.sum(regret > regret_xmax))
            bins = np.arange(0.0, regret_xmax + regret_bin_width, regret_bin_width)

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.hist(in_view, bins=bins)
            ax.set_xlim(0.0, regret_xmax)
            ax.set_xlabel("ML true distance - oracle true distance [mm]")
            ax.set_ylabel("Events")
            ax.set_title("Static ML regret relative to truth oracle")
            text_box = "shown: {0}/{1} events\n> {2:g} mm: {3} events".format(
                n_in_view, n_plot, regret_xmax, n_overflow
            )
            ax.text(0.98, 0.95, text_box, transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            plt.savefig(os.path.join(args.outdir, "regret_histogram.png"), dpi=150)
            plt.close(fig)

            # ------------------------------------------------------------
            # 2. True-rank plot: discrete bar chart with percentages
            # ------------------------------------------------------------
            rank_collapse_after = int(args.rank_collapse_after)
            if rank_collapse_after < 2:
                rank_collapse_after = 4

            labels = []
            counts = []
            for rnk in range(1, rank_collapse_after + 1):
                labels.append(str(rnk))
                counts.append(int(np.sum(ranks == rnk)))
            labels.append("{0}+".format(rank_collapse_after + 1))
            counts.append(int(np.sum(ranks > rank_collapse_after)))
            percents = [100.0 * c / n_plot if n_plot > 0 else 0.0 for c in counts]

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            x = np.arange(len(labels))
            ax.bar(x, counts)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Truth rank of ML-selected candidate")
            ax.set_ylabel("Events")
            ax.set_title("Truth rank of ML-selected candidate")
            ax.text(0.5, 0.96, "Rank 1 = exact oracle; higher rank can still have tiny regret",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9)
            ymax = max(counts) if counts else 1
            ax.set_ylim(0, 1.15 * ymax)
            for xi, c, pct in zip(x, counts, percents):
                ax.text(xi, c + 0.025 * ymax, "{0}\n({1:.1f}%)".format(c, pct),
                        ha="center", va="bottom", fontsize=9)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            plt.savefig(os.path.join(args.outdir, "ml_selected_true_rank_histogram.png"), dpi=150)
            plt.close(fig)

            # ------------------------------------------------------------
            # 3. Rank/regret interpretation categories with binned regret
            # ------------------------------------------------------------
            rank_regret_xmax = float(args.rank_regret_xmax)
            rank_regret_bin_width = float(args.rank_regret_bin_width)
            if rank_regret_xmax <= 0.0:
                rank_regret_xmax = 2.0
            if rank_regret_bin_width <= 0.0:
                rank_regret_bin_width = 0.25

            exact_oracle = ranks == 1
            non_oracle = ranks > 1
            cat_labels = ["Exact oracle\nrank 1"]
            cat_counts = [int(np.sum(exact_oracle))]

            edges = np.arange(0.0, rank_regret_xmax + rank_regret_bin_width, rank_regret_bin_width)
            for i in range(len(edges) - 1):
                lo = edges[i]
                hi = edges[i + 1]
                if i == 0:
                    mask = non_oracle & (regret >= lo) & (regret <= hi)
                    label = "Not rank 1\n0-{0:g} mm".format(hi)
                else:
                    mask = non_oracle & (regret > lo) & (regret <= hi)
                    label = "Not rank 1\n{0:g}-{1:g} mm".format(lo, hi)
                cat_labels.append(label)
                cat_counts.append(int(np.sum(mask)))
            cat_labels.append("Not rank 1\n> {0:g} mm".format(rank_regret_xmax))
            cat_counts.append(int(np.sum(non_oracle & (regret > rank_regret_xmax))))
            cat_percents = [100.0 * c / n_plot if n_plot > 0 else 0.0 for c in cat_counts]

            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(111)
            x = np.arange(len(cat_labels))
            ax.bar(x, cat_counts)
            ax.set_xticks(x)
            ax.set_xticklabels(cat_labels, rotation=30, ha="right")
            ax.set_ylabel("Events")
            ax.set_title("Oracle-rank result with regret bins")
            ymax = max(cat_counts) if cat_counts else 1
            ax.set_ylim(0, 1.18 * ymax)
            for xi, c, pct in zip(x, cat_counts, cat_percents):
                ax.text(xi, c + 0.025 * ymax, "{0}\n({1:.1f}%)".format(c, pct),
                        ha="center", va="bottom", fontsize=8)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            plt.savefig(os.path.join(args.outdir, "rank_regret_interpretation.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            sys.stderr.write("[WARN] could not make plots: {0}\n".format(e))

    # Console summary.
    print("")
    print("STATIC CANDIDATE ML VS TRUTH ORACLE")
    print("input:        {0}".format(args.scored))
    print("split:        {0}".format(args.split))
    print("rows used:    {0}".format(len(d)))
    print("events used:  {0}".format(n_events))
    print("pred col:     {0}".format(pred_col))
    print("truth col:    {0}".format(truth_col))
    print("")
    print("How often did ML select the truly closest candidate?")
    print("  exact same candidate_id: {0}/{1} = {2:.6f}".format(
        int(m["same_candidate_id"].sum()), n_events, float(m["same_candidate_id"].mean())
    ))
    print("  same candidate source:   {0}/{1} = {2:.6f}".format(
        int(m["same_candidate_source"].sum()), n_events, float(m["same_candidate_source"].mean())
    ))
    print("  true-rank <= 1:          {0}/{1} = {2:.6f}".format(
        int((m["ml_true_rank_in_event"] <= 1).sum()), n_events, float((m["ml_true_rank_in_event"] <= 1).mean())
    ))
    print("  true-rank <= 2:          {0}/{1} = {2:.6f}".format(
        int((m["ml_true_rank_in_event"] <= 2).sum()), n_events, float((m["ml_true_rank_in_event"] <= 2).mean())
    ))
    print("")
    print("How far off was ML from oracle?  regret = ml_true_dist - oracle_true_dist")
    print("  median regret: {0:.6f} mm".format(float(np.median(m["regret_mm"].values))))
    print("  mean regret:   {0:.6f} mm".format(float(np.mean(m["regret_mm"].values))))
    print("  q84 regret:    {0:.6f} mm".format(_safe_quantile(m["regret_mm"].values, 84)))
    print("  q95 regret:    {0:.6f} mm".format(_safe_quantile(m["regret_mm"].values, 95)))
    print("  max regret:    {0:.6f} mm".format(float(np.max(m["regret_mm"].values))))
    print("")
    for tol in tolerances:
        key = "within_{0:g}mm_of_oracle".format(tol).replace(".", "p")
        print("  within {0:g} mm of oracle: {1}/{2} = {3:.6f}".format(
            tol, int(m[key].sum()), n_events, float(m[key].mean())
        ))

    print("")
    print("Distances:")
    print("  oracle median true distance: {0:.6f} mm".format(float(np.median(m["oracle_true_dist_mm"].values))))
    print("  ML median true distance:     {0:.6f} mm".format(float(np.median(m["ml_true_dist_mm"].values))))
    print("  oracle mean true distance:   {0:.6f} mm".format(float(np.mean(m["oracle_true_dist_mm"].values))))
    print("  ML mean true distance:       {0:.6f} mm".format(float(np.mean(m["ml_true_dist_mm"].values))))
    print("")
    print("Worst misses:")
    print(worst[worst_cols].to_string(index=False))
    print("")
    print("Wrote:")
    print("  {0}".format(out_summary))
    print("  {0}".format(out_events))
    print("  {0}".format(out_worst))
    print("  {0}".format(os.path.join(args.outdir, "oracle_vs_ml_source_counts.csv")))
    print("  {0}".format(os.path.join(args.outdir, "ml_selected_true_rank_counts.csv")))
    if args.make_plots:
        print("  {0}".format(os.path.join(args.outdir, "regret_histogram.png")))
        print("  {0}".format(os.path.join(args.outdir, "ml_selected_true_rank_histogram.png")))

    return 0


if __name__ == "__main__":
    sys.exit(main())
