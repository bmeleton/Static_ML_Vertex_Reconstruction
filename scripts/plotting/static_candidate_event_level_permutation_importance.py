#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
static_candidate_event_level_permutation_importance.py

Held-out event-level permutation importance for the static candidate ML model.

This is different from the model's built-in ExtraTrees feature_importances_.
The built-in importance is training/model-internal. This script asks a more
physical question on a held-out feature CSV:

  If I randomly destroy one input feature, how much worse does the final
  event-level vertex selection become?

For each feature:
  1. Shuffle that raw input column across candidate rows.
  2. Re-run the trained model predictions.
  3. Select one candidate per event by minimum predicted distance.
  4. Compare event-level selected true distance/regret against baseline.

Outputs:
  permutation_importance_summary.csv
  permutation_importance_repeats.csv
  baseline_event_metrics.csv
  permutation_importance_top_mean_dist.png
  permutation_importance_top_oracle_frac.png
  permutation_importance_top_regret_q95.png

Run from the same directory as train_static_candidate_model.py so the same
preprocessing helpers can be imported.

Python 3.6 compatible.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    from sklearn.externals import joblib

try:
    from train_static_candidate_model import make_feature_matrix, align_matrix
except Exception as exc:
    sys.exit("[ERROR] Could not import train_static_candidate_model.py helpers: {0}".format(exc))


def parse_args():
    ap = argparse.ArgumentParser(
        description="Held-out event-level permutation importance for static candidate model."
    )
    ap.add_argument("--model", required=True,
                    help="static_candidate_quality_model.pkl")
    ap.add_argument("--features", required=True,
                    help="Held-out feature CSV, usually *_test*_generic.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--top-n", type=int, default=40,
                    help="Number of features shown in plots.")
    ap.add_argument("--top-model-features", type=int, default=80,
                    help=("Only test the top N raw features by grouped model impurity importance. "
                          "Use 0 to test all model input features. Default 80."))
    ap.add_argument("--feature-list", default=None,
                    help="Optional comma-separated list of raw feature columns to test.")
    ap.add_argument("--min-candidates-per-event", type=int, default=1,
                    help="Minimum finite candidates per event kept for evaluation. Default 1.")
    ap.add_argument("--target-col", default=None,
                    help="Truth-distance column. Default auto: target_dist_mm or dist_candidate_minus_truth.")
    ap.add_argument("--pred-tiebreak-col", default="candidate_score",
                    help="Tie breaker after predicted distance. Default candidate_score.")
    ap.add_argument("--make-plots", action="store_true")
    return ap.parse_args()


def ensure_target(df, target_col):
    if target_col is not None:
        if target_col not in df.columns:
            sys.exit("[ERROR] requested --target-col not found: {0}".format(target_col))
        df["target_dist_mm"] = pd.to_numeric(df[target_col], errors="coerce")
        return df

    if "target_dist_mm" in df.columns:
        df["target_dist_mm"] = pd.to_numeric(df["target_dist_mm"], errors="coerce")
        return df

    if "dist_candidate_minus_truth" in df.columns:
        df["target_dist_mm"] = pd.to_numeric(df["dist_candidate_minus_truth"], errors="coerce")
        return df

    sys.exit("[ERROR] Feature CSV has no target_dist_mm or dist_candidate_minus_truth. Need held-out simulation truth.")


def score_df(df, bundle):
    X, _, _, _ = make_feature_matrix(
        df,
        feature_columns=bundle["feature_columns"],
        categorical_columns=bundle["categorical_columns"],
        medians=bundle["medians"],
    )
    X = align_matrix(X, bundle["model_columns"])
    pred_log = bundle["model"].predict(X.values)
    out = df.copy()
    out["pred_log1p_dist"] = pred_log
    out["pred_dist_mm"] = np.expm1(pred_log)
    return out


def choose_selected(scored, pred_tiebreak_col):
    sort_cols = ["event", "pred_dist_mm"]
    ascending = [True, True]
    if pred_tiebreak_col in scored.columns:
        sort_cols.append(pred_tiebreak_col)
        ascending.append(False)
    elif "candidate_score" in scored.columns:
        sort_cols.append("candidate_score")
        ascending.append(False)

    selected = scored.sort_values(sort_cols, ascending=ascending).groupby("event", as_index=False).first()
    return selected


def choose_oracle(scored):
    sort_cols = ["event", "target_dist_mm"]
    ascending = [True, True]
    if "candidate_score" in scored.columns:
        sort_cols.append("candidate_score")
        ascending.append(False)
    return scored.sort_values(sort_cols, ascending=ascending).groupby("event", as_index=False).first()


def metric_row(scored, label, pred_tiebreak_col):
    selected = choose_selected(scored, pred_tiebreak_col)
    oracle = choose_oracle(scored)

    keep_cols = ["event", "target_dist_mm"]
    if "candidate_id" in selected.columns and "candidate_id" in oracle.columns:
        keep_cols_sel = ["event", "candidate_id", "target_dist_mm", "pred_dist_mm"]
        keep_cols_orc = ["event", "candidate_id", "target_dist_mm"]
        m = selected[keep_cols_sel].merge(
            oracle[keep_cols_orc],
            on="event",
            suffixes=("_selected", "_oracle")
        )
        same_oracle = (m["candidate_id_selected"].astype(str) == m["candidate_id_oracle"].astype(str))
    else:
        m = selected[["event", "target_dist_mm", "pred_dist_mm"]].merge(
            oracle[["event", "target_dist_mm"]],
            on="event",
            suffixes=("_selected", "_oracle")
        )
        same_oracle = pd.Series([False] * len(m))

    selected_dist = pd.to_numeric(m["target_dist_mm_selected"], errors="coerce")
    oracle_dist = pd.to_numeric(m["target_dist_mm_oracle"], errors="coerce")
    regret = selected_dist - oracle_dist

    finite = np.isfinite(selected_dist.values) & np.isfinite(oracle_dist.values) & np.isfinite(regret.values)
    selected_dist = selected_dist[finite]
    oracle_dist = oracle_dist[finite]
    regret = regret[finite]
    same_oracle = same_oracle[finite]

    row = {
        "label": label,
        "n_events": int(len(selected_dist)),
        "selected_median_dist_mm": float(selected_dist.median()) if len(selected_dist) else np.nan,
        "selected_mean_dist_mm": float(selected_dist.mean()) if len(selected_dist) else np.nan,
        "selected_rmse_dist_mm": float(np.sqrt(np.mean(selected_dist.values * selected_dist.values))) if len(selected_dist) else np.nan,
        "selected_frac_lt_0p5mm": float((selected_dist < 0.5).mean()) if len(selected_dist) else np.nan,
        "selected_frac_lt_1p0mm": float((selected_dist < 1.0).mean()) if len(selected_dist) else np.nan,
        "selected_frac_lt_2p0mm": float((selected_dist < 2.0).mean()) if len(selected_dist) else np.nan,
        "selected_frac_lt_5p0mm": float((selected_dist < 5.0).mean()) if len(selected_dist) else np.nan,
        "oracle_median_dist_mm": float(oracle_dist.median()) if len(oracle_dist) else np.nan,
        "oracle_mean_dist_mm": float(oracle_dist.mean()) if len(oracle_dist) else np.nan,
        "same_oracle_frac": float(same_oracle.mean()) if len(same_oracle) else np.nan,
        "regret_median_mm": float(regret.median()) if len(regret) else np.nan,
        "regret_mean_mm": float(regret.mean()) if len(regret) else np.nan,
        "regret_q95_mm": float(np.percentile(regret.values, 95)) if len(regret) else np.nan,
        "regret_max_mm": float(regret.max()) if len(regret) else np.nan,
    }
    return row


def grouped_model_importance(bundle):
    model = bundle.get("model")
    model_columns = list(bundle.get("model_columns", []))
    raw_features = list(bundle.get("feature_columns", []))
    cat_cols = set(bundle.get("categorical_columns", []))

    if (not hasattr(model, "feature_importances_")) or len(model_columns) == 0:
        return dict((f, np.nan) for f in raw_features)

    imp = np.asarray(model.feature_importances_, dtype=float)
    out = dict((f, 0.0) for f in raw_features)

    for col, val in zip(model_columns, imp):
        assigned = False

        if col in out:
            out[col] += float(val)
            assigned = True

        if not assigned:
            for cat in cat_cols:
                prefix = cat + "_"
                if col.startswith(prefix):
                    out[cat] = out.get(cat, 0.0) + float(val)
                    assigned = True
                    break

        if not assigned:
            # Last resort for unknown encoded columns.
            out[col] = out.get(col, 0.0) + float(val)

    return out


def select_features_to_test(df, bundle, args):
    raw_features = [f for f in bundle["feature_columns"] if f in df.columns]

    if args.feature_list:
        requested = [x.strip() for x in args.feature_list.split(",") if x.strip()]
        missing = [x for x in requested if x not in df.columns]
        if missing:
            print("[WARN] requested features missing from CSV:", ", ".join(missing))
        return [x for x in requested if x in df.columns]

    gimp = grouped_model_importance(bundle)
    raw_features = sorted(raw_features, key=lambda x: gimp.get(x, -1.0), reverse=True)

    if int(args.top_model_features) > 0:
        raw_features = raw_features[:int(args.top_model_features)]

    return raw_features


def plot_bar(df, value_col, title, out_png, top_n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[WARN] could not import matplotlib for plots:", exc)
        return

    d = df.sort_values(value_col, ascending=False).head(int(top_n)).copy()
    if len(d) == 0:
        return

    # Reverse so the largest appears at top in horizontal bar plot.
    d = d.iloc[::-1].copy()

    fig = plt.figure(figsize=(10, max(5, 0.24 * len(d) + 1.5)))
    ax = fig.add_subplot(111)
    y = np.arange(len(d))
    ax.barh(y, d[value_col].values)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"].values, fontsize=8)
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    bundle = joblib.load(args.model)
    df = pd.read_csv(args.features)
    print("[INFO] read features rows={0} events={1}".format(len(df), df["event"].nunique() if "event" in df.columns else "NA"))

    if "event" not in df.columns:
        sys.exit("[ERROR] Feature CSV must have event column")

    df = ensure_target(df, args.target_col)
    df = df[np.isfinite(df["target_dist_mm"].values)].copy()

    counts = df.groupby("event").size()
    keep_events = set(counts[counts >= int(args.min_candidates_per_event)].index.astype(int))
    df = df[df["event"].astype(int).isin(keep_events)].copy()

    print("[INFO] evaluation rows after target/event filters={0} events={1}".format(len(df), df["event"].nunique()))

    baseline_scored = score_df(df, bundle)
    baseline = metric_row(baseline_scored, "baseline", args.pred_tiebreak_col)

    pd.DataFrame([baseline]).to_csv(os.path.join(args.outdir, "baseline_event_metrics.csv"), index=False)
    baseline_scored.to_csv(os.path.join(args.outdir, "baseline_scored_rows.csv"), index=False)

    features = select_features_to_test(df, bundle, args)
    gimp = grouped_model_importance(bundle)

    print("[INFO] testing {0} raw features".format(len(features)))
    print("[INFO] baseline selected mean dist = {0:.6g} mm".format(baseline["selected_mean_dist_mm"]))
    print("[INFO] baseline same oracle frac = {0:.6g}".format(baseline["same_oracle_frac"]))
    print("[INFO] baseline regret q95 = {0:.6g} mm".format(baseline["regret_q95_mm"]))

    repeat_rows = []
    rng = np.random.RandomState(int(args.seed))

    for ifeat, feat in enumerate(features):
        values0 = df[feat].values.copy()

        for rep in range(int(args.n_repeats)):
            dperm = df.copy()
            order = rng.permutation(len(dperm))
            dperm[feat] = values0[order]

            scored = score_df(dperm, bundle)
            row = metric_row(scored, feat, args.pred_tiebreak_col)
            row["feature"] = feat
            row["repeat"] = int(rep)
            row["grouped_model_importance"] = float(gimp.get(feat, np.nan))
            repeat_rows.append(row)

        if (ifeat + 1) % 10 == 0 or (ifeat + 1) == len(features):
            print("[INFO] finished {0}/{1}: {2}".format(ifeat + 1, len(features), feat))

    reps = pd.DataFrame(repeat_rows)

    # Add degradation columns. Positive means permutation made event-level selection worse.
    for c in [
        "selected_median_dist_mm",
        "selected_mean_dist_mm",
        "selected_rmse_dist_mm",
        "regret_median_mm",
        "regret_mean_mm",
        "regret_q95_mm",
        "regret_max_mm",
    ]:
        reps["delta_" + c] = reps[c] - float(baseline[c])

    for c in [
        "selected_frac_lt_0p5mm",
        "selected_frac_lt_1p0mm",
        "selected_frac_lt_2p0mm",
        "selected_frac_lt_5p0mm",
        "same_oracle_frac",
    ]:
        reps["drop_" + c] = float(baseline[c]) - reps[c]

    repeat_path = os.path.join(args.outdir, "permutation_importance_repeats.csv")
    reps.to_csv(repeat_path, index=False)

    agg_spec = {}
    numeric_cols = [c for c in reps.columns if c not in ["feature", "label"]]
    for c in numeric_cols:
        if c == "repeat":
            continue
        agg_spec[c + "_mean"] = (c, "mean")
        agg_spec[c + "_std"] = (c, "std")

    # Python 3.6 pandas supports named aggregation only in later versions inconsistently.
    grouped = []
    for feat, g in reps.groupby("feature"):
        row = {"feature": feat}
        row["n_repeats"] = int(len(g))
        row["grouped_model_importance"] = float(g["grouped_model_importance"].iloc[0]) if "grouped_model_importance" in g.columns else np.nan
        for c in numeric_cols:
            if c in ["repeat", "grouped_model_importance"]:
                continue
            vals = pd.to_numeric(g[c], errors="coerce")
            vals = vals[np.isfinite(vals)]
            row[c + "_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[c + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        grouped.append(row)

    summary = pd.DataFrame(grouped)
    if "delta_selected_mean_dist_mm_mean" in summary.columns:
        summary.sort_values("delta_selected_mean_dist_mm_mean", ascending=False, inplace=True)

    summary_path = os.path.join(args.outdir, "permutation_importance_summary.csv")
    summary.to_csv(summary_path, index=False)

    if args.make_plots:
        plot_bar(
            summary,
            "delta_selected_mean_dist_mm_mean",
            "Held-out event-level permutation importance: mean selected distance",
            os.path.join(args.outdir, "permutation_importance_top_mean_dist.png"),
            args.top_n,
        )
        plot_bar(
            summary,
            "drop_same_oracle_frac_mean",
            "Held-out event-level permutation importance: oracle-selection fraction",
            os.path.join(args.outdir, "permutation_importance_top_oracle_frac.png"),
            args.top_n,
        )
        plot_bar(
            summary,
            "delta_regret_q95_mm_mean",
            "Held-out event-level permutation importance: q95 regret",
            os.path.join(args.outdir, "permutation_importance_top_regret_q95.png"),
            args.top_n,
        )

    print("")
    print("[INFO] wrote", os.path.join(args.outdir, "baseline_event_metrics.csv"))
    print("[INFO] wrote", repeat_path)
    print("[INFO] wrote", summary_path)
    print("")
    print("[BASELINE]")
    print(pd.DataFrame([baseline]).to_string(index=False))
    print("")
    print("[TOP PERMUTATION IMPORTANCE BY MEAN DISTANCE DEGRADATION]")
    cols = [
        "feature",
        "delta_selected_mean_dist_mm_mean",
        "delta_selected_median_dist_mm_mean",
        "drop_same_oracle_frac_mean",
        "delta_regret_q95_mm_mean",
        "grouped_model_importance",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
