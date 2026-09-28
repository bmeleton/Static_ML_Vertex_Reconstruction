from __future__ import print_function

"""
apply_static_candidate_model.py

Apply a trained static candidate-quality model to static_candidate_features.csv.
Chooses the candidate with the smallest predicted distance per event.
Works with or without truth columns; when truth exists, writes benchmark summaries.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    from sklearn.externals import joblib

# Import feature matrix helpers from trainer to ensure identical preprocessing.
try:
    from train_static_candidate_model import make_feature_matrix, align_matrix
except Exception as exc:
    sys.exit("[ERROR] Could not import train_static_candidate_model.py helper functions: %s" % exc)


def read_features(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "static_candidate_features.csv"), recursive=True)))
        else:
            files.extend(sorted(glob.glob(p)))
    files = sorted(set(files))
    if not files:
        sys.exit("[ERROR] No feature CSVs found")
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["_source_file"] = f
        dfs.append(d)
        print("[INFO] read %s rows=%d" % (f, len(d)))
    return pd.concat(dfs, ignore_index=True, sort=False)


def choose_by(df, sort_cols, ascending, name):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.sort_values(sort_cols, ascending=ascending).groupby("event", as_index=False).first()
    out["selection_rule"] = name
    return out


def summarize_selection(selected, all_df, label):
    row = dict(selection_rule=label, n_events=int(len(selected)), n_candidates=int(len(all_df)))
    if "target_dist_mm" in selected.columns:
        dist = pd.to_numeric(selected["target_dist_mm"], errors="coerce")
    elif "dist_candidate_minus_truth" in selected.columns:
        dist = pd.to_numeric(selected["dist_candidate_minus_truth"], errors="coerce")
    else:
        return row
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0:
        return row
    row["median_dist_mm"] = float(dist.median())
    row["mean_dist_mm"] = float(dist.mean())
    row["rms_dist_mm"] = float(np.sqrt(np.mean(dist.values * dist.values)))
    row["max_dist_mm"] = float(dist.max())
    for thr in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        row["frac_lt_%smm" % str(thr).replace(".", "p")] = float((dist < thr).mean())
    for ax in ["dx_candidate_minus_truth", "dy_candidate_minus_truth", "dz_candidate_minus_truth"]:
        if ax in selected.columns:
            v = pd.to_numeric(selected[ax], errors="coerce")
            v = v[np.isfinite(v)]
            if len(v):
                short = ax.split("_")[0]
                row[short + "_median_mm"] = float(v.median())
                row[short + "_fwhm68_mm"] = float(np.percentile(v, 84) - np.percentile(v, 16))
    return row


def parse_args():
    ap = argparse.ArgumentParser(description="Apply static candidate-quality model and select best candidate per event.")
    ap.add_argument("--model", required=True, help="static_candidate_quality_model.pkl")
    ap.add_argument("--features", nargs="+", required=True, help="Feature CSVs, dirs, or glob patterns")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--write-selected-vertices", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    bundle = joblib.load(args.model)
    df = read_features(args.features)
    if "event" not in df.columns:
        sys.exit("[ERROR] features must have event column")
    if "target_dist_mm" not in df.columns and "dist_candidate_minus_truth" in df.columns:
        df["target_dist_mm"] = pd.to_numeric(df["dist_candidate_minus_truth"], errors="coerce")

    X, _, _, _ = make_feature_matrix(
        df,
        feature_columns=bundle["feature_columns"],
        categorical_columns=bundle["categorical_columns"],
        medians=bundle["medians"],
    )
    X = align_matrix(X, bundle["model_columns"])
    pred_log = bundle["model"].predict(X.values)
    df = df.copy()
    df["pred_log1p_dist"] = pred_log
    df["pred_dist_mm"] = np.expm1(pred_log)

    scored_path = os.path.join(args.outdir, "static_candidate_scored_all.csv")
    df.sort_values(["event", "pred_dist_mm"], inplace=True)
    df.to_csv(scored_path, index=False)
    print("[INFO] wrote %s rows=%d" % (scored_path, len(df)))

    selected = choose_by(df, ["event", "pred_dist_mm", "candidate_score"], [True, True, False], "static_model")
    selected_path = os.path.join(args.outdir, "static_candidate_selected_by_model.csv")
    selected.to_csv(selected_path, index=False)
    print("[INFO] wrote %s rows=%d" % (selected_path, len(selected)))

    selections = [selected]
    labels = ["static_model"]
    if "candidate_score" in df.columns:
        gen = choose_by(df, ["event", "candidate_score"], [True, False], "generator_score")
        gen.to_csv(os.path.join(args.outdir, "static_candidate_selected_by_generator_score.csv"), index=False)
        selections.append(gen)
        labels.append("generator_score")
    if "target_dist_mm" in df.columns and np.isfinite(pd.to_numeric(df["target_dist_mm"], errors="coerce")).any():
        oracle = choose_by(df, ["event", "target_dist_mm", "candidate_score"], [True, True, False], "truth_oracle")
        oracle.to_csv(os.path.join(args.outdir, "static_candidate_selected_by_truth_oracle.csv"), index=False)
        selections.append(oracle)
        labels.append("truth_oracle")

    summary_rows = [summarize_selection(sel, df, lab) for sel, lab in zip(selections, labels)]
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "static_candidate_selection_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("[INFO] wrote %s rows=%d" % (summary_path, len(summary)))
    print("\n[STATIC CANDIDATE SELECTION SUMMARY]")
    print(summary.to_string(index=False))

    if bool(args.write_selected_vertices):
        vdir = os.path.join(args.outdir, "selected_vertices")
        if not os.path.isdir(vdir):
            os.makedirs(vdir)
        for _, r in selected.iterrows():
            ev = int(r["event"])
            with open(os.path.join(vdir, "event_%d_vertex.txt" % ev), "w") as f:
                f.write("%.9g %.9g %.9g\n" % (float(r["cand_x"]), float(r["cand_y"]), float(r["cand_z"])))
        print("[INFO] wrote selected vertex text files under %s" % vdir)


if __name__ == "__main__":
    main()
