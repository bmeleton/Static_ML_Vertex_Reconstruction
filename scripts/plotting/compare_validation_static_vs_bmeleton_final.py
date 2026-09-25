#!/usr/bin/env python3
"""
Compare the static ML-selected candidate against the Cloud_classes/BMeleton
final refit vertex on a held-out split, usually split == 'val'.

This script reads static_candidate_training_scored_rows.csv, so it uses the
same train/validation split that was created by train_static_candidate_model.py.
It does NOT use static_candidate_model_v1_apply_traincheck/static_candidate_scored_all.csv,
which contains train+val together.

Python 3.6 compatible.
"""
from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import pandas as pd

VERSION = "2026-08-07-val-static-vs-bmeleton-final-v1"


def finite_series(s):
    return pd.to_numeric(s, errors="coerce")


def pick_col(df, preferred, contains_any=None, required=True, description="column"):
    for c in preferred:
        if c in df.columns:
            return c
    if contains_any:
        lows = [(c, c.lower()) for c in df.columns]
        for c, lc in lows:
            ok = True
            for token in contains_any:
                if token.lower() not in lc:
                    ok = False
                    break
            if ok:
                return c
    if required:
        raise RuntimeError("Could not find %s. Available columns include: %s" % (description, list(df.columns)[:80]))
    return None


def fwhm68(series):
    x = finite_series(series).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.quantile(0.84) - x.quantile(0.16))


def summarize(name, df, dist_col):
    out = {
        "selection_rule": name,
        "n_events": int(df["event"].nunique()) if "event" in df.columns else int(len(df)),
        "n_rows": int(len(df)),
    }
    dist = finite_series(df[dist_col])
    out["median_dist_mm"] = float(dist.median()) if len(dist.dropna()) else np.nan
    out["mean_dist_mm"] = float(dist.mean()) if len(dist.dropna()) else np.nan
    out["rms_dist_mm"] = float(np.sqrt(np.nanmean(np.asarray(dist, dtype=float) ** 2))) if len(dist.dropna()) else np.nan
    out["max_dist_mm"] = float(dist.max()) if len(dist.dropna()) else np.nan
    for thr in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        out["frac_lt_%smm" % (str(thr).replace('.', 'p'))] = float((dist < thr).mean()) if len(dist) else np.nan

    for axis in ["dx", "dy", "dz"]:
        col = axis + "_candidate_minus_truth"
        if col in df.columns:
            vals = finite_series(df[col])
            out[axis + "_median_mm"] = float(vals.median()) if len(vals.dropna()) else np.nan
            out[axis + "_fwhm68_mm"] = fwhm68(vals)
    return out


def choose_by_pred(df, pred_col):
    v = df.copy()
    v[pred_col] = finite_series(v[pred_col])
    v = v[np.isfinite(v[pred_col])]
    if len(v) == 0:
        return v
    idx = v.groupby("event")[pred_col].idxmin()
    return v.loc[idx].copy()


def choose_oracle(df, dist_col):
    v = df.copy()
    v[dist_col] = finite_series(v[dist_col])
    v = v[np.isfinite(v[dist_col])]
    if len(v) == 0:
        return v
    idx = v.groupby("event")[dist_col].idxmin()
    return v.loc[idx].copy()


def choose_generator(df):
    if "candidate_score" not in df.columns:
        return None
    v = df.copy()
    v["candidate_score"] = finite_series(v["candidate_score"])
    v = v[np.isfinite(v["candidate_score"])]
    if len(v) == 0:
        return None
    # This matches the usual generator baseline: higher candidate_score is better.
    idx = v.groupby("event")["candidate_score"].idxmax()
    return v.loc[idx].copy()


def main():
    ap = argparse.ArgumentParser(description="Validation-only comparison: static ML candidate vs BMeleton/Cloud_classes final refit vertex.")
    ap.add_argument("--scored-rows", default="static_candidate_model_v1/static_candidate_training_scored_rows.csv",
                    help="Path to static_candidate_training_scored_rows.csv from train_static_candidate_model.py")
    ap.add_argument("--split", default="val", help="Split to evaluate, usually val. Use all to ignore split column.")
    ap.add_argument("--final-source", default="bmeleton_final_refit_vertex",
                    help="Candidate source name for the old Cloud_classes final refit vertex.")
    ap.add_argument("--outdir", default="static_candidate_model_v1/validation_static_vs_bmeleton_final",
                    help="Output directory for comparison CSV files.")
    args = ap.parse_args()

    print("[INFO] script version:", VERSION)
    print("[INFO] reading", args.scored_rows)
    if not os.path.isfile(args.scored_rows):
        raise RuntimeError("Could not find --scored-rows file: %s" % args.scored_rows)

    d = pd.read_csv(args.scored_rows)
    if "event" not in d.columns:
        raise RuntimeError("Expected an event column in %s" % args.scored_rows)

    pred_col = pick_col(d, ["pred_dist_mm", "predicted_dist_mm", "model_pred_dist_mm", "predicted_target_dist_mm"],
                        contains_any=["pred", "dist"], description="prediction-distance column")
    dist_col = pick_col(d, ["dist_candidate_minus_truth", "target_dist_mm", "candidate_dist_mm"],
                        contains_any=["dist"], description="truth-distance column")

    print("[INFO] prediction column:", pred_col)
    print("[INFO] truth-distance column:", dist_col)

    if args.split.lower() != "all":
        if "split" not in d.columns:
            raise RuntimeError("No split column found. Use --split all or provide a scored rows file with split labels.")
        available = sorted([str(x) for x in d["split"].dropna().unique()])
        print("[INFO] available splits:", ",".join(available))
        dsplit = d[d["split"].astype(str) == str(args.split)].copy()
    else:
        dsplit = d.copy()

    print("[INFO] rows in selected split:", len(dsplit))
    print("[INFO] events in selected split:", dsplit["event"].nunique())
    print("[INFO] candidate sources in selected split:")
    if "candidate_source" in dsplit.columns:
        print(dsplit["candidate_source"].value_counts().to_string())
    else:
        raise RuntimeError("No candidate_source column found.")

    model = choose_by_pred(dsplit, pred_col)
    final = dsplit[dsplit["candidate_source"] == args.final_source].copy()
    oracle = choose_oracle(dsplit, dist_col)
    generator = choose_generator(dsplit)

    common = sorted(set(model["event"]).intersection(set(final["event"])))
    model_common = model[model["event"].isin(common)].copy()
    final_common = final[final["event"].isin(common)].copy()
    oracle_common = oracle[oracle["event"].isin(common)].copy()
    generator_common = generator[generator["event"].isin(common)].copy() if generator is not None else None

    print("[INFO] model-selected events:", model["event"].nunique())
    print("[INFO] final-refit events:", final["event"].nunique())
    print("[INFO] common events:", len(common))

    rows = []
    rows.append(summarize("static_model_%s" % args.split, model_common, dist_col))
    rows.append(summarize("cloud_final_refit_%s" % args.split, final_common, dist_col))
    rows.append(summarize("truth_oracle_%s" % args.split, oracle_common, dist_col))
    if generator_common is not None:
        rows.append(summarize("generator_score_%s" % args.split, generator_common, dist_col))
    summary = pd.DataFrame(rows)

    # Wide event-by-event comparison.
    m = model_common.set_index("event")
    f = final_common.set_index("event")
    o = oracle_common.set_index("event")
    wide = pd.DataFrame(index=common)
    wide.index.name = "event"
    wide["model_candidate_id"] = m["candidate_id"] if "candidate_id" in m.columns else np.nan
    wide["model_candidate_source"] = m["candidate_source"]
    wide["model_pred_dist_mm"] = m[pred_col]
    wide["model_dist_mm"] = m[dist_col]
    wide["final_dist_mm"] = f[dist_col]
    wide["oracle_dist_mm"] = o[dist_col]
    wide["improvement_final_minus_model_mm"] = wide["final_dist_mm"] - wide["model_dist_mm"]
    wide["model_better_than_final"] = wide["improvement_final_minus_model_mm"] > 0.0
    wide["model_worse_than_final"] = wide["improvement_final_minus_model_mm"] < 0.0

    for axis in ["dx", "dy", "dz"]:
        col = axis + "_candidate_minus_truth"
        if col in m.columns and col in f.columns:
            wide["model_" + axis + "_mm"] = m[col]
            wide["final_" + axis + "_mm"] = f[col]

    if generator_common is not None and len(generator_common) > 0:
        g = generator_common.set_index("event")
        wide["generator_dist_mm"] = g[dist_col]

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "validation_static_vs_bmeleton_final_summary.csv")
    wide_path = os.path.join(args.outdir, "validation_static_vs_bmeleton_final_by_event.csv")
    summary.to_csv(summary_path, index=False)
    wide.to_csv(wide_path)

    print("")
    print("[VALIDATION COMPARISON SUMMARY]")
    print(summary.to_string(index=False))
    print("")
    if len(wide) > 0:
        print("event-by-event: ML better fraction", float(wide["model_better_than_final"].mean()))
        print("event-by-event: ML worse fraction", float(wide["model_worse_than_final"].mean()))
        print("median ML improvement over final refit mm", float(wide["improvement_final_minus_model_mm"].median()))
    print("")
    print("[INFO] wrote", summary_path)
    print("[INFO] wrote", wide_path)


if __name__ == "__main__":
    main()
