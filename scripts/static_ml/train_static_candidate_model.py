from __future__ import print_function

"""
train_static_candidate_model.py

Train a fast static vertex-candidate quality model.

The model predicts candidate distance-to-truth from static topology / Hough / refit
features. It is a candidate selector, not a rollout/action policy:
  candidate features -> predicted distance mm

Input files are static_candidate_features.csv outputs from
collect_static_candidate_features.py.

Python 3.6+ compatible.
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

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

EXCLUDE_EXACT = set([
    "event", "candidate_id", "cand_x", "cand_y", "cand_z",
    "truth_x", "truth_y", "truth_z",
    "dx_candidate_minus_truth", "dy_candidate_minus_truth", "dz_candidate_minus_truth",
    "dist_candidate_minus_truth", "truth_rank", "is_best_truth",
    "target_dist_mm", "target_log1p_dist", "target_truth_rank", "target_is_best_truth",
])

EXCLUDE_PREFIXES = [
    "target_within_",
]

DEFAULT_CATEGORICAL = [
    "candidate_source",
    "static_feature_status",
    "line1_refit_reason", "line2_refit_reason",
    "line1_far_endpoint", "line2_far_endpoint",
]


def read_features(paths, glob_pattern=None):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "static_candidate_features.csv"), recursive=True)))
        else:
            files.extend(sorted(glob.glob(p)))
    if glob_pattern:
        files.extend(sorted(glob.glob(glob_pattern)))
    files = sorted(set(files))
    if not files:
        sys.exit("[ERROR] No feature CSVs found")
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            d["_source_file"] = f
            dfs.append(d)
            print("[INFO] read %s rows=%d" % (f, len(d)))
        except Exception as exc:
            print("[WARN] could not read %s: %s" % (f, exc))
    if not dfs:
        sys.exit("[ERROR] No readable feature CSVs")
    return pd.concat(dfs, ignore_index=True, sort=False), files


def should_exclude(col):
    if col in EXCLUDE_EXACT:
        return True
    if col.startswith("_"):
        return True
    for p in EXCLUDE_PREFIXES:
        if col.startswith(p):
            return True
    # exclude raw string-ish IDs that should not be learned as labels
    if col.endswith("_file") or col.endswith("_path"):
        return True
    return False


def make_feature_matrix(df, feature_columns=None, categorical_columns=None, medians=None):
    if categorical_columns is None:
        categorical_columns = [c for c in DEFAULT_CATEGORICAL if c in df.columns]
    if feature_columns is None:
        usable = []
        for c in df.columns:
            if should_exclude(c):
                continue
            if c in categorical_columns:
                usable.append(c)
                continue
            # keep columns that can be numeric
            vals = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(vals).sum() > 0:
                usable.append(c)
        feature_columns = usable

    parts = []
    numeric_cols = []
    cat_cols = []
    for c in feature_columns:
        if c not in df.columns:
            continue
        if c in categorical_columns:
            cat_cols.append(c)
        else:
            numeric_cols.append(c)

    if numeric_cols:
        xnum = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if medians is None:
            medians = xnum.median(axis=0)
            medians = medians.fillna(0.0)
        else:
            # align medians with present numeric columns
            medians = pd.Series(medians)
        xnum = xnum.fillna(medians).fillna(0.0)
        parts.append(xnum)
    else:
        if medians is None:
            medians = pd.Series(dtype=float)

    if cat_cols:
        xcat = df[cat_cols].copy()
        for c in cat_cols:
            xcat[c] = xcat[c].astype(str).fillna("missing")
        xcat = pd.get_dummies(xcat, columns=cat_cols, dummy_na=False)
        parts.append(xcat)

    if parts:
        X = pd.concat(parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)
    return X, feature_columns, categorical_columns, medians


def align_matrix(X, columns):
    X = X.copy()
    for c in columns:
        if c not in X.columns:
            X[c] = 0.0
    extra = [c for c in X.columns if c not in columns]
    if extra:
        X = X.drop(columns=extra)
    return X[columns]


def event_split(events, val_frac, test_frac, seed):
    events = np.array(sorted(set([int(e) for e in events])))
    rng = np.random.RandomState(int(seed))
    rng.shuffle(events)
    n = len(events)
    n_test = int(round(float(test_frac) * n)) if test_frac > 0 else 0
    n_val = int(round(float(val_frac) * n)) if val_frac > 0 else 0
    test = set(events[:n_test])
    val = set(events[n_test:n_test + n_val])
    train = set(events[n_test + n_val:])
    if not train and len(events):
        train = set(events)
        val = set()
        test = set()
    return train, val, test


def event_selection_summary(df_scored, label):
    if df_scored is None or len(df_scored) == 0:
        return dict(split=label, n_events=0)
    rows = []
    gvalid = df_scored[np.isfinite(pd.to_numeric(df_scored["target_dist_mm"], errors="coerce"))].copy()
    if len(gvalid) == 0:
        return dict(split=label, n_events=0)
    chosen = gvalid.sort_values(["event", "pred_dist_mm", "candidate_score"], ascending=[True, True, False]).groupby("event", as_index=False).first()
    oracle = gvalid.sort_values(["event", "target_dist_mm", "candidate_score"], ascending=[True, True, False]).groupby("event", as_index=False).first()
    gen = gvalid.sort_values(["event", "candidate_score"], ascending=[True, False]).groupby("event", as_index=False).first()
    m = chosen[["event", "candidate_id", "candidate_source", "target_dist_mm", "pred_dist_mm"]].merge(
        oracle[["event", "candidate_id", "target_dist_mm"]], on="event", suffixes=("_model", "_oracle"))
    m = m.merge(gen[["event", "candidate_id", "target_dist_mm"]], on="event")
    m.rename(columns={"candidate_id": "candidate_id_generator", "target_dist_mm": "target_dist_mm_generator"}, inplace=True)
    out = dict(split=label, n_events=int(len(chosen)), n_candidates=int(len(gvalid)))
    for name, arr in [
        ("model", chosen["target_dist_mm"]),
        ("oracle", oracle["target_dist_mm"]),
        ("generator_score", gen["target_dist_mm"]),
    ]:
        arr = pd.to_numeric(arr, errors="coerce")
        out["%s_median_dist_mm" % name] = float(arr.median())
        out["%s_mean_dist_mm" % name] = float(arr.mean())
        for thr in [0.5, 1.0, 2.0, 5.0]:
            out["%s_frac_lt_%smm" % (name, str(thr).replace(".", "p"))] = float((arr < thr).mean())
    out["model_equals_oracle_frac"] = float((m["candidate_id_model"] == m["candidate_id_oracle"]).mean())
    out["model_minus_oracle_median_mm"] = float((m["target_dist_mm_model"] - m["target_dist_mm_oracle"]).median())
    out["model_minus_generator_median_mm"] = float((chosen["target_dist_mm"].values - gen["target_dist_mm"].values).mean()) if len(chosen) == len(gen) else np.nan
    return out


def fit_model(kind, n_estimators, random_state, n_jobs, min_samples_leaf):
    kind = str(kind).lower()
    if kind == "rf":
        return RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=int(n_jobs), min_samples_leaf=int(min_samples_leaf))
    if kind in ["gbr", "gb"]:
        return GradientBoostingRegressor(random_state=int(random_state))
    return ExtraTreesRegressor(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=int(n_jobs), min_samples_leaf=int(min_samples_leaf))


def parse_args():
    ap = argparse.ArgumentParser(description="Train a static TeBAT vertex candidate-quality model.")
    ap.add_argument("--features", nargs="+", required=True, help="Feature CSVs, directories, or glob patterns")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model-out", default="static_candidate_quality_model.pkl")
    ap.add_argument("--model-kind", choices=["extratrees", "rf", "gbr"], default="extratrees")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--random-state", type=int, default=123)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--test-frac", type=float, default=0.0)
    ap.add_argument("--target-max-mm", type=float, default=50.0, help="Clip distances above this for training target only")
    ap.add_argument("--min-candidates-per-event", type=int, default=2)
    ap.add_argument("--glob", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    df, files = read_features(args.features, glob_pattern=args.glob)
    if "target_dist_mm" not in df.columns:
        if "dist_candidate_minus_truth" in df.columns:
            df["target_dist_mm"] = pd.to_numeric(df["dist_candidate_minus_truth"], errors="coerce")
        else:
            sys.exit("[ERROR] Feature CSV has no target_dist_mm/dist_candidate_minus_truth. Need truth-labeled simulation features to train.")
    df["target_dist_mm"] = pd.to_numeric(df["target_dist_mm"], errors="coerce")
    df = df[np.isfinite(df["target_dist_mm"])]
    if "event" not in df.columns:
        sys.exit("[ERROR] Feature CSV must contain event column")
    counts = df.groupby("event").size()
    keep_events = set(counts[counts >= int(args.min_candidates_per_event)].index.astype(int))
    df = df[df["event"].astype(int).isin(keep_events)].copy()
    print("[INFO] training rows after target/event filters=%d events=%d" % (len(df), df["event"].nunique()))
    if len(df) < 10 or df["event"].nunique() < 3:
        sys.exit("[ERROR] Not enough labeled candidate rows/events to train")

    train_events, val_events, test_events = event_split(df["event"].astype(int).unique(), args.val_frac, args.test_frac, args.random_state)
    train_df = df[df["event"].astype(int).isin(train_events)].copy()
    val_df = df[df["event"].astype(int).isin(val_events)].copy()
    test_df = df[df["event"].astype(int).isin(test_events)].copy()
    print("[INFO] events train=%d val=%d test=%d" % (len(train_events), len(val_events), len(test_events)))
    print("[INFO] rows   train=%d val=%d test=%d" % (len(train_df), len(val_df), len(test_df)))

    X_train_raw, feature_columns, categorical_columns, medians = make_feature_matrix(train_df)
    model_columns = list(X_train_raw.columns)
    y_train_dist = np.minimum(pd.to_numeric(train_df["target_dist_mm"], errors="coerce").values.astype(float), float(args.target_max_mm))
    y_train = np.log1p(y_train_dist)
    # Mildly emphasize close candidates without ignoring bad ones.
    weights = 1.0 / (1.0 + np.minimum(y_train_dist, 10.0))
    weights = weights / np.mean(weights)

    model = fit_model(args.model_kind, args.n_estimators, args.random_state, args.n_jobs, args.min_samples_leaf)
    try:
        model.fit(X_train_raw.values, y_train, sample_weight=weights)
    except TypeError:
        model.fit(X_train_raw.values, y_train)

    def score_df(d):
        if d is None or len(d) == 0:
            return d.copy()
        X, _, _, _ = make_feature_matrix(d, feature_columns=feature_columns, categorical_columns=categorical_columns, medians=medians)
        X = align_matrix(X, model_columns)
        pred_log = model.predict(X.values)
        out = d.copy()
        out["pred_log1p_dist"] = pred_log
        out["pred_dist_mm"] = np.expm1(pred_log)
        return out

    scored = []
    summary_rows = []
    for label, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sd = score_df(d)
        if len(sd):
            mae = mean_absolute_error(sd["target_dist_mm"], sd["pred_dist_mm"])
            try:
                r2 = r2_score(sd["target_dist_mm"], sd["pred_dist_mm"])
            except Exception:
                r2 = np.nan
            row = event_selection_summary(sd, label)
            row["row_mae_mm"] = float(mae)
            row["row_r2"] = float(r2)
            summary_rows.append(row)
            sd["split"] = label
            scored.append(sd)
    scored_df = pd.concat(scored, ignore_index=True, sort=False) if scored else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    bundle = dict(
        model=model,
        model_kind=args.model_kind,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        medians=medians.to_dict() if hasattr(medians, "to_dict") else dict(medians),
        model_columns=model_columns,
        target_transform="log1p_dist_mm",
        target_max_mm=float(args.target_max_mm),
        training_files=files,
        exclude_exact=sorted(EXCLUDE_EXACT),
        exclude_prefixes=EXCLUDE_PREFIXES,
    )
    model_path = os.path.join(args.outdir, args.model_out)
    joblib.dump(bundle, model_path)
    print("[INFO] wrote model %s" % model_path)

    scored_path = os.path.join(args.outdir, "static_candidate_training_scored_rows.csv")
    summary_path = os.path.join(args.outdir, "static_candidate_training_summary.csv")
    scored_df.to_csv(scored_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print("[INFO] wrote %s rows=%d" % (scored_path, len(scored_df)))
    print("[INFO] wrote %s rows=%d" % (summary_path, len(summary_df)))
    print("\n[STATIC CANDIDATE TRAINING SUMMARY]")
    print(summary_df.to_string(index=False))

    # Feature importances when available.
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame(dict(feature=model_columns, importance=model.feature_importances_))
        imp.sort_values("importance", ascending=False, inplace=True)
        imp_path = os.path.join(args.outdir, "static_candidate_feature_importances.csv")
        imp.to_csv(imp_path, index=False)
        print("[INFO] wrote %s" % imp_path)
        print("\n[TOP FEATURES]")
        print(imp.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
