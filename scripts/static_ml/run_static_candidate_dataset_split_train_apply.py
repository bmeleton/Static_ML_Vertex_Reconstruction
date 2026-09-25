#!/usr/bin/env python3
"""
run_static_candidate_dataset_split_train_apply.py

General static vertex-candidate ML workflow for any simulated dataset.

It does:

  Hough lines + matching truth ROOT
  -> collect newest static candidate features
  -> split by event into train/test
  -> train static candidate-quality ML model
  -> apply model to held-out test events
  -> run oracle-regret diagnostics, confusion matrices, and residual plots

This script is Python 3.6 compatible and shell-neutral, so it is safe to run
from tcsh/csh.

Typical usage:

python3 run_static_candidate_dataset_split_train_apply.py \
  --label elastic_lowE \
  --linesdir reconstructed_lines_elastic_lowE \
  --rootfile TeBATSim_elastic_lowE.root \
  --train-frac 0.70 \
  --candidate-line-source bmeleton_merged \
  --max-pairs-for-refit 0 \
  --n-estimators 300 \
  --n-jobs 4 \
  --verbose-collect \
  --clean

For alpha simulation:

  python3 run_static_candidate_dataset_split_train_apply.py \
    --label alpha_sim \
    --linesdir reconstructed_lines_alpha_sim \
    --rootfile TeBATSim_alpha.root \
    --train-frac 0.70 \
    --clean
"""

from __future__ import print_function

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def run(cmd, dry_run=False):
    print("")
    print("[RUN]")
    print(" ".join(cmd))
    if dry_run:
        return 0
    p = subprocess.Popen(cmd)
    ret = p.wait()
    if ret != 0:
        raise RuntimeError("Command failed with exit code {0}: {1}".format(ret, " ".join(cmd)))
    return ret


def rm_rf(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def check_file(path, label):
    if not os.path.exists(path):
        raise IOError("{0} does not exist: {1}".format(label, path))


def check_dir(path, label):
    if not os.path.isdir(path):
        raise IOError("{0} does not exist or is not a directory: {1}".format(label, path))


def split_features_by_event(features_csv, split_dir, train_frac, seed, event_col):
    check_file(features_csv, "features CSV")
    ensure_dir(split_dir)

    d = pd.read_csv(features_csv)
    if event_col not in d.columns:
        raise RuntimeError("Event column '{0}' not found in {1}".format(event_col, features_csv))

    events = np.array(sorted(d[event_col].dropna().unique()))
    if len(events) == 0:
        raise RuntimeError("No events found in features CSV: {0}".format(features_csv))

    rng = np.random.RandomState(seed)
    rng.shuffle(events)

    n_train = int(round(float(train_frac) * len(events)))
    if n_train <= 0 or n_train >= len(events):
        raise RuntimeError(
            "Bad train/test split: train_frac={0}, events={1}, n_train={2}".format(
                train_frac, len(events), n_train
            )
        )

    train_events = set(events[:n_train])
    is_train = d[event_col].isin(train_events)

    train_csv = os.path.join(split_dir, "static_candidate_features_train.csv")
    test_csv = os.path.join(split_dir, "static_candidate_features_test.csv")
    event_split_csv = os.path.join(split_dir, "event_split.csv")

    d[is_train].to_csv(train_csv, index=False)
    d[~is_train].to_csv(test_csv, index=False)

    split_rows = []
    for ev in events[:n_train]:
        split_rows.append({event_col: ev, "split": "train"})
    for ev in events[n_train:]:
        split_rows.append({event_col: ev, "split": "test"})
    pd.DataFrame(split_rows).to_csv(event_split_csv, index=False)

    print("")
    print("[SPLIT SUMMARY]")
    print("features:", features_csv)
    print("events total:", len(events))
    print("events train:", n_train)
    print("events test:", len(events) - n_train)
    print("rows train:", int(is_train.sum()))
    print("rows test:", int((~is_train).sum()))
    print("train CSV:", train_csv)
    print("test CSV:", test_csv)
    print("event split CSV:", event_split_csv)

    return train_csv, test_csv, event_split_csv


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="General static candidate collection/train/test workflow for any simulation dataset."
    )

    ap.add_argument("--label", required=True,
                    help="Dataset label used to name output directories, e.g. elastic_lowE or alpha_sim.")
    ap.add_argument("--linesdir", default=None,
                    help="Hough-line directory. Required unless --features-csv is supplied with --skip-collect.")
    ap.add_argument("--rootfile", default=None,
                    help="Matching truth ROOT file. Required for supervised collection/training unless --features-csv is supplied.")
    ap.add_argument("--features-csv", default=None,
                    help="Use an existing static_candidate_features.csv instead of collecting features.")
    ap.add_argument("--skip-collect", action="store_true",
                    help="Skip feature collection and use --features-csv or <label>_all/static_candidate_features.csv.")
    ap.add_argument("--out-root", default=".",
                    help="Parent directory for output directories. Default: current directory.")
    ap.add_argument("--train-frac", type=float, default=0.70,
                    help="Fraction of events used for training. Default: 0.70.")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Random seed for event-level split. Default: 12345.")
    ap.add_argument("--event-col", default="event",
                    help="Event column name in feature table. Default: event.")

    ap.add_argument("--candidate-line-source", default="bmeleton_merged",
                    choices=["raw", "bmeleton_merged", "both"],
                    help="Candidate line source passed to collect_static_candidate_features.py. Default: bmeleton_merged.")
    ap.add_argument("--max-pairs-for-refit", type=int, default=0,
                    help="Passed to collect_static_candidate_features.py. Default: 0.")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Optional maximum number of events to collect. Useful for small tests.")
    ap.add_argument("--verbose-collect", action="store_true",
                    help="Pass --verbose to collect_static_candidate_features.py.")

    ap.add_argument("--model-kind", default="extratrees",
                    help="Model kind passed to train_static_candidate_model.py. Default: extratrees.")
    ap.add_argument("--n-estimators", type=int, default=300,
                    help="Number of trees. Default: 300.")
    ap.add_argument("--n-jobs", type=int, default=4,
                    help="Parallel jobs for training. Default: 4.")

    ap.add_argument("--clean", action="store_true",
                    help="Remove this dataset's output directories before running.")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="Skip oracle-miss, confusion-matrix, and residual-plot diagnostics.")
    ap.add_argument("--no-plots", action="store_true",
                    help="Do not ask diagnostic scripts to make PNG plots.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands but do not execute external scripts.")

    args = ap.parse_args(argv)

    label = args.label
    out_root = args.out_root

    all_dir = os.path.join(out_root, "static_candidate_{0}_all".format(label))
    split_dir = os.path.join(out_root, "static_candidate_{0}_split".format(label))
    model_dir = os.path.join(out_root, "static_candidate_{0}_model_train{1}".format(
        label, int(round(100.0 * args.train_frac))
    ))
    apply_dir = os.path.join(out_root, "static_candidate_{0}_apply_test{1}".format(
        label, int(round(100.0 * (1.0 - args.train_frac)))
    ))

    print("============================================================")
    print("General static candidate ML workflow")
    print("label:       {0}".format(label))
    print("linesdir:    {0}".format(args.linesdir))
    print("rootfile:    {0}".format(args.rootfile))
    print("features:    {0}".format(args.features_csv))
    print("train frac:  {0}".format(args.train_frac))
    print("seed:        {0}".format(args.seed))
    print("all dir:     {0}".format(all_dir))
    print("split dir:   {0}".format(split_dir))
    print("model dir:   {0}".format(model_dir))
    print("apply dir:   {0}".format(apply_dir))
    print("============================================================")

    if args.clean:
        print("")
        print("[CLEAN] Removing old output directories for label '{0}'".format(label))
        for p in [all_dir, split_dir, model_dir, apply_dir]:
            rm_rf(p)

    # Step 1: collect features, unless skipped.
    if args.skip_collect:
        if args.features_csv is not None:
            features_csv = args.features_csv
        else:
            features_csv = os.path.join(all_dir, "static_candidate_features.csv")
        check_file(features_csv, "features CSV")
        print("")
        print("[STEP 1] Skipping collection; using existing features:")
        print(features_csv)
    else:
        if args.linesdir is None:
            raise RuntimeError("--linesdir is required unless --skip-collect is used.")
        if args.rootfile is None:
            raise RuntimeError("--rootfile is required for supervised feature collection unless --skip-collect is used.")

        check_dir(args.linesdir, "Hough-line directory")
        check_file(args.rootfile, "truth ROOT file")

        cmd = [
            "python3", "collect_static_candidate_features.py",
            "--linesdir", args.linesdir,
            "--rootfile", args.rootfile,
            "--outdir", all_dir,
            "--candidate-line-source", args.candidate_line_source,
            "--max-pairs-for-refit", str(args.max_pairs_for_refit),
        ]
        if args.max_events is not None:
            cmd.extend(["--max-events", str(args.max_events)])
        if args.verbose_collect:
            cmd.append("--verbose")

        print("")
        print("[STEP 1] Collecting static candidate features...")
        run(cmd, dry_run=args.dry_run)
        features_csv = os.path.join(all_dir, "static_candidate_features.csv")

    if args.dry_run:
        print("")
        print("[DRY RUN COMPLETE]")
        return 0

    check_file(features_csv, "features CSV after collection")

    # Optional candidate/status summaries.
    candidates_csv = os.path.join(all_dir, "vertex_candidates_all.csv")
    status_csv = os.path.join(all_dir, "static_candidate_status.csv")
    if os.path.exists(candidates_csv):
        try:
            d = pd.read_csv(candidates_csv)
            print("")
            print("[CANDIDATE SOURCE COUNTS]")
            if "candidate_source" in d.columns:
                print(d["candidate_source"].value_counts().to_string())
            if "event" in d.columns:
                print("events_with_candidates", d["event"].nunique(), "candidate_rows", len(d))
        except Exception as e:
            print("[WARN] Could not summarize candidate sources:", e)

    if os.path.exists(status_csv):
        try:
            s = pd.read_csv(status_csv)
            print("")
            print("[CANDIDATE STATUS SUMMARY]")
            if "status" in s.columns:
                print(s["status"].value_counts().to_string())
            if "n_candidates" in s.columns:
                print("")
                print("events", len(s))
                print("with_candidates", int((s["n_candidates"] > 0).sum()))
                print("zero_candidates", int((s["n_candidates"] == 0).sum()))
                print("fraction_with_candidates", float((s["n_candidates"] > 0).mean()))
        except Exception as e:
            print("[WARN] Could not summarize candidate status:", e)

    # Step 2: split.
    print("")
    print("[STEP 2] Splitting feature table by event...")
    train_csv, test_csv, event_split_csv = split_features_by_event(
        features_csv, split_dir, args.train_frac, args.seed, args.event_col
    )

    # Step 3: train.
    print("")
    print("[STEP 3] Training static candidate model...")
    cmd = [
        "python3", "train_static_candidate_model.py",
        "--features", train_csv,
        "--outdir", model_dir,
        "--model-kind", args.model_kind,
        "--n-estimators", str(args.n_estimators),
        "--n-jobs", str(args.n_jobs),
    ]
    run(cmd)

    model_path = os.path.join(model_dir, "static_candidate_quality_model.pkl")
    check_file(model_path, "trained model")

    # Step 4: apply.
    print("")
    print("[STEP 4] Applying trained model to held-out test events...")
    cmd = [
        "python3", "apply_static_candidate_model.py",
        "--model", model_path,
        "--features", test_csv,
        "--outdir", apply_dir,
        "--write-selected-vertices",
    ]
    run(cmd)

    scored_csv = os.path.join(apply_dir, "static_candidate_scored_all.csv")
    selected_csv = os.path.join(apply_dir, "static_candidate_selected_by_model.csv")
    check_file(scored_csv, "scored test candidates")
    check_file(selected_csv, "selected test candidates")

    # Step 5: diagnostics.
    if not args.no_diagnostics:
        plot_flag = [] if args.no_plots else ["--make-plots"]

        print("")
        print("[STEP 5] Oracle-regret diagnostics...")
        cmd = [
            "python3", "summarize_static_candidate_oracle_misses.py",
            "--scored", scored_csv,
            "--outdir", os.path.join(apply_dir, "oracle_miss_summary"),
            "--split", "all",
            "--tolerances-mm", "0,0.05,0.1,0.25,0.5,1,2,5",
            "--top-n", "50",
        ] + plot_flag
        run(cmd)

        print("")
        print("[STEP 6] Confusion-matrix diagnostics...")
        cmd = [
            "python3", "static_candidate_confusion_matrices.py",
            "--scored", scored_csv,
            "--outdir", os.path.join(apply_dir, "confusion_matrices"),
            "--split", "all",
            "--thresholds", "0.5,1.0,2.0",
        ] + plot_flag
        run(cmd)

        print("")
        print("[STEP 7] Analysis-style residual plots...")
        cmd = [
            "python3", "plot_static_ml_residuals_analysis_style.py",
            "--selected-csv", selected_csv,
            "--outdir", os.path.join(apply_dir, "histograms_analysis_style"),
            "--histname", "{0}_static_ml_test_residuals.png".format(label),
        ]
        run(cmd)

    print("")
    print("============================================================")
    print("DONE")
    print("")
    print("Main outputs:")
    print("  features:          {0}".format(features_csv))
    print("  event split:       {0}".format(event_split_csv))
    print("  trained model:     {0}".format(model_path))
    print("  scored test rows:  {0}".format(scored_csv))
    print("  selected vertices: {0}".format(os.path.join(apply_dir, "selected_vertices")))
    print("  selected CSV:      {0}".format(selected_csv))
    if not args.no_diagnostics:
        print("  oracle summary:    {0}".format(os.path.join(apply_dir, "oracle_miss_summary", "ml_vs_oracle_summary.csv")))
        print("  residual plot:     {0}".format(os.path.join(apply_dir, "histograms_analysis_style", "{0}_static_ml_test_residuals.png".format(label))))
    print("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
