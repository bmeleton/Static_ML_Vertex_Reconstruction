#!/usr/bin/env python3
"""
make_static_candidate_combined_split_balanced.py

Merge multiple static_candidate_features.csv files, apply event offsets so event
IDs from different datasets cannot collide, and create combined event-level
train/test split files.

This version adds balanced round-down splitting:

  --balance-to-smallest --round-events-to 500

Example:
  Dataset A has 9351 events
  Dataset B has 8127 events
  smallest = 8127
  rounded common count = 8000
  train70 = 5600 per dataset
  test30  = 2400 per dataset

Python 3.6 compatible.

Input syntax:
  --input label:path
  --input label:offset:path

Examples:
  python3 make_static_candidate_combined_split_balanced.py \
    --input alpha_300:0:static_candidate_alpha_300_all/static_candidate_features.csv \
    --input alpha_400:1000000:static_candidate_alpha_400_all/static_candidate_features.csv \
    --input alpha_580:2000000:static_candidate_alpha_580_all/static_candidate_features.csv \
    --outdir static_candidate_combined_alpha_pressures/combined_balanced_70_30_validation \
    --train-frac 0.70 \
    --balance-to-smallest \
    --round-events-to 500 \
    --max-events-per-dataset 0

The script writes both full files with dataset bookkeeping columns and generic
files with those columns removed. Use the generic files for model training/apply
if you do not want the model to learn dataset identity as a shortcut.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd


def parse_input_spec(spec, auto_offset):
    parts = str(spec).split(":", 2)
    if len(parts) == 2:
        label, path = parts
        offset = auto_offset
    elif len(parts) == 3:
        label, offset_text, path = parts
        offset = int(offset_text)
    else:
        raise ValueError("Bad --input spec: {0}. Use label:path or label:offset:path".format(spec))
    return label, int(offset), path


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge static candidate feature tables and make combined train/test split."
    )
    p.add_argument("--input", action="append", required=True,
                   help="Dataset spec: label:path or label:offset:path. Repeat once per dataset.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--offset-step", type=int, default=1000000,
                   help="Automatic event offset step when offset is omitted. Default 1000000")
    p.add_argument("--max-events-per-dataset", type=int, default=0,
                   help="Randomly keep at most this many events from each dataset before split. 0 means all.")
    p.add_argument("--balance-to-smallest", action="store_true",
                   help=("Use the same number of events from every dataset. "
                         "The common event count is determined by the smallest dataset after any max-events cap."))
    p.add_argument("--round-events-to", type=int, default=1,
                   help=("Round the number of selected events per dataset down to a multiple of this value. "
                         "Use 500 for round 500-event blocks. Default 1."))
    p.add_argument("--event-col", default="event")
    p.add_argument("--summary-out", default="combined_dataset_summary.csv")
    return p.parse_args()


def pct_label(x):
    return str(int(round(100.0 * float(x))))


def round_down_to_multiple(n, step):
    n = int(n)
    step = int(step)
    if step <= 1:
        return n
    return (n // step) * step


def main():
    a = parse_args()

    if a.train_frac <= 0.0 or a.train_frac > 1.0:
        sys.exit("[ERROR] --train-frac must be in (0, 1]")
    if int(a.max_events_per_dataset) < 0:
        sys.exit("[ERROR] --max-events-per-dataset must be >= 0")
    if int(a.round_events_to) < 1:
        sys.exit("[ERROR] --round-events-to must be >= 1")

    if not os.path.isdir(a.outdir):
        os.makedirs(a.outdir)

    # First pass: read all datasets, discover unique events, and decide how
    # many events to use from each dataset.
    datasets = []
    eligible_counts = []

    for i, spec in enumerate(a.input):
        label, offset, path = parse_input_spec(spec, i * int(a.offset_step))
        if not os.path.exists(path):
            sys.exit("[ERROR] Missing input file for {0}: {1}".format(label, path))

        d = pd.read_csv(path)
        if a.event_col not in d.columns:
            sys.exit("[ERROR] Missing event column {0} in {1}".format(a.event_col, path))

        d[a.event_col] = pd.to_numeric(d[a.event_col], errors="coerce")
        d = d.dropna(subset=[a.event_col]).copy()
        d[a.event_col] = d[a.event_col].astype(int)

        events_sorted = np.array(sorted(d[a.event_col].unique()))
        n_before = len(events_sorted)

        if n_before <= 0:
            sys.exit("[ERROR] No events found for {0}: {1}".format(label, path))

        max_cap = int(a.max_events_per_dataset)
        eligible = n_before
        if max_cap > 0 and max_cap < eligible:
            eligible = max_cap

        eligible_counts.append(eligible)

        datasets.append({
            "index": i,
            "label": label,
            "offset": int(offset),
            "path": path,
            "data": d,
            "events_sorted": events_sorted,
            "events_before_limit": int(n_before),
            "eligible_events": int(eligible),
        })

    if a.balance_to_smallest:
        common = min(eligible_counts)
        common = round_down_to_multiple(common, int(a.round_events_to))
        if common <= 0:
            sys.exit("[ERROR] Balanced rounded event count is zero. Lower --round-events-to or check inputs.")
        target_counts = dict((ds["label"], int(common)) for ds in datasets)
        split_mode = "balanced_to_smallest"
    else:
        target_counts = {}
        for ds in datasets:
            n = int(ds["eligible_events"])
            n = round_down_to_multiple(n, int(a.round_events_to))
            if n <= 0:
                sys.exit("[ERROR] Rounded event count is zero for {0}. Lower --round-events-to.".format(ds["label"]))
            target_counts[ds["label"]] = int(n)
        split_mode = "independent_per_dataset"

    train_frames = []
    test_frames = []
    summary_rows = []

    for ds in datasets:
        i = ds["index"]
        label = ds["label"]
        offset = int(ds["offset"])
        d = ds["data"]

        events = np.array(ds["events_sorted"], copy=True)
        rng = np.random.RandomState(int(a.seed) + i)
        rng.shuffle(events)

        n_target = int(target_counts[label])
        if n_target > len(events):
            sys.exit("[ERROR] Target events exceeds available events for {0}".format(label))

        selected_events = events[:n_target]
        n_train = int(round(float(a.train_frac) * n_target))
        if n_train > n_target:
            n_train = n_target

        train_orig = set(selected_events[:n_train])
        test_orig = set(selected_events[n_train:])

        selected_set = set(selected_events)
        dsel = d[d[a.event_col].isin(selected_set)].copy()
        dsel["dataset"] = label
        dsel["original_event"] = dsel[a.event_col].astype(int)
        dsel["dataset_event_offset"] = offset
        dsel[a.event_col] = dsel["original_event"] + offset

        train = dsel[dsel["original_event"].isin(train_orig)].copy()
        test = dsel[dsel["original_event"].isin(test_orig)].copy()

        train_frames.append(train)
        test_frames.append(test)

        summary_rows.append({
            "dataset": label,
            "input_file": ds["path"],
            "event_offset": int(offset),
            "split_mode": split_mode,
            "events_before_limit": int(ds["events_before_limit"]),
            "eligible_after_max_cap": int(ds["eligible_events"]),
            "events_selected_after_rounding": int(n_target),
            "round_events_to": int(a.round_events_to),
            "train_frac": float(a.train_frac),
            "train_events": int(train["original_event"].nunique()),
            "test_events": int(test["original_event"].nunique()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "first_selected_original_event": int(selected_events[0]) if len(selected_events) else "",
            "last_selected_original_event": int(selected_events[-1]) if len(selected_events) else "",
        })

    train_combined = pd.concat(train_frames, ignore_index=True, sort=False) if train_frames else pd.DataFrame()
    test_combined = pd.concat(test_frames, ignore_index=True, sort=False) if test_frames else pd.DataFrame()

    train_label = "train{0}".format(pct_label(a.train_frac))
    test_label = "test{0}".format(pct_label(1.0 - a.train_frac))

    for split_name, df in [(train_label, train_combined), (test_label, test_combined)]:
        full_out = os.path.join(a.outdir, "static_candidate_features_combined_{0}.csv".format(split_name))
        generic_out = os.path.join(a.outdir, "static_candidate_features_combined_{0}_generic.csv".format(split_name))

        df.to_csv(full_out, index=False)

        generic = df.drop(
            columns=["dataset", "original_event", "dataset_event_offset"],
            errors="ignore"
        )
        generic.to_csv(generic_out, index=False)

        print(split_name)
        print("  rows:", len(df))
        print("  events:", df[a.event_col].nunique() if a.event_col in df.columns else 0)
        print("  full:", full_out)
        print("  generic:", generic_out)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(a.outdir, a.summary_out)
    summary.to_csv(summary_path, index=False)

    print("")
    print("summary:", summary_path)
    print("")
    print(summary.to_string(index=False))

    total_train_events = int(train_combined[a.event_col].nunique()) if a.event_col in train_combined.columns else 0
    total_test_events = int(test_combined[a.event_col].nunique()) if a.event_col in test_combined.columns else 0

    print("")
    print("combined totals:")
    print("  train events:", total_train_events)
    print("  test events:", total_test_events)
    print("  total selected events:", total_train_events + total_test_events)


if __name__ == "__main__":
    main()
