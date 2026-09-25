#!/usr/bin/env python3
"""
make_static_candidate_event_split.py

Create an event-level train/test split from one static_candidate_features.csv file.
Python 3.6 compatible.

Example:
  python3 make_static_candidate_event_split.py \
    --features static_candidate_alpha_sim_all/static_candidate_features.csv \
    --outdir static_candidate_alpha_sim_split \
    --train-frac 0.70 \
    --seed 12345 \
    --max-events 5000

If --max-events is supplied, the script randomly selects at most that many unique
input events before making the train/test split. This is useful when a directory
has, for example, 5142 events and you want to use exactly 5000.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Make event-level train/test split for static candidate features.")
    p.add_argument("--features", required=True, help="Input static_candidate_features.csv")
    p.add_argument("--outdir", required=True, help="Output split directory")
    p.add_argument("--train-frac", type=float, default=0.70, help="Fraction of selected events used for train. Default 0.70")
    p.add_argument("--seed", type=int, default=12345, help="Random seed. Default 12345")
    p.add_argument("--max-events", type=int, default=0,
                   help="Maximum number of unique events to keep before splitting. 0 means all.")
    p.add_argument("--event-col", default="event", help="Event column name. Default event")
    p.add_argument("--prefix", default="static_candidate_features", help="Output file prefix")
    p.add_argument("--selection-csv", default="selected_events_for_split.csv",
                   help="CSV recording train/test event assignments")
    return p.parse_args()


def pct_label(x):
    return str(int(round(100.0 * float(x))))


def main():
    a = parse_args()

    if not os.path.exists(a.features):
        sys.exit("[ERROR] Missing input features: {0}".format(a.features))
    if a.train_frac <= 0.0 or a.train_frac > 1.0:
        sys.exit("[ERROR] --train-frac must be in (0, 1]")

    if not os.path.isdir(a.outdir):
        os.makedirs(a.outdir)

    d = pd.read_csv(a.features)
    if a.event_col not in d.columns:
        sys.exit("[ERROR] Missing event column {0}. Columns: {1}".format(a.event_col, list(d.columns)))

    ev = np.array(sorted(pd.to_numeric(d[a.event_col], errors="coerce").dropna().astype(int).unique()))
    rng = np.random.RandomState(int(a.seed))
    rng.shuffle(ev)

    n_total_before = len(ev)
    if int(a.max_events) > 0 and int(a.max_events) < len(ev):
        ev = ev[:int(a.max_events)]

    n_selected = len(ev)
    n_train = int(round(float(a.train_frac) * n_selected))
    if n_train > n_selected:
        n_train = n_selected

    train_events = set(ev[:n_train])
    test_events = set(ev[n_train:])

    event_values = pd.to_numeric(d[a.event_col], errors="coerce").astype(int)
    train_df = d[event_values.isin(train_events)].copy()
    test_df = d[event_values.isin(test_events)].copy()

    train_label = "train{0}".format(pct_label(a.train_frac))
    test_label = "test{0}".format(pct_label(1.0 - a.train_frac))

    train_path = os.path.join(a.outdir, "{0}_{1}.csv".format(a.prefix, train_label))
    test_path = os.path.join(a.outdir, "{0}_{1}.csv".format(a.prefix, test_label))
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    rows = []
    for e in ev:
        rows.append({a.event_col: int(e), "split": "train" if int(e) in train_events else "test"})
    sel_path = os.path.join(a.outdir, a.selection_csv)
    pd.DataFrame(rows).to_csv(sel_path, index=False)

    print("input:", a.features)
    print("events before limit:", n_total_before)
    print("events after limit:", n_selected)
    print("train events:", len(train_events), "rows:", len(train_df))
    print("test events:", len(test_events), "rows:", len(test_df))
    print("wrote:", train_path)
    print("wrote:", test_path)
    print("wrote:", sel_path)


if __name__ == "__main__":
    main()
