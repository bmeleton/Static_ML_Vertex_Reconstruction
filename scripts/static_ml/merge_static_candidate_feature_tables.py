#!/usr/bin/env python3
"""
merge_static_candidate_feature_tables.py

Merge multiple static-candidate feature CSVs into one training table.

Why this exists:
  If you collect features for several simulated datasets separately, each
  dataset usually has event numbers 0, 1, 2, ...
  Before training one combined model, the event numbers must be made globally
  unique so that event-level train/validation splitting does not mix datasets
  incorrectly.

This script:
  - reads several feature CSVs
  - adds a dataset label column
  - preserves original event numbers in original_event
  - offsets event numbers to make them globally unique
  - optionally writes a dataset summary

Python 3.6 compatible.

Example:

  python3 merge_static_candidate_feature_tables.py \
    --input pp_high:static_candidate_pp_high_all/static_candidate_features.csv \
    --input pp_low:static_candidate_pp_low_all/static_candidate_features.csv \
    --input alphaalpha_low:static_candidate_alphaalpha_low_all/static_candidate_features.csv \
    --out static_candidate_combined_3datasets/static_candidate_features_combined.csv \
    --offset-step 1000000
"""

from __future__ import print_function

import argparse
import os
import sys

import pandas as pd


def ensure_dir_for_file(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)


def parse_input_spec(spec):
    # Supported:
    #   label:path
    #   label:offset:path
    parts = spec.split(":", 2)
    if len(parts) == 2:
        label, path = parts
        offset = None
    elif len(parts) == 3:
        label, offset_s, path = parts
        offset = int(offset_s)
    else:
        raise ValueError("Bad --input spec: {0}".format(spec))

    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError("Empty dataset label in --input spec: {0}".format(spec))
    if not path:
        raise ValueError("Empty path in --input spec: {0}".format(spec))
    return label, offset, path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Merge static-candidate feature CSVs from multiple datasets into one combined training table."
    )
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="Dataset input spec. Use label:path or label:offset:path. Repeat once per dataset.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output combined feature CSV.",
    )
    ap.add_argument(
        "--event-col",
        default="event",
        help="Event column to offset. Default: event",
    )
    ap.add_argument(
        "--dataset-col",
        default="dataset",
        help="Dataset-label column to create. Default: dataset",
    )
    ap.add_argument(
        "--original-event-col",
        default="original_event",
        help="Column that preserves the original event number. Default: original_event",
    )
    ap.add_argument(
        "--offset-step",
        type=int,
        default=1000000,
        help="Automatic event-number offset step when offset is not specified. Default: 1000000",
    )
    ap.add_argument(
        "--summary-out",
        default=None,
        help="Optional summary CSV. Default: <outdir>/combined_dataset_summary.csv",
    )
    args = ap.parse_args(argv)

    frames = []
    summary_rows = []

    for i, spec in enumerate(args.input):
        label, offset, path = parse_input_spec(spec)
        if offset is None:
            offset = i * args.offset_step

        if not os.path.exists(path):
            raise IOError("Input feature CSV does not exist: {0}".format(path))

        d = pd.read_csv(path)

        if args.event_col not in d.columns:
            raise RuntimeError(
                "Event column '{0}' not found in {1}".format(args.event_col, path)
            )

        d[args.dataset_col] = label
        d[args.original_event_col] = d[args.event_col]
        d["dataset_event_offset"] = offset
        d["source_feature_csv"] = path

        # Make event IDs globally unique.
        d[args.event_col] = d[args.event_col].astype(int) + int(offset)

        frames.append(d)

        summary_rows.append({
            "dataset": label,
            "source_feature_csv": path,
            "event_offset": offset,
            "rows": len(d),
            "original_events": d[args.original_event_col].nunique(),
            "global_events": d[args.event_col].nunique(),
            "global_event_min": d[args.event_col].min(),
            "global_event_max": d[args.event_col].max(),
        })

    if len(frames) == 0:
        raise RuntimeError("No inputs were read.")

    merged = pd.concat(frames, ignore_index=True, sort=False)

    ensure_dir_for_file(args.out)
    merged.to_csv(args.out, index=False)

    if args.summary_out is None:
        args.summary_out = os.path.join(os.path.dirname(os.path.abspath(args.out)), "combined_dataset_summary.csv")
    ensure_dir_for_file(args.summary_out)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.summary_out, index=False)

    print("")
    print("COMBINED STATIC-CANDIDATE FEATURE TABLE")
    print("output:", args.out)
    print("rows:", len(merged))
    print("global events:", merged[args.event_col].nunique())
    print("")
    print(summary.to_string(index=False))
    print("")
    print("dataset counts:")
    print(merged[args.dataset_col].value_counts().to_string())
    print("")
    print("summary:", args.summary_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
