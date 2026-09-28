#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
match_datasets_best_static_ml_events_analysis_valid.py

Select exactly N event IDs that:

  1. are present in EVERY supplied dataset,
  2. pass the same basic input cuts used by the SRIM comparison:
       - elastic_gate true, when present
       - finite E_cm
       - finite distance
       - distance > 0
       - ecm_min <= E_cm <= ecm_max
       - optional maximum distance
  3. are ranked by the static-ML selected-candidate quality,
     preferring smaller pred_dist_mm by default.

Then write filtered copies of every dataset containing the SAME EXACT
selected event IDs and write audit/event-list files.

Python 3.6+ compatible.
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd


EVENT_COL_CANDIDATES = [
    "event",
    "event_id",
    "Event",
    "EventID",
    "eventID",
    "EventId",
]

QUALITY_COL_CANDIDATES = [
    "pred_dist_mm",
    "predicted_dist_mm",
    "predicted_distance_mm",
    "pred_log1p_dist",
    "candidate_score",
]

ECM_COL_CANDIDATES = [
    "E_cm_elastic_equiv_MeV",
    "hough_E_cm_elastic_equiv_MeV",
    "E_cm_MeV",
    "Ecm_MeV",
    "Ecm",
    "E_cm",
]

DIST_COL_CANDIDATES = [
    "beam_path_from_entrance_mm",
    "hough_beam_path_from_entrance_mm",
    "reaction_distance_mm",
    "distance_mm",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Choose exactly N analysis-valid common events, "
            "ranked by static ML."
        )
    )

    p.add_argument(
        "--static-ml",
        required=True,
        help="static_candidate_selected_by_model.csv",
    )

    p.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset in NAME=PATH form. Repeat once per dataset. "
            "Example: --dataset ML=file1.csv --dataset Hough=file2.csv"
        ),
    )

    p.add_argument(
        "--n-events",
        type=int,
        default=700,
        help="Number of common events to retain. Default: 700",
    )

    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )

    p.add_argument(
        "--ecm-min",
        type=float,
        default=1.0,
        help="Minimum accepted E_cm. Default: 1 MeV",
    )

    p.add_argument(
        "--ecm-max",
        type=float,
        default=10.0,
        help="Maximum accepted E_cm. Default: 10 MeV",
    )

    p.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Optional maximum accepted beam/reaction distance",
    )

    p.add_argument(
        "--quality-col",
        default=None,
        help=(
            "Static-ML quality column. "
            "Default is auto-detection, preferring pred_dist_mm."
        ),
    )

    p.add_argument(
        "--quality-order",
        choices=["ascending", "descending"],
        default=None,
        help=(
            "Quality preference. Default: ascending for distance-like "
            "quantities, descending for score-like quantities."
        ),
    )

    return p.parse_args()


def parse_bool(v):
    if isinstance(v, bool):
        return v

    s = str(v).strip().lower()

    return s in (
        "1",
        "true",
        "t",
        "yes",
        "y",
    )


def parse_dataset_spec(spec):
    if "=" not in spec:
        raise RuntimeError(
            "--dataset must be NAME=PATH: {0}".format(spec)
        )

    name, path = spec.split("=", 1)

    name = name.strip()
    path = path.strip()

    if not name or not path:
        raise RuntimeError(
            "Bad --dataset specification: {0}".format(spec)
        )

    return name, path


def detect_col(
    df,
    candidates,
    label,
    path,
):
    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        "Could not find {0} column in {1}.\n"
        "Tried: {2}\n"
        "Available: {3}".format(
            label,
            path,
            ", ".join(candidates),
            ", ".join(df.columns),
        )
    )


def normalized_events(
    series,
    label,
):
    vals = pd.to_numeric(
        series,
        errors="coerce",
    )

    if vals.isna().any():
        raise RuntimeError(
            "{0}: nonnumeric event IDs found".format(label)
        )

    arr = vals.values.astype(float)
    rounded = np.rint(arr)

    if np.any(
        np.abs(arr - rounded) > 1.0e-9
    ):
        raise RuntimeError(
            "{0}: noninteger event IDs found".format(label)
        )

    return pd.Series(
        rounded.astype(np.int64),
        index=series.index,
    )


def detect_quality_col(
    df,
    requested,
):
    if requested:
        if requested not in df.columns:
            raise RuntimeError(
                "Quality column not found: {0}".format(
                    requested
                )
            )

        return requested

    for c in QUALITY_COL_CANDIDATES:
        if c in df.columns:
            return c

    raise RuntimeError(
        "Could not detect static-ML quality column.\n"
        "Tried: {0}".format(
            ", ".join(QUALITY_COL_CANDIDATES)
        )
    )


def quality_ascending(
    col,
    requested_order,
):
    if requested_order:
        return requested_order == "ascending"

    low = col.lower()

    # Distances/errors: smaller is better.
    # Generic scores: larger is better.
    if (
        "score" in low
        and "dist" not in low
        and "error" not in low
    ):
        return False

    return True


def sanitize(name):
    out = []

    for c in name:
        if (
            c.isalnum()
            or c in "_-"
        ):
            out.append(c)
        else:
            out.append("_")

    return "".join(out)


def analysis_valid_mask(
    df,
    ecm_col,
    dist_col,
    ecm_min,
    ecm_max,
    max_distance,
):
    ecm = pd.to_numeric(
        df[ecm_col],
        errors="coerce",
    )

    dist = pd.to_numeric(
        df[dist_col],
        errors="coerce",
    )

    ecm_values = ecm.values.astype(float)
    dist_values = dist.values.astype(float)

    mask = np.isfinite(ecm_values)

    mask &= np.isfinite(
        dist_values
    )

    mask &= (
        dist_values > 0.0
    )

    mask &= (
        ecm_values >= float(ecm_min)
    )

    mask &= (
        ecm_values <= float(ecm_max)
    )

    if max_distance is not None:
        mask &= (
            dist_values
            <= float(max_distance)
        )

    if "elastic_gate" in df.columns:
        gate = np.array(
            [
                parse_bool(v)
                for v
                in df["elastic_gate"].tolist()
            ],
            dtype=bool,
        )

        mask &= gate

    return mask


def main():
    args = parse_args()

    if args.n_events <= 0:
        sys.exit(
            "[ERROR] --n-events must be > 0"
        )

    if args.ecm_min >= args.ecm_max:
        sys.exit(
            "[ERROR] --ecm-min must be less than --ecm-max"
        )

    if not os.path.isfile(
        args.static_ml
    ):
        sys.exit(
            "[ERROR] Missing static-ML CSV: {0}".format(
                args.static_ml
            )
        )

    if not os.path.isdir(
        args.outdir
    ):
        os.makedirs(
            args.outdir
        )

    # ============================================================
    # Read the static-ML ranking information.
    # ============================================================

    ml = pd.read_csv(
        args.static_ml
    )

    ml_event_col = detect_col(
        ml,
        EVENT_COL_CANDIDATES,
        "event",
        args.static_ml,
    )

    qcol = detect_quality_col(
        ml,
        args.quality_col,
    )

    asc = quality_ascending(
        qcol,
        args.quality_order,
    )

    ml = ml.copy()

    ml["_event_norm"] = normalized_events(
        ml[ml_event_col],
        "Static ML",
    )

    ml["_quality_norm"] = pd.to_numeric(
        ml[qcol],
        errors="coerce",
    )

    qfinite = np.isfinite(
        ml["_quality_norm"].values.astype(float)
    )

    if not np.all(qfinite):
        print(
            "[WARN] dropping {0} static-ML rows "
            "with nonfinite {1}".format(
                int(
                    (~qfinite).sum()
                ),
                qcol,
            )
        )

        ml = ml.loc[
            qfinite
        ].copy()

    # Best static-ML row first.
    ml.sort_values(
        [
            "_quality_norm",
            "_event_norm",
        ],
        ascending=[
            asc,
            True,
        ],
        inplace=True,
    )

    # This file should already be one row per event,
    # but this protects against duplicates.
    ml_best = ml.drop_duplicates(
        "_event_norm",
        keep="first",
    ).copy()

    common_valid = set(
        int(x)
        for x
        in ml_best[
            "_event_norm"
        ].tolist()
    )

    print("")
    print("[STATIC ML]")
    print(
        "  file: {0}".format(
            args.static_ml
        )
    )
    print(
        "  event column: {0}".format(
            ml_event_col
        )
    )
    print(
        "  quality column: {0}".format(
            qcol
        )
    )
    print(
        "  quality preference: {0}".format(
            "smaller is better"
            if asc
            else "larger is better"
        )
    )
    print(
        "  usable unique events: {0}".format(
            len(common_valid)
        )
    )

    # ============================================================
    # Read every comparison dataset.
    # ============================================================

    dataset_infos = []

    print("")
    print("[ANALYSIS VALIDITY CUTS]")

    print(
        "  {0} <= E_cm <= {1} MeV".format(
            args.ecm_min,
            args.ecm_max,
        )
    )

    print(
        "  distance > 0 mm"
    )

    print(
        "  finite E_cm and distance"
    )

    print(
        "  elastic_gate = true when column exists"
    )

    if args.max_distance is not None:
        print(
            "  distance <= {0} mm".format(
                args.max_distance
            )
        )

    print("")
    print("[DATASETS]")

    seen_names = set()

    for spec in args.dataset:

        name, path = parse_dataset_spec(
            spec
        )

        if name in seen_names:
            sys.exit(
                "[ERROR] Duplicate dataset name: {0}".format(
                    name
                )
            )

        seen_names.add(
            name
        )

        if not os.path.isfile(
            path
        ):
            sys.exit(
                "[ERROR] Missing dataset: {0}".format(
                    path
                )
            )

        df = pd.read_csv(
            path
        )

        event_col = detect_col(
            df,
            EVENT_COL_CANDIDATES,
            "event",
            path,
        )

        ecm_col = detect_col(
            df,
            ECM_COL_CANDIDATES,
            "E_cm",
            path,
        )

        dist_col = detect_col(
            df,
            DIST_COL_CANDIDATES,
            "distance",
            path,
        )

        df = df.copy()

        df["_event_norm"] = normalized_events(
            df[event_col],
            "Dataset {0}".format(
                name
            ),
        )

        valid = analysis_valid_mask(
            df,
            ecm_col,
            dist_col,
            args.ecm_min,
            args.ecm_max,
            args.max_distance,
        )

        df[
            "_analysis_valid"
        ] = valid

        all_events = set(
            int(x)
            for x
            in df[
                "_event_norm"
            ].tolist()
        )

        valid_events = set(
            int(x)
            for x
            in df.loc[
                df["_analysis_valid"],
                "_event_norm",
            ].tolist()
        )

        print(
            "  {0}: rows={1}, "
            "unique={2}, "
            "analysis_valid_unique={3}".format(
                name,
                len(df),
                len(all_events),
                len(valid_events),
            )
        )

        print(
            "       event={0}, "
            "Ecm={1}, "
            "distance={2}".format(
                event_col,
                ecm_col,
                dist_col,
            )
        )

        dataset_infos.append(
            {
                "name": name,
                "path": path,
                "df": df,
                "event_col": event_col,
                "ecm_col": ecm_col,
                "dist_col": dist_col,
                "valid_events": valid_events,
            }
        )

        # Require event to pass in EVERY dataset.
        common_valid &= valid_events

    print("")
    print(
        "[COMMON ANALYSIS-VALID POOL]"
    )

    print(
        "  common valid events = {0}".format(
            len(common_valid)
        )
    )

    if len(common_valid) < args.n_events:
        sys.exit(
            "[ERROR] Need {0} events but only {1} events "
            "are present and analysis-valid in every dataset.".format(
                args.n_events,
                len(common_valid),
            )
        )

    # ============================================================
    # Rank the common pool using static ML.
    # ============================================================

    ranked = ml_best[
        ml_best[
            "_event_norm"
        ].isin(
            common_valid
        )
    ].copy()

    ranked.sort_values(
        [
            "_quality_norm",
            "_event_norm",
        ],
        ascending=[
            asc,
            True,
        ],
        inplace=True,
    )

    selected = ranked.head(
        args.n_events
    ).copy()

    selected[
        "_selection_rank"
    ] = np.arange(
        1,
        len(selected) + 1,
    )

    selected_events = [
        int(x)
        for x
        in selected[
            "_event_norm"
        ].tolist()
    ]

    selected_set = set(
        selected_events
    )

    if (
        len(selected_set)
        != args.n_events
    ):
        sys.exit(
            "[ERROR] Selected event set is not unique."
        )

    rank_map = dict(
        (
            ev,
            i + 1,
        )
        for i, ev
        in enumerate(
            selected_events
        )
    )

    # ============================================================
    # Write selected event audit information.
    # ============================================================

    audit_cols = []

    for c in [
        ml_event_col,
        "candidate_id",
        "candidate_source",
        "cand_x",
        "cand_y",
        "cand_z",
        qcol,
        "candidate_score",
    ]:
        if (
            c in selected.columns
            and c not in audit_cols
        ):
            audit_cols.append(
                c
            )

    audit = selected[
        audit_cols
    ].copy()

    audit.insert(
        0,
        "selection_rank",
        selected[
            "_selection_rank"
        ].values,
    )

    audit.insert(
        1,
        "event_normalized",
        selected[
            "_event_norm"
        ].values.astype(int),
    )

    ranked_csv = os.path.join(
        args.outdir,
        "selected_{0}_events_ranked_by_static_ml.csv".format(
            args.n_events
        ),
    )

    audit.to_csv(
        ranked_csv,
        index=False,
    )

    ranked_txt = os.path.join(
        args.outdir,
        "selected_{0}_event_ids_ranked.txt".format(
            args.n_events
        ),
    )

    with open(
        ranked_txt,
        "w",
    ) as f:

        for ev in selected_events:
            f.write(
                "{0}\n".format(
                    ev
                )
            )

    sorted_txt = os.path.join(
        args.outdir,
        "selected_{0}_event_ids_sorted.txt".format(
            args.n_events
        ),
    )

    with open(
        sorted_txt,
        "w",
    ) as f:

        for ev in sorted(
            selected_set
        ):
            f.write(
                "{0}\n".format(
                    ev
                )
            )

    # ============================================================
    # Filter every dataset to exactly the same events.
    # ============================================================

    print("")
    print(
        "[FILTERED OUTPUTS]"
    )

    output_sets = []

    for info in dataset_infos:

        df = info["df"]
        name = info["name"]

        out = df[
            df[
                "_event_norm"
            ].isin(
                selected_set
            )
            & df[
                "_analysis_valid"
            ]
        ].copy()

        out[
            "_selection_rank_tmp"
        ] = out[
            "_event_norm"
        ].map(
            rank_map
        )

        out.sort_values(
            [
                "_selection_rank_tmp",
                "_event_norm",
            ],
            inplace=True,
        )

        out_set = set(
            int(x)
            for x
            in out[
                "_event_norm"
            ].tolist()
        )

        if out_set != selected_set:

            missing = sorted(
                selected_set
                - out_set
            )

            sys.exit(
                "[ERROR] {0}: output missing selected events: {1}".format(
                    name,
                    missing[:20],
                )
            )

        duplicates = int(
            out[
                "_event_norm"
            ].duplicated().sum()
        )

        if duplicates:

            print(
                "[WARN] {0}: {1} duplicate event rows retained; "
                "unique event count is still exactly {2}.".format(
                    name,
                    duplicates,
                    args.n_events,
                )
            )

        out.drop(
            columns=[
                "_event_norm",
                "_analysis_valid",
                "_selection_rank_tmp",
            ],
            inplace=True,
            errors="ignore",
        )

        outpath = os.path.join(
            args.outdir,
            "{0}_matched_{1}_analysis_valid.csv".format(
                sanitize(name),
                args.n_events,
            ),
        )

        out.to_csv(
            outpath,
            index=False,
        )

        output_sets.append(
            out_set
        )

        print(
            "  {0}: rows={1}, unique_events={2}".format(
                name,
                len(out),
                len(out_set),
            )
        )

        print(
            "       {0}".format(
                outpath
            )
        )

    # ============================================================
    # Final exact-set verification.
    # ============================================================

    if not all(
        s == selected_set
        for s
        in output_sets
    ):
        sys.exit(
            "[ERROR] Final exact event-set verification failed."
        )

    summary = os.path.join(
        args.outdir,
        "matching_summary.txt",
    )

    with open(
        summary,
        "w",
    ) as f:

        f.write(
            "Requested events: {0}\n".format(
                args.n_events
            )
        )

        f.write(
            "Common analysis-valid pool: {0}\n".format(
                len(common_valid)
            )
        )

        f.write(
            "Ecm range: {0} to {1} MeV\n".format(
                args.ecm_min,
                args.ecm_max,
            )
        )

        f.write(
            "Distance requirement: > 0 mm\n"
        )

        f.write(
            "Quality column: {0}\n".format(
                qcol
            )
        )

        f.write(
            "Quality preference: {0}\n".format(
                "smaller is better"
                if asc
                else "larger is better"
            )
        )

        f.write(
            "All output datasets same event IDs: YES\n"
        )

        f.write(
            "\nSelected event IDs:\n"
        )

        for ev in selected_events:
            f.write(
                "{0}\n".format(
                    ev
                )
            )

    print("")
    print(
        "============================================================"
    )
    print(
        "MATCHING COMPLETE"
    )
    print(
        "============================================================"
    )

    print(
        "  selected unique events: {0}".format(
            len(selected_set)
        )
    )

    print(
        "  all inputs analysis-valid for these events: YES"
    )

    print(
        "  exact same event IDs in every output: YES"
    )

    print(
        "  ranked list: {0}".format(
            ranked_csv
        )
    )

    print(
        "  ranked IDs: {0}".format(
            ranked_txt
        )
    )

    print(
        "  sorted IDs: {0}".format(
            sorted_txt
        )
    )

    print(
        "  summary: {0}".format(
            summary
        )
    )

    print("")
    print(
        "Selected event IDs, best static-ML quality first:"
    )

    for ev in selected_events:
        print(
            ev
        )


if __name__ == "__main__":
    main()
