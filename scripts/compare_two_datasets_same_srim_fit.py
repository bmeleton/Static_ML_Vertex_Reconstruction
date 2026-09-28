#!/usr/bin/env python3
"""
compare_two_datasets_same_srim_fit.py

Fit a fixed-density SRIM E_cm-vs-distance curve to DATASET 1 only, freeze that
exact fit, then evaluate DATASET 2 against the SAME unchanged curve.

This is useful for comparing two reconstruction/measurement methods while
requiring that both be judged against one identical physical calibration.

Workflow
--------
1. Read one SRIM stopping table.
2. Read reference dataset 1.
3. Read comparison dataset 2.
4. Use a USER-SPECIFIED fixed gas density.
5. Fit ONLY the horizontal shift x0 to dataset 1.
6. Freeze density and x0.
7. Apply that exact same SRIM curve to dataset 2 with NO refit.
8. Report RMSE/MAE/median absolute residual for both datasets.
9. Plot both datasets together with the one common SRIM curve.
10. Make one horizontal reaction-distance residual histogram for each method.
11. Make one true transverse (shortest-distance) spread histogram for each
    method in normalized arbitrary units.
12. Preserve the optional Gaussian summary/fitting machinery.
13. Save per-event residual CSVs and a summary text file.
14. By default, force both datasets to use the same common event_id set.
15. Optionally cap the common event set with --max-events.

Supported column naming
-----------------------
The script automatically recognizes both unprefixed and prefixed names, e.g.:

    E_cm_elastic_equiv_MeV
    beam_path_from_entrance_mm

and:

    hough_E_cm_elastic_equiv_MeV
    hough_beam_path_from_entrance_mm

It also searches by suffix, so other prefixes are supported.

Usage:
python3 compare_two_datasets_same_srim_fit_publication.py SRIM_O14_in_96He_4CO2_580Torr.txt --data1 alphas_analysis/alpha_cm_pressure_combined_static_ml/alpha_cm_reconstruction.csv --data2 alphas_analysis/alpha_cm_pressure_combined_static_ml/alpha_cm_hough_reconstruction.csv --density 1.78e-4 --ecm-min 1 --ecm-max 10 --exclude-events-file excluded_alpha_events.txt --drop-data2-events-excluded-from-data1-fit --output-prefix srim_pressure_combined_ml_vs_hough --clip-sigma 0

Python 3.6 / SciPy 1.5 compatible.
"""

from __future__ import print_function

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Publication-oriented Matplotlib defaults.
# The PNG output defaults to 500 dpi, and a vector PDF is also written.
plt.rcParams.update({
    "font.size": 12.0,
    "axes.titlesize": 14.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 10.5,
    "figure.titlesize": 16.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})

try:
    from scipy.interpolate import PchipInterpolator
    try:
        from scipy.integrate import cumulative_trapezoid
    except ImportError:
        from scipy.integrate import cumtrapz as cumulative_trapezoid
except ImportError as exc:
    raise SystemExit("SciPy import failed: {}".format(exc))


ENERGY_TO_MEV = {
    "ev": 1.0e-6,
    "kev": 1.0e-3,
    "mev": 1.0,
    "gev": 1.0e3,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fit dataset 1 with a fixed-density SRIM curve, then evaluate "
            "dataset 2 using the exact same frozen curve."
        )
    )

    p.add_argument(
        "srim_file",
        type=Path,
        help="Native SRIM stopping/range text file",
    )

    p.add_argument(
        "--data1",
        type=Path,
        required=True,
        help="Reference dataset: this dataset determines the horizontal fit",
    )

    p.add_argument(
        "--data2",
        type=Path,
        required=True,
        help="Comparison dataset: evaluated with NO refitting",
    )

    p.add_argument(
        "--label1",
        default="Reference method",
        help="Legend/display label for dataset 1",
    )

    p.add_argument(
        "--label2",
        default="Comparison method",
        help="Legend/display label for dataset 2",
    )

    p.add_argument(
        "--density",
        type=float,
        required=True,
        help="Fixed gas density in g/cm^3, e.g. 1.78e-4",
    )

    p.add_argument(
        "--projectile-mass",
        type=float,
        default=14.0,
        help="Projectile mass in u (default 14)",
    )

    p.add_argument(
        "--target-mass",
        type=float,
        default=4.0,
        help="Target mass in u (default 4)",
    )

    p.add_argument(
        "--ecm-min",
        type=float,
        default=1.0,
        help="Minimum E_cm used/evaluated, MeV",
    )

    p.add_argument(
        "--ecm-max",
        type=float,
        default=10.0,
        help="Maximum E_cm used/evaluated, MeV",
    )

    p.add_argument(
        "--distance-column1",
        default=None,
        help="Optional explicit distance column for dataset 1",
    )

    p.add_argument(
        "--distance-column2",
        default=None,
        help="Optional explicit distance column for dataset 2",
    )

    p.add_argument(
        "--ecm-column1",
        default=None,
        help="Optional explicit E_cm column for dataset 1",
    )

    p.add_argument(
        "--ecm-column2",
        default=None,
        help="Optional explicit E_cm column for dataset 2",
    )

    p.add_argument(
        "--entrance-z1",
        type=float,
        default=-80.0,
        help="Fallback entrance z for dataset 1 if only vertex-z exists",
    )

    p.add_argument(
        "--entrance-z2",
        type=float,
        default=-80.0,
        help="Fallback entrance z for dataset 2 if only vertex-z exists",
    )

    p.add_argument(
        "--no-elastic-gate",
        action="store_true",
        help="Ignore elastic_gate in both datasets if present",
    )

    p.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Optional maximum accepted distance in both datasets",
    )

    p.add_argument(
        "--clip-sigma",
        type=float,
        default=3.5,
        help=(
            "MAD-based clipping threshold used ONLY while fitting dataset 1. "
            "Set <=0 to disable."
        ),
    )

    p.add_argument(
        "--clip-iterations",
        type=int,
        default=5,
        help="Maximum clipping iterations for dataset 1",
    )

    p.add_argument(
        "--clip-data2",
        action="store_true",
        help=(
            "If set, compute an additional clipped metric for dataset 2. "
            "The curve is still NOT refitted."
        ),
    )

    p.add_argument(
        "--metrics-use-all",
        action="store_true",
        help=(
            "Use all finite residuals for the primary reported Dataset 1 "
            "RMSE/RMS/MAE metrics, even if some points were excluded from the "
            "x0 fit by the MAD clip. The x0 fit itself still uses the clipped "
            "fit mask. This is recommended when outliers should remain visible "
            "in the RMS/RMSE."
        ),
    )

    p.add_argument(
        "--drop-data2-events-excluded-from-data1-fit",
        action="store_true",
        help=(
            "When Dataset 1 outliers are excluded from the x0 fit, also exclude "
            "the matching event_id entries from Dataset 2 primary metrics and "
            "plotting. This is useful for a fair ML-vs-Hough comparison when "
            "the same physical event should be removed from both methods."
        ),
    )

    p.add_argument(
        "--exclude-events-file",
        type=Path,
        default=None,
        help=(
            "Optional text file containing event IDs to remove from BOTH "
            "datasets before fitting, plotting, and metric calculations. "
            "Format: one event ID per line. Blank lines and lines beginning "
            "with # are ignored."
        ),
    )

    p.add_argument(
        "--require-common-events",
        dest="require_common_events",
        action="store_true",
        default=True,
        help=(
            "Require Dataset 1 and Dataset 2 to use exactly the same event_id "
            "set before fitting, plotting, and primary metric calculations. "
            "Default: enabled."
        ),
    )

    p.add_argument(
        "--no-require-common-events",
        dest="require_common_events",
        action="store_false",
        help=(
            "Disable common-event intersection filtering. Not recommended for "
            "method comparisons."
        ),
    )

    p.add_argument(
        "--max-events",
        type=int,
        default=0,
        help=(
            "After manual exclusions and common-event filtering, keep at most "
            "this many events from both datasets. 0 means all. Example: 700."
        ),
    )

    p.add_argument(
        "--event-selection",
        choices=("first", "random"),
        default="first",
        help=(
            "How to choose events when --max-events is positive. first keeps "
            "the first common event IDs in Dataset 1 order; random uses "
            "--event-selection-seed. Default: first."
        ),
    )

    p.add_argument(
        "--event-selection-seed",
        type=int,
        default=12345,
        help="Random seed used only when --event-selection random.",
    )

    p.add_argument(
        "--output-prefix",
        default="srim_same_fit_two_dataset_comparison",
        help="Output filename prefix",
    )

    p.add_argument(
        "--spread-bins",
        type=int,
        default=75,
        help=(
            "Number of bins in each transverse residual histogram "
            "(default 35)."
        ),
    )

    p.add_argument(
        "--spread-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_MM", "MAX_MM"),
        help=(
            "Optional common x-axis/histogram range for the two transverse "
            "spread plots. If omitted, a common symmetric range about zero "
            "is determined from the residuals used in the primary metrics."
        ),
    )

    p.add_argument(
        "--plot-x-range",
        type=float,
        nargs=2,
        default=(90.0, 230.0),
        metavar=("MIN_MM", "MAX_MM"),
        help=(
            "X-axis range for the main E_cm-vs-distance plot. The same range "
            "is used to normalize the distance coordinate for the true "
            "transverse-spread calculation (default 90 230 mm)."
        ),
    )

    p.add_argument(
        "--true-transverse-bins",
        type=int,
        default=None,
        help=(
            "Number of bins in the true transverse-spread histograms. "
            "If omitted, --spread-bins is used."
        ),
    )

    p.add_argument(
        "--true-transverse-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_AU", "MAX_AU"),
        help=(
            "Optional common histogram range for the true transverse residual "
            "in normalized arbitrary units. If omitted, a common symmetric "
            "range about zero is determined automatically."
        ),
    )

    p.add_argument(
        "--fit-true-transverse-gaussian",
        action="store_true",
        help=(
            "Overlay a Gaussian fit on the true transverse-spread histograms. "
            "By default the true transverse plots show only the histogram and "
            "the zero/SRIM reference."
        ),
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=500,
        help=(
            "PNG output resolution in dots per inch. Default: 500, appropriate "
            "for publication combination artwork. Vector PDF output is also saved."
        ),
    )

    p.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )

    return p.parse_args()


def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def parse_bool(x):
    return str(x).strip().lower() in (
        "1", "true", "t", "yes", "y"
    )


def parse_srim(path):
    text = path.read_text(errors="replace")

    ion_name = None
    ion_mass = None
    srim_density = None

    m = re.search(
        r"Ion\s*=\s*([^\[,]+).*?Mass\s*=\s*([0-9.+\-Ee]+)\s*amu",
        text,
        re.IGNORECASE,
    )
    if m:
        ion_name = m.group(1).strip()
        ion_mass = float(m.group(2))

    m = re.search(
        r"Target Density\s*=\s*([0-9.+\-Ee]+)\s*g/cm3",
        text,
        re.IGNORECASE,
    )
    if m:
        srim_density = float(m.group(1))

    row_re = re.compile(
        r"^\s*"
        r"([0-9.]+)\s+"
        r"(eV|keV|MeV|GeV)\s+"
        r"([0-9.+\-Ee]+)\s+"
        r"([0-9.+\-Ee]+)\s+",
        re.IGNORECASE,
    )

    energies = []
    stopping = []

    for line in text.splitlines():
        m = row_re.match(line)
        if not m:
            continue

        energy_mev = (
            float(m.group(1))
            * ENERGY_TO_MEV[m.group(2).lower()]
        )

        stopping_mass = (
            float(m.group(3))
            + float(m.group(4))
        )

        energies.append(energy_mev)
        stopping.append(stopping_mass)

    if len(energies) < 5:
        raise ValueError(
            "Could not parse enough SRIM stopping-power rows."
        )

    E = np.asarray(energies, dtype=float)
    S = np.asarray(stopping, dtype=float)

    order = np.argsort(E)
    E = E[order]
    S = S[order]

    E_unique, idx = np.unique(
        E,
        return_index=True,
    )
    E = E_unique
    S = S[idx]

    return (
        E,
        S,
        ion_name,
        ion_mass,
        srim_density,
    )


def choose_column(
    fields,
    requested,
    candidates,
    label,
    suffixes=None,
):
    if requested:
        if requested not in fields:
            raise ValueError(
                "{} column '{}' not found. Available: {}".format(
                    label,
                    requested,
                    ", ".join(fields),
                )
            )
        return requested

    for name in candidates:
        if name in fields:
            return name

    if suffixes:
        matches = []

        for field in fields:
            for suffix in suffixes:
                if field.endswith(suffix):
                    matches.append(field)
                    break

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise ValueError(
                "Multiple possible {} columns matched: {}. "
                "Select one explicitly."
                .format(
                    label,
                    ", ".join(matches),
                )
            )

    raise ValueError(
        "Could not find {} column. Available: {}".format(
            label,
            ", ".join(fields),
        )
    )


def choose_distance_column(
    fields,
    requested,
):
    if requested:
        if requested not in fields:
            raise ValueError(
                "Requested distance column '{}' not found."
                .format(requested)
            )
        return requested, "direct"

    exact_direct = [
        "beam_path_from_entrance_mm",
        "hough_beam_path_from_entrance_mm",
        "distance_from_entrance_mm",
    ]

    for name in exact_direct:
        if name in fields:
            return name, "direct"

    suffix_direct = [
        name for name in fields
        if name.endswith(
            "beam_path_from_entrance_mm"
        )
    ]

    if len(suffix_direct) == 1:
        return suffix_direct[0], "direct"

    if len(suffix_direct) > 1:
        raise ValueError(
            "Multiple possible beam-path columns: {}"
            .format(
                ", ".join(suffix_direct)
            )
        )

    exact_z = [
        "vertex_z_mm",
        "hough_vertex_z_mm",
        "vertexZ",
    ]

    for name in exact_z:
        if name in fields:
            return name, "z"

    suffix_z = [
        name for name in fields
        if name.endswith(
            "vertex_z_mm"
        )
    ]

    if len(suffix_z) == 1:
        return suffix_z[0], "z"

    if len(suffix_z) > 1:
        raise ValueError(
            "Multiple possible vertex-z columns: {}"
            .format(
                ", ".join(suffix_z)
            )
        )

    raise ValueError(
        "No beam-path distance or vertex-z column found."
    )


def read_dataset(
    path,
    requested_ecm,
    requested_distance,
    entrance_z,
    a,
):
    with path.open(
        "r",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        ecm_col = choose_column(
            fields,
            requested_ecm,
            [
                "E_cm_elastic_equiv_MeV",
                "hough_E_cm_elastic_equiv_MeV",
                "E_cm_MeV",
                "Ecm_MeV",
                "Ecm",
                "E_cm",
            ],
            "E_cm",
            suffixes=[
                "E_cm_elastic_equiv_MeV",
            ],
        )

        dist_col, mode = choose_distance_column(
            fields,
            requested_distance,
        )

        x = []
        y = []
        ids = []

        for i, row in enumerate(reader):
            if (
                not a.no_elastic_gate
                and "elastic_gate" in fields
                and not parse_bool(
                    row.get(
                        "elastic_gate",
                        "",
                    )
                )
            ):
                continue

            ecm = safe_float(
                row.get(ecm_col)
            )

            raw_x = safe_float(
                row.get(dist_col)
            )

            if (
                ecm is None
                or raw_x is None
            ):
                continue

            if mode == "z":
                distance = (
                    raw_x
                    - entrance_z
                )
            else:
                distance = raw_x

            if distance <= 0:
                continue

            if (
                ecm < a.ecm_min
                or ecm > a.ecm_max
            ):
                continue

            if (
                a.max_distance is not None
                and distance > a.max_distance
            ):
                continue

            x.append(distance)
            y.append(ecm)

            ids.append(
                row.get(
                    "event_id",
                    row.get(
                        "OriginalEvent",
                        str(i),
                    ),
                )
            )

    if len(x) < 5:
        raise ValueError(
            "{} has too few usable events."
            .format(path)
        )

    return {
        "x": np.asarray(
            x,
            dtype=float,
        ),
        "ecm": np.asarray(
            y,
            dtype=float,
        ),
        "ids": np.asarray(ids),
        "ecm_col": ecm_col,
        "dist_col": dist_col,
        "mode": mode,
        "path": path,
    }


def read_excluded_event_ids(path):
    excluded = set()

    if path is None:
        return excluded

    with path.open("r") as f:
        for line in f:
            s = line.strip()

            if not s:
                continue

            if s.startswith("#"):
                continue

            # Keep event IDs as strings so this works with either integer
            # event IDs or string event labels.
            excluded.add(str(s))

    return excluded


def filter_dataset_by_excluded_events(dataset, excluded_ids, label):
    if not excluded_ids:
        return dataset, 0

    ids_as_str = np.asarray(
        [str(x) for x in dataset["ids"]]
    )

    keep = np.asarray(
        [
            event_id not in excluded_ids
            for event_id in ids_as_str
        ],
        dtype=bool,
    )

    removed = int(
        len(keep)
        - keep.sum()
    )

    if removed == 0:
        return dataset, 0

    filtered = dict(dataset)
    filtered["x"] = dataset["x"][keep]
    filtered["ecm"] = dataset["ecm"][keep]
    filtered["ids"] = dataset["ids"][keep]

    if len(filtered["x"]) < 5:
        raise ValueError(
            "{} has too few usable events after applying --exclude-events-file."
            .format(label)
        )

    return filtered, removed




def canonical_event_id(value):
    return str(value)


def filter_dataset_by_keep_event_ids(dataset, keep_ids, label):
    keep_ids=set([canonical_event_id(x) for x in keep_ids])
    ids_as_str=np.asarray([canonical_event_id(x) for x in dataset["ids"]])
    keep=np.asarray([x in keep_ids for x in ids_as_str],dtype=bool)
    removed=int(len(keep)-keep.sum())

    filtered=dict(dataset)
    filtered["x"]=dataset["x"][keep]
    filtered["ecm"]=dataset["ecm"][keep]
    filtered["ids"]=dataset["ids"][keep]

    if len(filtered["x"])<5:
        raise ValueError(
            "{} has too few usable events after common/max-event filtering."
            .format(label)
        )

    return filtered, removed


def ordered_unique_ids(ids):
    seen=set()
    out=[]
    for eid in ids:
        key=canonical_event_id(eid)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def common_event_ids_in_dataset1_order(d1,d2):
    ids2=set([canonical_event_id(x) for x in d2["ids"]])
    return [eid for eid in ordered_unique_ids(d1["ids"]) if eid in ids2]


def choose_event_subset(event_ids,max_events,mode,seed):
    event_ids=list(event_ids)
    if int(max_events)<=0 or int(max_events)>=len(event_ids):
        return event_ids

    n=int(max_events)

    if mode=="random":
        rng=np.random.RandomState(int(seed))
        idx=np.arange(len(event_ids))
        rng.shuffle(idx)
        chosen=[event_ids[i] for i in idx[:n]]
        # Restore Dataset-1/common order so output is deterministic and tidy.
        chosen_set=set(chosen)
        return [eid for eid in event_ids if eid in chosen_set]

    return event_ids[:n]


def enforce_common_and_max_events(d1,d2,a):
    before1=len(d1["x"])
    before2=len(d2["x"])

    if a.require_common_events:
        common_ids=common_event_ids_in_dataset1_order(d1,d2)
    else:
        common_ids=ordered_unique_ids(d1["ids"])

    before_common=len(common_ids)
    selected_ids=choose_event_subset(
        common_ids,
        a.max_events,
        a.event_selection,
        a.event_selection_seed,
    )

    if a.require_common_events or int(a.max_events)>0:
        d1, removed1=filter_dataset_by_keep_event_ids(
            d1,
            selected_ids,
            "Dataset 1",
        )
        d2, removed2=filter_dataset_by_keep_event_ids(
            d2,
            selected_ids,
            "Dataset 2",
        )
    else:
        removed1=0
        removed2=0

    print()
    print("[INFO] Common-event comparison:")
    print("       require common events =", bool(a.require_common_events))
    print("       Dataset 1 events before common/max filter =", before1)
    print("       Dataset 2 events before common/max filter =", before2)
    print("       common event IDs before max filter =", before_common)
    if int(a.max_events)>0:
        print("       max-events requested =", int(a.max_events))
        print("       event selection mode =", a.event_selection)
    print("       common event IDs selected =", len(selected_ids))
    print("       Dataset 1 events removed by common/max filter =", removed1)
    print("       Dataset 2 events removed by common/max filter =", removed2)

    return d1,d2,{
        "before1":before1,
        "before2":before2,
        "common_before_max":before_common,
        "selected_common":len(selected_ids),
        "removed1":removed1,
        "removed2":removed2,
        "selected_ids":selected_ids,
    }

def build_areal_coordinate(
    E_tab,
    S_tab,
    ecm_values,
    cm_factor,
    ecm_ref,
):
    ebeam = (
        ecm_values
        / cm_factor
    )

    eref = (
        ecm_ref
        / cm_factor
    )

    emin = min(
        float(
            np.min(ebeam)
        ),
        eref,
    )

    emax = max(
        float(
            np.max(ebeam)
        ),
        eref,
    )

    if (
        emin < E_tab.min()
        or emax > E_tab.max()
    ):
        raise ValueError(
            "Required beam-energy range [{:.6g}, {:.6g}] MeV "
            "is outside SRIM range [{:.6g}, {:.6g}] MeV."
            .format(
                emin,
                emax,
                E_tab.min(),
                E_tab.max(),
            )
        )

    stop_interp = PchipInterpolator(
        E_tab,
        S_tab,
        extrapolate=False,
    )

    grid = np.linspace(
        emin,
        emax,
        30000,
    )

    stop = stop_interp(grid)

    if (
        np.any(
            ~np.isfinite(stop)
        )
        or np.any(
            stop <= 0
        )
    ):
        raise ValueError(
            "Invalid SRIM stopping-power interpolation."
        )

    integ = cumulative_trapezoid(
        1.0 / stop,
        grid,
        initial=0.0,
    )

    integ_interp = PchipInterpolator(
        grid,
        integ,
        extrapolate=False,
    )

    I_ref = float(
        integ_interp(eref)
    )

    I_E = integ_interp(
        ebeam
    )

    return np.asarray(
        I_ref - I_E,
        dtype=float,
    )


def robust_sigma(values):
    med = np.median(values)
    mad = np.median(
        np.abs(
            values - med
        )
    )

    if mad <= 0:
        return 0.0

    return (
        1.4826
        * mad
    )


def fit_reference_shift(
    x_data,
    d_srim,
    a,
):
    mask = np.ones(
        len(x_data),
        dtype=bool,
    )

    for iteration in range(
        max(
            0,
            a.clip_iterations,
        )
    ):
        x0 = float(
            np.mean(
                x_data[mask]
                - d_srim[mask]
            )
        )

        pred = (
            x0
            + d_srim
        )

        resid = (
            x_data
            - pred
        )

        good = (
            mask
            & np.isfinite(resid)
        )

        if a.clip_sigma <= 0:
            break

        sigma = robust_sigma(
            resid[good]
        )

        if sigma <= 0:
            break

        med = np.median(
            resid[good]
        )

        new_mask = (
            mask
            & np.isfinite(resid)
            & (
                np.abs(
                    resid - med
                )
                <= (
                    a.clip_sigma
                    * sigma
                )
            )
        )

        removed = int(
            mask.sum()
            - new_mask.sum()
        )

        print(
            "[INFO] Reference clip {}: "
            "x0={:.4f} mm, "
            "sigma_x={:.4f} mm, "
            "removed {}"
            .format(
                iteration + 1,
                x0,
                sigma,
                removed,
            )
        )

        if np.array_equal(
            new_mask,
            mask,
        ):
            break

        mask = new_mask

    x0 = float(
        np.mean(
            x_data[mask]
            - d_srim[mask]
        )
    )

    return (
        x0,
        mask,
    )


def metrics(
    residual,
    mask=None,
):
    if mask is None:
        mask = np.isfinite(
            residual
        )
    else:
        mask = (
            mask
            & np.isfinite(
                residual
            )
        )

    r = residual[mask]

    if len(r) == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "median_abs": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    return {
        "n": len(r),
        "rmse": float(
            np.sqrt(
                np.mean(
                    r ** 2
                )
            )
        ),
        "mae": float(
            np.mean(
                np.abs(r)
            )
        ),
        "median_abs": float(
            np.median(
                np.abs(r)
            )
        ),
        "mean": float(
            np.mean(r)
        ),
        "std": float(
            np.std(
                r,
                ddof=1,
            )
        )
        if len(r) > 1
        else 0.0,
    }


def data2_clip_mask(
    residual,
    sigma_cut,
):
    good = np.isfinite(
        residual
    )

    if (
        sigma_cut <= 0
        or good.sum() < 5
    ):
        return good

    med = np.median(
        residual[good]
    )

    sigma = robust_sigma(
        residual[good]
    )

    if sigma <= 0:
        return good

    return (
        good
        & (
            np.abs(
                residual - med
            )
            <= sigma_cut * sigma
        )
    )


def write_event_csv(
    path,
    dataset,
    pred,
    residual,
    used_mask,
):
    with path.open(
        "w",
        newline="",
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "event_id",
            "distance_data_mm",
            "Ecm_data_MeV",
            "distance_same_SRIM_fit_mm",
            "distance_residual_mm",
            "used_in_metric",
        ])

        for i in range(
            len(dataset["x"])
        ):
            w.writerow([
                dataset["ids"][i],
                dataset["x"][i],
                dataset["ecm"][i],
                pred[i],
                residual[i],
                bool(
                    used_mask[i]
                ),
            ])


def gaussian_fit_to_residuals(residual, mask=None):
    """
    Unbinned maximum-likelihood Gaussian fit to the horizontal distance
    residuals about the common SRIM curve.

    The residual is

        Delta d = d_reco - d_SRIM(E_cm)

    so zero corresponds exactly to the SRIM curve.  The fitted mean measures
    any horizontal bias and sigma measures the transverse spread.
    """
    residual = np.asarray(residual, dtype=float)

    if mask is None:
        good = np.isfinite(residual)
    else:
        good = np.asarray(mask, dtype=bool) & np.isfinite(residual)

    r = residual[good]

    if len(r) == 0:
        return {
            "n": 0,
            "mu": np.nan,
            "sigma": np.nan,
            "fwhm": np.nan,
            "values": r,
        }

    mu = float(np.mean(r))

    # ddof=0 is the maximum-likelihood estimate for a normal distribution.
    sigma = float(np.sqrt(np.mean((r - mu) ** 2)))

    return {
        "n": int(len(r)),
        "mu": mu,
        "sigma": sigma,
        "fwhm": float(2.354820045 * sigma),
        "values": r,
    }


def common_spread_range(fit1, fit2, requested_range=None):
    """Choose one x range for both spread plots so widths are comparable."""
    if requested_range is not None:
        lo = float(requested_range[0])
        hi = float(requested_range[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("--spread-range requires MIN_MM < MAX_MM")
        return (lo, hi)

    vals = []
    if fit1["n"]:
        vals.append(fit1["values"])
    if fit2["n"]:
        vals.append(fit2["values"])

    if not vals:
        return (-1.0, 1.0)

    all_values = np.concatenate(vals)
    max_abs = float(np.max(np.abs(all_values)))

    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    # Slight visual padding while keeping zero exactly centered.
    max_abs *= 1.05
    return (-max_abs, max_abs)



def true_transverse_residuals(
    x_data,
    ecm_data,
    x_curve,
    ecm_curve,
    x_range,
    y_range,
):
    """
    Compute the signed shortest Euclidean distance from each data point to the
    common SRIM curve AFTER both axes are normalized to dimensionless plot
    coordinates.

    Normalization:
        x_n = (x - x_min) / (x_max - x_min)
        y_n = (Ecm - y_min) / (y_max - y_min)

    The shortest point-to-polyline distance in this normalized plane is the
    "true transverse" residual reported here.  Its unit is therefore arbitrary
    (a.u.): one normalized unit corresponds to the full selected x-axis span
    or the full selected y-axis span.

    The magnitude is the genuine shortest distance to the SRIM polyline in the
    normalized 2D geometry.  The sign is chosen so that positive values
    generally correspond to a reconstructed distance lying to the right
    (larger x) of the closest point on the SRIM curve.

    This is intentionally NOT a physical distance in mm because the original
    plot mixes mm and MeV on its two axes.
    """
    x_data = np.asarray(x_data, dtype=float)
    ecm_data = np.asarray(ecm_data, dtype=float)
    x_curve = np.asarray(x_curve, dtype=float)
    ecm_curve = np.asarray(ecm_curve, dtype=float)

    xmin, xmax = map(float, x_range)
    ymin, ymax = map(float, y_range)

    if not (
        np.isfinite(xmin)
        and np.isfinite(xmax)
        and np.isfinite(ymin)
        and np.isfinite(ymax)
        and xmax > xmin
        and ymax > ymin
    ):
        raise ValueError(
            "Invalid normalization ranges for true transverse spread."
        )

    # Convert data and curve to a common dimensionless coordinate system.
    px = (x_data - xmin) / (xmax - xmin)
    py = (ecm_data - ymin) / (ymax - ymin)

    cx = (x_curve - xmin) / (xmax - xmin)
    cy = (ecm_curve - ymin) / (ymax - ymin)

    curve = np.column_stack((cx, cy))

    # Remove non-finite curve points before forming the polyline.
    finite_curve = np.all(np.isfinite(curve), axis=1)
    curve = curve[finite_curve]

    if len(curve) < 2:
        raise ValueError(
            "Too few finite SRIM-curve points for transverse-distance calculation."
        )

    seg_a = curve[:-1]
    seg_b = curve[1:]
    seg_v = seg_b - seg_a
    seg_len2 = np.sum(seg_v * seg_v, axis=1)

    valid_seg = np.isfinite(seg_len2) & (seg_len2 > 0.0)
    seg_a = seg_a[valid_seg]
    seg_v = seg_v[valid_seg]
    seg_len2 = seg_len2[valid_seg]

    if len(seg_a) == 0:
        raise ValueError(
            "No finite nonzero SRIM-curve segments are available."
        )

    out = np.full(len(x_data), np.nan, dtype=float)

    for i in range(len(x_data)):
        if not (
            np.isfinite(px[i])
            and np.isfinite(py[i])
        ):
            continue

        p = np.array([px[i], py[i]], dtype=float)

        ap = p[None, :] - seg_a

        t = np.sum(ap * seg_v, axis=1) / seg_len2
        t = np.clip(t, 0.0, 1.0)

        closest = seg_a + t[:, None] * seg_v
        delta = p[None, :] - closest
        dist2 = np.sum(delta * delta, axis=1)

        j = int(np.argmin(dist2))
        dist = float(np.sqrt(max(0.0, dist2[j])))

        # Use the horizontal side of the closest SRIM point for the sign.
        # This keeps the sign convention intuitive and roughly consistent with
        # the existing horizontal reaction-distance residual.
        dx_side = p[0] - closest[j, 0]

        if abs(dx_side) > 1.0e-14:
            sign = 1.0 if dx_side > 0.0 else -1.0
        else:
            # Rare fallback when the point is vertically above/below the
            # closest location. Use the local segment cross product.
            cross = (
                seg_v[j, 0] * delta[j, 1]
                - seg_v[j, 1] * delta[j, 0]
            )
            if cross > 0.0:
                sign = 1.0
            elif cross < 0.0:
                sign = -1.0
            else:
                sign = 1.0

        out[i] = sign * dist

    return out


def common_true_transverse_range(
    fit1,
    fit2,
    requested_range=None,
):
    """Choose one common symmetric arbitrary-unit range for both methods."""
    if requested_range is not None:
        lo = float(requested_range[0])
        hi = float(requested_range[1])

        if (
            not np.isfinite(lo)
            or not np.isfinite(hi)
            or lo >= hi
        ):
            raise ValueError(
                "--true-transverse-range requires MIN_AU < MAX_AU"
            )

        return (lo, hi)

    vals = []

    if fit1["n"]:
        vals.append(fit1["values"])

    if fit2["n"]:
        vals.append(fit2["values"])

    if not vals:
        return (-0.1, 0.1)

    all_values = np.concatenate(vals)
    max_abs = float(np.max(np.abs(all_values)))

    if not np.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 0.1

    max_abs *= 1.05

    return (-max_abs, max_abs)


def plot_true_transverse_spread(
    fit,
    label,
    out_path,
    bins,
    hist_range,
    color,
    overlay_gaussian=False,
    dpi=500,
):
    """
    Plot the signed shortest distance to the common SRIM curve in normalized
    2D (distance, E_cm) coordinates.

    Because the original axes have different physical units, the two axes are
    normalized before the Euclidean shortest distance is calculated.  The
    resulting transverse coordinate is dimensionless and is labeled in
    arbitrary units.
    """
    r = fit["values"]

    fig, ax = plt.subplots(
        figsize=(8.2, 5.6)
    )

    counts, edges, _ = ax.hist(
        r,
        bins=int(bins),
        range=hist_range,
        alpha=0.72,
        edgecolor="black",
        linewidth=0.7,
        color=color,
        label="Data",
    )

    ax.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="SRIM curve",
    )

    if overlay_gaussian:
        mu = fit["mu"]
        sigma = fit["sigma"]

        if (
            fit["n"] > 0
            and np.isfinite(mu)
            and np.isfinite(sigma)
            and sigma > 0
        ):
            xx = np.linspace(
                hist_range[0],
                hist_range[1],
                1200,
            )

            bin_width = float(
                edges[1]
                - edges[0]
            )

            yy = (
                fit["n"]
                * bin_width
                / (
                    sigma
                    * np.sqrt(
                        2.0
                        * np.pi
                    )
                )
                * np.exp(
                    -0.5
                    * (
                        (
                            xx
                            - mu
                        )
                        / sigma
                    )
                    ** 2
                )
            )

            ax.plot(
                xx,
                yy,
                linewidth=2.5,
                label=(
                    "Gaussian fit: "
                    + r"$\mu$={:.4f}, $\sigma$={:.4f} a.u."
                    .format(
                        mu,
                        sigma,
                    )
                ),
            )

            ax.axvline(
                mu,
                linestyle=":",
                linewidth=1.6,
                label=r"Gaussian $\mu$",
            )

    ax.set_xlim(
        hist_range
    )

    ax.set_xlabel(
        r"Signed perpendicular residual to SRIM curve (a.u.)",
        fontsize=13,
    )

    ax.set_ylabel(
        "Events",
        fontsize=13,
    )

    ax.set_title(
        "True transverse spread about the common SRIM curve: {}".format(
            label
        ),
        fontsize=14,
        pad=8,
    )

    ax.tick_params(axis="both", which="major", labelsize=11, width=1.0, length=4.5)
    ax.grid(
        alpha=0.25
    )

    ax.legend(fontsize=10.5, framealpha=0.92)

    fig.tight_layout()

    fig.savefig(
        str(
            out_path
        ),
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        str(out_path.with_suffix(".pdf")),
        bbox_inches="tight",
        pad_inches=0.04,
    )

    return fig


def plot_transverse_spread(
    fit,
    label,
    out_path,
    bins,
    hist_range,
    color,
    dpi=500,
):
    """
    Plot horizontal distance residuals about the common SRIM locus and overlay
    the unbinned Gaussian fit, scaled to histogram counts.
    """
    r = fit["values"]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    counts, edges, _ = ax.hist(
        r,
        bins=int(bins),
        range=hist_range,
        alpha=0.72,
        edgecolor="black",
        color=color,
        linewidth=0.7,
        label="Data residuals",
    )

    # SRIM curve is Delta d = 0 by construction.
    ax.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Kinematic curve",
    )
    # Optional Gaussian Fitting
    r"""
    mu = fit["mu"]
    sigma = fit["sigma"]

    if (
        fit["n"] > 0
        and np.isfinite(mu)
        and np.isfinite(sigma)
        and sigma > 0
    ):
        xx = np.linspace(hist_range[0], hist_range[1], 1200)

        # Scale a normalized Gaussian PDF to histogram counts.
        bin_width = float(edges[1] - edges[0])
        yy = (
            fit["n"]
            * bin_width
            / (sigma * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((xx - mu) / sigma) ** 2)
        )

        ax.plot(
            xx,
            yy,
            linewidth=2.5,
            label=(
                "Gaussian fit: "
                + r"$\mu$={:.2f} mm, $\sigma$={:.2f} mm"
                .format(mu, sigma)
            ),
        )

        ax.axvline(
            mu,
            linestyle=":",
            linewidth=1.6,
            label=r"Gaussian $\mu$",
        )

    """
    ax.set_xlim(hist_range)
    ax.set_xlabel(
        r"Horizontal distance, "
        r"$\Delta d=d_{\rm Reco}-d_{\rm Kin}$ (mm)",
        fontsize=13,
    )
    ax.set_ylabel("Events", fontsize=13)
    ax.set_title(
        "Horizontal spread about common kinematic curve: {}".format(label),
        fontsize=14,
        pad=8,
    )
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.0, length=4.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(
        str(out_path),
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        str(out_path.with_suffix(".pdf")),
        bbox_inches="tight",
        pad_inches=0.04,
    )

    return fig



def symmetric_range_for_one_histogram(values, padding=1.05):
    """
    Return an x-axis range centered exactly on zero and wide enough to contain
    all finite values in the supplied residual distribution.

    In the combined-comparison figure this helper can be used either for one
    method alone or for a concatenated set of residuals from both methods,
    depending on whether matched axes are desired.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (-1.0, 1.0)

    max_abs = float(np.max(np.abs(values)))

    if not np.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 1.0

    max_abs *= float(padding)
    return (-max_abs, max_abs)


def draw_horizontal_residual_histogram(
    ax,
    values,
    bins,
    hist_range,
    color,
    title=None,
    label=None,
    alpha=0.72,
    show_legend=True,
    xlabel=True,
    ylabel=True,
):
    """
    Draw one horizontal reaction-distance residual histogram on an existing
    Matplotlib Axes.

    Zero is the common SRIM/kinematic curve by construction.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    ax.hist(
        values,
        bins=int(bins),
        range=hist_range,
        alpha=float(alpha),
        color=color,
        edgecolor="black",
        linewidth=0.55,
        label=(label if label is not None else "Data residuals"),
    )

    ax.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Kinematic curve",
    )

    ax.set_xlim(hist_range)

    if xlabel:
        ax.set_xlabel(
            r"Horizontal Residual, "
            r"$\Delta d=d_{\rm Reco}-d_{\rm Kin}$ [mm]",
            fontsize=12,
        )

    if ylabel:
        ax.set_ylabel("Events", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, pad=7)

    ax.tick_params(axis="both", which="major", labelsize=10.5, width=1.0, length=4.0)
    ax.grid(alpha=0.22)

    if show_legend:
        ax.legend(
            fontsize=10.0,
            framealpha=0.92,
            facecolor="white",
            edgecolor="0.75",
            borderaxespad=0.35,
            labelspacing=0.25,
            handlelength=1.8,
        )



def main():
    a = parse_args()

    if a.density <= 0:
        raise SystemExit(
            "--density must be positive."
        )

    if a.ecm_min >= a.ecm_max:
        raise SystemExit(
            "--ecm-min must be less than --ecm-max."
        )

    if a.spread_bins < 3:
        raise SystemExit(
            "--spread-bins must be at least 3."
        )

    if (
        a.spread_range is not None
        and a.spread_range[0] >= a.spread_range[1]
    ):
        raise SystemExit(
            "--spread-range requires MIN_MM < MAX_MM."
        )

    if (
        a.plot_x_range[0] >= a.plot_x_range[1]
    ):
        raise SystemExit(
            "--plot-x-range requires MIN_MM < MAX_MM."
        )

    if (
        a.true_transverse_bins is not None
        and a.true_transverse_bins < 3
    ):
        raise SystemExit(
            "--true-transverse-bins must be at least 3."
        )

    if (
        a.true_transverse_range is not None
        and a.true_transverse_range[0] >= a.true_transverse_range[1]
    ):
        raise SystemExit(
            "--true-transverse-range requires MIN_AU < MAX_AU."
        )

    (
        E_tab,
        S_tab,
        ion_name,
        ion_mass,
        srim_density,
    ) = parse_srim(
        a.srim_file
    )

    d1 = read_dataset(
        a.data1,
        a.ecm_column1,
        a.distance_column1,
        a.entrance_z1,
        a,
    )

    d2 = read_dataset(
        a.data2,
        a.ecm_column2,
        a.distance_column2,
        a.entrance_z2,
        a,
    )

    excluded_event_ids = read_excluded_event_ids(
        a.exclude_events_file
    )

    removed_excluded1 = 0
    removed_excluded2 = 0

    if excluded_event_ids:
        d1, removed_excluded1 = filter_dataset_by_excluded_events(
            d1,
            excluded_event_ids,
            "Dataset 1",
        )

        d2, removed_excluded2 = filter_dataset_by_excluded_events(
            d2,
            excluded_event_ids,
            "Dataset 2",
        )

        print()
        print(
            "[INFO] Loaded {} excluded event IDs from {}"
            .format(
                len(excluded_event_ids),
                a.exclude_events_file,
            )
        )
        print(
            "[INFO] Removed {} matching events from Dataset 1"
            .format(
                removed_excluded1
            )
        )
        print(
            "[INFO] Removed {} matching events from Dataset 2"
            .format(
                removed_excluded2
            )
        )

    d1, d2, common_filter_info = enforce_common_and_max_events(
        d1,
        d2,
        a,
    )

    cm_factor = (
        a.target_mass
        / (
            a.projectile_mass
            + a.target_mass
        )
    )

    ecm_ref = (
        a.ecm_max
    )

    A1 = build_areal_coordinate(
        E_tab,
        S_tab,
        d1["ecm"],
        cm_factor,
        ecm_ref,
    )

    A2 = build_areal_coordinate(
        E_tab,
        S_tab,
        d2["ecm"],
        cm_factor,
        ecm_ref,
    )

    srim_distance1 = (
        A1
        / (
            100.0
            * a.density
        )
    )

    srim_distance2 = (
        A2
        / (
            100.0
            * a.density
        )
    )

    print()
    print(
        "[INFO] Dataset 1/reference:"
    )
    print(
        "       file =", d1["path"]
    )
    print(
        "       E_cm column =", d1["ecm_col"]
    )
    print(
        "       distance column =", d1["dist_col"]
    )
    print(
        "       events =", len(d1["x"])
    )

    print()
    print(
        "[INFO] Dataset 2/comparison:"
    )
    print(
        "       file =", d2["path"]
    )
    print(
        "       E_cm column =", d2["ecm_col"]
    )
    print(
        "       distance column =", d2["dist_col"]
    )
    print(
        "       events =", len(d2["x"])
    )

    if srim_density is not None:
        print()
        print(
            "[INFO] SRIM header density {:.6e} g/cm^3 is ignored."
            .format(
                srim_density
            )
        )

    print(
        "[INFO] Physical density held fixed at {:.10e} g/cm^3"
        .format(
            a.density
        )
    )

    # Fit DATASET 1 ONLY.
    x0, fit_mask1 = fit_reference_shift(
        d1["x"],
        srim_distance1,
        a,
    )

    # Freeze exact curve.
    pred1 = (
        x0
        + srim_distance1
    )

    pred2 = (
        x0
        + srim_distance2
    )

    residual1 = (
        d1["x"]
        - pred1
    )

    residual2 = (
        d2["x"]
        - pred2
    )

    fit_metric_mask1 = (
        fit_mask1
        & np.isfinite(
            residual1
        )
    )

    all_metric_mask1 = np.isfinite(
        residual1
    )

    if a.metrics_use_all:
        metric_mask1 = all_metric_mask1
    else:
        metric_mask1 = fit_metric_mask1

    metric_mask2_all_finite = np.isfinite(
        residual2
    )

    metric_mask2 = metric_mask2_all_finite.copy()

    data2_matched_excluded_mask = np.zeros(
        len(d2["x"]),
        dtype=bool,
    )

    excluded_data1_event_ids = set()

    # Keep the primary Dataset 1 and Dataset 2 metrics paired.
    #
    # If Dataset 1 primary metrics use the x0-fit mask, then any event removed
    # from Dataset 1 by the x0-fit clipping should also be removed from Dataset 2.
    # This is automatic when --require-common-events is enabled, unless
    # --metrics-use-all is requested.  The legacy explicit flag is still
    # honored.
    match_data2_to_data1_primary_mask = (
        bool(a.drop_data2_events_excluded_from_data1_fit)
        or (bool(a.require_common_events) and (not bool(a.metrics_use_all)))
    )

    if match_data2_to_data1_primary_mask:
        for eid, keep in zip(
            d1["ids"],
            metric_mask1,
        ):
            if not bool(keep):
                excluded_data1_event_ids.add(
                    str(eid)
                )

        if excluded_data1_event_ids:
            for i, eid in enumerate(
                d2["ids"]
            ):
                if str(eid) in excluded_data1_event_ids:
                    data2_matched_excluded_mask[i] = True

            metric_mask2 = (
                metric_mask2
                & ~data2_matched_excluded_mask
            )

        print()
        print(
            "[INFO] Dataset 1 events excluded from primary Dataset 1 metric mask: {}"
            .format(
                len(excluded_data1_event_ids)
            )
        )
        print(
            "[INFO] Matching Dataset 2 events excluded from primary metrics: {}"
            .format(
                int(
                    data2_matched_excluded_mask.sum()
                )
            )
        )

    # Primary metrics.  For Dataset 1 these can either follow the clipped
    # x0-fit mask or include all finite residuals, depending on
    # --metrics-use-all.
    m1 = metrics(
        residual1,
        metric_mask1,
    )

    # Always compute both Dataset 1 versions so the output file documents
    # exactly how much the clipping mattered.
    m1_fitmask = metrics(
        residual1,
        fit_metric_mask1,
    )

    m1_all = metrics(
        residual1,
        all_metric_mask1,
    )

    m2 = metrics(
        residual2,
        metric_mask2,
    )

    clipped_mask2 = None
    m2_clipped = None

    if a.clip_data2:
        clipped_mask2 = (
            data2_clip_mask(
                residual2,
                a.clip_sigma,
            )
            & metric_mask2
        )

        m2_clipped = metrics(
            residual2,
            clipped_mask2,
        )

    # Smooth one common SRIM curve.
    ecm_curve = np.linspace(
        a.ecm_max,
        a.ecm_min,
        1500,
    )

    A_curve = build_areal_coordinate(
        E_tab,
        S_tab,
        ecm_curve,
        cm_factor,
        ecm_ref,
    )

    x_curve = (
        x0
        + (
            A_curve
            / (
                100.0
                * a.density
            )
        )
    )

    prefix = Path(
        a.output_prefix
    )

    png_path = prefix.with_suffix(
        ".png"
    )

    txt_path = prefix.with_name(
        prefix.name
        + "_results.txt"
    )

    data1_path = prefix.with_name(
        prefix.name
        + "_dataset1_events.csv"
    )

    data2_path = prefix.with_name(
        prefix.name
        + "_dataset2_events.csv"
    )

    curve_path = prefix.with_name(
        prefix.name
        + "_common_curve.csv"
    )

    spread1_path = prefix.with_name(
        prefix.name
        + "_dataset1_transverse_spread.png"
    )

    spread2_path = prefix.with_name(
        prefix.name
        + "_dataset2_transverse_spread.png"
    )

    true_transverse1_path = prefix.with_name(
        prefix.name
        + "_dataset1_true_transverse_spread.png"
    )

    true_transverse2_path = prefix.with_name(
        prefix.name
        + "_dataset2_true_transverse_spread.png"
    )

    # Use the same residual samples as the primary reported metrics so the
    # histogram widths correspond directly to the method comparison.
    gaussian1 = gaussian_fit_to_residuals(
        residual1,
        metric_mask1,
    )

    gaussian2 = gaussian_fit_to_residuals(
        residual2,
        metric_mask2,
    )

    spread_range = common_spread_range(
        gaussian1,
        gaussian2,
        requested_range=a.spread_range,
    )


    # ------------------------------------------------------------------
    # TRUE TRANSVERSE SPREAD
    #
    # The original plot mixes millimeters and MeV, so a literal Euclidean
    # perpendicular distance in raw coordinates would have no meaningful unit.
    # Instead, map the displayed x and y ranges independently to [0,1] and
    # measure the shortest Euclidean distance to the common SRIM polyline in
    # that normalized plane.  The result is a dimensionless arbitrary-unit
    # transverse residual.
    # ------------------------------------------------------------------
    transverse_y_range = (
        float(a.ecm_min),
        float(a.ecm_max),
    )

    true_transverse1 = true_transverse_residuals(
        d1["x"],
        d1["ecm"],
        x_curve,
        ecm_curve,
        a.plot_x_range,
        transverse_y_range,
    )

    true_transverse2 = true_transverse_residuals(
        d2["x"],
        d2["ecm"],
        x_curve,
        ecm_curve,
        a.plot_x_range,
        transverse_y_range,
    )

    true_transverse_fit1 = gaussian_fit_to_residuals(
        true_transverse1,
        metric_mask1,
    )

    true_transverse_fit2 = gaussian_fit_to_residuals(
        true_transverse2,
        metric_mask2,
    )

    true_transverse_range = common_true_transverse_range(
        true_transverse_fit1,
        true_transverse_fit2,
        requested_range=a.true_transverse_range,
    )

    true_transverse_bins = (
        a.true_transverse_bins
        if a.true_transverse_bins is not None
        else a.spread_bins
    )

    write_event_csv(
        data1_path,
        d1,
        pred1,
        residual1,
        metric_mask1,
    )

    write_event_csv(
        data2_path,
        d2,
        pred2,
        residual2,
        (
            clipped_mask2
            if clipped_mask2 is not None
            else metric_mask2
        ),
    )

    with curve_path.open(
        "w",
        newline="",
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "distance_mm",
            "Ecm_SRIM_MeV",
        ])

        for xx, yy in zip(
            x_curve,
            ecm_curve,
        ):
            w.writerow([
                xx,
                yy,
            ])

    with txt_path.open(
        "w"
    ) as f:
        f.write(
            "Two-dataset comparison using ONE identical SRIM fit\n"
        )
        f.write(
            "===================================================\n\n"
        )

        f.write(
            "REFERENCE DATASET 1: {}\n".format(
                a.data1
            )
        )

        f.write(
            "COMPARISON DATASET 2: {}\n\n".format(
                a.data2
            )
        )

        f.write(
            "Manual excluded-events file = {}\n".format(
                a.exclude_events_file
            )
        )
        f.write(
            "Manual excluded event IDs loaded = {}\n".format(
                len(
                    excluded_event_ids
                )
            )
        )
        f.write(
            "Manual exclusions removed from Dataset 1 = {}\n".format(
                removed_excluded1
            )
        )
        f.write(
            "Manual exclusions removed from Dataset 2 = {}\n\n".format(
                removed_excluded2
            )
        )

        f.write(
            "Common-event filtering enabled = {}\n".format(
                bool(a.require_common_events)
            )
        )
        f.write(
            "Dataset 1 events before common/max filter = {}\n".format(
                common_filter_info["before1"]
            )
        )
        f.write(
            "Dataset 2 events before common/max filter = {}\n".format(
                common_filter_info["before2"]
            )
        )
        f.write(
            "Common event IDs before max filter = {}\n".format(
                common_filter_info["common_before_max"]
            )
        )
        f.write(
            "Max events requested = {}\n".format(
                int(a.max_events)
            )
        )
        f.write(
            "Event selection mode = {}\n".format(
                a.event_selection
            )
        )
        f.write(
            "Common event IDs selected = {}\n\n".format(
                common_filter_info["selected_common"]
            )
        )

        f.write(
            "Fixed density = {:.12e} g/cm^3\n"
            .format(
                a.density
            )
        )

        f.write(
            "Best-fit x0 from DATASET 1 ONLY = {:.12g} mm\n"
            .format(
                x0
            )
        )

        f.write(
            "E_cm range = {:.12g} to {:.12g} MeV\n\n"
            .format(
                a.ecm_min,
                a.ecm_max,
            )
        )

        f.write(
            "Dataset 1 primary metric mode = {}\n".format(
                "all finite residuals"
                if a.metrics_use_all
                else "same clipped mask used for x0 fit"
            )
        )
        f.write(
            "Dataset 1 x0-fit-mask N = {}\n".format(
                m1_fitmask["n"]
            )
        )
        f.write(
            "Dataset 1 all-finite N = {}\n\n".format(
                m1_all["n"]
            )
        )

        f.write(
            "Dataset 2 matched-event exclusion enabled = {}\n".format(
                bool(
                    match_data2_to_data1_primary_mask
                )
            )
        )
        f.write(
            "Dataset 1 events excluded from primary Dataset 1 metric mask = {}\n".format(
                len(
                    excluded_data1_event_ids
                )
            )
        )
        f.write(
            "Dataset 2 events excluded by Dataset 1 outlier mask = {}\n\n".format(
                int(
                    data2_matched_excluded_mask.sum()
                )
            )
        )

        f.write(
            "DATASET 1 ({}) -- PRIMARY METRICS\n".format(
                a.label1
            )
        )
        f.write(
            "  N = {}\n".format(
                m1["n"]
            )
        )
        f.write(
            "  RMSE_x = {:.9g} mm\n".format(
                m1["rmse"]
            )
        )
        f.write(
            "  MAE_x = {:.9g} mm\n".format(
                m1["mae"]
            )
        )
        f.write(
            "  Median |residual| = {:.9g} mm\n".format(
                m1["median_abs"]
            )
        )
        f.write(
            "  Mean residual = {:.9g} mm\n".format(
                m1["mean"]
            )
        )
        f.write(
            "  Residual std = {:.9g} mm\n\n".format(
                m1["std"]
            )
        )

        f.write(
            "DATASET 1 ({}) -- x0-fit-mask metrics\n".format(
                a.label1
            )
        )
        f.write(
            "  N = {}\n".format(
                m1_fitmask["n"]
            )
        )
        f.write(
            "  RMSE_x = {:.9g} mm\n".format(
                m1_fitmask["rmse"]
            )
        )
        f.write(
            "  MAE_x = {:.9g} mm\n".format(
                m1_fitmask["mae"]
            )
        )
        f.write(
            "  Median |residual| = {:.9g} mm\n".format(
                m1_fitmask["median_abs"]
            )
        )
        f.write(
            "  Mean residual = {:.9g} mm\n".format(
                m1_fitmask["mean"]
            )
        )
        f.write(
            "  Residual std = {:.9g} mm\n\n".format(
                m1_fitmask["std"]
            )
        )

        f.write(
            "DATASET 1 ({}) -- all-finite-residual metrics\n".format(
                a.label1
            )
        )
        f.write(
            "  N = {}\n".format(
                m1_all["n"]
            )
        )
        f.write(
            "  RMSE_x = {:.9g} mm\n".format(
                m1_all["rmse"]
            )
        )
        f.write(
            "  MAE_x = {:.9g} mm\n".format(
                m1_all["mae"]
            )
        )
        f.write(
            "  Median |residual| = {:.9g} mm\n".format(
                m1_all["median_abs"]
            )
        )
        f.write(
            "  Mean residual = {:.9g} mm\n".format(
                m1_all["mean"]
            )
        )
        f.write(
            "  Residual std = {:.9g} mm\n\n".format(
                m1_all["std"]
            )
        )

        f.write(
            "DATASET 2 ({}) -- NO REFIT\n".format(
                a.label2
            )
        )
        f.write(
            "  N = {}\n".format(
                m2["n"]
            )
        )
        f.write(
            "  RMSE_x = {:.9g} mm\n".format(
                m2["rmse"]
            )
        )
        f.write(
            "  MAE_x = {:.9g} mm\n".format(
                m2["mae"]
            )
        )
        f.write(
            "  Median |residual| = {:.9g} mm\n".format(
                m2["median_abs"]
            )
        )
        f.write(
            "  Mean residual = {:.9g} mm\n".format(
                m2["mean"]
            )
        )
        f.write(
            "  Residual std = {:.9g} mm\n".format(
                m2["std"]
            )
        )

        if m2_clipped is not None:
            f.write(
                "\nDATASET 2 clipped diagnostic only "
                "(curve still unchanged)\n"
            )
            f.write(
                "  N = {}\n".format(
                    m2_clipped["n"]
                )
            )
            f.write(
                "  RMSE_x = {:.9g} mm\n".format(
                    m2_clipped["rmse"]
                )
            )
            f.write(
                "  MAE_x = {:.9g} mm\n".format(
                    m2_clipped["mae"]
                )
            )
            f.write(
                "  Median |residual| = {:.9g} mm\n".format(
                    m2_clipped["median_abs"]
                )
            )

        f.write(
            "\nHORIZONTAL REACTION-DISTANCE SPREAD ABOUT THE COMMON SRIM CURVE\n"
        )
        f.write(
            "Residual definition: Delta d = d_reco - d_SRIM(E_cm)\n"
        )
        f.write(
            "Gaussian summaries use the same event masks as the primary metrics.\n\n"
        )

        f.write(
            "Dataset 1 ({}) Gaussian residual fit\n".format(
                a.label1
            )
        )
        f.write(
            "  N = {}\n".format(
                gaussian1["n"]
            )
        )
        f.write(
            "  mu = {:.9g} mm\n".format(
                gaussian1["mu"]
            )
        )
        f.write(
            "  sigma = {:.9g} mm\n".format(
                gaussian1["sigma"]
            )
        )
        f.write(
            "  FWHM = {:.9g} mm\n\n".format(
                gaussian1["fwhm"]
            )
        )

        f.write(
            "Dataset 2 ({}) Gaussian residual fit\n".format(
                a.label2
            )
        )
        f.write(
            "  N = {}\n".format(
                gaussian2["n"]
            )
        )
        f.write(
            "  mu = {:.9g} mm\n".format(
                gaussian2["mu"]
            )
        )
        f.write(
            "  sigma = {:.9g} mm\n".format(
                gaussian2["sigma"]
            )
        )
        f.write(
            "  FWHM = {:.9g} mm\n".format(
                gaussian2["fwhm"]
            )
        )

        f.write(
            "\nTRUE TRANSVERSE SPREAD ABOUT THE COMMON SRIM CURVE\n"
        )
        f.write(
            "Definition: signed shortest Euclidean distance to the SRIM "
            "polyline after independent axis normalization.\n"
        )
        f.write(
            "Distance normalization range = {:.9g} to {:.9g} mm -> 0 to 1\n"
            .format(
                a.plot_x_range[0],
                a.plot_x_range[1],
            )
        )
        f.write(
            "E_cm normalization range = {:.9g} to {:.9g} MeV -> 0 to 1\n"
            .format(
                a.ecm_min,
                a.ecm_max,
            )
        )
        f.write(
            "Units = dimensionless arbitrary units (a.u.).\n"
        )
        f.write(
            "The sign is positive on the larger-distance side of the nearest "
            "SRIM point.\n\n"
        )

        f.write(
            "Dataset 1 ({}) true transverse spread\n".format(
                a.label1
            )
        )
        f.write(
            "  N = {}\n".format(
                true_transverse_fit1["n"]
            )
        )
        f.write(
            "  mean = {:.9g} a.u.\n".format(
                true_transverse_fit1["mu"]
            )
        )
        f.write(
            "  sigma = {:.9g} a.u.\n".format(
                true_transverse_fit1["sigma"]
            )
        )
        f.write(
            "  FWHM = {:.9g} a.u.\n\n".format(
                true_transverse_fit1["fwhm"]
            )
        )

        f.write(
            "Dataset 2 ({}) true transverse spread\n".format(
                a.label2
            )
        )
        f.write(
            "  N = {}\n".format(
                true_transverse_fit2["n"]
            )
        )
        f.write(
            "  mean = {:.9g} a.u.\n".format(
                true_transverse_fit2["mu"]
            )
        )
        f.write(
            "  sigma = {:.9g} a.u.\n".format(
                true_transverse_fit2["sigma"]
            )
        )
        f.write(
            "  FWHM = {:.9g} a.u.\n".format(
                true_transverse_fit2["fwhm"]
            )
        )

        f.write(
            "\nRELATIVE PERFORMANCE\n"
        )

        if (
            np.isfinite(m1["rmse"])
            and m1["rmse"] > 0
            and np.isfinite(m2["rmse"])
        ):
            f.write(
                "  RMSE ratio dataset2/dataset1 = {:.9g}\n"
                .format(
                    m2["rmse"]
                    / m1["rmse"]
                )
            )

        if (
            np.isfinite(m1["mae"])
            and m1["mae"] > 0
            and np.isfinite(m2["mae"])
        ):
            f.write(
                "  MAE ratio dataset2/dataset1 = {:.9g}\n"
                .format(
                    m2["mae"]
                    / m1["mae"]
                )
            )

    # ==================================================================
    # COMBINED PUBLICATION-STYLE COMPARISON FIGURE
    #
    # Layout:
    #
    #   +--------------------------+-------------+-------------+
    #   |                          | Dataset 1   | Dataset 2   |
    #   |   E_cm vs distance      | residuals   | residuals   |
    #   |   scatter + common      +-------------+-------------+
    #   |   kinematic curve       |                           |
    #   |                          |  overlaid residuals        |
    #   +--------------------------+---------------------------+
    #
    # The two upper-right histograms use independent symmetric x ranges,
    # allowing each method's complete residual distribution to fill its panel.
    # The lower-right overlay uses one common symmetric range so the two widths
    # can be compared directly.
    # ==================================================================

    fig = plt.figure(
        figsize=(17.2, 9.0),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=(2.35, 1.0, 1.0),
        height_ratios=(1.0, 1.0),
    )

    # Main scatter plot spans the full left side.
    ax = fig.add_subplot(
        gs[:, 0]
    )

    # Separate residual histograms.
    ax_resid1 = fig.add_subplot(
        gs[0, 1]
    )

    ax_resid2 = fig.add_subplot(
        gs[0, 2]
    )

    # Overlay comparison spans the full lower-right side.
    ax_resid_overlay = fig.add_subplot(
        gs[1, 1:3]
    )

    # --------------------------------------------------------------
    # LEFT: original scatter comparison
    # --------------------------------------------------------------
    ax.scatter(
        d1["x"][fit_metric_mask1],
        d1["ecm"][fit_metric_mask1],
        s=28,
        alpha=0.70,
        color="tab:blue",
        label="ML Reconstructed Vertices".format(
            a.label1
        ),
    )

    rejected1 = (
        all_metric_mask1
        & ~fit_metric_mask1
    )

    if np.any(
        rejected1
    ):
        ax.scatter(
            d1["x"][rejected1],
            d1["ecm"][rejected1],
            s=24,
            alpha=0.35,
            color="tab:blue",
            marker="x",
            label="{} excluded from x0 fit".format(
                a.label1
            ),
        )

    ax.scatter(
        d2["x"][metric_mask2],
        d2["ecm"][metric_mask2],
        s=28,
        alpha=0.65,
        color="tab:orange",
        marker="s",
        label="Hough Closest Approach Vertices".format(
            a.label2
        ),
    )

    if np.any(
        data2_matched_excluded_mask
    ):
        ax.scatter(
            d2["x"][data2_matched_excluded_mask],
            d2["ecm"][data2_matched_excluded_mask],
            s=26,
            alpha=0.35,
            marker="x",
            color="tab:orange",
            label="{} matched excluded".format(
                a.label2
            ),
        )

    ax.plot(
        x_curve,
        ecm_curve,
        linewidth=2.7,
        label="Elastic Kinematic Curve",
        color="black"
    )

    ax.set_xlabel(
        "Vertex Distance from Chamber Entrance (Along Beam Path) [mm]",
        fontsize=13,
    )

    ax.set_ylabel(
        r"$E_{\rm cm}$ [MeV]",
        fontsize=13,
    )

    ax.set_ylim(
        a.ecm_min,
        a.ecm_max,
    )

    ax.set_xlim(
        a.plot_x_range[0],
        a.plot_x_range[1],
    )

    ax.set_title(
        r"Center-of-Mass Energy ($E_{\rm cm}$) vs Vertex Distance from Chamber Entrance",
        fontsize=15,
        pad=9,
    )

    ax.grid(
        alpha=0.3
    )

    ax.tick_params(axis="both", which="major", labelsize=11.5, width=1.0, length=4.5)
    ax.legend(fontsize=10.5, framealpha=0.92)

    rmse_text = (
        "ML Horizontal RMSE = {:.2f} mm\n"
        "Hough Horizontal RMSE = {:.2f} mm"
    ).format(
        m1["rmse"],
        m2["rmse"],
    )

    ax.text(
        0.03,
        0.05,
        rmse_text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=11.5,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="0.35",
            alpha=0.92,
        ),
    )

    # --------------------------------------------------------------
    # UPPER RIGHT: separate residual histograms.
    #
    # Use a single common x-axis range centered at zero and wide enough to
    # contain the broader of the two residual distributions.  This ensures
    # the ML and Hough histograms can be compared directly with identical
    # horizontal and vertical scaling.
    # --------------------------------------------------------------
    residual_values1 = gaussian1["values"]
    residual_values2 = gaussian2["values"]

    common_upper_range = symmetric_range_for_one_histogram(
        np.concatenate([residual_values1, residual_values2])
    )

    draw_horizontal_residual_histogram(
        ax=ax_resid1,
        values=residual_values1,
        bins=a.spread_bins,
        hist_range=common_upper_range,
        color="tab:blue",
        title="ML Horizontal Residuals",
        label="ML residuals",
        alpha=0.78,
        show_legend=True,
        xlabel=True,
        ylabel=True,
    )

    draw_horizontal_residual_histogram(
        ax=ax_resid2,
        values=residual_values2,
        bins=a.spread_bins,
        hist_range=common_upper_range,
        color="tab:orange",
        title="Hough Horizontal Residuals",
        label="Hough residuals",
        alpha=0.78,
        show_legend=True,
        xlabel=True,
        ylabel=False,
    )

    # Make the two upper histograms use the same y-axis scale
    ymax = max(
        ax_resid1.get_ylim()[1],
        ax_resid2.get_ylim()[1],
    )

    ax_resid1.set_ylim(0, ymax)
    ax_resid2.set_ylim(0, ymax)

    # Remove y-axis numbers/ticks from the right histogram
    ax_resid2.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )

    ax_resid2.set_ylabel("")

    # --------------------------------------------------------------
    # LOWER RIGHT: both horizontal residual distributions overlaid.
    #
    # Use the existing common `spread_range`, which is symmetric around zero,
    # so both methods are shown on the identical horizontal scale.
    # --------------------------------------------------------------
    common_edges = np.linspace(
        spread_range[0],
        spread_range[1],
        int(a.spread_bins) + 1,
    )

    ax_resid_overlay.hist(
        residual_values1,
        bins=common_edges,
        alpha=0.46,
        color="tab:blue",
        edgecolor="tab:blue",
        linewidth=0.65,
        label="ML residuals",
    )

    ax_resid_overlay.hist(
        residual_values2,
        bins=common_edges,
        alpha=0.46,
        color="tab:orange",
        edgecolor="tab:orange",
        linewidth=0.65,
        label="Hough residuals",
    )

    ax_resid_overlay.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Kinematic curve",
    )

    ax_resid_overlay.set_xlim(
        spread_range
    )

    ax_resid_overlay.set_xlabel(
        r"Horizontal Residual, "
        r"$\Delta d=d_{\rm Reco}-d_{\rm Kin}$ [mm]",
        fontsize=13,
    )

    ax_resid_overlay.set_ylabel(
        "Events",
        fontsize=13,
    )

    ax_resid_overlay.set_title(
        "Horizontal Residual Comparison",
        fontsize=14,
        pad=7,
    )

    ax_resid_overlay.tick_params(
        axis="both",
        which="major",
        labelsize=11,
        width=1.0,
        length=4.5,
    )
    ax_resid_overlay.grid(
        alpha=0.22
    )

    ax_resid_overlay.legend(
        fontsize=10.5,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.75",
    )

    # Save the entire multi-panel comparison as both a high-resolution PNG
    # and a vector PDF. For manuscript submission, the PDF is preferred when
    # the journal workflow accepts vector artwork.
    fig.savefig(
        str(
            png_path
        ),
        dpi=int(a.dpi),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        str(png_path.with_suffix(".pdf")),
        bbox_inches="tight",
        pad_inches=0.04,
    )

    spread_fig1 = plot_transverse_spread(
        gaussian1,
        a.label1,
        spread1_path,
        a.spread_bins,
        spread_range,
        color="tab:blue",
        dpi=a.dpi,
    )

    spread_fig2 = plot_transverse_spread(
        gaussian2,
        a.label2,
        spread2_path,
        a.spread_bins,
        spread_range,
        color="tab:orange",
        dpi=a.dpi,
    )

    true_transverse_fig1 = plot_true_transverse_spread(
        true_transverse_fit1,
        a.label1,
        true_transverse1_path,
        true_transverse_bins,
        true_transverse_range,
        color="tab:blue",
        overlay_gaussian=a.fit_true_transverse_gaussian,
        dpi=a.dpi,
    )

    true_transverse_fig2 = plot_true_transverse_spread(
        true_transverse_fit2,
        a.label2,
        true_transverse2_path,
        true_transverse_bins,
        true_transverse_range,
        color="tab:orange",
        overlay_gaussian=a.fit_true_transverse_gaussian,
        dpi=a.dpi,
    )

    print()
    print(
        "============================================================"
    )
    print(
        " ONE COMMON SRIM FIT -- DATASET 1 FIT, DATASET 2 NO REFIT"
    )
    print(
        "============================================================"
    )
    if excluded_event_ids:
        print(
            " Manual excluded-events file: {}".format(
                a.exclude_events_file
            )
        )
        print(
            " Manual exclusions removed: Dataset 1 = {}, Dataset 2 = {}"
            .format(
                removed_excluded1,
                removed_excluded2,
            )
        )

    print(
        " Fixed rho = {:.10e} g/cm^3".format(
            a.density
        )
    )
    print(
        " x0 fitted from dataset 1 only = {:.6f} mm".format(
            x0
        )
    )
    print()
    print(
        " DATASET 1: {} -- PRIMARY METRICS ({})".format(
            a.label1,
            "all finite residuals"
            if a.metrics_use_all
            else "x0-fit clipped mask",
        )
    )
    print(
        "   N    = {}".format(
            m1["n"]
        )
    )
    print(
        "   RMSE = {:.6g} mm".format(
            m1["rmse"]
        )
    )
    print(
        "   MAE  = {:.6g} mm".format(
            m1["mae"]
        )
    )
    print(
        "   fit-mask RMSE = {:.6g} mm, all-finite RMSE = {:.6g} mm".format(
            m1_fitmask["rmse"],
            m1_all["rmse"],
        )
    )
    print()
    if a.drop_data2_events_excluded_from_data1_fit:
        print()
        print(
            " Dataset 2 matched-event exclusions = {}"
            .format(
                int(
                    data2_matched_excluded_mask.sum()
                )
            )
        )

    print(
        " DATASET 2: {} (NO REFIT)".format(
            a.label2
        )
    )
    print(
        "   N    = {}".format(
            m2["n"]
        )
    )
    print(
        "   RMSE = {:.6g} mm".format(
            m2["rmse"]
        )
    )
    print(
        "   MAE  = {:.6g} mm".format(
            m2["mae"]
        )
    )

    if (
        m1["rmse"] > 0
        and np.isfinite(
            m2["rmse"]
        )
    ):
        print()
        print(
            " RMSE ratio (dataset2 / dataset1) = {:.6g}"
            .format(
                m2["rmse"]
                / m1["rmse"]
            )
        )

    print()
    print(
        " HORIZONTAL RESIDUAL GAUSSIAN SUMMARY -- {}".format(
            a.label1
        )
    )
    print(
        "   mu    = {:.6g} mm".format(
            gaussian1["mu"]
        )
    )
    print(
        "   sigma = {:.6g} mm".format(
            gaussian1["sigma"]
        )
    )
    print(
        "   FWHM  = {:.6g} mm".format(
            gaussian1["fwhm"]
        )
    )

    print()
    print(
        " HORIZONTAL RESIDUAL GAUSSIAN SUMMARY -- {}".format(
            a.label2
        )
    )
    print(
        "   mu    = {:.6g} mm".format(
            gaussian2["mu"]
        )
    )
    print(
        "   sigma = {:.6g} mm".format(
            gaussian2["sigma"]
        )
    )
    print(
        "   FWHM  = {:.6g} mm".format(
            gaussian2["fwhm"]
        )
    )

    print()
    print(
        " TRUE TRANSVERSE SPREAD -- {}".format(
            a.label1
        )
    )
    print(
        "   mean  = {:.6g} a.u.".format(
            true_transverse_fit1["mu"]
        )
    )
    print(
        "   sigma = {:.6g} a.u.".format(
            true_transverse_fit1["sigma"]
        )
    )
    print(
        "   FWHM  = {:.6g} a.u.".format(
            true_transverse_fit1["fwhm"]
        )
    )

    print()
    print(
        " TRUE TRANSVERSE SPREAD -- {}".format(
            a.label2
        )
    )
    print(
        "   mean  = {:.6g} a.u.".format(
            true_transverse_fit2["mu"]
        )
    )
    print(
        "   sigma = {:.6g} a.u.".format(
            true_transverse_fit2["sigma"]
        )
    )
    print(
        "   FWHM  = {:.6g} a.u.".format(
            true_transverse_fit2["fwhm"]
        )
    )

    print(
        "============================================================"
    )

    print(
        "[OUTPUT]",
        png_path,
    )

    print(
        "[OUTPUT]",
        spread1_path,
    )

    print(
        "[OUTPUT]",
        spread2_path,
    )

    print(
        "[OUTPUT]",
        true_transverse1_path,
    )

    print(
        "[OUTPUT]",
        true_transverse2_path,
    )

    print(
        "[OUTPUT]",
        txt_path,
    )

    print(
        "[OUTPUT]",
        data1_path,
    )

    print(
        "[OUTPUT]",
        data2_path,
    )

    print(
        "[OUTPUT]",
        curve_path,
    )

    if a.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(spread_fig1)
        plt.close(spread_fig2)
        plt.close(true_transverse_fig1)
        plt.close(true_transverse_fig2)


if __name__ == "__main__":
    main()