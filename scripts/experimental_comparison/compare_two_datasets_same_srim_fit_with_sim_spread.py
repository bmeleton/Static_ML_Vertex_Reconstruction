#!/usr/bin/env python3
"""
compare_two_datasets_same_srim_fit_with_sim_spread.py

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
10. Optionally read TeBATSim ROOT truth points using lightE + heavyE.
11. Horizontally align the simulated cloud to the frozen SRIM curve.
12. Save a second plot with a min-max SRIM-shaped simulated spread envelope.

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
    python3 compare_two_datasets_same_srim_fit.py SRIM_O14_in_96He_4CO2_580Torr.txt --data1 alphas_analysis/alpha_cm_compare_static_ml/alpha_cm_reconstruction.csv --data2 alphas_analysis/alpha_cm_compare_static_ml/alpha_cm_hough_reconstruction.csv --density 1.78e-4 --ecm-min 1 --ecm-max 10 --label1 "ML vertex" --label2 "Hough vertex" --show


Python 3.6 / SciPy 1.5 compatible.
"""

from __future__ import print_function

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
        "--sim-root",
        type=Path,
        default=None,
        help=(
            "Optional TeBATSim ROOT file containing simulated spread points. "
            "The simulation is shifted horizontally only; the common SRIM "
            "curve fitted to Dataset 1 is NOT changed."
        ),
    )

    p.add_argument(
        "--sim-tree",
        default="simData",
        help="TTree name in --sim-root (default: simData)",
    )

    p.add_argument(
        "--sim-vertex-z-branch",
        default="vertexZ",
        help="Simulation reaction-z branch (default: vertexZ)",
    )

    p.add_argument(
        "--sim-light-energy-branch",
        default="lightE",
        help="Simulation outgoing alpha energy branch (default: lightE)",
    )

    p.add_argument(
        "--sim-heavy-energy-branch",
        default="heavyE",
        help="Simulation outgoing 14O energy branch (default: heavyE)",
    )

    p.add_argument(
        "--sim-label",
        default="TeBATSim spread",
        help="Legend label for simulated points",
    )

    p.add_argument(
        "--sim-max-events",
        type=int,
        default=None,
        help="Optional maximum number of simulated ROOT entries to read",
    )

    p.add_argument(
        "--output-prefix",
        default="srim_same_fit_two_dataset_comparison",
        help="Output filename prefix",
    )

    p.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively",
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


def read_simulation_root(path, a, cm_factor):
    """Read TeBATSim truth points and calculate E_cm from lightE + heavyE."""
    try:
        import ROOT
    except ImportError:
        raise SystemExit(
            "--sim-root was supplied, but PyROOT could not be imported.\n"
            "Run from the same ROOT environment used for TeBATSim."
        )

    root_file = ROOT.TFile.Open(str(path), "READ")

    if not root_file or root_file.IsZombie():
        raise ValueError(
            "Could not open simulation ROOT file: {}".format(path)
        )

    tree = root_file.Get(a.sim_tree)

    if tree is None:
        root_file.Close()
        raise ValueError(
            "Could not find TTree '{}' in {}".format(
                a.sim_tree,
                path,
            )
        )

    for branch in (
        a.sim_vertex_z_branch,
        a.sim_light_energy_branch,
        a.sim_heavy_energy_branch,
    ):
        if tree.GetBranch(branch) is None:
            root_file.Close()
            raise ValueError(
                "Required simulation branch '{}' was not found in tree '{}'."
                .format(branch, a.sim_tree)
            )

    n_entries = int(tree.GetEntries())

    if a.sim_max_events is not None:
        n_process = min(n_entries, a.sim_max_events)
    else:
        n_process = n_entries

    entries = []
    x_raw = []
    ecm = []
    light_e = []
    heavy_e = []

    for i in range(n_process):
        tree.GetEntry(i)

        z = float(
            getattr(tree, a.sim_vertex_z_branch)
        )
        e_light = float(
            getattr(tree, a.sim_light_energy_branch)
        )
        e_heavy = float(
            getattr(tree, a.sim_heavy_energy_branch)
        )

        e_cm = cm_factor * (e_light + e_heavy)

        if not (
            np.isfinite(z)
            and np.isfinite(e_light)
            and np.isfinite(e_heavy)
            and np.isfinite(e_cm)
        ):
            continue

        if e_cm < a.ecm_min or e_cm > a.ecm_max:
            continue

        entries.append(i)
        x_raw.append(z)
        ecm.append(e_cm)
        light_e.append(e_light)
        heavy_e.append(e_heavy)

    root_file.Close()

    if len(x_raw) < 5:
        raise ValueError(
            "Simulation ROOT file has too few usable events in the requested E_cm range."
        )

    return {
        "entry": np.asarray(entries, dtype=int),
        "x_raw": np.asarray(x_raw, dtype=float),
        "ecm": np.asarray(ecm, dtype=float),
        "lightE": np.asarray(light_e, dtype=float),
        "heavyE": np.asarray(heavy_e, dtype=float),
        "path": path,
    }


def align_simulation_horizontally(
    sim,
    E_tab,
    S_tab,
    cm_factor,
    ecm_ref,
    density,
    common_x0,
):
    """
    Horizontally translate the simulated points to the already-fitted common
    SRIM curve. No density or SRIM parameter is changed.

    For each simulated E_cm, calculate the x-position expected from the common
    SRIM curve. The least-squares best horizontal translation is the mean of
    (x_expected - x_sim_raw).
    """
    A_sim = build_areal_coordinate(
        E_tab,
        S_tab,
        sim["ecm"],
        cm_factor,
        ecm_ref,
    )

    expected_x = (
        common_x0
        + A_sim / (100.0 * density)
    )

    shift_mm = float(
        np.mean(
            expected_x - sim["x_raw"]
        )
    )

    x_shifted = sim["x_raw"] + shift_mm
    x_residual = x_shifted - expected_x

    return shift_mm, x_shifted, expected_x, x_residual


def simulation_vertical_band(
    x_shifted,
    ecm_sim,
    x_curve,
    ecm_curve,
):
    """
    Construct a constant vertical-residual envelope around the SRIM curve.

    After the simulated cloud has been horizontally aligned, evaluate the SRIM
    E_cm at each simulated x.  The minimum and maximum vertical residuals

        delta_E = E_cm(sim) - E_cm(SRIM at same x)

    define two translated copies of the SRIM contour.  These become the lower
    and upper boundaries of the shaded spread band.
    """
    order = np.argsort(x_curve)
    x_sorted = np.asarray(x_curve[order], dtype=float)
    e_sorted = np.asarray(ecm_curve[order], dtype=float)

    e_of_x = PchipInterpolator(
        x_sorted,
        e_sorted,
        extrapolate=False,
    )

    inside = (
        np.isfinite(x_shifted)
        & np.isfinite(ecm_sim)
        & (x_shifted >= x_sorted.min())
        & (x_shifted <= x_sorted.max())
    )

    if int(inside.sum()) < 3:
        raise ValueError(
            "Too few shifted simulated points overlap the plotted SRIM curve "
            "to construct the spread band."
        )

    e_expected = np.asarray(
        e_of_x(x_shifted[inside]),
        dtype=float,
    )

    vertical_residual = (
        ecm_sim[inside] - e_expected
    )

    delta_low = float(
        np.min(vertical_residual)
    )
    delta_high = float(
        np.max(vertical_residual)
    )

    band_low = ecm_curve + delta_low
    band_high = ecm_curve + delta_high

    return (
        band_low,
        band_high,
        delta_low,
        delta_high,
        inside,
        vertical_residual,
    )


def write_simulation_csv(
    path,
    sim,
    x_shifted,
    expected_x,
    x_residual,
):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "entry",
            "vertexZ_raw_mm",
            "vertexZ_shifted_mm",
            "Ecm_energy_sum_MeV",
            "lightE_MeV",
            "heavyE_MeV",
            "distance_common_SRIM_mm",
            "horizontal_residual_mm",
        ])

        for i in range(len(sim["x_raw"])):
            w.writerow([
                sim["entry"][i],
                sim["x_raw"][i],
                x_shifted[i],
                sim["ecm"][i],
                sim["lightE"][i],
                sim["heavyE"][i],
                expected_x[i],
                x_residual[i],
            ])


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

    if a.drop_data2_events_excluded_from_data1_fit:
        for eid, keep in zip(
            d1["ids"],
            fit_metric_mask1,
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
            "[INFO] Dataset 1 events excluded from x0 fit: {}"
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

    # Optional TeBATSim spread sample.  The common SRIM fit remains frozen.
    sim = None
    sim_shift_mm = None
    sim_x_shifted = None
    sim_expected_x = None
    sim_x_residual = None
    sim_metrics = None
    band_low = None
    band_high = None
    band_delta_low = None
    band_delta_high = None
    sim_band_inside = None
    sim_vertical_residual = None

    if a.sim_root is not None:
        sim = read_simulation_root(
            a.sim_root,
            a,
            cm_factor,
        )

        (
            sim_shift_mm,
            sim_x_shifted,
            sim_expected_x,
            sim_x_residual,
        ) = align_simulation_horizontally(
            sim,
            E_tab,
            S_tab,
            cm_factor,
            ecm_ref,
            a.density,
            x0,
        )

        sim_metrics = metrics(
            sim_x_residual
        )

        (
            band_low,
            band_high,
            band_delta_low,
            band_delta_high,
            sim_band_inside,
            sim_vertical_residual,
        ) = simulation_vertical_band(
            sim_x_shifted,
            sim["ecm"],
            x_curve,
            ecm_curve,
        )

        print()
        print("[INFO] Optional TeBATSim spread sample:")
        print("       file =", sim["path"])
        print("       events =", len(sim["x_raw"]))
        print(
            "       horizontal shift = {:.6f} mm".format(
                sim_shift_mm
            )
        )
        print(
            "       horizontal RMSE to common SRIM = {:.6f} mm".format(
                sim_metrics["rmse"]
            )
        )
        print(
            "       vertical spread relative to SRIM = [{:.6f}, {:.6f}] MeV"
            .format(
                band_delta_low,
                band_delta_high,
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

    sim_csv_path = prefix.with_name(
        prefix.name
        + "_simulation_spread.csv"
    )

    band_png_path = prefix.with_name(
        prefix.name
        + "_spread_band.png"
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

    if sim is not None:
        write_simulation_csv(
            sim_csv_path,
            sim,
            sim_x_shifted,
            sim_expected_x,
            sim_x_residual,
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
                    a.drop_data2_events_excluded_from_data1_fit
                )
            )
        )
        f.write(
            "Dataset 1 events excluded from x0 fit = {}\n".format(
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

        if sim is not None:
            f.write(
                "\nSIMULATION SPREAD SAMPLE -- HORIZONTAL SHIFT ONLY\n"
            )
            f.write(
                "  ROOT file = {}\n".format(
                    a.sim_root
                )
            )
            f.write(
                "  E_cm definition = target/(projectile+target) * (lightE + heavyE)\n"
            )
            f.write(
                "  N = {}\n".format(
                    sim_metrics["n"]
                )
            )
            f.write(
                "  Applied horizontal shift = {:.9g} mm\n".format(
                    sim_shift_mm
                )
            )
            f.write(
                "  Horizontal RMSE to common SRIM = {:.9g} mm\n".format(
                    sim_metrics["rmse"]
                )
            )
            f.write(
                "  Horizontal MAE to common SRIM = {:.9g} mm\n".format(
                    sim_metrics["mae"]
                )
            )
            f.write(
                "  Vertical residual minimum = {:.9g} MeV\n".format(
                    band_delta_low
                )
            )
            f.write(
                "  Vertical residual maximum = {:.9g} MeV\n".format(
                    band_delta_high
                )
            )
            f.write(
                "  Full min/max band width = {:.9g} MeV\n".format(
                    band_delta_high - band_delta_low
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

    fig, ax = plt.subplots(
        figsize=(
            10.0,
            6.8,
        )
    )

    ax.scatter(
        d1["x"][fit_metric_mask1],
        d1["ecm"][fit_metric_mask1],
        s=28,
        alpha=0.70,
        label="{} (used for x0 fit)".format(
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
        marker="s",
        label="{} (no refit)".format(
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
            label="{} matched excluded".format(
                a.label2
            ),
        )

    if sim is not None:
        ax.scatter(
            sim_x_shifted,
            sim["ecm"],
            s=18,
            alpha=0.32,
            marker=".",
            label="{} (horizontally aligned)".format(
                a.sim_label
            ),
        )

    ax.plot(
        x_curve,
        ecm_curve,
        linewidth=2.7,
        label="One common SRIM curve",
    )

    ax.set_xlabel(
        "Beam path / distance coordinate (mm)"
    )

    ax.set_ylabel(
        r"$E_{\rm cm}$ (MeV)"
    )

    ax.set_ylim(
        a.ecm_min,
        a.ecm_max,
    )

    ax.set_xlim(90, 230)

    ax.set_title(
        "Two datasets compared to the exact same SRIM fit"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fit_text = (
        r"$\rho$ = "
        + "{:.4e}".format(
            a.density
        )
        + r" g/cm$^3$"
        + "\n"
        + r"$x_0$ = "
        + "{:.2f}".format(
            x0
        )
        + " mm"
        + "\n"
        + "{} RMSE = {:.2f} mm".format(
            a.label1,
            m1["rmse"],
        )
        + "\n"
        + "{} RMSE = {:.2f} mm".format(
            a.label2,
            m2["rmse"],
        )
    )

    ax.text(
        0.03,
        0.05,
        fit_text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.88,
        ),
    )

    fig.tight_layout()

    fig.savefig(
        str(
            png_path
        ),
        dpi=220,
    )

    # A second presentation-style plot replaces the simulated point cloud
    # with a dark envelope.  Its upper/lower edges are translated copies of
    # the SRIM contour using the minimum and maximum vertical residuals of the
    # horizontally aligned TeBATSim points.
    band_fig = None
    if sim is not None:
        band_fig, band_ax = plt.subplots(
            figsize=(
                10.0,
                6.8,
            )
        )

        band_ax.scatter(
            d1["x"][fit_metric_mask1],
            d1["ecm"][fit_metric_mask1],
            s=28,
            alpha=0.70,
            label="{} (used for x0 fit)".format(
                a.label1
            ),
        )

        if np.any(rejected1):
            band_ax.scatter(
                d1["x"][rejected1],
                d1["ecm"][rejected1],
                s=24,
                alpha=0.35,
                marker="x",
                label="{} excluded from x0 fit".format(
                    a.label1
                ),
            )

        band_ax.scatter(
            d2["x"][metric_mask2],
            d2["ecm"][metric_mask2],
            s=28,
            alpha=0.65,
            marker="s",
            label="{} (no refit)".format(
                a.label2
            ),
        )

        if np.any(data2_matched_excluded_mask):
            band_ax.scatter(
                d2["x"][data2_matched_excluded_mask],
                d2["ecm"][data2_matched_excluded_mask],
                s=26,
                alpha=0.35,
                marker="x",
                label="{} matched excluded".format(
                    a.label2
                ),
            )

        # Faint raw simulated points remain visible so the origin of the band
        # is clear, while the shaded region emphasizes the full spread.
        band_ax.scatter(
            sim_x_shifted,
            sim["ecm"],
            s=12,
            alpha=0.12,
            marker=".",
            label="{} points".format(
                a.sim_label
            ),
        )

        band_ax.fill_between(
            x_curve,
            band_low,
            band_high,
            alpha=0.38,
            label="{} full min-max envelope".format(
                a.sim_label
            ),
        )

        band_ax.plot(
            x_curve,
            ecm_curve,
            linewidth=2.7,
            label="One common SRIM curve",
        )

        band_ax.plot(
            x_curve,
            band_low,
            linewidth=1.1,
            alpha=0.75,
        )

        band_ax.plot(
            x_curve,
            band_high,
            linewidth=1.1,
            alpha=0.75,
        )

        band_ax.set_xlabel(
            "Beam path / distance coordinate (mm)"
        )
        band_ax.set_ylabel(
            r"$E_{\rm cm}$ (MeV)"
        )
        band_ax.set_ylim(
            a.ecm_min,
            a.ecm_max,
        )
        band_ax.set_xlim(90, 230)
        band_ax.set_title(
            "Experimental datasets with TeBATSim spread envelope"
        )
        band_ax.grid(alpha=0.3)
        band_ax.legend()

        band_text = (
            "{} horizontal shift = {:.2f} mm".format(
                a.sim_label,
                sim_shift_mm,
            )
            + "\n"
            + "Band residuals = [{:.3f}, {:.3f}] MeV".format(
                band_delta_low,
                band_delta_high,
            )
        )

        band_ax.text(
            0.03,
            0.05,
            band_text,
            transform=band_ax.transAxes,
            va="bottom",
            ha="left",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.88,
            ),
        )

        band_fig.tight_layout()
        band_fig.savefig(
            str(band_png_path),
            dpi=220,
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

    print(
        "============================================================"
    )

    print(
        "[OUTPUT]",
        png_path,
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

    if sim is not None:
        print(
            "[OUTPUT]",
            sim_csv_path,
        )
        print(
            "[OUTPUT]",
            band_png_path,
        )

    if a.show:
        plt.show()
    else:
        plt.close(
            fig
        )
        if band_fig is not None:
            plt.close(
                band_fig
            )


if __name__ == "__main__":
    main()
