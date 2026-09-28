#!/usr/bin/env python3
"""
Reconstruct alpha kinetic energy at the reaction vertex.
Python 3.6+ compatible.

Inputs:
  * reconstructed vertices, accepted in any of these forms:
      - legacy CSV: event_id, vert_1_x, vert_1_y, vert_1_z
      - static-ML selected_vertices directory: event_<N>_vertex.txt
      - static-ML selected CSV: event, cand_x, cand_y, cand_z
  * export_manifest.csv OR alpha_mapping.csv for event-number mapping
  * Morgan tracks_run_*.csv files (SiChan, eSi)
  * silicon calibration parameters (cobo asad aget channel intercept slope)
  * SRIM stopping/range table
  * silicon channel -> physical quadrant-center CSV

Calibration:
  raw_amplitude = raw_sign * eSi              [default raw_sign = -1]
  E_a = intercept + slope * raw_amplitude
  E_Si = 0.0017*E_a^2 + 1.13*E_a - 0.57

The alpha is assumed to stop fully in the silicon. E_Si is therefore treated
as its kinetic energy upon reaching the Si. The gas loss is reconstructed by
integrating SRIM stopping power backward from the Si quadrant center to the
reaction vertex using RK4.

By default the detector map assumes the RIGHT side of the experimental layout
is +x. Use --flip-detector-x to reverse that orientation.

Optional global geometry corrections:
    --detector-y-offset DY_MM
    --detector-z-offset DZ_MM

These are added to every silicon quadrant-center position before the gas path
length and SRIM energy-loss correction are calculated.
"""
from __future__ import print_function

import argparse
import csv
import glob
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def clean_text(value):
    if value is None:
        return ""
    return str(value).replace("\r", "").replace("^M", "").strip()


def finite_float(value):
    try:
        x = float(clean_text(value))
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def parse_args():
    p = argparse.ArgumentParser(description="Reconstruct alpha kinetic energy at the reaction vertex.")
    p.add_argument(
        "--vertices",
        default="alphas_analysis/alphas/events_dataframe.csv",
        help=(
            "Vertex input. Supported formats: legacy CSV with "
            "event_id,vert_1_x,vert_1_y,vert_1_z; static-ML "
            "selected_vertices directory containing event_<N>_vertex.txt; "
            "or static selected-candidate CSV with event,cand_x,cand_y,cand_z."
        )
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="export_manifest.csv; preferred mapping OutputEvent -> Run/Subrun/OriginalEvent")
    g.add_argument("--mapping-csv", help="alpha_mapping.csv fallback; event_id is zero-based mapping row index")

    p.add_argument("--tracks-dir", default="morgan_events")
    p.add_argument("--tracks-glob", default="tracks_run_*.csv")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--unsuffixed-subrun", type=int, default=0)
    p.add_argument("--subrun-offset", type=int, default=0)

    p.add_argument("--calibration", required=True)
    p.add_argument("--srim", required=True)
    p.add_argument("--detector-map", required=True)

    p.add_argument("--output-dir", default="alphas_analysis/alpha_energy")
    p.add_argument("--output-csv", default="alpha_kinetic_energy.csv")
    p.add_argument("--skipped-csv", default="alpha_energy_skipped.csv")
    p.add_argument("--plot-name", default="alpha_kinetic_energy_vs_vertexZ.png")
    p.add_argument(
        "--annotate-events",
        action="store_true",
        help="Write event_id centered on each E_alpha vs vertex-Z data point"
    )
    p.add_argument(
        "--annotation-fontsize",
        type=float,
        default=5.0,
        help="Font size for --annotate-events labels (default: 5)"
    )

    p.add_argument("--raw-sign", type=float, default=-1.0,
                   help="raw_amplitude = raw_sign*eSi; default -1 for negative-going signals")
    p.add_argument("--cobo", type=int, default=1)
    p.add_argument("--asad", type=int, default=0)
    p.add_argument("--aget", type=int, default=0)

    p.add_argument("--poly-a2", type=float, default=0.0017)
    p.add_argument("--poly-a1", type=float, default=1.13)
    p.add_argument("--poly-a0", type=float, default=-0.57)

    p.add_argument("--step-mm", type=float, default=0.10,
                   help="RK4 step size for backward gas-loss integration")
    p.add_argument("--flip-detector-x", action="store_true",
                   help="Multiply every detector-map x coordinate by -1")
    p.add_argument("--flip-detector-y", action="store_true",
                   help="Diagnostic option: multiply every detector-map y coordinate by -1")
    p.add_argument("--si-thickness-um", type=float, default=650.0,
                   help="Recorded as metadata; not used while full stopping in Si is assumed")
    p.add_argument(
        "--detector-y-offset",
        type=float,
        default=0.0,
        help="Global offset added to all silicon detector y positions [mm]. Default: 0.0"
    )

    p.add_argument(
        "--detector-z-offset",
        type=float,
        default=0.0,
        help="Global offset added to all silicon detector z positions [mm]. Default: 0.0"
    )

    p.add_argument("--allow-placeholder-calibration", action="store_true",
                   help="Allow calibration entries exactly intercept=0.1,slope=1.0")
    p.add_argument("--allow-srim-extrapolation", action="store_true")
    return p.parse_args()


def _clean_fieldnames(reader):
    if reader.fieldnames is not None:
        reader.fieldnames = [clean_text(x) for x in reader.fieldnames]
    return reader.fieldnames or []


def _first_existing(fields, candidates):
    field_set = set(fields)
    for name in candidates:
        if name in field_set:
            return name
    return None


def _choose_xyz_columns(fields):
    """
    Return (label, xcol, ycol, zcol) for supported vertex CSV formats.
    """
    coordinate_sets = [
        ("legacy_vert_1_xyz", "vert_1_x", "vert_1_y", "vert_1_z"),
        ("static_ml_cand_xyz", "cand_x", "cand_y", "cand_z"),
        ("static_ml_candidate_xyz", "candidate_x", "candidate_y", "candidate_z"),
        ("selected_xyz_mm", "selected_x_mm", "selected_y_mm", "selected_z_mm"),
        ("selected_xyz", "selected_x", "selected_y", "selected_z"),
        ("final_xyz_mm", "final_x_mm", "final_y_mm", "final_z_mm"),
        ("final_xyz", "final_x", "final_y", "final_z"),
        ("vertex_xyz_mm", "vertex_x_mm", "vertex_y_mm", "vertex_z_mm"),
        ("vertex_xyz", "vertex_x", "vertex_y", "vertex_z"),
        ("seed_xyz", "seed_x", "seed_y", "seed_z"),
        ("plain_xyz", "x", "y", "z"),
    ]

    field_set = set(fields)
    for label, xcol, ycol, zcol in coordinate_sets:
        if xcol in field_set and ycol in field_set and zcol in field_set:
            return label, xcol, ycol, zcol

    raise ValueError(
        "Could not identify vertex coordinate columns. Supported examples: "
        "vert_1_x/y/z, cand_x/y/z, selected_x_mm/y_mm/z_mm, "
        "final_x_mm/y_mm/z_mm, vertex_x_mm/y_mm/z_mm, seed_x/y/z, x/y/z. "
        "Available columns: {0}".format(", ".join(fields))
    )


def _event_id_from_vertex_filename(path):
    base = os.path.basename(path)
    patterns = [
        r"event[_-]?(\d+)[_-]?vertex",
        r"event[_-]?(\d+)",
        r"(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, base, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def _read_xyz_from_text_vertex(path):
    """
    Read one selected_vertices/event_<N>_vertex.txt style file.
    The first three numeric tokens are interpreted as x y z.
    """
    with open(path, "r") as f:
        for raw in f:
            s = clean_text(raw)
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            nums = []
            for item in parts:
                value = finite_float(item)
                if value is not None:
                    nums.append(value)
                if len(nums) >= 3:
                    return nums[0], nums[1], nums[2]
    return None, None, None


def read_vertices_directory(path):
    out = {}

    patterns = [
        os.path.join(path, "event_*_vertex.txt"),
        os.path.join(path, "event_*vertex*.txt"),
        os.path.join(path, "event_*.txt"),
    ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    for file_path in files:
        event_id = _event_id_from_vertex_filename(file_path)
        if event_id is None:
            continue

        x, y, z = _read_xyz_from_text_vertex(file_path)
        if x is None or y is None or z is None:
            continue

        out[event_id] = {
            "x": x,
            "y": y,
            "z": z,
            "in_line_file": os.path.basename(file_path),
            "vertex_format": "selected_vertices_directory",
        }

    if not out:
        raise ValueError(
            "No usable event_<N>_vertex.txt files found in vertex directory: {0}".format(path)
        )

    print("[INFO] Vertex input format: selected_vertices directory")
    print("[INFO] Vertex directory entries read: {0}".format(len(out)))
    return out


def read_vertices_csv(path):
    out = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        fields = _clean_fieldnames(r)
        if not fields:
            raise ValueError("Vertex CSV has no header")

        event_col = _first_existing(
            fields,
            ["event_id", "event", "Event", "OutputEvent", "output_event"]
        )
        if event_col is None:
            raise ValueError(
                "Could not identify event id column in vertex CSV. "
                "Expected one of event_id, event, Event, OutputEvent. "
                "Available columns: {0}".format(", ".join(fields))
            )

        fmt_label, xcol, ycol, zcol = _choose_xyz_columns(fields)

        for row in r:
            try:
                event_id = int(float(clean_text(row.get(event_col))))
            except Exception:
                continue

            x = finite_float(row.get(xcol))
            y = finite_float(row.get(ycol))
            z = finite_float(row.get(zcol))
            if x is None or y is None or z is None:
                continue

            out[event_id] = {
                "x": x,
                "y": y,
                "z": z,
                "in_line_file": clean_text(row.get("in_line_file", "")),
                "vertex_format": fmt_label,
            }

    if not out:
        raise ValueError("No usable vertex rows read from CSV: {0}".format(path))

    print("[INFO] Vertex CSV event column: {0}".format(event_col))
    print("[INFO] Vertex CSV coordinate format: {0}".format(fmt_label))
    print("[INFO] Vertex CSV entries read: {0}".format(len(out)))
    return out


def read_vertices(path):
    """
    Read vertex input in one of several formats:
      1. legacy CSV: event_id, vert_1_x, vert_1_y, vert_1_z
      2. selected_vertices directory: event_<N>_vertex.txt with one line x y z
      3. static selected-candidate CSV: event, cand_x, cand_y, cand_z
    """
    if os.path.isdir(path):
        return read_vertices_directory(path)
    return read_vertices_csv(path)


def read_manifest(path):
    out = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        required = ("OutputEvent", "Run", "Subrun", "OriginalEvent")
        if r.fieldnames is None:
            raise ValueError("Manifest has no header")
        missing = [x for x in required if x not in r.fieldnames]
        if missing:
            raise ValueError("Manifest missing: {0}".format(", ".join(missing)))
        for row in r:
            try:
                out[int(row["OutputEvent"])] = (
                    int(row["Run"]), int(row["Subrun"]), int(row["OriginalEvent"])
                )
            except Exception:
                continue
    return out


def read_mapping_csv(path):
    rows = []
    seen = set()
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                key = (int(float(clean_text(row["Run"]))),
                       int(float(clean_text(row["Subrun"]))),
                       int(float(clean_text(row["Event"]))))
            except Exception:
                continue
            # Match the exporter: it de-duplicated mapping rows while preserving order.
            if key not in seen:
                seen.add(key)
                rows.append(key)
    return dict((i, key) for i, key in enumerate(rows))


def read_calibration(path):
    out = {}
    with open(path, "r") as f:
        for line in f:
            text = clean_text(line)
            if not text or text.startswith("#"):
                continue
            p = text.split()
            if len(p) < 6:
                continue
            try:
                key = (int(p[0]), int(p[1]), int(p[2]), int(p[3]))
                out[key] = (float(p[4]), float(p[5]))
            except Exception:
                continue
    return out


def read_detector_map(path, flip_x=False, flip_y=False,
                      y_offset=0.0, z_offset=0.0):
    out = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        required = ("SiChan", "DetectorID", "Quadrant", "x_mm", "y_mm", "z_mm")
        if r.fieldnames is None:
            raise ValueError("Detector map has no header")
        missing = [x for x in required if x not in r.fieldnames]
        if missing:
            raise ValueError("Detector map missing: {0}".format(", ".join(missing)))
        for row in r:
            try:
                ch = int(row["SiChan"])
                x = float(row["x_mm"])
                y = float(row["y_mm"])
                z = float(row["z_mm"])
                if flip_x:
                    x = -x
                if flip_y:
                    y = -y

                # Apply global detector offsets
                y += y_offset
                z += z_offset

                out[ch] = {
                    "detector_id": int(row["DetectorID"]),
                    "quadrant": clean_text(row["Quadrant"]),
                    "x": x, "y": y, "z": z
                }
            except Exception:
                continue
    return out


def srim_energy_to_mev(value, unit):
    unit = unit.lower()
    if unit == "ev":
        return value * 1.0e-6
    if unit == "kev":
        return value * 1.0e-3
    if unit == "mev":
        return value
    if unit == "gev":
        return value * 1.0e3
    raise ValueError("Unknown SRIM energy unit: {0}".format(unit))


def read_srim(path):
    energies = []
    stopping_mass = []
    factor = None

    data_re = re.compile(
        r"^\s*([0-9.+\-Ee]+)\s*(eV|keV|MeV|GeV)\s+"
        r"([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)\s+"
    )
    factor_re = re.compile(r"^\s*([0-9.+\-Ee]+)\s+MeV\s*/\s*mm\s*$", re.IGNORECASE)

    with open(path, "r") as f:
        for line in f:
            m = data_re.match(line)
            if m:
                energies.append(srim_energy_to_mev(float(m.group(1)), m.group(2)))
                stopping_mass.append(float(m.group(3)) + float(m.group(4)))
                continue
            m = factor_re.match(line.strip())
            if m:
                factor = float(m.group(1))

    if not energies:
        raise ValueError("No SRIM data rows parsed")
    if factor is None:
        raise ValueError("Could not find SRIM MeV/mm conversion factor")

    pairs = sorted(zip(energies, stopping_mass))
    e = [x[0] for x in pairs]
    s = [x[1] * factor for x in pairs]
    return e, s, factor


class StoppingInterpolator(object):
    def __init__(self, e, s, extrapolate=False):
        self.e = e
        self.s = s
        self.extrapolate = extrapolate

    def segment(self, energy, i0, i1):
        e0, e1 = self.e[i0], self.e[i1]
        s0, s1 = self.s[i0], self.s[i1]
        if e1 == e0:
            return s0
        return s0 + (energy-e0) * (s1-s0) / (e1-e0)

    def __call__(self, energy):
        if energy < self.e[0]:
            if not self.extrapolate:
                raise ValueError("Energy below SRIM minimum: {0:.6g} MeV".format(energy))
            return self.segment(energy, 0, 1)
        if energy > self.e[-1]:
            if not self.extrapolate:
                raise ValueError("Energy above SRIM maximum: {0:.6g} MeV".format(energy))
            return self.segment(energy, len(self.e)-2, len(self.e)-1)
        if energy == self.e[-1]:
            return self.s[-1]
        lo, hi = 0, len(self.e)-1
        while hi-lo > 1:
            mid = (lo+hi)//2
            if self.e[mid] <= energy:
                lo = mid
            else:
                hi = mid
        return self.segment(energy, lo, lo+1)


def backward_integrate(e_si, path_mm, stopping, step_mm):
    if e_si <= 0.0:
        raise ValueError("Non-positive silicon energy")
    e = e_si
    left = path_mm
    while left > 0.0:
        h = min(step_mm, left)
        k1 = stopping(e)
        k2 = stopping(e + 0.5*h*k1)
        k3 = stopping(e + 0.5*h*k2)
        k4 = stopping(e + h*k3)
        e += (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        left -= h
    return e


def parse_track_run_subrun(path, unsuffixed_subrun, subrun_offset):
    m = re.match(r"^tracks_run_(\d+)\.dat\..*?(?:\.(\d+))?\.csv$",
                 os.path.basename(path), flags=re.IGNORECASE)
    if not m:
        return None
    run = int(m.group(1))
    subrun = unsuffixed_subrun if m.group(2) is None else int(m.group(2))
    return run, subrun + subrun_offset


def discover_track_files(args):
    pattern = os.path.join(args.tracks_dir, "**", args.tracks_glob) if args.recursive else os.path.join(args.tracks_dir, args.tracks_glob)
    out = {}
    for path in sorted(glob.glob(pattern, recursive=args.recursive)):
        key = parse_track_run_subrun(path, args.unsuffixed_subrun, args.subrun_offset)
        if key is not None:
            out.setdefault(key, []).append(os.path.abspath(path))
    return out, pattern


def iter_track_rows(path):
    with open(path, "r", newline="") as f:
        header = None
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            header = [clean_text(x) for x in next(csv.reader([clean_text(line)]))]
            break
        if header is None:
            return
        r = csv.DictReader(f, fieldnames=header)
        for row in r:
            yield row


def load_needed_tracks(track_files, needed_keys):
    needed_by_group = {}
    for run, subrun, event in needed_keys:
        needed_by_group.setdefault((run, subrun), set()).add(event)

    hits = {}
    for group, wanted in needed_by_group.items():
        for path in track_files.get(group, []):
            print("[SCAN] {0}".format(path))
            source_row = 0
            for row in iter_track_rows(path):
                source_row += 1
                try:
                    event = int(float(clean_text(row.get("Event"))))
                except Exception:
                    continue
                if event not in wanted:
                    continue
                row = dict(row)
                row["_source_file"] = path
                row["_source_row"] = source_row
                hits.setdefault((group[0], group[1], event), []).append(row)
    return hits


def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fmt(v):
    if isinstance(v, float):
        return "{0:.8f}".format(v)
    return v


def main():
    args = parse_args()
    args.vertices = os.path.abspath(args.vertices)
    args.tracks_dir = os.path.abspath(args.tracks_dir)
    args.calibration = os.path.abspath(args.calibration)
    args.srim = os.path.abspath(args.srim)
    args.detector_map = os.path.abspath(args.detector_map)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.manifest:
        args.manifest = os.path.abspath(args.manifest)
    if args.mapping_csv:
        args.mapping_csv = os.path.abspath(args.mapping_csv)

    if args.step_mm <= 0:
        print("[ERROR] --step-mm must be positive", file=sys.stderr)
        return 1
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    vertices = read_vertices(args.vertices)

    if args.manifest:
        event_map = read_manifest(args.manifest)
        mapping_mode = "manifest"

        if not event_map:
            print(
                "[ERROR] The manifest contains no usable OutputEvent mappings. "
                "Use --mapping-csv alpha_mapping.csv instead, or regenerate the "
                "manifest with the corrected exporter.",
                file=sys.stderr
            )
            return 1
    else:
        event_map = read_mapping_csv(args.mapping_csv)
        mapping_mode = "alpha_mapping.csv zero-based mapping index"

    calibration = read_calibration(args.calibration)
    detector_map = read_detector_map(
        args.detector_map,
        args.flip_detector_x,
        args.flip_detector_y,
        args.detector_y_offset,
        args.detector_z_offset
    )
    srim_e, srim_s, srim_factor = read_srim(args.srim)
    stopping = StoppingInterpolator(srim_e, srim_s, args.allow_srim_extrapolation)
    track_files, track_pattern = discover_track_files(args)

    print("[INFO] Vertices loaded: {0}".format(len(vertices)))
    print("[INFO] Event mapping mode: {0}".format(mapping_mode))
    print("[INFO] Event mappings loaded: {0}".format(len(event_map)))
    print("[INFO] Calibration rows loaded: {0}".format(len(calibration)))
    print("[INFO] Detector channels mapped: {0}".format(len(detector_map)))
    print("[INFO] Track search: {0}".format(track_pattern))
    print("[INFO] SRIM range: {0:.6g}-{1:.6g} MeV; multiplier to MeV/mm={2:.8g}".format(srim_e[0], srim_e[-1], srim_factor))
    print("[INFO] Detector x orientation: {0}".format("flipped" if args.flip_detector_x else "layout right = +x"))
    print("[INFO] Silicon thickness metadata: {0:.1f} um (not used in full-stop approximation)".format(args.si_thickness_um))

    skipped = []
    vertex_to_key = {}
    for event_id in sorted(vertices):
        v = vertices[event_id]
        if v["x"] is None or v["y"] is None or v["z"] is None:
            skipped.append({"event_id":event_id,"Run":"","Subrun":"","OriginalEvent":"","SiChan":"","reason":"missing_vertex","detail":"blank/non-finite vertex coordinate"})
            continue
        key = event_map.get(event_id)
        if key is None:
            skipped.append({"event_id":event_id,"Run":"","Subrun":"","OriginalEvent":"","SiChan":"","reason":"missing_event_mapping","detail":"event_id absent from manifest/mapping"})
            continue
        vertex_to_key[event_id] = key

    hits = load_needed_tracks(track_files, set(vertex_to_key.values()))
    out_rows = []

    for event_id in sorted(vertex_to_key):
        run, subrun, original_event = vertex_to_key[event_id]
        v = vertices[event_id]
        rows = hits.get((run, subrun, original_event), [])
        if not rows:
            skipped.append({"event_id":event_id,"Run":run,"Subrun":subrun,"OriginalEvent":original_event,"SiChan":"","reason":"missing_track_hit","detail":"no matching tracks row"})
            continue

        for hit_number, hit in enumerate(rows):
            try:
                ch = int(float(clean_text(hit.get("SiChan"))))
            except Exception:
                skipped.append({"event_id":event_id,"Run":run,"Subrun":subrun,"OriginalEvent":original_event,"SiChan":clean_text(hit.get("SiChan")),"reason":"invalid_si_channel","detail":""})
                continue
            eSi = finite_float(hit.get("eSi"))
            if eSi is None:
                skipped.append({"event_id":event_id,"Run":run,"Subrun":subrun,"OriginalEvent":original_event,"SiChan":ch,"reason":"invalid_eSi","detail":clean_text(hit.get("eSi"))})
                continue

            cal = calibration.get((args.cobo, args.asad, args.aget, ch))
            if cal is None:
                skipped.append({"event_id":event_id,"Run":run,"Subrun":subrun,"OriginalEvent":original_event,"SiChan":ch,"reason":"missing_calibration","detail":"no calibration row for (1,0,0,SiChan)"})
                continue
            intercept, slope = cal
            placeholder = abs(intercept-0.1) < 1e-12 and abs(slope-1.0) < 1e-12
            if placeholder and not args.allow_placeholder_calibration:
                skipped.append({"event_id":event_id,"Run":run,"Subrun":subrun,"OriginalEvent":original_event,"SiChan":ch,"reason":"placeholder_calibration","detail":"intercept=0.1 slope=1.0"})
                continue

            det = detector_map.get(ch)
            if det is None:
                skipped.append({"event_id":event_id,"Run":run,"Subrun":subrun,"OriginalEvent":original_event,"SiChan":ch,"reason":"missing_detector_position","detail":"channel absent from detector map"})
                continue

            raw_amp = args.raw_sign * eSi
            E_a = intercept + slope * raw_amp
            E_si = args.poly_a2*E_a*E_a + args.poly_a1*E_a + args.poly_a0
            if E_si <= 0 or not math.isfinite(E_si):
                skipped.append({"event_id":event_id,"Run":run,"Subrun":subrun,"OriginalEvent":original_event,"SiChan":ch,"reason":"nonpositive_calibrated_energy","detail":"Ea={0:.5g}, ESi={1:.5g}".format(E_a,E_si)})
                continue

            vxyz = (v["x"], v["y"], v["z"])
            sixyz = (det["x"], det["y"], det["z"])
            path_mm = distance(vxyz, sixyz)
            try:
                E_vertex = backward_integrate(E_si, path_mm, stopping, args.step_mm)
                """
                print(
                    "[SRIM DEBUG] event={0} SiChan={1} "
                    "vertex=({2:.2f},{3:.2f},{4:.2f}) "
                    "Si=({5:.2f},{6:.2f},{7:.2f}) "
                    "path={8:.2f} mm "
                    "E_si={9:.4f} MeV "
                    "dE_gas={10:.4f} MeV "
                    "E_vertex={11:.4f} MeV".format(
                        event_id,
                        ch,
                        vxyz[0], vxyz[1], vxyz[2],
                        sixyz[0], sixyz[1], sixyz[2],
                        path_mm,
                        E_si,
                        E_vertex - E_si,
                        E_vertex
                    )
                )
                """
            except Exception as exc:
                skipped.append({"event_id":event_id,"Run":run,"Subrun":subrun,"OriginalEvent":original_event,"SiChan":ch,"reason":"srim_integration_failed","detail":str(exc)})
                continue

            out_rows.append({
                "event_id":event_id,
                "Run":run,
                "Subrun":subrun,
                "OriginalEvent":original_event,
                "HitNumber":hit_number,
                "Zone":clean_text(hit.get("Zone","")),
                "SiChan":ch,
                "DetectorID":det["detector_id"],
                "Quadrant":det["quadrant"],
                "raw_eSi_ADC":eSi,
                "raw_amplitude_positive_ADC":raw_amp,
                "cal_intercept_MeV":intercept,
                "cal_slope_MeV_per_ADC":slope,
                "E_a_linear_MeV":E_a,
                "E_Si_corrected_MeV":E_si,
                "vertex_x_mm":v["x"],
                "vertex_y_mm":v["y"],
                "vertex_z_mm":v["z"],
                "Si_x_mm":det["x"],
                "Si_y_mm":det["y"],
                "Si_z_mm":det["z"],
                "gas_path_mm":path_mm,
                "gas_energy_loss_MeV":E_vertex-E_si,
                "E_kinetic_alpha_vertex_MeV":E_vertex,
                "si_thickness_um_metadata":args.si_thickness_um,
                "detector_y_offset_mm":args.detector_y_offset,
                "detector_z_offset_mm":args.detector_z_offset,
                "SRIM_step_mm":args.step_mm,
                "psi_tracks_deg":clean_text(hit.get("psi","")),
                "theta_tracks_deg":clean_text(hit.get("theta","")),
                "phi_tracks_deg":clean_text(hit.get("phi","")),
                "TrackSourceFile":os.path.basename(hit.get("_source_file","")),
                "TrackSourceRow":hit.get("_source_row","")
            })

    output_fields = [
        "event_id","Run","Subrun","OriginalEvent","HitNumber","Zone","SiChan","DetectorID","Quadrant",
        "raw_eSi_ADC","raw_amplitude_positive_ADC","cal_intercept_MeV","cal_slope_MeV_per_ADC",
        "E_a_linear_MeV","E_Si_corrected_MeV","vertex_x_mm","vertex_y_mm","vertex_z_mm",
        "Si_x_mm","Si_y_mm","Si_z_mm","gas_path_mm","gas_energy_loss_MeV","E_kinetic_alpha_vertex_MeV",
        "si_thickness_um_metadata","detector_y_offset_mm","detector_z_offset_mm",
        "SRIM_step_mm","psi_tracks_deg","theta_tracks_deg","phi_tracks_deg",
        "TrackSourceFile","TrackSourceRow"
    ]
    skipped_fields = ["event_id","Run","Subrun","OriginalEvent","SiChan","reason","detail"]

    out_path = os.path.join(args.output_dir, args.output_csv)
    skip_path = os.path.join(args.output_dir, args.skipped_csv)
    plot_path = os.path.join(args.output_dir, args.plot_name)
    formatted = []
    for row in out_rows:
        formatted.append(dict((k,fmt(v)) for k,v in row.items()))
    write_csv(out_path, output_fields, formatted)
    write_csv(skip_path, skipped_fields, skipped)

    if out_rows:
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111)
        x_values = [r["vertex_z_mm"] for r in out_rows]
        y_values = [r["E_kinetic_alpha_vertex_MeV"] for r in out_rows]

        ax.scatter(
            x_values,
            y_values,
            s=24 if args.annotate_events else 18,
            alpha=0.8
        )

        if args.annotate_events:
            for row, x_value, y_value in zip(out_rows, x_values, y_values):
                ax.annotate(
                    str(row["event_id"]),
                    (x_value, y_value),
                    ha="center",
                    va="center",
                    fontsize=args.annotation_fontsize
                )

        ax.set_xlabel("Vertex Z [mm]")
        ax.set_ylabel("Alpha kinetic energy at vertex [MeV]")
        ax.set_title("Alpha kinetic energy vs reconstructed vertex Z")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

    counts = {}
    for row in skipped:
        counts[row["reason"]] = counts.get(row["reason"],0) + 1

    print("\nAlpha energy reconstruction complete")
    print("  successful silicon hits: {0}".format(len(out_rows)))
    print("  output CSV: {0}".format(out_path))
    print("  skipped CSV: {0}".format(skip_path))
    if out_rows:
        print("  plot: {0}".format(plot_path))
    if counts:
        print("  skipped/rejected:")
        for reason in sorted(counts):
            print("    {0}: {1}".format(reason, counts[reason]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
