from __future__ import print_function

"""
generate_vertex_candidates_from_hough.py

Standalone candidate-vertex generator for TeBAT Hough-line outputs.

What it does
------------
For each event_N/lines.txt file:
  1. Parse all Hough lines and their assigned points.
  2. Generate original pairwise closest-approach vertex candidates.
  3. For each original candidate, optionally generate a conservative
     pivot-refit candidate:
        - keep the endpoint farthest from the proposed vertex fixed,
        - use only that Hough line's assigned points,
        - reject points within r_exclude of the proposed vertex,
        - reject points on the wrong side of the proposed vertex,
        - reject points outside a tube/angle gate around that line branch,
        - optionally veto points that look more like the competing branch,
        - refit only the line direction through the fixed far endpoint.
  4. If truth ROOT is provided, compute distance from every candidate to the
     truth reaction vertex and write the per-event truth-best candidate.

This script does NOT modify Cloud_classes.py or any original Hough-line files.
It is intended as a diagnostic / candidate-source generator for downstream ML
seed assessment.

Python: compatible with Python 3.6+
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd


STATIC_BMELETON_MERGE_VERSION = "2026-08-07-v4-bmeleton-final-refit-candidate"

EPS = 1.0e-12


def clamp(x, lo=-1.0, hi=1.0):
    try:
        x = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))


def unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= EPS:
        return np.zeros_like(v, dtype=float)
    return v / n


def angle_deg(a, b):
    ua = unit(a)
    ub = unit(b)
    if float(np.linalg.norm(ua)) <= EPS or float(np.linalg.norm(ub)) <= EPS:
        return np.nan
    return float(np.degrees(np.arccos(clamp(float(np.dot(ua, ub))))))


def line_point_distances(points, anchor, direction):
    points = np.asarray(points, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    direction = unit(direction)
    if points.size == 0 or float(np.linalg.norm(direction)) <= EPS:
        return np.zeros(0, dtype=float)
    q = points - anchor[None, :]
    return np.linalg.norm(np.cross(q, direction[None, :]), axis=1)


def closest_approach(anchor_a, dir_a, anchor_b, dir_b):
    """Closest approach between two infinite 3D lines.

    Returns: sep_mm, midpoint_xyz, point_on_a, point_on_b, nearly_parallel
    """
    pa = np.asarray(anchor_a, dtype=float)
    pb = np.asarray(anchor_b, dtype=float)
    da = unit(dir_a)
    db = unit(dir_b)
    if float(np.linalg.norm(da)) <= EPS or float(np.linalg.norm(db)) <= EPS:
        nan = np.array([np.nan, np.nan, np.nan], dtype=float)
        return np.nan, nan, nan, nan, True

    w0 = pa - pb
    aa = float(np.dot(da, da))
    bb = float(np.dot(da, db))
    cc = float(np.dot(db, db))
    dd = float(np.dot(da, w0))
    ee = float(np.dot(db, w0))
    denom = aa * cc - bb * bb

    if abs(denom) < 1.0e-10:
        # Parallel approximation: use line-a perpendicular distance from pb.
        ca = pa
        cb = pb
        sep = float(np.linalg.norm(np.cross(pb - pa, da)))
        mid = 0.5 * (ca + cb)
        return sep, mid, ca, cb, True

    sc = (bb * ee - cc * dd) / denom
    tc = (aa * ee - bb * dd) / denom
    ca = pa + sc * da
    cb = pb + tc * db
    sep = float(np.linalg.norm(ca - cb))
    mid = 0.5 * (ca + cb)
    return sep, mid, ca, cb, False


def finite_xyz(x):
    x = np.asarray(x, dtype=float)
    return x.shape == (3,) and bool(np.all(np.isfinite(x)))


# ----------------------------- Hough parser -----------------------------


def parse_hough_lines(path):
    """Parse event_N/lines.txt into simple dictionaries.

    Header format expected, based on existing Cloud_classes.py parser:
      # Line 1: n=85 a=(...) b=(...) start=(...) end=(...)

    Data rows are expected to contain at least:
      x y z globalTime peakAmplitude integratedCharge
    """
    lines = []
    current = None
    vec = r"\(\s*([^)]+?)\s*\)"
    header_re = re.compile(
        r"#\s*Line\s*(\d+).*?n=(\d+)\s*"
        r"a=\s*" + vec + r".*?"
        r"b=\s*" + vec + r".*?"
        r"start=\s*" + vec + r".*?"
        r"end=\s*" + vec,
        re.IGNORECASE,
    )

    def close_current():
        if current is None:
            return
        pts = np.asarray(current.get("points", []), dtype=float)
        if pts.size == 0:
            pts = np.zeros((0, 6), dtype=float)
        elif pts.ndim == 1:
            pts = pts.reshape((1, -1))
        current["points_array"] = pts
        if pts.shape[1] >= 3:
            current["xyz"] = pts[:, 0:3].astype(float)
        else:
            current["xyz"] = np.zeros((0, 3), dtype=float)
        current["n_points_parsed"] = int(current["xyz"].shape[0])
        lines.append(current)

    try:
        fh = open(path, "r")
    except Exception as exc:
        raise RuntimeError("Could not open %s: %s" % (path, exc))

    with fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            m = header_re.search(line)
            if m:
                close_current()
                try:
                    line_num = int(m.group(1))
                    n_header = int(m.group(2))
                    a = np.fromstring(m.group(3), sep=",", dtype=float)
                    b = np.fromstring(m.group(4), sep=",", dtype=float)
                    start = np.fromstring(m.group(5), sep=",", dtype=float)
                    end = np.fromstring(m.group(6), sep=",", dtype=float)
                except Exception:
                    current = None
                    continue
                b = unit(b)
                # Match the spirit of the old parser: orient start/end by norm.
                if finite_xyz(start) and finite_xyz(end):
                    if float(np.linalg.norm(end)) < float(np.linalg.norm(start)):
                        start, end = end.copy(), start.copy()
                        b = -b
                current = dict(
                    line_num=line_num,
                    n_header=n_header,
                    a=a,
                    b=b,
                    start=start,
                    end=end,
                    points=[],
                    source_file=path,
                )
                continue

            if not line or line.startswith("#"):
                continue
            if current is None:
                continue
            parts = line.replace(",", " ").split()
            vals = []
            ok = True
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    ok = False
                    break
            if ok and len(vals) >= 3:
                # Pad rows to 6 values for consistency.
                while len(vals) < 6:
                    vals.append(np.nan)
                current["points"].append(vals[:6])

    close_current()
    return lines


# ----------------------------- truth reader -----------------------------


def read_truth_reaction_vertices(rootfile, only_siHitE=False):
    """Return dict[event_id] = np.array([vertexX, vertexY, vertexZ])."""
    if rootfile is None or str(rootfile).lower() in ("", "none", "null"):
        return {}
    if not os.path.isfile(rootfile):
        raise RuntimeError("Missing truth ROOT file: %s" % rootfile)

    try:
        import ROOT
    except Exception as exc:
        raise RuntimeError("Could not import ROOT to read truth file: %s" % exc)

    f = ROOT.TFile.Open(rootfile)
    if not f or f.IsZombie():
        raise RuntimeError("Could not open ROOT file: %s" % rootfile)
    t = f.Get("simData")
    if not t:
        raise RuntimeError("Could not find tree 'simData' in %s" % rootfile)

    truth = {}
    try:
        t.SetBranchStatus("*", 0)
        for br in ["siHitE", "vertexX", "vertexY", "vertexZ"]:
            try:
                t.SetBranchStatus(br, 1)
            except Exception:
                pass
    except Exception:
        pass

    for iev, entry in enumerate(t):
        if only_siHitE:
            try:
                if len(entry.siHitE) == 0:
                    continue
            except Exception:
                pass
        try:
            v = np.array([float(entry.vertexX), float(entry.vertexY), float(entry.vertexZ)], dtype=float)
        except Exception:
            continue
        if finite_xyz(v):
            truth[int(iev)] = v
    return truth


# ----------------------------- support selection and refit -----------------------------


def far_endpoint(line, v0):
    start = np.asarray(line["start"], dtype=float)
    end = np.asarray(line["end"], dtype=float)
    v0 = np.asarray(v0, dtype=float)
    ds = float(np.linalg.norm(start - v0)) if finite_xyz(start) else -np.inf
    de = float(np.linalg.norm(end - v0)) if finite_xyz(end) else -np.inf
    if de >= ds:
        return end.copy(), "end", de
    return start.copy(), "start", ds


def select_assigned_branch_points(line, v0, branch_out, other_branch_out, args):
    """Select safe assigned points for pivot refit.

    branch_out points from candidate vertex toward the far endpoint of this line.
    Only line['xyz'] points are considered.
    """
    pts = np.asarray(line.get("xyz", np.zeros((0, 3))), dtype=float)
    if pts.shape[0] == 0:
        return pts, dict(
            support_n_initial=0,
            support_n_after_side=0,
            support_n_after_tube=0,
            support_n_after_veto=0,
        )

    v0 = np.asarray(v0, dtype=float)
    d = unit(branch_out)
    od = unit(other_branch_out)
    q = pts - v0[None, :]
    dist = np.linalg.norm(q, axis=1)
    s = np.dot(q, d)
    perp = np.linalg.norm(q - s[:, None] * d[None, :], axis=1)
    branch_len = float(np.linalg.norm(branch_out))

    finite = np.isfinite(dist) & np.isfinite(s) & np.isfinite(perp)
    side_mask = finite & (dist > float(args.exclude_radius_mm)) & (s > float(args.exclude_radius_mm))
    if branch_len > EPS:
        side_mask = side_mask & (s < branch_len + float(args.branch_margin_mm))
    n_after_side = int(np.sum(side_mask))

    # Angle gate and tube gate.  Either gate alone can sometimes be too harsh;
    # require both by default for conservative assigned-only behavior.
    cosang = np.zeros_like(dist)
    good_dist = dist > EPS
    cosang[good_dist] = s[good_dist] / dist[good_dist]
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    tube_mask = side_mask & (perp <= float(args.tube_radius_mm)) & (ang <= float(args.angle_gate_deg))
    n_after_tube = int(np.sum(tube_mask))

    keep = tube_mask.copy()

    if bool(args.competitor_veto) and float(np.linalg.norm(od)) > EPS:
        so = np.dot(q, od)
        coso = np.zeros_like(dist)
        coso[good_dist] = so[good_dist] / dist[good_dist]
        ango = np.degrees(np.arccos(np.clip(coso, -1.0, 1.0)))
        # Drop only if the point is clearly better aligned with the other branch.
        other_side = so > float(args.exclude_radius_mm)
        steal = other_side & (ango + float(args.competitor_angle_margin_deg) < ang)
        keep = keep & (~steal)

    n_after_veto = int(np.sum(keep))
    info = dict(
        support_n_initial=int(pts.shape[0]),
        support_n_after_side=n_after_side,
        support_n_after_tube=n_after_tube,
        support_n_after_veto=n_after_veto,
    )
    return pts[keep], info


def pivot_refit_line(line, v0, other_line, args):
    """Refit line direction through its far endpoint using assigned points only."""
    p_far, far_name, branch_len = far_endpoint(line, v0)
    p_other_far, other_far_name, other_branch_len = far_endpoint(other_line, v0)

    branch_out = p_far - np.asarray(v0, dtype=float)
    other_out = p_other_far - np.asarray(v0, dtype=float)
    if float(np.linalg.norm(branch_out)) <= EPS:
        return None

    pts, selinfo = select_assigned_branch_points(line, v0, branch_out, other_out, args)
    support_n = int(pts.shape[0])
    if support_n < int(args.min_support_points):
        out = dict(valid=False, reason="too_few_support_points")
        out.update(selinfo)
        out.update(dict(
            p_far_x=p_far[0], p_far_y=p_far[1], p_far_z=p_far[2],
            far_endpoint=far_name,
            branch_length_mm=branch_len,
            support_n=support_n,
        ))
        return out

    # Fit a line constrained to pass through p_far.  Since the anchor is fixed,
    # use an uncentered second-moment/SVD of vectors from p_far to points.
    q_anchor = pts - p_far[None, :]
    try:
        uu, ss, vv = np.linalg.svd(q_anchor, full_matrices=False)
        new_dir = unit(vv[0])
    except Exception:
        out = dict(valid=False, reason="svd_failed")
        out.update(selinfo)
        return out

    # Orient direction from far endpoint back toward candidate vertex.
    toward_vertex = unit(np.asarray(v0, dtype=float) - p_far)
    if float(np.dot(new_dir, toward_vertex)) < 0:
        new_dir = -new_dir

    original_toward_vertex = toward_vertex
    angle_change = angle_deg(new_dir, original_toward_vertex)
    residuals = line_point_distances(pts, p_far, new_dir)
    med_resid = float(np.median(residuals)) if residuals.size else np.nan
    rms_resid = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else np.nan
    max_resid = float(np.max(residuals)) if residuals.size else np.nan

    valid = True
    reason = "ok"
    if not np.isfinite(angle_change) or angle_change > float(args.max_angle_change_deg):
        valid = False
        reason = "angle_change_too_large"
    if not np.isfinite(med_resid) or med_resid > float(args.max_median_residual_mm):
        valid = False
        reason = "median_residual_too_large"

    out = dict(
        valid=bool(valid),
        reason=reason,
        p_far_x=float(p_far[0]),
        p_far_y=float(p_far[1]),
        p_far_z=float(p_far[2]),
        far_endpoint=far_name,
        branch_length_mm=float(branch_len),
        support_n=support_n,
        direction_x=float(new_dir[0]),
        direction_y=float(new_dir[1]),
        direction_z=float(new_dir[2]),
        angle_change_deg=float(angle_change) if np.isfinite(angle_change) else np.nan,
        resid_median_mm=med_resid,
        resid_rms_mm=rms_resid,
        resid_max_mm=max_resid,
    )
    out.update(selinfo)
    return out


# ----------------------------- candidate rows -----------------------------


def candidate_score(pair_sep, current_score_like, support1=np.nan, support2=np.nan,
                    resid1=np.nan, resid2=np.nan, angle_change1=np.nan, angle_change2=np.nan):
    """A simple geometry-only ranking score. Truth is NOT used."""
    if not np.isfinite(pair_sep):
        return -np.inf
    sep_penalty = 1.0 / (1.0 + max(0.0, float(pair_sep)) / 5.0)
    base = max(0.0, float(current_score_like)) * sep_penalty

    support_factor = 1.0
    if np.isfinite(support1) and np.isfinite(support2):
        support_factor = math.sqrt(max(0.0, float(support1)) * max(0.0, float(support2))) / 20.0
        support_factor = max(0.2, min(3.0, support_factor))

    resid_penalty = 1.0
    vals = [x for x in [resid1, resid2] if np.isfinite(x)]
    if vals:
        resid_penalty = 1.0 / (1.0 + float(np.mean(vals)) / 2.0)

    angle_penalty = 1.0
    vals = [x for x in [angle_change1, angle_change2] if np.isfinite(x)]
    if vals:
        angle_penalty = 1.0 / (1.0 + float(np.mean(vals)) / 20.0)

    return float(base * support_factor * resid_penalty * angle_penalty)


def add_truth_columns(row, truth_xyz):
    if truth_xyz is None or not finite_xyz(truth_xyz):
        row.update(dict(
            truth_x=np.nan, truth_y=np.nan, truth_z=np.nan,
            dx_candidate_minus_truth=np.nan,
            dy_candidate_minus_truth=np.nan,
            dz_candidate_minus_truth=np.nan,
            dist_candidate_minus_truth=np.nan,
        ))
        return row
    cand = np.array([row.get("cand_x", np.nan), row.get("cand_y", np.nan), row.get("cand_z", np.nan)], dtype=float)
    diff = cand - truth_xyz
    row.update(dict(
        truth_x=float(truth_xyz[0]), truth_y=float(truth_xyz[1]), truth_z=float(truth_xyz[2]),
        dx_candidate_minus_truth=float(diff[0]),
        dy_candidate_minus_truth=float(diff[1]),
        dz_candidate_minus_truth=float(diff[2]),
        dist_candidate_minus_truth=float(np.linalg.norm(diff)),
    ))
    return row


def make_base_candidate_row(event_id, candidate_id, source, line1, line2, cand, pair_sep, pair_parallel, truth_xyz):
    b1 = unit(line1["b"])
    b2 = unit(line2["b"])
    dot12 = float(np.dot(b1, b2)) if float(np.linalg.norm(b1)) > EPS and float(np.linalg.norm(b2)) > EPS else np.nan
    current_score_like = 1.0 - dot12 if np.isfinite(dot12) else np.nan
    nonparallel_score = 1.0 - abs(dot12) if np.isfinite(dot12) else np.nan
    pair_angle = angle_deg(b1, b2)

    row = dict(
        event=int(event_id),
        candidate_id=int(candidate_id),
        candidate_source=source,
        cand_x=float(cand[0]),
        cand_y=float(cand[1]),
        cand_z=float(cand[2]),
        line1=int(line1["line_num"]),
        line2=int(line2["line_num"]),
        line1_n_header=int(line1.get("n_header", -1)),
        line2_n_header=int(line2.get("n_header", -1)),
        line1_n_points=int(line1.get("n_points_parsed", 0)),
        line2_n_points=int(line2.get("n_points_parsed", 0)),
        pair_closest_approach_mm=float(pair_sep),
        pair_nearly_parallel=bool(pair_parallel),
        pair_angle_deg=float(pair_angle) if np.isfinite(pair_angle) else np.nan,
        current_hough_score_like=float(current_score_like) if np.isfinite(current_score_like) else np.nan,
        nonparallel_score=float(nonparallel_score) if np.isfinite(nonparallel_score) else np.nan,
    )
    score = candidate_score(pair_sep, current_score_like,
                            support1=line1.get("n_points_parsed", np.nan),
                            support2=line2.get("n_points_parsed", np.nan))
    row["candidate_score"] = score
    add_truth_columns(row, truth_xyz)
    return row


def make_refit_candidate_row(event_id, candidate_id, source, line1, line2, refit1, refit2,
                             original_mid, truth_xyz):
    p1 = np.array([refit1["p_far_x"], refit1["p_far_y"], refit1["p_far_z"]], dtype=float)
    d1 = np.array([refit1["direction_x"], refit1["direction_y"], refit1["direction_z"]], dtype=float)
    p2 = np.array([refit2["p_far_x"], refit2["p_far_y"], refit2["p_far_z"]], dtype=float)
    d2 = np.array([refit2["direction_x"], refit2["direction_y"], refit2["direction_z"]], dtype=float)
    sep, mid, ca, cb, parallel = closest_approach(p1, d1, p2, d2)

    dot12 = float(np.dot(unit(d1), unit(d2)))
    current_score_like = 1.0 - dot12
    nonparallel_score = 1.0 - abs(dot12)
    pair_angle = angle_deg(d1, d2)
    shift = float(np.linalg.norm(mid - np.asarray(original_mid, dtype=float))) if finite_xyz(mid) else np.nan

    row = dict(
        event=int(event_id),
        candidate_id=int(candidate_id),
        candidate_source=source,
        cand_x=float(mid[0]),
        cand_y=float(mid[1]),
        cand_z=float(mid[2]),
        line1=int(line1["line_num"]),
        line2=int(line2["line_num"]),
        line1_n_header=int(line1.get("n_header", -1)),
        line2_n_header=int(line2.get("n_header", -1)),
        line1_n_points=int(line1.get("n_points_parsed", 0)),
        line2_n_points=int(line2.get("n_points_parsed", 0)),
        pair_closest_approach_mm=float(sep),
        pair_nearly_parallel=bool(parallel),
        pair_angle_deg=float(pair_angle) if np.isfinite(pair_angle) else np.nan,
        current_hough_score_like=float(current_score_like) if np.isfinite(current_score_like) else np.nan,
        nonparallel_score=float(nonparallel_score) if np.isfinite(nonparallel_score) else np.nan,
        refit_shift_from_original_candidate_mm=shift,
        line1_refit_valid=bool(refit1.get("valid", False)),
        line2_refit_valid=bool(refit2.get("valid", False)),
        line1_refit_reason=refit1.get("reason", ""),
        line2_refit_reason=refit2.get("reason", ""),
        line1_support_n=int(refit1.get("support_n", 0)),
        line2_support_n=int(refit2.get("support_n", 0)),
        line1_support_n_after_side=int(refit1.get("support_n_after_side", 0)),
        line2_support_n_after_side=int(refit2.get("support_n_after_side", 0)),
        line1_support_n_after_tube=int(refit1.get("support_n_after_tube", 0)),
        line2_support_n_after_tube=int(refit2.get("support_n_after_tube", 0)),
        line1_support_n_after_veto=int(refit1.get("support_n_after_veto", 0)),
        line2_support_n_after_veto=int(refit2.get("support_n_after_veto", 0)),
        line1_resid_median_mm=float(refit1.get("resid_median_mm", np.nan)),
        line2_resid_median_mm=float(refit2.get("resid_median_mm", np.nan)),
        line1_resid_rms_mm=float(refit1.get("resid_rms_mm", np.nan)),
        line2_resid_rms_mm=float(refit2.get("resid_rms_mm", np.nan)),
        line1_angle_change_deg=float(refit1.get("angle_change_deg", np.nan)),
        line2_angle_change_deg=float(refit2.get("angle_change_deg", np.nan)),
        line1_far_endpoint=refit1.get("far_endpoint", ""),
        line2_far_endpoint=refit2.get("far_endpoint", ""),
    )
    row["candidate_score"] = candidate_score(
        sep,
        current_score_like,
        support1=row["line1_support_n"],
        support2=row["line2_support_n"],
        resid1=row["line1_resid_median_mm"],
        resid2=row["line2_resid_median_mm"],
        angle_change1=row["line1_angle_change_deg"],
        angle_change2=row["line2_angle_change_deg"],
    )
    add_truth_columns(row, truth_xyz)
    return row


def original_line_as_refit(line, v0):
    p_far, far_name, branch_len = far_endpoint(line, v0)
    toward_vertex = unit(np.asarray(v0, dtype=float) - p_far)
    pts = np.asarray(line.get("xyz", np.zeros((0, 3))), dtype=float)
    residuals = line_point_distances(pts, p_far, toward_vertex)
    return dict(
        valid=True,
        reason="original_line",
        p_far_x=float(p_far[0]), p_far_y=float(p_far[1]), p_far_z=float(p_far[2]),
        far_endpoint=far_name,
        branch_length_mm=float(branch_len),
        support_n=int(pts.shape[0]),
        support_n_after_side=int(pts.shape[0]),
        support_n_after_tube=int(pts.shape[0]),
        support_n_after_veto=int(pts.shape[0]),
        direction_x=float(toward_vertex[0]),
        direction_y=float(toward_vertex[1]),
        direction_z=float(toward_vertex[2]),
        angle_change_deg=0.0,
        resid_median_mm=float(np.median(residuals)) if residuals.size else np.nan,
        resid_rms_mm=float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else np.nan,
        resid_max_mm=float(np.max(residuals)) if residuals.size else np.nan,
    )




# ----------------------------- BMeleton-compatible merged-line candidates -----------------------------

_CLOUD_IMPORT_ERROR = None

def _import_cloud_classes():
    """Import Cloud_classes only when BMeleton-compatible mode is requested."""
    global _CLOUD_IMPORT_ERROR
    try:
        import Cloud_classes as cloud
        return cloud
    except Exception as exc:
        _CLOUD_IMPORT_ERROR = exc
        raise


def _quiet_call(func, args_obj):
    """Suppress noisy Cloud_classes printout unless --verbose-bmeleton-merge is set."""
    if bool(getattr(args_obj, 'verbose_bmeleton_merge', False)):
        return func()
    # Python 3.6-safe stdout redirection.
    old_stdout = sys.stdout
    try:
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        return func()
    finally:
        try:
            sys.stdout = old_stdout
            devnull.close()
        except Exception:
            sys.stdout = old_stdout


def _cloud_line_to_dict(line_obj, line_index, stage):
    """Convert a Cloud_classes.Line object to the simple line dict used here.

    Cloud_classes may use float line numbers after splitting/refitting.  For the
    static candidate pipeline we assign a stable integer line_num within the
    chosen stage and preserve the original Cloud line number separately.
    """
    try:
        pts_df = getattr(line_obj, 'points', None)
        if pts_df is not None and hasattr(pts_df, 'columns') and all(c in pts_df.columns for c in ['x','y','z']):
            xyz = pts_df[['x','y','z']].values.astype(float)
            arr_cols = []
            for c in ['x','y','z','gTime','peakAmp','integQ']:
                if c in pts_df.columns:
                    arr_cols.append(pts_df[c].values.astype(float))
                else:
                    arr_cols.append(np.full(len(pts_df), np.nan, dtype=float))
            points_array = np.vstack(arr_cols).T if len(arr_cols) else np.zeros((0,6), dtype=float)
        else:
            xyz = np.zeros((0,3), dtype=float)
            points_array = np.zeros((0,6), dtype=float)
    except Exception:
        xyz = np.zeros((0,3), dtype=float)
        points_array = np.zeros((0,6), dtype=float)
    try:
        cloud_num = float(getattr(line_obj, 'lineNum', line_index))
    except Exception:
        cloud_num = float(line_index)
    out = dict(
        line_num=int(line_index),
        cloud_line_num=float(cloud_num),
        line_stage=str(stage),
        n_header=int(getattr(line_obj, 'numOfPts', xyz.shape[0])),
        a=np.asarray(getattr(line_obj, 'a', np.zeros(3)), dtype=float),
        b=unit(np.asarray(getattr(line_obj, 'b', np.zeros(3)), dtype=float)),
        start=np.asarray(getattr(line_obj, 'start', np.zeros(3)), dtype=float),
        end=np.asarray(getattr(line_obj, 'end', np.zeros(3)), dtype=float),
        xyz=xyz,
        points_array=points_array,
        n_points_parsed=int(xyz.shape[0]),
        source_file='',
    )
    return out


def _build_bmeleton_merged_data(line_file, args):
    """Run the BMeleton/Cloud_classes merge + intersection-finding stage.

    This intentionally mirrors the old seed finder through:
      merge_all_lines(); find_all_intersections()
    but it does not immediately convert one best intersection to a vertex.
    That lets the static ML model choose among the post-merge intersections.
    """
    cloud = _import_cloud_classes()

    def work():
        line_objs = cloud.parse_hough_lines(line_file)
        tree = cloud.Mytree(lines=line_objs)
        tree.merge_all_lines()
        tree.find_all_intersections()
        return tree

    tree = _quiet_call(work, args)
    try:
        sorted_lines = tree.sort_lines()
    except Exception:
        sorted_lines = list(getattr(tree, 'lines', []))
    line_dicts = []
    obj_to_dict = {}
    for idx, line_obj in enumerate(sorted_lines):
        try:
            if int(getattr(line_obj, 'numOfPts', 0)) <= 0:
                continue
        except Exception:
            pass
        d = _cloud_line_to_dict(line_obj, len(line_dicts), 'bmeleton_merged')
        d['source_file'] = line_file
        line_dicts.append(d)
        obj_to_dict[id(line_obj)] = d

    intersections = []
    try:
        inters = tree.sorted_intersections()
    except Exception:
        inters = list(getattr(tree, 'intersections', []))
    for inter in inters:
        l1 = obj_to_dict.get(id(getattr(inter, 'line1', None)))
        l2 = obj_to_dict.get(id(getattr(inter, 'line2', None)))
        if l1 is None or l2 is None:
            continue
        try:
            p1 = np.asarray(inter.point1, dtype=float)
            p2 = np.asarray(inter.point2, dtype=float)
            sep = float(np.linalg.norm(p1 - p2))
        except Exception:
            sep = np.nan
        try:
            mid = np.asarray(inter.point, dtype=float)
        except Exception:
            sep2, mid, ca, cb, parallel = closest_approach(l1['a'], l1['b'], l2['a'], l2['b'])
            if not np.isfinite(sep):
                sep = sep2
        intersections.append(dict(
            line1=l1,
            line2=l2,
            cand=mid,
            pair_sep=sep,
            nearly_parallel=False,
            bmeleton_intersection_score=float(getattr(inter, 'score', np.nan)),
            bmeleton_intersection_id=int(getattr(inter, 'id', -1)),
        ))
    return dict(lines=line_dicts, intersections=intersections)


def _generate_pair_candidates_from_line_pairs(event_id, pair_iter, args, truth_xyz, cand_id_start,
                                             base_source, pivot_source_prefix, line_stage):
    rows = []
    cand_id = int(cand_id_start)
    pair_records = []

    for rec in pair_iter:
        l1 = rec['line1']
        l2 = rec['line2']
        mid = np.asarray(rec['cand'], dtype=float)
        sep = float(rec.get('pair_sep', np.nan))
        parallel = bool(rec.get('nearly_parallel', False))
        if not np.isfinite(sep) or not finite_xyz(mid):
            continue
        if sep > float(args.max_initial_pair_dist_mm):
            continue
        row = make_base_candidate_row(event_id, cand_id, base_source, l1, l2, mid, sep, parallel, truth_xyz)
        row['line_stage'] = str(line_stage)
        if 'bmeleton_intersection_score' in rec:
            row['bmeleton_intersection_score'] = rec.get('bmeleton_intersection_score', np.nan)
        if 'bmeleton_intersection_id' in rec:
            row['bmeleton_intersection_id'] = rec.get('bmeleton_intersection_id', -1)
        rows.append(row)
        pair_records.append((row['candidate_score'], cand_id, l1, l2, mid, sep, parallel))
        cand_id += 1

    if not pair_records:
        return rows, cand_id

    pair_records.sort(key=lambda x: x[0], reverse=True)
    if int(args.max_pairs_for_refit) > 0:
        pair_records_refit = pair_records[:int(args.max_pairs_for_refit)]
    else:
        pair_records_refit = pair_records

    if bool(args.enable_pivot_refit):
        for score0, base_cand_id, l1, l2, mid, sep, parallel in pair_records_refit:
            r1 = pivot_refit_line(l1, mid, l2, args)
            r2 = pivot_refit_line(l2, mid, l1, args)
            if r1 is None or r2 is None:
                continue
            if bool(r1.get('valid', False)) and bool(r2.get('valid', False)):
                row = make_refit_candidate_row(event_id, cand_id, pivot_source_prefix + 'pivot_refit_both', l1, l2, r1, r2, mid, truth_xyz)
                row['line_stage'] = str(line_stage)
                row['parent_candidate_id'] = int(base_cand_id)
                row['parent_pair_closest_approach_mm'] = float(sep)
                if np.isfinite(row['pair_closest_approach_mm']) and row['pair_closest_approach_mm'] <= float(args.max_refit_pair_dist_mm):
                    rows.append(row)
                    cand_id += 1

                if bool(args.include_mixed_refits):
                    o2 = original_line_as_refit(l2, mid)
                    row = make_refit_candidate_row(event_id, cand_id, pivot_source_prefix + 'pivot_refit_line1_only', l1, l2, r1, o2, mid, truth_xyz)
                    row['line_stage'] = str(line_stage)
                    row['parent_candidate_id'] = int(base_cand_id)
                    row['parent_pair_closest_approach_mm'] = float(sep)
                    if np.isfinite(row['pair_closest_approach_mm']) and row['pair_closest_approach_mm'] <= float(args.max_refit_pair_dist_mm):
                        rows.append(row)
                        cand_id += 1
                    o1 = original_line_as_refit(l1, mid)
                    row = make_refit_candidate_row(event_id, cand_id, pivot_source_prefix + 'pivot_refit_line2_only', l1, l2, o1, r2, mid, truth_xyz)
                    row['line_stage'] = str(line_stage)
                    row['parent_candidate_id'] = int(base_cand_id)
                    row['parent_pair_closest_approach_mm'] = float(sep)
                    if np.isfinite(row['pair_closest_approach_mm']) and row['pair_closest_approach_mm'] <= float(args.max_refit_pair_dist_mm):
                        rows.append(row)
                        cand_id += 1
    return rows, cand_id


def _raw_pair_iter_from_lines(lines):
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            l1 = lines[i]
            l2 = lines[j]
            sep, mid, ca, cb, parallel = closest_approach(l1['a'], l1['b'], l2['a'], l2['b'])
            yield dict(line1=l1, line2=l2, cand=mid, pair_sep=sep, nearly_parallel=parallel)


def _merged_pair_iter_from_data(data):
    for rec in data.get('intersections', []):
        yield rec




def _choose_two_vertex_lines(vertex, obj_to_dict):
    """Choose two representative final-refit lines attached to a Cloud_classes.Vertex."""
    chosen = []
    try:
        lines = list(getattr(vertex, 'lines', []))
    except Exception:
        lines = []
    for lo in lines:
        d = obj_to_dict.get(id(lo))
        if d is not None:
            chosen.append(d)
    # Prefer the highest-support surviving lines.  This keeps the feature vector
    # stable even if Cloud_classes returns a set with arbitrary order.
    chosen.sort(key=lambda x: (int(x.get('n_points_parsed', 0)), -int(x.get('line_num', 0))), reverse=True)
    if len(chosen) >= 2:
        return chosen[0], chosen[1], len(chosen)
    return None, None, len(chosen)


def _build_bmeleton_final_refit_data(line_file, args):
    """Run the full BMeleton/Cloud_classes vertex finder and return final vertices.

    This mirrors the original final seed path:
      Event(...).identify_vertex()
    which calls tree.identify_vertex(), i.e. merge_all_lines(),
    find_all_intersections(), convert_best_intersection(), and refit_vertices().
    The resulting refitted Vertex.point values are added as explicit candidates.
    """
    cloud = _import_cloud_classes()

    def work():
        ev = cloud.Event(event_num=-1, out_dir='.', line_file=line_file)
        ev.identify_vertex()
        return ev.tree

    tree = _quiet_call(work, args)
    try:
        sorted_lines = tree.sort_lines()
    except Exception:
        sorted_lines = list(getattr(tree, 'lines', []))

    line_dicts = []
    obj_to_dict = {}
    for idx, line_obj in enumerate(sorted_lines):
        try:
            if int(getattr(line_obj, 'numOfPts', 0)) <= 0:
                continue
        except Exception:
            pass
        d = _cloud_line_to_dict(line_obj, len(line_dicts), 'bmeleton_final_refit')
        d['source_file'] = line_file
        line_dicts.append(d)
        obj_to_dict[id(line_obj)] = d

    vertices = []
    try:
        verts = sorted(list(getattr(tree, 'vertices', [])), key=lambda v: int(getattr(v, 'id', 0)))
    except Exception:
        verts = list(getattr(tree, 'vertices', []))

    for v in verts:
        try:
            pt = np.asarray(getattr(v, 'point', [np.nan, np.nan, np.nan]), dtype=float)
        except Exception:
            continue
        if not finite_xyz(pt):
            continue
        l1, l2, n_vlines = _choose_two_vertex_lines(v, obj_to_dict)
        if l1 is None or l2 is None:
            # The downstream static feature extractor currently needs two lines.
            # Keep this vertex out rather than creating a row that will only have
            # missing-line features.
            continue
        sep, mid, ca, cb, parallel = closest_approach(l1['a'], l1['b'], l2['a'], l2['b'])
        vertices.append(dict(
            line1=l1,
            line2=l2,
            cand=pt,
            pair_sep=sep,
            nearly_parallel=parallel,
            bmeleton_final_vertex_id=int(getattr(v, 'id', -1)),
            bmeleton_final_vertex_radius=float(getattr(v, 'radius', np.nan)),
            bmeleton_final_n_vertex_lines=int(n_vlines),
        ))

    return dict(lines=line_dicts, vertices=vertices)


def _generate_final_vertex_candidates_from_data(event_id, final_data, args, truth_xyz, cand_id_start):
    rows = []
    cand_id = int(cand_id_start)
    for rec in final_data.get('vertices', []):
        l1 = rec['line1']
        l2 = rec['line2']
        cand = np.asarray(rec['cand'], dtype=float)
        sep = float(rec.get('pair_sep', np.nan))
        parallel = bool(rec.get('nearly_parallel', False))
        if not finite_xyz(cand):
            continue
        # Do NOT cut on max_initial_pair_dist here: this is the actual final
        # BMeleton refitted vertex.  It should remain in the candidate pool even
        # if its two representative final lines are not close as infinite lines.
        row = make_base_candidate_row(event_id, cand_id, 'bmeleton_final_refit_vertex', l1, l2, cand, sep, parallel, truth_xyz)
        row['line_stage'] = 'bmeleton_final_refit'
        row['bmeleton_final_vertex_id'] = rec.get('bmeleton_final_vertex_id', -1)
        row['bmeleton_final_vertex_radius'] = rec.get('bmeleton_final_vertex_radius', np.nan)
        row['bmeleton_final_n_vertex_lines'] = rec.get('bmeleton_final_n_vertex_lines', np.nan)
        # Give the hand-score baseline a recognizable value, but let the ML use
        # the full feature vector.  The truth-independent candidate_score is still
        # based on geometry only.
        row['bmeleton_final_candidate_score'] = row.get('candidate_score', np.nan)
        rows.append(row)
        cand_id += 1
    return rows, cand_id

def get_feature_line_maps_for_event(line_file, args):
    """Return stage -> {line_num: line_dict} for static feature extraction."""
    maps = {}
    # Raw stage is always cheap and useful for legacy/raw candidate CSVs.
    try:
        raw_lines = parse_hough_lines(line_file)
        maps['raw'] = dict((int(l['line_num']), l) for l in raw_lines)
    except Exception:
        maps['raw'] = {}

    source = getattr(args, 'candidate_line_source', 'bmeleton_merged')
    need_merged = source in ('bmeleton_merged', 'both')
    need_final = bool(getattr(args, 'include_bmeleton_final_vertex', True)) and source in ('bmeleton_merged', 'both')
    # Also build maps when rows in an existing CSV request them.
    if bool(getattr(args, 'force_bmeleton_feature_lines', False)):
        need_merged = True
    if bool(getattr(args, 'force_bmeleton_final_feature_lines', False)):
        need_final = True
    if need_merged:
        try:
            data = _build_bmeleton_merged_data(line_file, args)
            maps['bmeleton_merged'] = dict((int(l['line_num']), l) for l in data.get('lines', []))
        except Exception as exc:
            maps['bmeleton_merged'] = {}
            if bool(getattr(args, 'verbose', False)):
                print('[WARN] BMeleton merged feature lines failed for %s: %s' % (line_file, exc))
    if need_final:
        try:
            fdata = _build_bmeleton_final_refit_data(line_file, args)
            maps['bmeleton_final_refit'] = dict((int(l['line_num']), l) for l in fdata.get('lines', []))
        except Exception as exc:
            maps['bmeleton_final_refit'] = {}
            if bool(getattr(args, 'verbose', False)):
                print('[WARN] BMeleton final-refit feature lines failed for %s: %s' % (line_file, exc))
    return maps


def generate_candidates_for_event(event_id, line_file, args, truth_xyz=None):
    """Generate candidate vertices for one event.

    candidate_line_source controls which line set is used:
      raw              : original raw Hough line pairs, previous behavior
      bmeleton_merged  : Cloud_classes merge_all_lines()+find_all_intersections(), default
      both             : include both raw and BMeleton-merged candidates
    """
    source = getattr(args, 'candidate_line_source', 'bmeleton_merged')
    rows = []
    status_messages = []
    cand_id = 0
    n_lines_total = 0

    if source in ('raw', 'both'):
        try:
            raw_lines = parse_hough_lines(line_file)
        except Exception as exc:
            raw_lines = []
            status_messages.append('raw_parse_failed:%s' % exc)
        n_lines_total += len(raw_lines)
        if len(raw_lines) >= 2:
            raw_rows, cand_id = _generate_pair_candidates_from_line_pairs(
                event_id,
                _raw_pair_iter_from_lines(raw_lines),
                args,
                truth_xyz,
                cand_id,
                'original_hough_pair',
                '',
                'raw',
            )
            rows.extend(raw_rows)
        elif source == 'raw':
            return [], dict(event=int(event_id), status='too_few_lines', message='', n_lines=len(raw_lines), n_candidates=0)

    if source in ('bmeleton_merged', 'both'):
        try:
            data = _build_bmeleton_merged_data(line_file, args)
            merged_lines = data.get('lines', [])
            n_lines_total += len(merged_lines)
            if len(merged_lines) < 2:
                status_messages.append('bmeleton_too_few_merged_lines')
            merged_rows, cand_id = _generate_pair_candidates_from_line_pairs(
                event_id,
                _merged_pair_iter_from_data(data),
                args,
                truth_xyz,
                cand_id,
                'bmeleton_merged_intersection',
                'bmeleton_merged_',
                'bmeleton_merged',
            )
            rows.extend(merged_rows)
            if not merged_rows:
                status_messages.append('bmeleton_no_valid_intersections')

            if bool(getattr(args, 'include_bmeleton_final_vertex', True)):
                try:
                    fdata = _build_bmeleton_final_refit_data(line_file, args)
                    final_rows, cand_id = _generate_final_vertex_candidates_from_data(
                        event_id, fdata, args, truth_xyz, cand_id)
                    rows.extend(final_rows)
                    if not final_rows:
                        status_messages.append('bmeleton_no_final_refit_vertex')
                except Exception as fexc:
                    status_messages.append('bmeleton_final_failed:%s' % fexc)
        except Exception as exc:
            status_messages.append('bmeleton_failed:%s' % exc)
            if source == 'bmeleton_merged':
                return [], dict(event=int(event_id), status='bmeleton_failed', message=str(exc), n_lines=0, n_candidates=0)

    if rows:
        status = 'ok'
    else:
        status = 'no_candidates'
    return rows, dict(event=int(event_id), status=status, message=';'.join(status_messages), n_lines=int(n_lines_total), n_candidates=len(rows))


# ----------------------------- event discovery/output -----------------------------


def parse_event_id_from_name(name):
    m = re.search(r"event_(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def discover_event_line_files(linesdir):
    out = []
    for entry in sorted(os.listdir(linesdir)):
        ev = parse_event_id_from_name(entry)
        if ev is None:
            continue
        d = os.path.join(linesdir, entry)
        if os.path.isdir(d):
            f = os.path.join(d, "lines.txt")
        else:
            continue
        if os.path.isfile(f):
            out.append((ev, f))
    out.sort(key=lambda x: x[0])
    return out


def write_candidate_text_files(df, outdir, only_best_truth=False):
    if df is None or len(df) == 0:
        return
    root = os.path.join(outdir, "candidate_vertices")
    if only_best_truth:
        root = os.path.join(outdir, "best_truth_seed_vertices")
    if not os.path.isdir(root):
        os.makedirs(root)
    for _, row in df.iterrows():
        ev = int(row["event"])
        if only_best_truth:
            path = os.path.join(root, "event_%d_vertex.txt" % ev)
        else:
            cid = int(row["candidate_id"])
            src = str(row.get("candidate_source", "candidate"))
            path = os.path.join(root, "event_%d_candidate_%04d_%s.txt" % (ev, cid, src))
        with open(path, "w") as f:
            f.write("%.9g %.9g %.9g\n" % (float(row["cand_x"]), float(row["cand_y"]), float(row["cand_z"])))


def make_event_summary(df_all, status_rows):
    status_df = pd.DataFrame(status_rows)
    if df_all is None or len(df_all) == 0:
        return status_df
    summaries = []
    has_truth = "dist_candidate_minus_truth" in df_all.columns and np.isfinite(df_all["dist_candidate_minus_truth"]).any()
    for ev, g in df_all.groupby("event"):
        row = dict(event=int(ev), n_candidates=int(len(g)))
        row["n_original_hough_pair"] = int((g["candidate_source"] == "original_hough_pair").sum())
        row["n_pivot_refit_both"] = int((g["candidate_source"] == "pivot_refit_both").sum())
        row["n_pivot_refit_line1_only"] = int((g["candidate_source"] == "pivot_refit_line1_only").sum())
        row["n_pivot_refit_line2_only"] = int((g["candidate_source"] == "pivot_refit_line2_only").sum())
        row["n_bmeleton_merged_intersection"] = int((g["candidate_source"] == "bmeleton_merged_intersection").sum())
        row["n_bmeleton_merged_pivot_refit_both"] = int((g["candidate_source"] == "bmeleton_merged_pivot_refit_both").sum())
        row["n_bmeleton_merged_pivot_refit_line1_only"] = int((g["candidate_source"] == "bmeleton_merged_pivot_refit_line1_only").sum())
        row["n_bmeleton_merged_pivot_refit_line2_only"] = int((g["candidate_source"] == "bmeleton_merged_pivot_refit_line2_only").sum())
        row["n_bmeleton_final_refit_vertex"] = int((g["candidate_source"] == "bmeleton_final_refit_vertex").sum())
        row["best_score"] = float(g["candidate_score"].max()) if "candidate_score" in g.columns else np.nan
        if has_truth:
            gg = g[np.isfinite(g["dist_candidate_minus_truth"])]
            if len(gg):
                best = gg.sort_values("dist_candidate_minus_truth").iloc[0]
                row["best_truth_candidate_id"] = int(best["candidate_id"])
                row["best_truth_source"] = str(best["candidate_source"])
                row["best_truth_dist_mm"] = float(best["dist_candidate_minus_truth"])
                orig = gg[gg["candidate_source"] == "original_hough_pair"]
                pivot = gg[gg["candidate_source"] == "pivot_refit_both"]
                row["best_original_truth_dist_mm"] = float(orig["dist_candidate_minus_truth"].min()) if len(orig) else np.nan
                row["best_pivot_both_truth_dist_mm"] = float(pivot["dist_candidate_minus_truth"].min()) if len(pivot) else np.nan
        summaries.append(row)
    s = pd.DataFrame(summaries)
    if len(status_df):
        s = pd.merge(status_df, s, on="event", how="outer", suffixes=("", "_from_candidates"))
        if "n_candidates_from_candidates" in s.columns:
            s["n_candidates"] = s["n_candidates_from_candidates"].fillna(s.get("n_candidates", 0))
            s.drop(columns=["n_candidates_from_candidates"], inplace=True)
    return s.sort_values("event")


def add_truth_rank_columns(df):
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if "dist_candidate_minus_truth" not in df.columns:
        df["truth_rank"] = np.nan
        df["is_best_truth"] = False
        return df
    df["truth_rank"] = df.groupby("event")["dist_candidate_minus_truth"].rank(method="first", ascending=True)
    df["is_best_truth"] = df["truth_rank"] == 1.0
    return df


# ----------------------------- CLI -----------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="Generate multi-candidate vertex seeds from Hough lines with conservative pivot-refit options.")
    ap.add_argument("--linesdir", required=True, help="Directory containing event_*/lines.txt")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--rootfile", default=None, help="Optional truth ROOT file with simData.vertexX/Y/Z")
    ap.add_argument("--only-siHitE", action="store_true", help="When reading truth ROOT, skip events without siHitE")
    ap.add_argument("--events", nargs="+", default=[], help="Optional event numbers to process")
    ap.add_argument("--skip-events", nargs="+", default=[], help="Optional event numbers to skip")
    ap.add_argument("--max-events", type=int, default=0, help="Maximum number of selected events to process; 0 means all")

    ap.add_argument("--candidate-line-source", choices=["raw", "bmeleton_merged", "both"], default="bmeleton_merged",
                    help="Candidate source lines. bmeleton_merged mirrors Cloud_classes merge_all_lines()+find_all_intersections(); raw reproduces the earlier raw-Hough behavior; both includes both. Default: bmeleton_merged")
    ap.add_argument("--verbose-bmeleton-merge", action="store_true",
                    help="Do not suppress verbose Cloud_classes merge/intersection/refit printout")
    ap.add_argument("--include-bmeleton-final-vertex", action="store_true", default=True,
                    help="Include the full Cloud_classes Event.identify_vertex() final refitted vertex as a candidate. Default: enabled")
    ap.add_argument("--no-bmeleton-final-vertex", action="store_false", dest="include_bmeleton_final_vertex",
                    help="Do not include the full BMeleton final refitted vertex candidate")

    ap.add_argument("--max-initial-pair-dist-mm", type=float, default=25.0,
                    help="Only original Hough line pairs with closest approach <= this are used")
    ap.add_argument("--max-refit-pair-dist-mm", type=float, default=25.0,
                    help="Only pivot-refit candidates with pair closest approach <= this are kept")
    ap.add_argument("--max-pairs-for-refit", type=int, default=0,
                    help="Refit only top N original pair candidates by geometry score; 0 means all")

    ap.add_argument("--enable-pivot-refit", action="store_true", default=True,
                    help="Enable pivoted line-direction refit candidates. Default: enabled")
    ap.add_argument("--no-pivot-refit", action="store_false", dest="enable_pivot_refit",
                    help="Disable pivoted refit candidates")
    ap.add_argument("--include-mixed-refits", action="store_true",
                    help="Also write candidates where only one of the two lines is pivot-refit")

    ap.add_argument("--exclude-radius-mm", type=float, default=5.0,
                    help="Ignore assigned line points within this radius of candidate vertex")
    ap.add_argument("--tube-radius-mm", type=float, default=5.0,
                    help="Require support points to lie within this perpendicular tube around the branch")
    ap.add_argument("--angle-gate-deg", type=float, default=25.0,
                    help="Require support points to be within this angle of the branch direction")
    ap.add_argument("--branch-margin-mm", type=float, default=5.0,
                    help="Allow support points this far beyond the far endpoint branch length")
    ap.add_argument("--min-support-points", type=int, default=8,
                    help="Minimum assigned support points after cuts for a pivot refit")
    ap.add_argument("--max-angle-change-deg", type=float, default=25.0,
                    help="Reject refit directions rotated more than this from original endpoint-to-vertex direction")
    ap.add_argument("--max-median-residual-mm", type=float, default=5.0,
                    help="Reject refit line if median support-point residual exceeds this")

    ap.add_argument("--competitor-veto", action="store_true", default=True,
                    help="Drop support points clearly better aligned with the competing branch. Default: enabled")
    ap.add_argument("--no-competitor-veto", action="store_false", dest="competitor_veto",
                    help="Disable competing-branch support-point veto")
    ap.add_argument("--competitor-angle-margin-deg", type=float, default=3.0,
                    help="Point is vetoed only when other branch angle is this much smaller")

    ap.add_argument("--write-candidate-files", action="store_true",
                    help="Also write one text file per candidate vertex")
    ap.add_argument("--write-best-truth-seeds", action="store_true", default=True,
                    help="If truth exists, write event_N_vertex.txt files for truth-best candidate. Default: enabled")
    ap.add_argument("--no-best-truth-seeds", action="store_false", dest="write_best_truth_seeds",
                    help="Do not write truth-best seed text files")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.linesdir):
        sys.exit("[ERROR] Missing linesdir: %s" % args.linesdir)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    truth = {}
    if args.rootfile:
        print("[INFO] Reading truth from %s" % args.rootfile)
        truth = read_truth_reaction_vertices(args.rootfile, only_siHitE=args.only_siHitE)
        print("[INFO] Loaded truth vertices for %d events" % len(truth))

    requested = set([int(x) for x in args.events]) if args.events else None
    skipped = set([int(x) for x in args.skip_events]) if args.skip_events else set()

    event_files = discover_event_line_files(args.linesdir)
    selected = []
    for ev, lf in event_files:
        if requested is not None and ev not in requested:
            continue
        if ev in skipped:
            continue
        selected.append((ev, lf))
    if int(args.max_events) > 0:
        selected = selected[:int(args.max_events)]

    print("[INFO] linesdir=%s selected_events=%d" % (args.linesdir, len(selected)))
    print("[INFO] output=%s" % args.outdir)

    all_rows = []
    status_rows = []
    for idx, (ev, lf) in enumerate(selected):
        txyz = truth.get(int(ev), None)
        rows, status = generate_candidates_for_event(ev, lf, args, truth_xyz=txyz)
        all_rows.extend(rows)
        status_rows.append(status)
        if args.verbose or (idx + 1) % 25 == 0:
            msg = "[EVENT] %d/%d event=%d lines_status=%s candidates=%d" % (
                idx + 1, len(selected), ev, status.get("status", ""), len(rows))
            if rows and txyz is not None:
                vals = [r.get("dist_candidate_minus_truth", np.nan) for r in rows]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    msg += " best_truth=%.4g mm" % min(vals)
            print(msg)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = add_truth_rank_columns(df)
        df.sort_values(["event", "candidate_id"], inplace=True)
    else:
        df = pd.DataFrame()

    all_csv = os.path.join(args.outdir, "vertex_candidates_all.csv")
    df.to_csv(all_csv, index=False)
    print("[INFO] wrote %s rows=%d" % (all_csv, len(df)))

    # Best by truth and best by score.
    if len(df):
        if "dist_candidate_minus_truth" in df.columns and np.isfinite(df["dist_candidate_minus_truth"]).any():
            best_truth = df[np.isfinite(df["dist_candidate_minus_truth"])].sort_values(
                ["event", "dist_candidate_minus_truth", "candidate_score"], ascending=[True, True, False]
            ).groupby("event", as_index=False).first()
            best_truth_csv = os.path.join(args.outdir, "vertex_candidates_best_by_truth.csv")
            best_truth.to_csv(best_truth_csv, index=False)
            print("[INFO] wrote %s rows=%d" % (best_truth_csv, len(best_truth)))
            if bool(args.write_best_truth_seeds):
                write_candidate_text_files(best_truth, args.outdir, only_best_truth=True)
                print("[INFO] wrote truth-best seed files under %s" % os.path.join(args.outdir, "best_truth_seed_vertices"))
        else:
            best_truth = pd.DataFrame()

        best_score = df.sort_values(["event", "candidate_score"], ascending=[True, False]).groupby("event", as_index=False).first()
        best_score_csv = os.path.join(args.outdir, "vertex_candidates_best_by_score.csv")
        best_score.to_csv(best_score_csv, index=False)
        print("[INFO] wrote %s rows=%d" % (best_score_csv, len(best_score)))

        if bool(args.write_candidate_files):
            write_candidate_text_files(df, args.outdir, only_best_truth=False)
            print("[INFO] wrote candidate vertex files under %s" % os.path.join(args.outdir, "candidate_vertices"))

    summary = make_event_summary(df, status_rows)
    summary_csv = os.path.join(args.outdir, "vertex_candidate_summary_by_event.csv")
    summary.to_csv(summary_csv, index=False)
    print("[INFO] wrote %s rows=%d" % (summary_csv, len(summary)))

    status_csv = os.path.join(args.outdir, "vertex_candidate_status.csv")
    pd.DataFrame(status_rows).to_csv(status_csv, index=False)
    print("[INFO] wrote %s rows=%d" % (status_csv, len(status_rows)))

    if len(df) and "dist_candidate_minus_truth" in df.columns and np.isfinite(df["dist_candidate_minus_truth"]).any():
        bt = pd.read_csv(os.path.join(args.outdir, "vertex_candidates_best_by_truth.csv"))
        print("\n[TRUTH-BEST CANDIDATE SUMMARY]")
        for col in ["dist_candidate_minus_truth", "dx_candidate_minus_truth", "dy_candidate_minus_truth", "dz_candidate_minus_truth"]:
            vals = pd.to_numeric(bt[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                print("  %s: median=%.5g mean=%.5g max_abs=%.5g" % (
                    col, float(np.median(vals)), float(np.mean(vals)), float(np.max(np.abs(vals)))))
        if "candidate_source" in bt.columns:
            print("\n[TRUTH-BEST SOURCE COUNTS]")
            print(bt["candidate_source"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
