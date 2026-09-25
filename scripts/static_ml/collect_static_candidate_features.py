from __future__ import print_function

"""
collect_static_candidate_features.py

Static candidate-quality feature collection for TeBAT vertex candidates.

This script is intentionally different from the rollout/action-policy code:
  - it generates/reads vertex candidates,
  - computes one static topology/geometry feature vector per candidate,
  - uses truth only to label/benchmark candidates when a ROOT file is supplied,
  - does NOT scan actions, compute rollout gradients, or move vertices.

It can either:
  A) read an existing vertex_candidates_all.csv with --candidates-csv, or
  B) generate candidates directly from Hough event_*/lines.txt using the existing
     generate_vertex_candidates_from_hough.py module.

Python 3.6+ compatible.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

STATIC_BMELETON_MERGE_VERSION = "2026-08-07-v4-bmeleton-final-refit-candidate"

EPS = 1.0e-12


def _import_candidate_generator():
    try:
        import generate_vertex_candidates_from_hough as gen
        return gen
    except Exception as exc:
        sys.exit("[ERROR] Could not import generate_vertex_candidates_from_hough.py. Put it in this directory first. Details: %s" % exc)


def safe_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def safe_int(x, default=-1):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= EPS:
        return np.zeros_like(v, dtype=float)
    return v / n


def clamp(x, lo=-1.0, hi=1.0):
    try:
        x = float(x)
    except Exception:
        return lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def angle_deg(a, b):
    ua = unit(a)
    ub = unit(b)
    if float(np.linalg.norm(ua)) <= EPS or float(np.linalg.norm(ub)) <= EPS:
        return np.nan
    return float(np.degrees(np.arccos(clamp(float(np.dot(ua, ub))))))


def wrap180(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def circular_mean_deg(alpha_deg):
    a = np.asarray(alpha_deg, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    rad = np.radians(a)
    s = float(np.mean(np.sin(rad)))
    c = float(np.mean(np.cos(rad)))
    if abs(s) <= EPS and abs(c) <= EPS:
        return np.nan
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def centered_alpha(alpha_deg, center_deg):
    return wrap180(np.asarray(alpha_deg, dtype=float) - float(center_deg))


def robust_stats_1d(x, prefix):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {}
    n = int(x.size)
    out[prefix + "_n"] = n
    if n == 0:
        for key in ["mean", "median", "std", "rms", "mad", "q05", "q16", "q84", "q95", "width68", "width90", "skew", "asym68", "asym90"]:
            out[prefix + "_" + key] = np.nan
        return out
    mean = float(np.mean(x))
    med = float(np.median(x))
    std = float(np.std(x)) if n > 1 else 0.0
    rms = float(np.sqrt(np.mean(x * x)))
    mad = float(np.median(np.abs(x - med)))
    qs = np.percentile(x, [5, 16, 84, 95]) if n > 1 else np.array([x[0], x[0], x[0], x[0]], dtype=float)
    q05, q16, q84, q95 = [float(v) for v in qs]
    w68 = q84 - q16
    w90 = q95 - q05
    if std > EPS and n > 2:
        skew = float(np.mean(((x - mean) / std) ** 3))
    else:
        skew = 0.0
    asym68 = abs((q84 - med) - (med - q16)) / (abs(w68) + EPS)
    asym90 = abs((q95 - med) - (med - q05)) / (abs(w90) + EPS)
    out.update({
        prefix + "_mean": mean,
        prefix + "_median": med,
        prefix + "_std": std,
        prefix + "_rms": rms,
        prefix + "_mad": mad,
        prefix + "_q05": q05,
        prefix + "_q16": q16,
        prefix + "_q84": q84,
        prefix + "_q95": q95,
        prefix + "_width68": float(w68),
        prefix + "_width90": float(w90),
        prefix + "_skew": skew,
        prefix + "_asym68": float(asym68),
        prefix + "_asym90": float(asym90),
    })
    return out


def xyz_to_alpha_beta(points, vertex, convention="refiner"):
    pts = np.asarray(points, dtype=float)
    v = np.asarray(vertex, dtype=float)
    rel = pts - v[None, :]
    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]
    rho_xz = np.sqrt(x * x + z * z)
    if convention == "gui":
        # Original GUI convention used alpha=atan2(z,x).
        alpha = np.degrees(np.arctan2(z, x)) % 360.0
    else:
        # Refiner/policy convention used alpha=atan2(x,z), alpha=0 along +z.
        alpha = np.degrees(np.arctan2(x, z)) % 360.0
    beta = np.degrees(np.arctan2(y, rho_xz))
    r = np.sqrt(x * x + y * y + z * z)
    return alpha, beta, r, rel


def line_point_distances(points, anchor, direction):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    direction = unit(direction)
    q = pts - anchor[None, :]
    return np.linalg.norm(np.cross(q, direction[None, :]), axis=1)


def choose_far_endpoint(line, cand):
    cand = np.asarray(cand, dtype=float)
    start = np.asarray(line.get("start", [np.nan, np.nan, np.nan]), dtype=float)
    end = np.asarray(line.get("end", [np.nan, np.nan, np.nan]), dtype=float)
    ds = float(np.linalg.norm(start - cand)) if np.all(np.isfinite(start)) else -1.0
    de = float(np.linalg.norm(end - cand)) if np.all(np.isfinite(end)) else -1.0
    if de >= ds:
        return end, "end", de
    return start, "start", ds


def select_branch_points(line, cand, args):
    pts = np.asarray(line.get("xyz", np.zeros((0, 3))), dtype=float)
    cand = np.asarray(cand, dtype=float)
    if pts.size == 0:
        return pts.reshape((0, 3)), dict(branch_len=np.nan, far_endpoint="none", side_count=0, tube_count=0, total_count=0)
    pfar, far_name, branch_len = choose_far_endpoint(line, cand)
    d = unit(pfar - cand)
    if float(np.linalg.norm(d)) <= EPS:
        return pts[:0], dict(branch_len=np.nan, far_endpoint=far_name, side_count=0, tube_count=0, total_count=int(len(pts)))
    q = pts - cand[None, :]
    r = np.linalg.norm(q, axis=1)
    s = np.dot(q, d)
    perp = q - s[:, None] * d[None, :]
    rho = np.linalg.norm(perp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = s / np.maximum(r, EPS)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    keep_side = (r >= float(args.exclude_radius_mm)) & (s >= float(args.exclude_radius_mm)) & (s <= float(branch_len) + float(args.branch_margin_mm))
    keep = keep_side.copy()
    if float(args.static_tube_radius_mm) > 0:
        keep = keep & (rho <= float(args.static_tube_radius_mm))
    if float(args.static_angle_gate_deg) > 0:
        keep = keep & (ang <= float(args.static_angle_gate_deg))
    return pts[keep], dict(
        branch_len=float(branch_len),
        far_endpoint=far_name,
        total_count=int(len(pts)),
        side_count=int(np.sum(keep_side)),
        tube_count=int(np.sum(keep)),
        side_fraction=float(np.sum(keep_side)) / float(max(len(pts), 1)),
        kept_fraction=float(np.sum(keep)) / float(max(len(pts), 1)),
    )


def cluster_features_for_line(prefix, line, cand, args):
    out = {}
    cand = np.asarray(cand, dtype=float)
    line_b = unit(line.get("b", np.zeros(3)))
    pfar, far_name, branch_len = choose_far_endpoint(line, cand)
    branch_dir = unit(pfar - cand)
    all_pts = np.asarray(line.get("xyz", np.zeros((0, 3))), dtype=float)
    pts, sel = select_branch_points(line, cand, args)
    out[prefix + "_line_num"] = safe_int(line.get("line_num", -1))
    out[prefix + "_n_points_total"] = int(sel.get("total_count", len(all_pts)))
    out[prefix + "_n_points_side"] = int(sel.get("side_count", 0))
    out[prefix + "_n_points_kept"] = int(sel.get("tube_count", len(pts)))
    out[prefix + "_side_fraction"] = float(sel.get("side_fraction", np.nan))
    out[prefix + "_kept_fraction"] = float(sel.get("kept_fraction", np.nan))
    out[prefix + "_branch_len_mm"] = float(sel.get("branch_len", branch_len))
    out[prefix + "_far_endpoint_is_end"] = 1 if sel.get("far_endpoint", far_name) == "end" else 0
    out[prefix + "_hough_vs_branch_angle_deg"] = angle_deg(line_b, branch_dir)

    if pts.size == 0 or pts.shape[0] == 0:
        # Fill a broad list of expected numerical features with NaN.
        for name in [
            "alpha_mean_deg", "alpha_std_deg", "alpha_rms_deg", "alpha_width68_deg", "alpha_width90_deg",
            "alpha_asym68", "alpha_asym90", "alpha_skew", "beta_mean_deg", "beta_std_deg", "beta_rms_deg",
            "beta_width68_deg", "beta_width90_deg", "pca_linearity", "pca_minor_width_deg", "pca_major_width_deg",
            "pca_axis_beta_abs", "branch_resid_median_mm", "branch_resid_rms_mm", "branch_resid_max_mm",
            "branch_angle_median_deg", "branch_angle_rms_deg", "radial_median_mm", "radial_width68_mm"
        ]:
            out[prefix + "_" + name] = np.nan
        return out

    alpha, beta, r, rel = xyz_to_alpha_beta(pts, cand, convention=args.alpha_convention)
    amean = circular_mean_deg(alpha)
    da = centered_alpha(alpha, amean if np.isfinite(amean) else 0.0)
    out[prefix + "_alpha_mean_deg"] = amean
    s = robust_stats_1d(da, prefix + "_alpha_delta")
    out[prefix + "_alpha_std_deg"] = s[prefix + "_alpha_delta_std"]
    out[prefix + "_alpha_rms_deg"] = s[prefix + "_alpha_delta_rms"]
    out[prefix + "_alpha_width68_deg"] = s[prefix + "_alpha_delta_width68"]
    out[prefix + "_alpha_width90_deg"] = s[prefix + "_alpha_delta_width90"]
    out[prefix + "_alpha_asym68"] = s[prefix + "_alpha_delta_asym68"]
    out[prefix + "_alpha_asym90"] = s[prefix + "_alpha_delta_asym90"]
    out[prefix + "_alpha_skew"] = s[prefix + "_alpha_delta_skew"]

    bs = robust_stats_1d(beta, prefix + "_beta")
    out[prefix + "_beta_mean_deg"] = bs[prefix + "_beta_mean"]
    out[prefix + "_beta_std_deg"] = bs[prefix + "_beta_std"]
    out[prefix + "_beta_rms_deg"] = bs[prefix + "_beta_rms"]
    out[prefix + "_beta_width68_deg"] = bs[prefix + "_beta_width68"]
    out[prefix + "_beta_width90_deg"] = bs[prefix + "_beta_width90"]

    rs = robust_stats_1d(r, prefix + "_radial")
    out[prefix + "_radial_median_mm"] = rs[prefix + "_radial_median"]
    out[prefix + "_radial_width68_mm"] = rs[prefix + "_radial_width68"]

    if pts.shape[0] >= 3:
        x2 = np.vstack([da, beta - np.nanmean(beta)]).T
        ok = np.isfinite(x2).all(axis=1)
        x2 = x2[ok]
        if x2.shape[0] >= 3:
            cov = np.cov(x2.T)
            try:
                vals, vecs = np.linalg.eigh(cov)
                vals = np.sort(np.maximum(vals, 0.0))
                minor = float(np.sqrt(vals[0]))
                major = float(np.sqrt(vals[1]))
                lin = float((vals[1] - vals[0]) / (vals[1] + vals[0] + EPS))
                # principal eigenvector associated with major axis
                vals2, vecs2 = np.linalg.eigh(cov)
                imax = int(np.argmax(vals2))
                axis = vecs2[:, imax]
                out[prefix + "_pca_linearity"] = lin
                out[prefix + "_pca_minor_width_deg"] = minor
                out[prefix + "_pca_major_width_deg"] = major
                out[prefix + "_pca_axis_beta_abs"] = abs(float(axis[1]))
            except Exception:
                out[prefix + "_pca_linearity"] = np.nan
                out[prefix + "_pca_minor_width_deg"] = np.nan
                out[prefix + "_pca_major_width_deg"] = np.nan
                out[prefix + "_pca_axis_beta_abs"] = np.nan
        else:
            out[prefix + "_pca_linearity"] = np.nan
            out[prefix + "_pca_minor_width_deg"] = np.nan
            out[prefix + "_pca_major_width_deg"] = np.nan
            out[prefix + "_pca_axis_beta_abs"] = np.nan
    else:
        out[prefix + "_pca_linearity"] = np.nan
        out[prefix + "_pca_minor_width_deg"] = np.nan
        out[prefix + "_pca_major_width_deg"] = np.nan
        out[prefix + "_pca_axis_beta_abs"] = np.nan

    # Branch residuals relative to ray from candidate to far endpoint.
    q = pts - cand[None, :]
    proj = np.dot(q, branch_dir)
    perp = q - proj[:, None] * branch_dir[None, :]
    resid = np.linalg.norm(perp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = proj / np.maximum(np.linalg.norm(q, axis=1), EPS)
    ad = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    rs2 = robust_stats_1d(resid, prefix + "_branch_resid")
    as2 = robust_stats_1d(ad, prefix + "_branch_angle")
    out[prefix + "_branch_resid_median_mm"] = rs2[prefix + "_branch_resid_median"]
    out[prefix + "_branch_resid_rms_mm"] = rs2[prefix + "_branch_resid_rms"]
    out[prefix + "_branch_resid_max_mm"] = rs2[prefix + "_branch_resid_q95"]
    out[prefix + "_branch_angle_median_deg"] = as2[prefix + "_branch_angle_median"]
    out[prefix + "_branch_angle_rms_deg"] = as2[prefix + "_branch_angle_rms"]
    return out


def combine_two_line_features(out):
    # Aggregate line1/line2 shape metrics. Missing values are okay.
    metrics = [
        "n_points_kept", "kept_fraction", "alpha_std_deg", "alpha_rms_deg", "alpha_width68_deg",
        "alpha_width90_deg", "alpha_asym68", "alpha_asym90", "beta_std_deg", "beta_rms_deg",
        "pca_linearity", "pca_minor_width_deg", "pca_major_width_deg", "pca_axis_beta_abs",
        "branch_resid_median_mm", "branch_resid_rms_mm", "branch_angle_median_deg", "branch_angle_rms_deg",
        "radial_median_mm", "radial_width68_mm"
    ]
    for m in metrics:
        vals = []
        for p in ["line1_static", "line2_static"]:
            v = out.get(p + "_" + m, np.nan)
            if np.isfinite(safe_float(v)):
                vals.append(float(v))
        if vals:
            out["static_mean_" + m] = float(np.mean(vals))
            out["static_max_" + m] = float(np.max(vals))
            out["static_min_" + m] = float(np.min(vals))
        else:
            out["static_mean_" + m] = np.nan
            out["static_max_" + m] = np.nan
            out["static_min_" + m] = np.nan

    # Inter-cluster angular separation and balance.
    a1 = safe_float(out.get("line1_static_alpha_mean_deg"))
    a2 = safe_float(out.get("line2_static_alpha_mean_deg"))
    b1 = safe_float(out.get("line1_static_beta_mean_deg"))
    b2 = safe_float(out.get("line2_static_beta_mean_deg"))
    if np.isfinite(a1) and np.isfinite(a2):
        out["static_alpha_mean_sep_deg"] = abs(float(wrap180(a1 - a2)))
    else:
        out["static_alpha_mean_sep_deg"] = np.nan
    if np.isfinite(b1) and np.isfinite(b2):
        out["static_beta_mean_sep_deg"] = abs(b1 - b2)
    else:
        out["static_beta_mean_sep_deg"] = np.nan
    n1 = safe_float(out.get("line1_static_n_points_kept"), 0.0)
    n2 = safe_float(out.get("line2_static_n_points_kept"), 0.0)
    out["static_support_total"] = n1 + n2
    out["static_support_balance"] = min(n1, n2) / (max(n1, n2) + EPS) if max(n1, n2) > 0 else np.nan
    return out


def compute_features_for_candidate(row, lines_by_num, args):
    cand = np.array([safe_float(row.get("cand_x")), safe_float(row.get("cand_y")), safe_float(row.get("cand_z"))], dtype=float)
    out = {}

    # Preserve all input candidate columns; the trainer can decide which are usable.
    for col in row.index:
        val = row[col]
        if isinstance(val, (np.generic,)):
            val = val.item()
        out[col] = val

    if not np.all(np.isfinite(cand)):
        out["static_feature_status"] = "bad_candidate_xyz"
        return out

    l1n = safe_int(row.get("line1", -1))
    l2n = safe_int(row.get("line2", -1))
    l1 = lines_by_num.get(l1n)
    l2 = lines_by_num.get(l2n)
    out["static_candidate_r_xy_mm"] = float(np.sqrt(cand[0] * cand[0] + cand[1] * cand[1]))
    out["static_candidate_abs_x_mm"] = abs(float(cand[0]))
    out["static_candidate_abs_y_mm"] = abs(float(cand[1]))
    out["static_candidate_z_mm"] = float(cand[2])

    if l1 is None or l2 is None:
        out["static_feature_status"] = "missing_line"
        return out

    out.update(cluster_features_for_line("line1_static", l1, cand, args))
    out.update(cluster_features_for_line("line2_static", l2, cand, args))
    combine_two_line_features(out)
    out["static_feature_status"] = "ok"
    return out


def add_static_labels(df):
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if "dist_candidate_minus_truth" in df.columns:
        vals = pd.to_numeric(df["dist_candidate_minus_truth"], errors="coerce")
        df["target_dist_mm"] = vals
        df["target_log1p_dist"] = np.log1p(vals)
        for thr in [0.25, 0.5, 1.0, 2.0, 5.0]:
            name = "target_within_%smm" % str(thr).replace(".", "p")
            df[name] = (vals <= thr).astype(float)
        ranks = vals.groupby(df["event"]).rank(method="first", ascending=True)
        df["target_truth_rank"] = ranks
        df["target_is_best_truth"] = (ranks == 1.0).astype(float)
    return df


def event_subset_from_args(event_files, args):
    event_files = sorted(event_files, key=lambda x: x[0])
    requested = set([int(x) for x in args.events]) if args.events else None
    skipped = set([int(x) for x in args.skip_events]) if args.skip_events else set()
    selected = []
    for ev, lf in event_files:
        if requested is not None and ev not in requested:
            continue
        if ev in skipped:
            continue
        selected.append((ev, lf))
    if int(args.event_index_start) >= 0 or int(args.event_index_end) >= 0:
        start = max(0, int(args.event_index_start)) if int(args.event_index_start) >= 0 else 0
        end = int(args.event_index_end) if int(args.event_index_end) >= 0 else len(selected)
        selected = selected[start:end]
    if int(args.max_events) > 0:
        selected = selected[:int(args.max_events)]
    return selected


def collect_from_generated_candidates(args):
    gen = _import_candidate_generator()
    if not os.path.isdir(args.linesdir):
        sys.exit("[ERROR] Missing linesdir: %s" % args.linesdir)

    truth = {}
    if args.rootfile:
        print("[INFO] Reading truth from %s" % args.rootfile)
        truth = gen.read_truth_reaction_vertices(args.rootfile, only_siHitE=args.only_siHitE)
        print("[INFO] Loaded truth vertices for %d events" % len(truth))

    event_files = gen.discover_event_line_files(args.linesdir)
    selected = event_subset_from_args(event_files, args)
    print("[INFO] selected events=%d from linesdir=%s" % (len(selected), args.linesdir))

    all_candidate_rows = []
    all_feature_rows = []
    status_rows = []

    # Re-use the candidate generator argparse shape by passing args itself. It only needs the matching attributes.
    for idx, (ev, lf) in enumerate(selected):
        txyz = truth.get(int(ev), None)
        cand_rows, status = gen.generate_candidates_for_event(ev, lf, args, truth_xyz=txyz)
        status_rows.append(status)
        all_candidate_rows.extend(cand_rows)
        try:
            line_maps = gen.get_feature_line_maps_for_event(lf, args)
        except Exception as exc:
            print("[WARN] event %s failed static feature line-map parse: %s" % (ev, exc))
            continue
        for cr in cand_rows:
            srow = pd.Series(cr)
            stage = str(cr.get("line_stage", "raw"))
            lines_by_num = line_maps.get(stage, line_maps.get("raw", {}))
            feat = compute_features_for_candidate(srow, lines_by_num, args)
            all_feature_rows.append(feat)
        if args.verbose or (idx + 1) % 25 == 0:
            msg = "[EVENT] %d/%d event=%d candidates=%d" % (idx + 1, len(selected), ev, len(cand_rows))
            if cand_rows and txyz is not None:
                vals = [r.get("dist_candidate_minus_truth", np.nan) for r in cand_rows]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    msg += " oracle_best=%.4g mm" % min(vals)
            print(msg)

    cand_df = gen.add_truth_rank_columns(pd.DataFrame(all_candidate_rows)) if all_candidate_rows else pd.DataFrame()
    feat_df = add_static_labels(pd.DataFrame(all_feature_rows)) if all_feature_rows else pd.DataFrame()
    return cand_df, feat_df, pd.DataFrame(status_rows)


def collect_from_candidate_csv(args):
    gen = _import_candidate_generator()
    if not args.candidates_csv or not os.path.isfile(args.candidates_csv):
        sys.exit("[ERROR] Missing --candidates-csv: %s" % args.candidates_csv)
    df = pd.read_csv(args.candidates_csv)
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if args.linesdir is None or not os.path.isdir(args.linesdir):
        sys.exit("[ERROR] --linesdir is required and must exist when reading a candidate CSV")

    if args.events:
        wanted = set([int(x) for x in args.events])
        df = df[df["event"].astype(int).isin(wanted)]
    if args.skip_events:
        skipped = set([int(x) for x in args.skip_events])
        df = df[~df["event"].astype(int).isin(skipped)]
    if int(args.max_events) > 0:
        events = sorted(df["event"].astype(int).unique())[:int(args.max_events)]
        df = df[df["event"].astype(int).isin(events)]

    # Existing candidate CSVs may contain BMeleton-merged candidate rows.  If so,
    # force construction of the BMeleton merged line map for feature extraction.
    if "line_stage" in df.columns:
        try:
            stages = df["line_stage"].astype(str)
            args.force_bmeleton_feature_lines = bool(stages.str.contains("bmeleton").any())
            args.force_bmeleton_final_feature_lines = bool(stages.str.contains("bmeleton_final_refit").any())
        except Exception:
            args.force_bmeleton_feature_lines = False
            args.force_bmeleton_final_feature_lines = False
    else:
        args.force_bmeleton_feature_lines = False
        args.force_bmeleton_final_feature_lines = False

    feature_rows = []
    status_rows = []
    for idx, ev in enumerate(sorted(df["event"].astype(int).unique())):
        lf = os.path.join(args.linesdir, "event_%d" % ev, "lines.txt")
        if not os.path.isfile(lf):
            status_rows.append(dict(event=int(ev), status="missing_lines", n_candidates=int((df["event"].astype(int) == ev).sum())))
            continue
        line_maps = gen.get_feature_line_maps_for_event(lf, args)
        g = df[df["event"].astype(int) == ev]
        for _, row in g.iterrows():
            stage = str(row.get("line_stage", "raw"))
            lines_by_num = line_maps.get(stage, line_maps.get("raw", {}))
            feature_rows.append(compute_features_for_candidate(row, lines_by_num, args))
        status_rows.append(dict(event=int(ev), status="ok", n_candidates=int(len(g))))
        if args.verbose or (idx + 1) % 25 == 0:
            print("[EVENT] %d event=%d candidates=%d" % (idx + 1, ev, len(g)))
    feat_df = add_static_labels(pd.DataFrame(feature_rows)) if feature_rows else pd.DataFrame()
    return df, feat_df, pd.DataFrame(status_rows)


def parse_args():
    ap = argparse.ArgumentParser(description="Collect static candidate-quality features without action scans or rollout.")
    ap.add_argument("--linesdir", default=None, help="Directory containing event_*/lines.txt")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--candidates-csv", default=None, help="Existing vertex_candidates_all.csv; if omitted, candidates are generated from --linesdir")
    ap.add_argument("--rootfile", default=None, help="Optional truth ROOT file, used only when generating candidates")
    ap.add_argument("--only-siHitE", action="store_true")
    ap.add_argument("--events", nargs="+", default=[])
    ap.add_argument("--skip-events", nargs="+", default=[])
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--event-index-start", type=int, default=-1, help="0-based index in discovered event list; inclusive")
    ap.add_argument("--event-index-end", type=int, default=-1, help="0-based index in discovered event list; exclusive")

    ap.add_argument("--candidate-line-source", choices=["raw", "bmeleton_merged", "both"], default="bmeleton_merged",
                    help="Candidate source when generating candidates. Default bmeleton_merged mirrors BMeleton/Cloud_classes merging before intersection generation. Use raw to reproduce the previous raw-Hough behavior.")
    ap.add_argument("--verbose-bmeleton-merge", action="store_true",
                    help="Do not suppress verbose Cloud_classes merge/intersection/refit printout")
    ap.add_argument("--include-bmeleton-final-vertex", action="store_true", default=True,
                    help="Include the full Cloud_classes Event.identify_vertex() final refitted vertex as a candidate. Default: enabled")
    ap.add_argument("--no-bmeleton-final-vertex", action="store_false", dest="include_bmeleton_final_vertex",
                    help="Do not include the full BMeleton final refitted vertex candidate")

    # Candidate generation options. These mirror generate_vertex_candidates_from_hough.py.
    ap.add_argument("--max-initial-pair-dist-mm", type=float, default=25.0)
    ap.add_argument("--max-refit-pair-dist-mm", type=float, default=25.0)
    ap.add_argument("--max-pairs-for-refit", type=int, default=0, help="0 means refit all initial pairs")
    ap.add_argument("--enable-pivot-refit", action="store_true", default=True)
    ap.add_argument("--no-pivot-refit", action="store_false", dest="enable_pivot_refit")
    ap.add_argument("--include-mixed-refits", action="store_true")
    ap.add_argument("--exclude-radius-mm", type=float, default=5.0)
    ap.add_argument("--tube-radius-mm", type=float, default=5.0)
    ap.add_argument("--angle-gate-deg", type=float, default=25.0)
    ap.add_argument("--branch-margin-mm", type=float, default=5.0)
    ap.add_argument("--min-support-points", type=int, default=8)
    ap.add_argument("--max-angle-change-deg", type=float, default=25.0)
    ap.add_argument("--max-median-residual-mm", type=float, default=5.0)
    ap.add_argument("--competitor-veto", action="store_true", default=True)
    ap.add_argument("--no-competitor-veto", action="store_false", dest="competitor_veto")
    ap.add_argument("--competitor-angle-margin-deg", type=float, default=3.0)

    # Static feature options.
    ap.add_argument("--alpha-convention", choices=["refiner", "gui"], default="refiner")
    ap.add_argument("--static-tube-radius-mm", type=float, default=10.0, help="Tube cut for static branch features; 0 disables")
    ap.add_argument("--static-angle-gate-deg", type=float, default=40.0, help="Angle cut for static branch features; 0 disables")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    if args.candidates_csv:
        cand_df, feat_df, status_df = collect_from_candidate_csv(args)
    else:
        cand_df, feat_df, status_df = collect_from_generated_candidates(args)

    cand_path = os.path.join(args.outdir, "vertex_candidates_all.csv")
    feat_path = os.path.join(args.outdir, "static_candidate_features.csv")
    status_path = os.path.join(args.outdir, "static_candidate_feature_status.csv")
    cand_df.to_csv(cand_path, index=False)
    feat_df.to_csv(feat_path, index=False)
    status_df.to_csv(status_path, index=False)
    print("[INFO] wrote %s rows=%d" % (cand_path, len(cand_df)))
    print("[INFO] wrote %s rows=%d" % (feat_path, len(feat_df)))
    print("[INFO] wrote %s rows=%d" % (status_path, len(status_df)))

    if len(feat_df) and "target_dist_mm" in feat_df.columns:
        valid = feat_df[np.isfinite(pd.to_numeric(feat_df["target_dist_mm"], errors="coerce"))]
        if len(valid):
            best = valid.sort_values(["event", "target_dist_mm"]).groupby("event", as_index=False).first()
            best_path = os.path.join(args.outdir, "static_candidate_oracle_best_by_truth.csv")
            best.to_csv(best_path, index=False)
            print("[INFO] wrote %s rows=%d" % (best_path, len(best)))
            print("[INFO] oracle median best candidate dist = %.4g mm" % float(best["target_dist_mm"].median()))


if __name__ == "__main__":
    main()
