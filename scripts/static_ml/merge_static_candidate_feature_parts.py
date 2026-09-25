from __future__ import print_function

import argparse
import glob
import os
import sys

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Merge static candidate feature collection parts.")
    ap.add_argument("--parts-root", required=True, help="Directory containing part_*/static_candidate_features.csv")
    ap.add_argument("--out", required=True, help="Merged output CSV path")
    ap.add_argument("--candidates-out", default=None, help="Optional merged candidates CSV path")
    ap.add_argument("--status-out", default=None, help="Optional merged status CSV path")
    return ap.parse_args()


def merge_files(pattern, outpath):
    files = sorted(glob.glob(pattern))
    if not files:
        print("[WARN] no files matched %s" % pattern)
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            d["_part_file"] = f
            dfs.append(d)
            print("[INFO] read %s rows=%d" % (f, len(d)))
        except Exception as exc:
            print("[WARN] could not read %s: %s" % (f, exc))
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True, sort=False)
    if "event" in out.columns and "candidate_id" in out.columns:
        out.sort_values(["event", "candidate_id"], inplace=True)
    elif "event" in out.columns:
        out.sort_values(["event"], inplace=True)
    odir = os.path.dirname(outpath)
    if odir and not os.path.isdir(odir):
        os.makedirs(odir)
    out.to_csv(outpath, index=False)
    print("[INFO] wrote %s rows=%d" % (outpath, len(out)))
    return out


def main():
    args = parse_args()
    if not os.path.isdir(args.parts_root):
        sys.exit("[ERROR] missing parts root: %s" % args.parts_root)
    merge_files(os.path.join(args.parts_root, "part_*", "static_candidate_features.csv"), args.out)
    if args.candidates_out:
        merge_files(os.path.join(args.parts_root, "part_*", "vertex_candidates_all.csv"), args.candidates_out)
    if args.status_out:
        merge_files(os.path.join(args.parts_root, "part_*", "static_candidate_feature_status.csv"), args.status_out)


if __name__ == "__main__":
    main()
