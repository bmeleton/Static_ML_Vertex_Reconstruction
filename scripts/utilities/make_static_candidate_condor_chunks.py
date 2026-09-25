from __future__ import print_function

import argparse
import os
import re
import sys


def parse_event_id(name):
    m = re.search(r"event_(\d+)", name)
    return int(m.group(1)) if m else None


def discover(linesdir):
    out = []
    for e in sorted(os.listdir(linesdir)):
        ev = parse_event_id(e)
        if ev is None:
            continue
        lf = os.path.join(linesdir, e, "lines.txt")
        if os.path.isfile(lf):
            out.append(ev)
    out.sort()
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Make Condor chunk file for static candidate feature collection.")
    ap.add_argument("--linesdir", required=True)
    ap.add_argument("--events-per-job", type=int, default=50)
    ap.add_argument("--out", required=True, help="Output chunks txt file")
    ap.add_argument("--max-events", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.linesdir):
        sys.exit("[ERROR] missing linesdir: %s" % args.linesdir)
    events = discover(args.linesdir)
    if args.max_events > 0:
        events = events[:args.max_events]
    n = len(events)
    if n == 0:
        sys.exit("[ERROR] no event_*/lines.txt files found")
    epj = max(1, int(args.events_per_job))
    odir = os.path.dirname(args.out)
    if odir and not os.path.isdir(odir):
        os.makedirs(odir)
    part = 0
    with open(args.out, "w") as f:
        start = 0
        while start < n:
            end = min(n, start + epj)
            f.write("%03d %d %d\n" % (part, start, end))
            part += 1
            start = end
    print("[INFO] events=%d events_per_job=%d chunks=%d" % (n, epj, part))
    print("[INFO] wrote %s" % args.out)


if __name__ == "__main__":
    main()
