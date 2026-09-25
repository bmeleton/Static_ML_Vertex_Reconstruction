import argparse, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import Cloud_classes as cloud


def parse_args(argParser=None):
    if argParser is None: 
        argParser = argparse.ArgumentParser()
        argParser.add_argument("--outdir", required=True, type=cloud.out_dir, help="Dir to save results to")
        argParser.add_argument("--linesdir", required=True, help="Dir containing hough line data formated like event_*/lines.txt")

    argParser.add_argument("--analysis", default="default", choices=["default", "assess_vert_id", "no_lines"], help="Determines which analysis function(s) is/are run on given data")
    argParser.add_argument("--pklfile", nargs='?', const="non", help="pkl file to save dataframe to. auto stored in outdir")
    argParser.add_argument("--loadpkl", nargs='?', const="non", help="pkl file to load analyzed event data from")
    argParser.add_argument("--only_siHitE", action='store_true', help="only process events with siHitE values in their root truth file")

    # Seed-vertex outputs for ML assessment.
    # These are mainly used by --analysis assess_vert_id.
    argParser.add_argument("--seedcsv", default=None,
                           help="CSV file to write detected seed vertices. Default: <outdir>/vertex_seeds_all.csv")
    argParser.add_argument("--primary-seedcsv", default=None,
                           help="CSV file to write one selected seed vertex per event. Default: <outdir>/vertex_seeds_primary.csv")
    argParser.add_argument("--seeddir", default=None,
                           help="Directory to write one ML seed file per event: event_<id>_vertex.txt. Default: <outdir>/seed_vertices")
    argParser.add_argument("--seed-select", default="first", choices=["first", "smallest_radius", "largest_nlines"],
                           help="How to choose the primary seed if an event has multiple vertices.")

    argParser.add_argument("--histname", help="what to name saved histogram if running vert idd assesment")
    argParser.add_argument("--events", nargs='+', default=[], help="event numbers to analyze")
    argParser.add_argument("--skip_events", nargs='+', default=[], help="event numbers to skip")
    argParser.add_argument("--plotlines", action='store_true', help="plots each event one at a time")
    argParser.add_argument("--plotpoints", action='store_true', help="plots each event points cloud one at a time")
    argParser.add_argument("--plothough", action='store_true', help="plot the hough lines for each event")
    argParser.add_argument("--plotall", action='store_true', help="plot all of the other options")
    argParser.add_argument("--show", action='store_true', help="saves and shows each plot one at a time")
    argParser.add_argument("--showall", action='store_true', help="shows all plots at the end")

    args = argParser.parse_args()
    #argParser = argparse.ArgumentParser()
    return args

def check_args(args, no_lines=False):
    if no_lines:    req_attributes = ["pointssdir", "outdir"]
    else:           req_attributes = ["linesdir", "outdir"]
    for attribute in req_attributes:
        attr = getattr(args, attribute, None)
        if attr is None: sys.exit(f"  args is missing {attr}, args must have: {req_attributes}")

    attributes = ["rootfile", "pointsfile", "histname"]
    for attribute in attributes:
        attr = getattr(args, attribute, None)
        if attr is None: setattr(args, attribute, None)

    if args.pklfile is not None:    args.pklfile = os.path.join(args.outdir, args.pklfile)
        
    if args.loadpkl is not None:
        if args.loadpkl == "non":   args.loadpkl = args.pklfile
        else:                       args.loadpkl = os.path.join(args.outdir, args.loadpkl)
        if not os.path.isfile(args.loadpkl):    sys.exit(f"  Failed to find pkl file to load: '{args.loadpkl}'")

    return

def analyze(argParser, config_args, process_event_files):
    args = parse_args(argParser)

    if args.analysis == "default": return default(args, config_args, process_event_files)
    if args.analysis == "no_lines": return no_lines(args, config_args, process_event_files)
    if args.analysis == "assess_vert_id": return assess_vert_id(args, config_args, process_event_files)


    return


def stream_events(args, process_event_files):

    full_truth = cloud.get_truth(args.rootfile)

    event_dir = getattr(args, 'linesdir', getattr(args, 'pointsdir', None))
    if event_dir == "No line dir" or not os.path.exists(event_dir):
        event_dir = args.pointsdir

    for entry in sorted(os.listdir(event_dir)):
        entry_info = process_event_files(args, entry)
        if entry_info is None:
            continue
        
        event_tuple = {key : entry_info[key] for key in ['event_num', 'out_dir', 'line_file', 'points_file'] if key in entry_info}
        print(event_tuple)
        yield cloud.Event(**event_tuple, full_truth=full_truth)

def default(args, config_args, process_event_files):
    """args must have at least args.linesdir and args.outdir after config"""
        # region arg setup
    config_args(args)
    check_args(args)
    figures = []
    events_df = None

    if args.rootfile is not None:
        full_truth = cloud.get_truth(args.rootfile)
    else:
        full_truth = None

    if args.loadpkl is None:
        event_dicts = []
        if args.linesdir == "No line dir": eventdir=args.pointsdir # legacy stuff I don't feel like rewriting but doesn't get used anymore
        else: eventdir = args.linesdir
        for entry in sorted(os.listdir(eventdir)): # loop through each file/(folder) in point cloud/(hough line) directory 
            # region entry proccessing
            entry_info = process_event_files(args, entry) # handels selecting a specific event_id amung other things
            if entry_info is None: continue
            event = cloud.Event(*[entry_info[key] for key in ['event_num', 'out_dir', 'line_file', 'points_file'] if key in entry_info])

            event.get_truth(full_truth=full_truth)

            if args.plothough: 
                #out_dir = os.path.join(args.outdir, "hough_plots")
                #os.makedirs(out_dir, exist_ok=True)
                event.plot(name=f'event_{event.id}_hough.png')

            event.identify_vertex()

            event_dicts.append(event.get_dict())

            if args.plotlines or args.plotpoints or args.showall or args.show:
                figs = event.plot(plot_cloud=args.plotpoints)
                if args.show:
                    plt.show()
                    plt.close()
                if args.showall:
                    figures.extend(figs)

        if len(args.events) == 0: # if analyzing whole data set, save analysis
            events_df:pd.DataFrame = pd.DataFrame(event_dicts)
            events_df.set_index("event_id", inplace=True)
            events_df.index.name = "event_id"

            pklsave = os.path.join(args.outdir,"events_dataframe.pkl")
            csvsave = os.path.join(args.outdir,"events_dataframe.csv")
            events_df.to_pickle(pklsave)
            events_df.to_csv(csvsave)
        
    else:
        print(f"  === Reading from {args.loadpkl} ===")
        events_df = pd.read_pickle(args.loadpkl)

    #if events_df is not None:
    
    if args.showall == True:
        plt.show()
        plt.close()
    return


def no_lines_archive(args, config_args, process_event_files):
    config_args(args)
    check_args(args)
    args.linesdir = "No line dir"
    
    figures = []
    events_df = None

    if args.rootfile is not None:
        full_truth = cloud.get_truth(args.rootfile)
    else:
        full_truth = None

    if args.loadpkl is None:
        event_dicts = []
        eventdir = args.pointsdir
        for entry in sorted(os.listdir(eventdir)): # loop through each file/(folder) in point cloud/(hough line) directory 
            # region entry proccessing
            entry_info = process_event_files(args, entry) # handels selecting a specific event_id amung other things
            if entry_info is None: continue
            event = cloud.Event(*[entry_info[key] for key in ['event_num', 'out_dir', 'line_file', 'points_file']])

            if args.plotlines or args.showall or args.show:
                figs = event.plot(plot_cloud=True)
                if args.show:
                    plt.show()
                    plt.close()
                if args.showall:
                    figures.extend(figs)

        if not args.events: # if analyzing whole data set, save analysis
            events_df:pd.DataFrame = pd.DataFrame(event_dicts)
            events_df.set_index("event_id", inplace=True)
            events_df.index.name = "event_id"

            pklsave = os.path.join(args.outdir,"events_dataframe.pkl")
            csvsave = os.path.join(args.outdir,"events_dataframe.csv")
            events_df.to_pickle(pklsave)
            events_df.to_csv(csvsave)
        
    else:
        print(f"  === Reading from {args.loadpkl} ===")
        events_df = pd.read_pickle(args.loadpkl)

    #if events_df is not None:
    
    if args.showall == True:
        plt.show()
        plt.close()
    return

def no_lines(args, config_args, process_event_files):
    config_args(args)
    check_args(args)
    args.linesdir = "No line dir"
    figures = []
    events_df = None

    if args.loadpkl is None:
        event_dicts = []
        for event in stream_events(args, process_event_files):
            
            if args.plotlines or args.showall or args.show or args.plotpoints:
                figs = event.plot(plot_cloud=True)
                if args.show:
                    plt.show()
                    plt.close()
                if args.showall:
                    figures.extend(figs)

        if not args.events: # if analyzing whole data set, save analysis
            events_df:pd.DataFrame = pd.DataFrame(event_dicts)
            events_df.set_index("event_id", inplace=True)
            events_df.index.name = "event_id"

            pklsave = os.path.join(args.outdir,"events_dataframe.pkl")
            csvsave = os.path.join(args.outdir,"events_dataframe.csv")
            events_df.to_pickle(pklsave)
            events_df.to_csv(csvsave)
        
    else:
        print(f"  === Reading from {args.loadpkl} ===")
        events_df = pd.read_pickle(args.loadpkl)

    #if events_df is not None:
    
    if args.showall == True:
        plt.show()
        plt.close()
    return

def gaussian_with_baseline(x, amplitude, mean, stddev, baseline):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def get_gaussian(data, bincount):
    counts, bin_edges = np.histogram(data, bins=bincount)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    est_amp = max(counts)
    est_mean = data.mean()
    est_std = data.std()
    est_base = 0

    init_guess = [est_amp, est_mean, est_std, est_base]

    popt, pcov = curve_fit(gaussian_with_baseline, bin_centers, counts, p0=init_guess)
    fit_amp, fit_mean, fit_sigma, fit_base = popt

    fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(fit_sigma)
    print(f"  gaussian amp:    {fit_amp:.4f}")
    print(f"  gaussian mean:   {fit_mean:.4f}")
    print(f"  gaussian STD:   {fit_sigma:.4f}")
    print(f"  calculated FWHM: {fwhm:.4f}")

    x_fine = np.linspace(bin_edges[0], bin_edges[-1], 200)
    y_fine = gaussian_with_baseline(x_fine, *popt)
    fit_data = {'x_val' : x_fine, 'y_val' : y_fine}
    plot_config = {
                    'data': fit_data,
                    'color': 'red',
                    'linestyle': '--',
                    'linewidth': 2,
                    'label': 'Gaussian Fit'
                    }   
    return fit_amp, fit_mean, fwhm, plot_config, fit_sigma

def make_histograms(truth, binwidth, rnge, histdir, histname="vert_assessment_hist.png"):
    fig, axes = plt.subplots(2, 2, figsize=(17, 9))

    things = ['dx', 'dy', 'dz', 'dist']
    for idx in things:
        data = truth[idx]
        span = data.max() - data.min()
        bincount = int(span/binwidth)
        print(f"\n  == {idx} info ==\n")
        fit_amp, fit_mean, fwhm, plot_config, fit_sigma = get_gaussian(data, bincount)
        dic = data.describe()
        print(dic)
        dic = dic.to_string()

        if idx=='dx':
            ax = axes[0,0]
        elif idx=='dy':
            ax = axes[0,1]
        elif idx=='dz':
            ax = axes[1,0]
        elif idx=='dist':
            ax = axes[1,1]

        if idx != 'dist':
            ax.hist(data, bins=bincount)
            ax.plot('x_val', 'y_val', **plot_config) # gaussian plot
            ax.set_xlim(-rnge,rnge)
            ax.set_title(f'Vertex residual {idx}')
            ax.set_xlabel('Found - True [mm]')
        else:
            ax.hist(data, bins=bincount)
            ax.plot('x_val', 'y_val', **plot_config) # gaussian plot
            ax.set_xlim(-0.1, 20)
            ax.set_title('Vertex distance')
            ax.set_xlabel('Norm(Found - True) [mm]')

        ax.set_ylabel('Number of Events')
        ax.grid(True, alpha=0.3)
        ax.text(
            0.95, 0.95, f"{dic}\ngauss amp: {fit_amp:.4f}\ngauss mean: {fit_mean:.4f}\ngauss FWHM: {fwhm:.4f}\ngauss STD: {fit_sigma:.4f}\nbin width (mm): {binwidth}",
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', alpha=0.3)
        )

    plt.tight_layout()
    plt.savefig(os.path.join(histdir, histname), dpi=220)
    return 

# -----------------------------------------------------------------------------
# Seed vertex outputs for ML assessment
# -----------------------------------------------------------------------------

def _safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return np.nan


def _vertex_nlines(vert):
    try:
        if vert.lines is None:
            return 0
        return len(vert.lines)
    except Exception:
        return 0


def _sorted_vertices(vertices, seed_select="first"):
    verts = list(vertices) if vertices is not None else []

    def vid(v):
        return getattr(v, "id", 999999)

    if seed_select == "smallest_radius":
        return sorted(verts, key=lambda v: (_safe_float(getattr(v, "radius", np.inf)), vid(v)))
    if seed_select == "largest_nlines":
        return sorted(verts, key=lambda v: (-_vertex_nlines(v), vid(v)))
    return sorted(verts, key=lambda v: vid(v))


def _truth_reaction_xyz(truth_df, event_id):
    if truth_df is None:
        return None
    try:
        if int(event_id) not in truth_df.index:
            return None
        row = truth_df.loc[int(event_id)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if "reaction" not in row.index:
            return None
        tv = row["reaction"]
        if isinstance(tv, float) and np.isnan(tv):
            return None
        arr = np.asarray(tv, dtype=float).reshape(-1)
        if len(arr) < 3 or not np.isfinite(arr[:3]).all():
            return None
        return arr[:3]
    except Exception:
        return None


def _vertex_point_xyz(vert):
    try:
        arr = np.asarray(vert.point, dtype=float).reshape(-1)
        if len(arr) >= 3:
            return arr[:3]
    except Exception:
        pass
    return np.array([np.nan, np.nan, np.nan], dtype=float)


def collect_seed_rows(event, truth_df=None, seed_select="first"):
    """
    Return two values:
      rows: list of all detected vertices for this event, including no_vertex rows
      primary_vertex: selected Vertex object or None

    The CSV columns are designed to be directly usable by the ML seed-rollout
    assessment code. The per-event seed text files contain just x y z.
    """
    rows = []
    truth_xyz = _truth_reaction_xyz(truth_df, event.id)
    vertices = _sorted_vertices(getattr(event.tree, "vertices", []), seed_select=seed_select)

    if not vertices:
        row = {
            "event": int(event.id),
            "status": "no_vertex",
            "is_primary": 0,
            "vertex_id": "",
            "seed_x": np.nan,
            "seed_y": np.nan,
            "seed_z": np.nan,
            "vertex_radius": np.nan,
            "n_vertex_lines": 0,
            "n_tree_vertices": 0,
        }
        if truth_xyz is not None:
            row.update({
                "truth_x": truth_xyz[0],
                "truth_y": truth_xyz[1],
                "truth_z": truth_xyz[2],
                "dx_seed_minus_truth": np.nan,
                "dy_seed_minus_truth": np.nan,
                "dz_seed_minus_truth": np.nan,
                "dist_seed_minus_truth": np.nan,
            })
        rows.append(row)
        return rows, None

    primary = vertices[0]

    for vert in vertices:
        xyz = _vertex_point_xyz(vert)
        is_primary = 1 if vert is primary else 0
        row = {
            "event": int(event.id),
            "status": "ok" if np.isfinite(xyz).all() else "bad_vertex",
            "is_primary": is_primary,
            "vertex_id": getattr(vert, "id", ""),
            "seed_x": xyz[0],
            "seed_y": xyz[1],
            "seed_z": xyz[2],
            "vertex_radius": _safe_float(getattr(vert, "radius", np.nan)),
            "n_vertex_lines": _vertex_nlines(vert),
            "n_tree_vertices": len(vertices),
        }
        if truth_xyz is not None:
            diff = xyz - truth_xyz
            row.update({
                "truth_x": truth_xyz[0],
                "truth_y": truth_xyz[1],
                "truth_z": truth_xyz[2],
                "dx_seed_minus_truth": diff[0],
                "dy_seed_minus_truth": diff[1],
                "dz_seed_minus_truth": diff[2],
                "dist_seed_minus_truth": float(np.linalg.norm(diff)) if np.isfinite(diff).all() else np.nan,
            })
        rows.append(row)

    return rows, primary


def write_seed_outputs(seed_rows, args):
    """
    Write:
      1. all detected vertices to vertex_seeds_all.csv
      2. one selected primary seed per event to vertex_seeds_primary.csv
      3. one x y z text file per primary seed to seed_vertices/event_<id>_vertex.txt
    """
    if seed_rows is None or len(seed_rows) == 0:
        print("[SEEDS] no seed rows to write")
        return None, None, None

    seedcsv = args.seedcsv
    primary_seedcsv = args.primary_seedcsv
    seeddir = args.seeddir

    if seedcsv is None:
        seedcsv = os.path.join(args.outdir, "vertex_seeds_all.csv")
    if primary_seedcsv is None:
        primary_seedcsv = os.path.join(args.outdir, "vertex_seeds_primary.csv")
    if seeddir is None:
        seeddir = os.path.join(args.outdir, "seed_vertices")

    os.makedirs(os.path.dirname(seedcsv) if os.path.dirname(seedcsv) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(primary_seedcsv) if os.path.dirname(primary_seedcsv) else ".", exist_ok=True)
    os.makedirs(seeddir, exist_ok=True)

    df = pd.DataFrame(seed_rows)
    df.to_csv(seedcsv, index=False)

    good = df[(df["status"] == "ok") & (df["is_primary"] == 1)].copy()
    for c in ["seed_x", "seed_y", "seed_z"]:
        good = good[np.isfinite(pd.to_numeric(good[c], errors="coerce"))]

    good.to_csv(primary_seedcsv, index=False)

    n_written = 0
    for _, row in good.iterrows():
        ev = int(row["event"])
        path = os.path.join(seeddir, "event_{}_vertex.txt".format(ev))
        with open(path, "w") as f:
            f.write("{:.10f} {:.10f} {:.10f}\n".format(float(row["seed_x"]), float(row["seed_y"]), float(row["seed_z"])))
        n_written += 1

    print("[SEEDS] wrote all seed CSV:      {}".format(seedcsv))
    print("[SEEDS] wrote primary seed CSV:  {}".format(primary_seedcsv))
    print("[SEEDS] wrote seed vertex dir:   {}  n_files={}".format(seeddir, n_written))
    return seedcsv, primary_seedcsv, seeddir


def assess_vert_id(args, config_args, process_event_files):
    
    config_args(args)
    check_args(args)
    if args.loadpkl is None:
        events = [] 
        all_verts = []
        seed_rows = []
        truth = cloud.get_truth(args.rootfile, only_siHitE=args.only_siHitE)
        #print(truth.index)
        for entry in sorted(os.listdir(args.linesdir)): # loop through each folder in lines directory 
            entry_info = process_event_files(args, entry)
            if entry_info == None: continue

            event = cloud.Event(*[entry_info[key] for key in ['event_num', 'out_dir', 'line_file', 'points_file'] if key in entry_info])
            event.identify_vertex()
            event.get_truth(truth)


            if args.plotlines:
                event.plot()

            rows, primary_vert = collect_seed_rows(event, truth_df=truth, seed_select=args.seed_select)
            seed_rows.extend(rows)

            if primary_vert is not None:
                all_verts.append(primary_vert.point)
                events.append(event.id)
            else: 
                all_verts.append(np.nan)
                events.append(event.id)

        write_seed_outputs(seed_rows, args)

        guess_series = pd.Series(all_verts, index=events)
        truth['found'] = guess_series

        if args.pklfile is not None:
            truth.to_pickle(args.pklfile)
    else:
        truth = pd.read_pickle(args.loadpkl)


    far_param = 100.0 # how far is 'far' from truth
    truth = cloud.get_truth_distances(truth, far_param)
    print(f"Furthest guesses: \n{truth[['dist', 'dx', 'dy', 'dz']].nlargest(10, 'dist')}")
    print(f"Smallest dy: \n{truth[['dist', 'dx', 'dy', 'dz']].nsmallest(10, 'dy')}")

    binwidth = 0.075
    rnge = 10
    args.histdir = os.path.join(args.outdir, 'histograms')
    if args.histname is not None: histname = args.histname
    else: histname = "vertex_residuals_hist.png"
    os.makedirs(args.histdir, exist_ok=True)
    fig = make_histograms(truth=truth, binwidth=binwidth, rnge=rnge, histdir=args.histdir, histname=histname)

    return
