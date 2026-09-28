"""
Program to look for data in Ben Meletons data dir to be analyzed using Cloud_classes.py

Usage: 

Give datatype ( single_[2p, 3p, 4p] )


it will find hough line dir, point dir, and root file in /data/grgroup/gr02/bmeleton/TeBATAnalysis_v1.0.0/ (see :py:func:`check_args`)
then run :py:func:`analyze` from Analysis.py
by default it will output to ./bmeleton_outputs/
    per event data stored as    /events/event_[id]
    histograms stored in        /histograms/[datatype]_hist.png
    hough plots in              /hough_plots/event_[id]_lines3D.png
    pkl files stored in         /bmeleton_outputs/

    
honestly just run python3 BMeleton.py -h as all the arguments span multiple files. above is whats needed to run vert_id analysis

python3 BMeleton.py 
datatype        data type to look for in bens dir
--outdir        directory to put outputs in *
--plot          plots each event one by one *
--event         selects a specific event to analyse *

* implies not needed to run

Ex: 
python3 BMeleton.py single_2p --analysis assess_vert_id 
    Check bmeleton_outputs/histograms for results
python3 BMeleton.py single_2p --plotlines


python3 BMeleton.py single_2p --outdir elastic_analysis --rootfile TeBATSim_elastic.root --analysis assess_vert_id --histname test

 python3 BMeleton.py elastic_lowE --outdir elastic_lowE_analysis --rootfile TeBATSim_elastic_low_E.root --analysis assess_vert_id --histname test

python3 BMeleton.py alpha_sim --outdir alpha_sim_analysis --rootfile TeBATSim_alpha.root --analysis assess_vert_id --histname alpha_vertex_residuals

"""
import argparse, os, re, sys
import pandas as pd

# Custom classes and functions
import Cloud_classes as cloud
from Analysis import analyze
#import Analysis

DATA_TYPES = ["single_2p", "single_3p", "single_4p", "alphas", "elastic_lowE", "alpha_sim", "custom"]

# ++++++++++++++++++++++++++++++++++++++++ main +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ======================================== argument parsing helper function ============================================================ has docstring
def parse_args():
    '''
    Parses all the arguments 
    Returns:
        args: args has attributes with same name as each program argument without '--':
        datatype      determines what type of data to pull from bens data dir
        --outdir        directory to store program outputs
        --rootfile
    '''
    argParser = argparse.ArgumentParser()
    datatypes=DATA_TYPES
    argParser.add_argument("datatype", type=str, choices=datatypes, help=f"Data type to look for. must be one of these: {datatypes}")
    argParser.add_argument("--outdir", default="bmeleton_outputs", type=cloud.out_dir, help="Dir to save results to")
    argParser.add_argument("--linesdir", type=cloud.in_dir, help="Dir to look for hough line data in")
    argParser.add_argument("--pointsdir", type=cloud.in_dir, help="Dir to look for reconstructucted point clouds in")
    argParser.add_argument("--rootfile", nargs='?', const="non", help="Will search for .root file containing true vertices for all events, unless given one")

    return argParser

# ======================================== argument parsing helper function ============================================================
def check_args(args):

    args.outdir = os.path.join(args.outdir, args.datatype)

    datadir = "/data/grgroup/gr02/bmeleton/TeBATAnalysis_v1.0.0/"
    if args.datatype == "single_2p":
        linesdir = os.path.join(datadir, "reconstructed_lines_elastic")
        pointsdir = os.path.join(datadir, "reconstructed_points_and_plots_elastic/reconstructed_dat/")
        rootfile = os.path.join(datadir, "TeBATSim_elastic.root")
    elif args.datatype == "single_3p":
        linesdir = os.path.join(datadir, "reconstructed_lines_3_prong")
        pointsdir = os.path.join(datadir, "reconstructed_points_and_plots_3_prong/reconstructed_dat/")
        rootfile = os.path.join(datadir, "3_prong_events.root")
    elif args.datatype == "single_4p":
        linesdir = os.path.join(datadir, "reconstructed_lines_4_prong")
        pointsdir = os.path.join(datadir, "reconstructed_points_and_plots_4_prong/reconstructed_dat/")
        rootfile = os.path.join(datadir, "4_prong_events.root")
    elif args.datatype == "alphas":
        linesdir = os.path.join(datadir, "reconstructed_lines_alphas")
        pointsdir = os.path.join(datadir, "reconstructed_points_and_plots_alphas/reconstructed_dat/")
    elif args.datatype == "elastic_lowE":
        linesdir = os.path.join(datadir, "reconstructed_lines_elastic_lowE")
        pointsdir = os.path.join(datadir, "reconstructed_points_and_plots_elastic_lowE/reconstructed_dat/")
    elif args.datatype == "alpha_sim":
        linesdir = os.path.join(datadir, "reconstructed_lines_alpha_sim")
        pointsdir = os.path.join(datadir, "reconstructed_points_and_plots_alpha_sim/reconstructed_dat/")
    
    if args.linesdir is None: 
        if args.datatype == "custom":
            sys.exit("If using 'custom' datatype, you must provide linesdir and/or pointsdir")
        args.linesdir = linesdir
    if args.pointsdir is None:
        if args.datatype == "custom":
            sys.exit("If using 'custom' datatype, you must provide linesdir and/or pointsdir")
        args.pointsdir = pointsdir
        
    if args.rootfile == "non":
        args.rootfile = rootfile
    if args.analysis == "assess_vert_id" and args.rootfile is None:
        args.rootfile = rootfile

    elif args.rootfile is not None:
        args.rootfile = os.path.join(datadir, args.rootfile)

    if args.histname is None:
        args.histname = f"{args.datatype}_hist.png"
    if not args.histname.lower().endswith(".png"):
        args.histname = args.histname + ".png"

    if args.pklfile == "non":
        args.pklfile = f"{args.datatype}.pkl"

    return args


# ======================================== event parsing helper function ============================================================ has docstring
def process_event_files(args, entry):
    '''
    Handles all the file work, be careful about how its assigned to other variables in main
    Parameters:
        args: program arguments from parse_args
        entry: current file entry to pull info from
    Returns:
        dict: {'event', 'event_line_dir', 'event_out_dir', 'line_file'}
    '''
    cloud.Intersection._id_counter = 1 # each entry should restart the id counters for intersections and vertices
    cloud.Vertex._id_counter = 1

    label = re.search(r"(event_\d+)", entry) # looks for event_{event_num} in the file name
    if label:
        event_label = str(label.group(1))
        event = int( event_label.split("_",1)[1] )
    else:
        print(f"  Couldn't find event number from file name, {entry}. Make sure 'event_<event_number>' is in the file name. Assigning to event -1")
        event_label="event_-1"
        event = -1
    if args.events and str(event) not in args.events: return None # if looking for specific event, and it isn't this one, skip
    if str(event) in args.skip_events:
        print(f"   == Event {event} has been set to skip ==")
        return None
    print(f"\n===== Processing event {event} =====")
    
    if args.linesdir != "No line dir":
        event_line_dir = os.path.join(args.linesdir, event_label) # directory with event data to be analyzed
        if not os.path.isdir(event_line_dir) or not event_label.startswith("event_"): # if can't find the folder, or its not labled properly, skip
            print("  Error finding {event_label) in lines directory, skipping")
            return None
        line_file = os.path.join(event_line_dir, "lines.txt")
        if not os.path.isfile(line_file):
            print(f"  No lines file found, skipping")
            return None
    else: event_line_dir, line_file = None, None
    
    if args.pointsdir != "No point dir":
        points_file = os.path.join(args.pointsdir, f"event_{event}.dat")
        if not os.path.isfile(points_file):
            print(f"  No points file found at '{points_file}', skipping")
            return None
    else: points_file = None

    event_out_dir = os.path.join(args.outdir, 'events') # directory to store event specific outputs
    event_out_dir = os.path.join(event_out_dir, event_label) # directory to store event specific outputs
    os.makedirs(event_out_dir, exist_ok=True)
    
    return {'event_num':event, 'event_line_dir':event_line_dir, 'out_dir':event_out_dir, 'line_file':line_file, 'points_file':points_file}

def main():
    # region arg setup
    argParser=parse_args()
    analyze(argParser=argParser, config_args=check_args, process_event_files=process_event_files)

    #Analysis.boiler_plate(argParser=argParser, config_args=check_args, process_event_files=process_event_files)
    #Analysis.assess_vert_id(argParser=argParser, config_args=check_args, process_event_files=process_event_files)
    return


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    main()
