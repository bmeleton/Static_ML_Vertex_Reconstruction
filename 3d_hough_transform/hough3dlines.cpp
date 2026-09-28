// hough3dlines.cpp
#include "vector3d.h"
#include "pointcloud.h"
#include "hough.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <Eigen/Dense>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <vector>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <limits>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

const char* usage =
  "Usage:\n"
  "  hough3dlines -d <datdir> -o <outdir> [options]\n"
  "Options:\n"
  "  -d <datdir>      directory of event_*.dat files\n"
  "  -o <outdir>      base output directory\n"
  "  -dx <dx>         Hough bin size (voting discretization) [0 auto]\n"
  "  -rline <r>       inlier radius around a line (mm). If <=0, defaults to dx_event\n"
  "  -nlines <nl>     max lines per event [0=unlimited]\n"
  "  -nstart <ns>     minimum number of lines to attempt before coverage stopping [0]\n"
  "  -cover <c>       coverage target in [0,1]; stop after nstart when assigned/N0 >= cover [0.90]\n"
  "  -minvotes <nv>   minimum votes (min points near a line) [2]\n"
  "  -maxmem <MB>     maximum Hough accumulator memory per event [512]\n"
  "                   use 0 for no explicit cap\n"
  "  -memmode <mode>  oversize handling: coarsen or skip [coarsen]\n"
  "  -v/-vv           verbosity\n";

static double orthogonal_LSQ(const PointCloud &pc, Vector3d* a, Vector3d* b) {
    pc.meanValue(a);
    int n = (int)pc.points.size();
    Eigen::MatrixXf pts(n,3);
    for (int i = 0; i < n; ++i) {
        pts(i,0) = pc.points[i].p.x;
        pts(i,1) = pc.points[i].p.y;
        pts(i,2) = pc.points[i].p.z;
    }
    Eigen::MatrixXf centered = pts.rowwise() - pts.colwise().mean();
    Eigen::MatrixXf scatter = centered.adjoint() * centered;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(scatter);
    b->x = eig.eigenvectors()(0,2);
    b->y = eig.eigenvectors()(1,2);
    b->z = eig.eigenvectors()(2,2);
    return eig.eigenvalues()(2);
}

struct LineFit {
    Vector3d a_world; // point on line in WORLD coords (unshifted)
    Vector3d b;       // direction (unit from eig), same in shifted/world
    PointCloud pts;   // points assigned to this line (stored in SHIFTED coords)
};

static double dist2_point_line_world(const Vector3d &p_world, const Vector3d &a_world, const Vector3d &b) {
    // distance^2 from point p to line a + t b, all in world coords
    double dx = p_world.x - a_world.x;
    double dy = p_world.y - a_world.y;
    double dz = p_world.z - a_world.z;
    double t  = dx*b.x + dy*b.y + dz*b.z;
    double px = a_world.x + t*b.x;
    double py = a_world.y + t*b.y;
    double pz = a_world.z + t*b.z;
    double ex = p_world.x - px;
    double ey = p_world.y - py;
    double ez = p_world.z - pz;
    return ex*ex + ey*ey + ez*ez;
}

static void compute_segment_endpoints_world(const LineFit &lf, const Vector3d &shift,
                                           Vector3d *start_world, Vector3d *end_world) {
    // We stored points in SHIFTED coords. Convert line anchor to shifted for t computation.
    Vector3d a_shifted;
    a_shifted.x = lf.a_world.x - shift.x;
    a_shifted.y = lf.a_world.y - shift.y;
    a_shifted.z = lf.a_world.z - shift.z;

    double tmin = 1e300, tmax = -1e300;
    for (size_t i = 0; i < lf.pts.points.size(); ++i) {
        const Vector3d &p = lf.pts.points[i].p; // shifted
        double dx = p.x - a_shifted.x;
        double dy = p.y - a_shifted.y;
        double dz = p.z - a_shifted.z;
        double t  = dx*lf.b.x + dy*lf.b.y + dz*lf.b.z;
        if (t < tmin) tmin = t;
        if (t > tmax) tmax = t;
    }
    start_world->x = lf.a_world.x + lf.b.x*tmin;
    start_world->y = lf.a_world.y + lf.b.y*tmin;
    start_world->z = lf.a_world.z + lf.b.z*tmin;

    end_world->x   = lf.a_world.x + lf.b.x*tmax;
    end_world->y   = lf.a_world.y + lf.b.y*tmax;
    end_world->z   = lf.a_world.z + lf.b.z*tmax;
}


static double bytes_to_mb(long double bytes) {
    return static_cast<double>(bytes / (1024.0L * 1024.0L));
}

static bool write_skipped_event(const char *outfn,
                                const char *evs,
                                const char *reason,
                                size_t n0,
                                double dx_requested,
                                double rline_event,
                                const HoughMemoryPlan *plan) {
    FILE *out = fopen(outfn, "w");
    if (!out) {
        perror("fopen");
        return false;
    }

    fprintf(out, "# Event %s\n", evs);
    fprintf(out, "# SKIPPED: %s\n", reason ? reason : "unknown reason");
    fprintf(out,
            "# Params: dx_requested=%.6f rline(inlier)=%.6f\n",
            dx_requested, rline_event);

    if (plan) {
        fprintf(out,
                "# HoughMemory: requested_dx=%.6f requested_num_x=%lu "
                "directions=%lu requested_MB=%.3f max_x=%.6f range_x=%.6f\n",
                plan->requested_dx,
                (unsigned long)plan->requested_num_x,
                (unsigned long)plan->num_b,
                bytes_to_mb(plan->requested_bytes),
                plan->max_x,
                plan->range_x);
    }

    fprintf(out,
            "# N0=%lu assigned=0 remaining_after_force=%lu\n",
            (unsigned long)n0,
            (unsigned long)n0);

    fclose(out);
    return true;
}

int main(int argc, char**argv) {
    const char *datdir = NULL, *outdir = NULL;

    // dx_event is used for voting discretization; rline_event is inlier radius
    double opt_dx = 0.0;
    double opt_rline = 0.0; // NEW

    int opt_nlines = 0, opt_minvotes = 2, opt_verbose = 0;
    int opt_nstart = 0;
    double opt_cover = 0.90;

    // Per-event Hough accumulator memory guard.
    //
    // "coarsen" keeps the event but increases ONLY the voting-grid dx enough
    // to fit under the cap.  rline_event is left unchanged.
    //
    // "skip" writes a lines.txt containing a SKIPPED reason and moves on.
    double opt_maxmem_mb = 512.0;
    std::string opt_memmode = "coarsen";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i],"-d")==0 && i+1<argc) datdir = argv[++i];
        else if (strcmp(argv[i],"-o")==0 && i+1<argc) outdir = argv[++i];
        else if (strcmp(argv[i],"-dx")==0 && i+1<argc) opt_dx = atof(argv[++i]);
        else if (strcmp(argv[i],"-rline")==0 && i+1<argc) opt_rline = atof(argv[++i]); // NEW
        else if (strcmp(argv[i],"-nlines")==0 && i+1<argc) opt_nlines = atoi(argv[++i]);
        else if (strcmp(argv[i],"-nstart")==0 && i+1<argc) opt_nstart = atoi(argv[++i]);
        else if (strcmp(argv[i],"-cover")==0 && i+1<argc) opt_cover = atof(argv[++i]);
        else if (strcmp(argv[i],"-minvotes")==0 && i+1<argc) opt_minvotes = atoi(argv[++i]);
        else if (strcmp(argv[i],"-maxmem")==0 && i+1<argc) opt_maxmem_mb = atof(argv[++i]);
        else if (strcmp(argv[i],"-memmode")==0 && i+1<argc) opt_memmode = argv[++i];
        else if (strcmp(argv[i],"-v")==0) opt_verbose = 1;
        else if (strcmp(argv[i],"-vv")==0) opt_verbose = 2;
        else { fprintf(stderr,"%s",usage); return 1; }
    }
    if (!datdir || !outdir) { fprintf(stderr,"%s",usage); return 1; }
    mkdir(outdir,0755);

    if (opt_nstart < 0) opt_nstart = 0;
    if (opt_cover < 0.0) opt_cover = 0.0;
    if (opt_cover > 1.0) opt_cover = 1.0;

    // rline can be <=0 to mean "use dx_event"
    if (opt_rline < 0.0) opt_rline = 0.0;

    if (!std::isfinite(opt_maxmem_mb) || opt_maxmem_mb < 0.0) {
        fprintf(stderr, "ERROR: -maxmem must be >= 0 MB\n");
        return 1;
    }

    if (opt_memmode != "coarsen" && opt_memmode != "skip") {
        fprintf(stderr,
                "ERROR: -memmode must be either 'coarsen' or 'skip'\n");
        return 1;
    }

    size_t max_hough_bytes = 0;
    if (opt_maxmem_mb > 0.0) {
        const long double requested =
            static_cast<long double>(opt_maxmem_mb) *
            1024.0L * 1024.0L;

        if (requested >
            static_cast<long double>(std::numeric_limits<size_t>::max())) {
            fprintf(stderr,
                    "ERROR: -maxmem is too large for this platform\n");
            return 1;
        }

        max_hough_bytes = static_cast<size_t>(requested);
    }

    if (opt_verbose) {
        fprintf(stderr,
                "[mem] Hough memory guard: maxmem=%.3f MB mode=%s\n",
                opt_maxmem_mb,
                opt_memmode.c_str());
    }

    DIR *dir = opendir(datdir);
    if (!dir) { perror("opendir"); return 1; }
    struct dirent *ent;

    while ((ent = readdir(dir)) != NULL) {
        const char *fname = ent->d_name;
        if (strncmp(fname,"event_",6)!=0) continue;
        int len = (int)strlen(fname);
        if (len<=10 || strcmp(fname+len-4,".dat")!=0) continue;

        char evs[64];
        int evlen = len - 6 - 4;
        strncpy(evs, fname+6, evlen); evs[evlen]='\0';

        char evout[256];
        snprintf(evout,sizeof(evout),"%s/event_%s",outdir,evs);
        mkdir(evout,0755);

        char outfn[300]; snprintf(outfn,sizeof(outfn),"%s/lines.txt",evout);

        char path[300]; snprintf(path,sizeof(path),"%s/%s",datdir,fname);
        PointCloud X;
        if (X.readFromFile(path)!=0) {
            fprintf(stderr,"fail read %s\n",fname);
            continue;
        }
        if (opt_verbose) fprintf(stderr,"[dbg] event %s -> got %lu points\n", evs, (unsigned long)X.points.size());
        if (X.points.size() < 2) {
            FILE *out = fopen(outfn,"w");
            if (out) { fprintf(out,"# Event %s\n",evs); fclose(out); }
            continue;
        }

        Vector3d minP_world, maxP_world;
        X.getMinMax3D(&minP_world, &maxP_world);

        const double d = (maxP_world - minP_world).norm();

        if (!std::isfinite(d) || d <= 0.0) {
            fprintf(stderr,
                    "[mem] event %s skipped: invalid/zero point-cloud extent d=%g\n",
                    evs, d);
            write_skipped_event(
                outfn, evs,
                "invalid or zero point-cloud extent",
                X.points.size(),
                0.0, 0.0, NULL);
            continue;
        }

        // Keep the existing TeBAT convention: shift points before Hough voting.
        X.shiftToOrigin();

        // IMPORTANT MEMORY FIX:
        // The Hough accumulator must be sized from the SAME shifted coordinate
        // system used by h->add(X).  The previous TeBAT driver passed the
        // pre-shift world min/max here, which can unnecessarily enlarge max_x
        // and therefore num_x^2.
        Vector3d minP_hough, maxP_hough;
        X.getMinMax3D(&minP_hough, &maxP_hough);

        // Requested voting discretization.
        double dx_requested = opt_dx;
        if (dx_requested <= 0.0) dx_requested = d / 64.0;

        if (!std::isfinite(dx_requested) || dx_requested <= 0.0) {
            fprintf(stderr,
                    "[mem] event %s skipped: invalid dx_requested=%g\n",
                    evs, dx_requested);
            write_skipped_event(
                outfn, evs,
                "invalid Hough voting resolution",
                X.points.size(),
                dx_requested, 0.0, NULL);
            continue;
        }

        // Inlier-tube radius remains based on the originally requested dx.
        // If memory protection coarsens the Hough voting grid, this radius is
        // deliberately NOT changed.
        double rline_event = opt_rline;
        if (rline_event <= 0.0) rline_event = dx_requested;

        const bool allow_coarsen = (opt_memmode == "coarsen");

        HoughMemoryPlan memplan;
        std::string plan_error;

        if (!Hough::planVotingSpace(
                minP_hough,
                maxP_hough,
                dx_requested,
                4,
                max_hough_bytes,
                allow_coarsen,
                &memplan,
                &plan_error)) {

            fprintf(stderr,
                    "[mem] event %s SKIPPED: %s. "
                    "requested_dx=%.6f requested_num_x=%lu "
                    "directions=%lu requested=%.3f MB cap=%.3f MB\n",
                    evs,
                    plan_error.c_str(),
                    memplan.requested_dx,
                    (unsigned long)memplan.requested_num_x,
                    (unsigned long)memplan.num_b,
                    bytes_to_mb(memplan.requested_bytes),
                    opt_maxmem_mb);

            write_skipped_event(
                outfn,
                evs,
                plan_error.c_str(),
                X.points.size(),
                dx_requested,
                rline_event,
                &memplan);

            continue;
        }

        const double dx_vote = memplan.used_dx;

        if (opt_verbose || memplan.coarsened) {
            fprintf(stderr,
                    "[mem] event %s: requested_dx=%.6f requested=%.3f MB; "
                    "used_dx=%.6f num_x=%lu directions=%lu allocated=%.3f MB%s\n",
                    evs,
                    memplan.requested_dx,
                    bytes_to_mb(memplan.requested_bytes),
                    memplan.used_dx,
                    (unsigned long)memplan.num_x,
                    (unsigned long)memplan.num_b,
                    bytes_to_mb(
                        static_cast<long double>(memplan.bytes)),
                    memplan.coarsened ? " [COARSENED]" : "");
        }

        std::unique_ptr<Hough> h;

        try {
            h.reset(new Hough(
                minP_hough,
                maxP_hough,
                dx_vote,
                4));

            h->add(X);
        }
        catch (const std::bad_alloc &e) {
            fprintf(stderr,
                    "[mem] event %s SKIPPED after std::bad_alloc: %s "
                    "(planned %.3f MB)\n",
                    evs,
                    e.what(),
                    bytes_to_mb(
                        static_cast<long double>(memplan.bytes)));

            write_skipped_event(
                outfn,
                evs,
                "std::bad_alloc while creating/filling Hough voting space",
                X.points.size(),
                dx_requested,
                rline_event,
                &memplan);

            continue;
        }
        catch (const std::exception &e) {
            fprintf(stderr,
                    "[mem] event %s SKIPPED after Hough exception: %s\n",
                    evs,
                    e.what());

            write_skipped_event(
                outfn,
                evs,
                e.what(),
                X.points.size(),
                dx_requested,
                rline_event,
                &memplan);

            continue;
        }

        const size_t N0 = X.points.size();
        size_t assigned = 0;

        std::vector<LineFit> fitted;
        fitted.reserve((opt_nlines > 0) ? opt_nlines : 16);

        PointCloud Y; // last extracted inliers (for subtract)
        int nln = 0;

        if (opt_verbose) {
            fprintf(stderr,
                "[dbg] Starting loop: N0=%lu dx_requested=%.6f dx_event(vote)=%.6f rline(inlier)=%.6f nstart=%d cover=%.3f nlines=%d\n",
                (unsigned long)N0, dx_requested, dx_vote, rline_event, opt_nstart, opt_cover, opt_nlines);
        }

        while (X.points.size() > 1) {
            // remove last inliers from voting
            h->subtract(Y);

            Vector3d a_shifted, b;
            h->getLine(&a_shifted, &b);

            // collect points close to that line (in shifted coords)
            // NEW: use rline_event instead of dx_event
            X.pointsCloseToLine(a_shifted, b, rline_event, &Y);

            if ((int)Y.points.size() < opt_minvotes) {
                if (opt_verbose) fprintf(stderr,"[dbg] break: Y.size=%lu < minvotes=%d\n",
                                        (unsigned long)Y.points.size(), opt_minvotes);
                break;
            }

            // refit line via orthogonal LSQ using the inliers (still in shifted coords)
            orthogonal_LSQ(Y, &a_shifted, &b);

            // convert anchor to WORLD coords (unshifted)
            Vector3d a_world;
            a_world.x = a_shifted.x + X.shift.x;
            a_world.y = a_shifted.y + X.shift.y;
            a_world.z = a_shifted.z + X.shift.z;

            LineFit lf;
            lf.a_world = a_world;
            lf.b = b;
            lf.pts = Y; // store points (SHIFTED coords)
            fitted.push_back(lf);

            // remove these inliers from the remaining pool
            X.removePoints(Y);

            nln++;
            assigned = N0 - X.points.size();
            double frac = (N0 > 0) ? ((double)assigned / (double)N0) : 1.0;

            if (opt_verbose) {
                fprintf(stderr,"[dbg] Line %d: inliers=%lu assigned=%lu/%lu (%.3f) remaining=%lu\n",
                        nln,
                        (unsigned long)Y.points.size(),
                        (unsigned long)assigned,
                        (unsigned long)N0,
                        frac,
                        (unsigned long)X.points.size());
            }

            // stopping conditions
            bool hitMax = (opt_nlines != 0 && nln >= opt_nlines);

            bool hitCover = false;
            if (opt_cover > 0.0) {
                if (nln >= opt_nstart && frac >= opt_cover) hitCover = true;
            }

            if (hitMax) {
                if (opt_verbose) fprintf(stderr,"[dbg] stop: reached max lines (%d)\n", opt_nlines);
                break;
            }
            if (hitCover) {
                if (opt_verbose) fprintf(stderr,"[dbg] stop: reached coverage target %.3f\n", opt_cover);
                break;
            }
        }

        // Force-assign all remaining points to closest fitted line (if we found at least one)
        if (!fitted.empty() && !X.points.empty()) {
            if (opt_verbose) fprintf(stderr,"[dbg] Forcing %lu leftover points onto nearest line\n",
                                    (unsigned long)X.points.size());

            for (size_t ip = 0; ip < X.points.size(); ++ip) {
                const PointCloud::Point &pt = X.points[ip]; // SHIFTED coords
                Vector3d p_world;
                p_world.x = pt.p.x + X.shift.x;
                p_world.y = pt.p.y + X.shift.y;
                p_world.z = pt.p.z + X.shift.z;

                size_t best = 0;
                double bestD2 = 1e300;
                for (size_t il = 0; il < fitted.size(); ++il) {
                    double d2 = dist2_point_line_world(p_world, fitted[il].a_world, fitted[il].b);
                    if (d2 < bestD2) { bestD2 = d2; best = il; }
                }
                fitted[best].pts.points.push_back(pt);
            }
        }

        // Write output file at end (now includes forced points)
        FILE *out = fopen(outfn,"w");
        if (!out) { perror("fopen"); continue; }

        fprintf(out,"# Event %s\n",evs);
        fprintf(out,
                "# Params: dx_requested=%.6f dx_event(vote)=%.6f "
                "rline(inlier)=%.6f minvotes=%d nstart=%d cover=%.3f nlines=%d\n",
                dx_requested, dx_vote, rline_event,
                opt_minvotes, opt_nstart, opt_cover, opt_nlines);

        fprintf(out,
                "# HoughMemory: requested_MB=%.3f allocated_MB=%.3f "
                "num_x=%lu directions=%lu coarsened=%d maxmem_MB=%.3f\n",
                bytes_to_mb(memplan.requested_bytes),
                bytes_to_mb(static_cast<long double>(memplan.bytes)),
                (unsigned long)memplan.num_x,
                (unsigned long)memplan.num_b,
                memplan.coarsened ? 1 : 0,
                opt_maxmem_mb);
        fprintf(out,"# N0=%lu assigned=%lu remaining_after_force=0\n",
                (unsigned long)N0, (unsigned long)(N0)); // after force, everything is assigned

        for (size_t i = 0; i < fitted.size(); ++i) {
            LineFit &lf = fitted[i];

            // segment endpoints based on ALL points assigned to this line
            Vector3d start_world, end_world;
            compute_segment_endpoints_world(lf, X.shift, &start_world, &end_world);

            fprintf(out,
                "# Line %lu: n=%lu a=(%.3f,%.3f,%.3f) b=(%.3f,%.3f,%.3f) start=(%.3f,%.3f,%.3f) end=(%.3f,%.3f,%.3f)\n",
                (unsigned long)(i+1),
                (unsigned long)lf.pts.points.size(),
                lf.a_world.x, lf.a_world.y, lf.a_world.z,
                lf.b.x, lf.b.y, lf.b.z,
                start_world.x, start_world.y, start_world.z,
                end_world.x, end_world.y, end_world.z
            );
            fprintf(out, "# x y z globalTime peakAmplitude integratedCharge\n");

            for (size_t ip = 0; ip < lf.pts.points.size(); ++ip) {
                const PointCloud::Point &pt = lf.pts.points[ip]; // SHIFTED coords
                double Xw = pt.p.x + X.shift.x;
                double Yw = pt.p.y + X.shift.y;
                double Zw = pt.p.z + X.shift.z;
                fprintf(out, "%f %f %f %f %f %f\n",
                    Xw, Yw, Zw,
                    pt.globalTime, pt.peakAmplitude, pt.integratedCharge
                );
            }
        }

        fclose(out);

        // Release the large accumulator before moving to the next event.
        h.reset();

#if defined(__GLIBC__)
        // Large std::vector allocations are usually returned by free(), but
        // malloc_trim helps keep the process RSS from retaining unused heap
        // pages during long multi-event jobs on glibc-based cluster nodes.
        malloc_trim(0);
#endif

        if (opt_verbose) printf("Wrote %s\n",outfn);
    }

    closedir(dir);
    return 0;
}
