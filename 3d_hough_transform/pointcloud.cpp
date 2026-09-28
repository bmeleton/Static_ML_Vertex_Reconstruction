// pointcloud.cpp
#include "pointcloud.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

static std::vector<std::string> splitWhitespace(const std::string &line) {
    std::istringstream iss(line);
    std::vector<std::string> fields;
    std::string token;

    while (iss >> token) {
        fields.push_back(token);
    }

    return fields;
}

static int findColumn(const std::vector<std::string> &columns,
                      const std::string &name) {
    for (size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static bool parseDoubleToken(const std::string &text, double *value) {
    if (!value) return false;

    char *end = NULL;
    const char *start = text.c_str();

    const double parsed = std::strtod(start, &end);

    if (end == start || (end && *end != '\0')) {
        return false;
    }

    if (!std::isfinite(parsed)) {
        return false;
    }

    *value = parsed;
    return true;
}

static bool getNamedDouble(const std::vector<std::string> &tokens,
                           int index,
                           double *value) {
    if (index < 0 ||
        static_cast<size_t>(index) >= tokens.size()) {
        return false;
    }

    return parseDoubleToken(tokens[static_cast<size_t>(index)], value);
}

} // namespace


int PointCloud::readFromFile(const std::string &fileName) {
    std::ifstream in(fileName.c_str());
    if (!in) return -1;

    points.clear();

    std::string line;
    int idCounter = 0;

    // -------------------------------------------------------------------------
    // Supported input formats
    //
    // 1. Current TeBAT reconstruction .dat format, whose header contains:
    //
    //      eventID padX padY x_fit y_fit z_fit t_global ... peakAmplitude
    //      padEnergyMeV x_orig z_orig
    //
    //    For this format we map by COLUMN NAME, not by physical position.
    //
    // 2. Legacy six-column Hough format:
    //
    //      x y z globalTime peakAmplitude integratedCharge
    //
    //    This remains supported when no recognized named header is present.
    // -------------------------------------------------------------------------

    bool useNamedColumns = false;

    int idx_x = -1;
    int idx_y = -1;
    int idx_z = -1;

    int idx_global_time = -1;
    int idx_peak_amplitude = -1;
    int idx_integrated_charge = -1;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        // Find the first non-whitespace character.
        size_t first = 0;
        while (first < line.size() &&
               (line[first] == ' ' || line[first] == '\t')) {
            ++first;
        }

        if (first >= line.size()) continue;

        // Comment lines are ignored.
        if (line[first] == '#') continue;

        // Header / non-numeric line.
        if ((line[first] >= 'A' && line[first] <= 'Z') ||
            (line[first] >= 'a' && line[first] <= 'z')) {

            const std::vector<std::string> columns =
                splitWhitespace(line);

            const int candidate_x = findColumn(columns, "x_fit");
            const int candidate_y = findColumn(columns, "y_fit");
            const int candidate_z = findColumn(columns, "z_fit");

            if (candidate_x >= 0 &&
                candidate_y >= 0 &&
                candidate_z >= 0) {

                useNamedColumns = true;

                idx_x = candidate_x;
                idx_y = candidate_y;
                idx_z = candidate_z;

                // Prefer the same timing/response quantities used by the
                // current TeBAT reconstruction output.
                idx_global_time = findColumn(columns, "t_global");
                if (idx_global_time < 0) {
                    idx_global_time = findColumn(columns, "globalTime");
                }
                if (idx_global_time < 0) {
                    idx_global_time = findColumn(columns, "t_waveform");
                }

                idx_peak_amplitude =
                    findColumn(columns, "peakAmplitude");
                if (idx_peak_amplitude < 0) {
                    idx_peak_amplitude =
                        findColumn(columns, "observedPeak");
                }

                // Preserve the sixth Hough-output quantity.  If a true
                // integratedCharge column exists, use it.  Current TeBAT
                // reconstruction files instead contain padEnergyMeV, which is
                // the most appropriate per-pad signal-strength quantity.
                idx_integrated_charge =
                    findColumn(columns, "integratedCharge");

                if (idx_integrated_charge < 0) {
                    idx_integrated_charge =
                        findColumn(columns, "padEnergyMeV");
                }
            }

            continue;
        }

        Point pt;

        if (useNamedColumns) {
            const std::vector<std::string> tokens =
                splitWhitespace(line);

            double x = 0.0;
            double y = 0.0;
            double z = 0.0;

            if (!getNamedDouble(tokens, idx_x, &x) ||
                !getNamedDouble(tokens, idx_y, &y) ||
                !getNamedDouble(tokens, idx_z, &z)) {
                continue;
            }

            // Coordinates must be finite; otherwise this point could create an
            // invalid Hough bounding box.
            if (!std::isfinite(x) ||
                !std::isfinite(y) ||
                !std::isfinite(z)) {
                continue;
            }

            double gt = 0.0;
            double pa = 0.0;
            double iq = 0.0;

            // These quantities are useful metadata but are not required for
            // line finding, so missing/non-numeric values safely become zero.
            getNamedDouble(tokens, idx_global_time, &gt);
            getNamedDouble(tokens, idx_peak_amplitude, &pa);
            getNamedDouble(tokens, idx_integrated_charge, &iq);

            pt.p.x = x;
            pt.p.y = y;
            pt.p.z = z;

            pt.globalTime = gt;
            pt.peakAmplitude = pa;
            pt.integratedCharge = iq;

            pt.id = idCounter++;
            points.push_back(pt);
        }
        else {
            // Legacy six-column format:
            // x y z globalTime peakAmplitude integratedCharge
            std::istringstream iss(line);

            double x, y, z, gt, pa, iq;

            if (!(iss >> x >> y >> z >> gt >> pa >> iq)) {
                continue;
            }

            if (!std::isfinite(x) ||
                !std::isfinite(y) ||
                !std::isfinite(z)) {
                continue;
            }

            // Optional legacy extra columns are ignored safely.
            double x_orig, z_orig;
            if (iss >> x_orig >> z_orig) {
                // intentionally ignored
            }

            pt.p.x = x;
            pt.p.y = y;
            pt.p.z = z;

            pt.globalTime = std::isfinite(gt) ? gt : 0.0;
            pt.peakAmplitude = std::isfinite(pa) ? pa : 0.0;
            pt.integratedCharge = std::isfinite(iq) ? iq : 0.0;

            pt.id = idCounter++;
            points.push_back(pt);
        }
    }

    return 0;
}


void PointCloud::getMinMax3D(Vector3d *minP, Vector3d *maxP) const {
    if (points.empty()) return;

    minP->x = maxP->x = points[0].p.x;
    minP->y = maxP->y = points[0].p.y;
    minP->z = maxP->z = points[0].p.z;

    for (size_t i = 1; i < points.size(); ++i) {
        const Vector3d &pt = points[i].p;

        minP->x = std::min(minP->x, pt.x);
        minP->y = std::min(minP->y, pt.y);
        minP->z = std::min(minP->z, pt.z);

        maxP->x = std::max(maxP->x, pt.x);
        maxP->y = std::max(maxP->y, pt.y);
        maxP->z = std::max(maxP->z, pt.z);
    }
}


void PointCloud::shiftToOrigin() {
    Vector3d minP, maxP;
    getMinMax3D(&minP, &maxP);

    shift = minP;

    for (size_t i = 0; i < points.size(); ++i) {
        points[i].p.x -= shift.x;
        points[i].p.y -= shift.y;
        points[i].p.z -= shift.z;
    }
}


void PointCloud::meanValue(Vector3d *mean) const {
    mean->x = mean->y = mean->z = 0.0;

    for (size_t i = 0; i < points.size(); ++i) {
        mean->x += points[i].p.x;
        mean->y += points[i].p.y;
        mean->z += points[i].p.z;
    }

    mean->x /= points.size();
    mean->y /= points.size();
    mean->z /= points.size();
}


void PointCloud::pointsCloseToLine(const Vector3d &a,
                                   const Vector3d &b,
                                   double dx,
                                   PointCloud *out) const {
    out->points.clear();

    for (size_t i = 0; i < points.size(); ++i) {
        const Vector3d &p = points[i].p;

        const double diffx = p.x - a.x;
        const double diffy = p.y - a.y;
        const double diffz = p.z - a.z;

        const double t =
            diffx*b.x + diffy*b.y + diffz*b.z;

        const double projx = a.x + t*b.x;
        const double projy = a.y + t*b.y;
        const double projz = a.z + t*b.z;

        const double dx2 = p.x - projx;
        const double dy2 = p.y - projy;
        const double dz2 = p.z - projz;

        if (dx2*dx2 + dy2*dy2 + dz2*dz2 <= dx*dx) {
            out->points.push_back(points[i]);
        }
    }
}


// Faster & robust removal using integer IDs (no float equality).
void PointCloud::removePoints(const PointCloud &sub) {
    if (sub.points.empty() || points.empty()) return;

    std::vector<int> ids;
    ids.reserve(sub.points.size());

    for (size_t j = 0; j < sub.points.size(); ++j) {
        ids.push_back(sub.points[j].id);
    }

    std::sort(ids.begin(), ids.end());

    std::vector<Point> remain;
    remain.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        const int id = points[i].id;
        const bool found =
            std::binary_search(ids.begin(), ids.end(), id);

        if (!found) {
            remain.push_back(points[i]);
        }
    }

    points.swap(remain);
}
