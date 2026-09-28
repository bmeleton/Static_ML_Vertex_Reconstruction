// pointcloud.h
#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <vector>
#include <string>
#include "vector3d.h"

class PointCloud {
public:
    struct Point {
        Vector3d p;
        double globalTime;
        double peakAmplitude;
        double integratedCharge;
        int id; // NEW: stable ID for robust removal/assignment
    };

    std::vector<Point> points;
    Vector3d shift;

    int readFromFile(const std::string &fileName);
    void getMinMax3D(Vector3d *minP, Vector3d *maxP) const;
    void shiftToOrigin();
    void meanValue(Vector3d *mean) const;
    void pointsCloseToLine(const Vector3d &a, const Vector3d &b, double dx, PointCloud *out) const;
    void removePoints(const PointCloud &sub);
};

#endif // POINTCLOUD_H
