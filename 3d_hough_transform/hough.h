//
// hough.h
//     Implementation of Algorithm 2 (Hough transform) from IPOL paper
//
// Original authors: Tilman Schramke, Manuel Jeltsch, Christoph Dalitz
// Memory-safety additions for TeBATAnalysis:
//   - pre-allocation voting-space memory planning
//   - checked size_t arithmetic
//   - 64-bit-safe accumulator indexing
//

#ifndef HOUGH_H_
#define HOUGH_H_

#include "vector3d.h"
#include "sphere.h"
#include "pointcloud.h"

#include <cstddef>
#include <string>
#include <vector>

// Information returned before a Hough accumulator is allocated.
//
// requested_* describes the grid requested by the caller.
// used_* describes the grid that will actually be allocated after an optional
// memory-driven coarsening step.
struct HoughMemoryPlan {
  double requested_dx;
  double used_dx;
  double max_x;
  double range_x;

  std::size_t num_b;

  std::size_t requested_num_x;
  long double requested_num_cells;
  long double requested_bytes;

  std::size_t num_x;
  std::size_t num_cells;
  std::size_t bytes;

  bool coarsened;

  HoughMemoryPlan()
      : requested_dx(0.0),
        used_dx(0.0),
        max_x(0.0),
        range_x(0.0),
        num_b(0),
        requested_num_x(0),
        requested_num_cells(0.0L),
        requested_bytes(0.0L),
        num_x(0),
        num_cells(0),
        bytes(0),
        coarsened(false) {}
};

class Hough {
public:
  // accumulator array A
  std::vector<unsigned int> VotingSpace;

  // Directions B
  Sphere *sphere;
  std::size_t num_b;

  // x' and y'
  double dx, max_x;
  std::size_t num_x;

  // Plan the voting-space allocation without actually allocating the large
  // accumulator.  If maxBytes == 0, no explicit memory cap is applied.
  //
  // If allowCoarsen == true and the requested grid exceeds maxBytes, used_dx
  // is increased just enough to bring the accumulator under the cap.
  //
  // Returns false if the geometry/resolution is invalid, integer arithmetic
  // would overflow, the memory limit cannot support even a 1x1 grid, or the
  // requested grid exceeds the cap while allowCoarsen == false.
  static bool planVotingSpace(
      const Vector3d& minP,
      const Vector3d& maxP,
      double requestedDx,
      unsigned int sphereGranularity,
      std::size_t maxBytes,
      bool allowCoarsen,
      HoughMemoryPlan *plan,
      std::string *errorMessage);

  // Parameter-space discretization and allocation of voting space.
  // This constructor is hardened against size_t overflow and invalid grids.
  Hough(const Vector3d& minP, const Vector3d& maxP, double dx,
        unsigned int sphereGranularity);

  ~Hough();

  // returns the line with most votes (rc = number of votes)
  unsigned int getLine(Vector3d* point, Vector3d* direction);

  // add all points from point cloud to voting space
  void add(const PointCloud &pc);

  // subtract all points from point cloud from voting space
  void subtract(const PointCloud &pc);

  std::size_t votingBytes() const {
    return VotingSpace.size() * sizeof(unsigned int);
  }

private:
  // add or subtract (add==false) one point from voting space
  void pointVote(const Vector3d& point, bool add);
};

#endif /* HOUGH_H_ */
