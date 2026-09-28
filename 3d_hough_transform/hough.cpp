// hough.cpp
//     Implementation of Algorithm 2 (Hough transform) from IPOL paper
//
// Original authors: Tilman Schramke, Manuel Jeltsch, Christoph Dalitz
// Memory-safety additions for TeBATAnalysis:
//   - pre-allocation voting-space memory planning
//   - checked size_t arithmetic
//   - 64-bit-safe accumulator indexing
//   - signed bounds checks before converting Hough bin indices to size_t
//

#include "hough.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

static double roundToNearest(double num) {
  return (num > 0.0) ? std::floor(num + 0.5) : std::ceil(num - 0.5);
}

static bool checkedMultiply(std::size_t a, std::size_t b, std::size_t *out) {
  if (!out) return false;

  if (a == 0 || b == 0) {
    *out = 0;
    return true;
  }

  if (a > std::numeric_limits<std::size_t>::max() / b) {
    return false;
  }

  *out = a * b;
  return true;
}

static bool roundedNumX(double range_x, double dx, std::size_t *num_x) {
  if (!num_x) return false;

  if (!std::isfinite(range_x) || !std::isfinite(dx) ||
      range_x <= 0.0 || dx <= 0.0) {
    return false;
  }

  const double raw = roundToNearest(range_x / dx);

  if (!std::isfinite(raw) || raw < 1.0) {
    *num_x = 1;
    return true;
  }

  const long double raw_ld = static_cast<long double>(raw);
  const long double max_ld =
      static_cast<long double>(std::numeric_limits<std::size_t>::max());

  if (raw_ld > max_ld) {
    return false;
  }

  *num_x = static_cast<std::size_t>(raw);
  if (*num_x < 1) *num_x = 1;

  return true;
}

bool Hough::planVotingSpace(
    const Vector3d& minP,
    const Vector3d& maxP,
    double requestedDx,
    unsigned int sphereGranularity,
    std::size_t maxBytes,
    bool allowCoarsen,
    HoughMemoryPlan *plan,
    std::string *errorMessage) {

  if (!plan) {
    if (errorMessage) *errorMessage = "null HoughMemoryPlan pointer";
    return false;
  }

  *plan = HoughMemoryPlan();

  // Direction count is small compared with the accumulator and can be
  // generated safely here without allocating the voting array.
  Sphere tmpSphere;
  tmpSphere.fromIcosahedron(sphereGranularity);

  plan->num_b = tmpSphere.vertices.size();

  if (plan->num_b == 0) {
    if (errorMessage) {
      *errorMessage = "direction sphere contains zero vertices";
    }
    return false;
  }

  plan->max_x = std::max(maxP.norm(), minP.norm());
  plan->range_x = 2.0 * plan->max_x;

  if (!std::isfinite(plan->max_x) ||
      !std::isfinite(plan->range_x) ||
      plan->range_x <= 0.0) {
    if (errorMessage) {
      *errorMessage = "invalid/non-positive Hough coordinate range";
    }
    return false;
  }

  plan->requested_dx = requestedDx;
  if (!std::isfinite(plan->requested_dx) || plan->requested_dx <= 0.0) {
    plan->requested_dx = plan->range_x / 64.0;
  }

  if (!std::isfinite(plan->requested_dx) || plan->requested_dx <= 0.0) {
    if (errorMessage) {
      *errorMessage = "invalid Hough voting resolution";
    }
    return false;
  }

  if (!roundedNumX(
          plan->range_x,
          plan->requested_dx,
          &plan->requested_num_x)) {
    if (errorMessage) {
      *errorMessage = "requested Hough grid dimension is invalid or overflows size_t";
    }
    return false;
  }

  // Long-double estimate is intentionally used here so we can still report a
  // requested memory size even if the exact size_t product would overflow.
  plan->requested_num_cells =
      static_cast<long double>(plan->requested_num_x) *
      static_cast<long double>(plan->requested_num_x) *
      static_cast<long double>(plan->num_b);

  plan->requested_bytes =
      plan->requested_num_cells *
      static_cast<long double>(sizeof(unsigned int));

  plan->used_dx = plan->requested_dx;
  plan->num_x = plan->requested_num_x;

  if (maxBytes > 0) {
    const std::size_t maxCells =
        maxBytes / sizeof(unsigned int);

    const std::size_t maxNxSquared =
        maxCells / plan->num_b;

    if (maxNxSquared < 1) {
      if (errorMessage) {
        *errorMessage =
            "memory cap is too small for even one Hough x/y bin";
      }
      return false;
    }

    const long double maxNxLd =
        std::floor(std::sqrt(static_cast<long double>(maxNxSquared)));

    if (maxNxLd < 1.0L) {
      if (errorMessage) {
        *errorMessage =
            "memory cap yields zero allowable Hough grid width";
      }
      return false;
    }

    std::size_t maxNumX =
        static_cast<std::size_t>(maxNxLd);

    if (plan->num_x > maxNumX) {
      if (!allowCoarsen) {
        if (errorMessage) {
          *errorMessage =
              "requested Hough voting space exceeds configured memory cap";
        }
        return false;
      }

      plan->coarsened = true;

      // Choose a dx that targets maxNumX bins.  The tiny multiplicative margin
      // protects against roundoff causing round(range/dx) to become maxNumX+1.
      plan->used_dx =
          (plan->range_x / static_cast<double>(maxNumX)) *
          1.0000001;

      if (!roundedNumX(
              plan->range_x,
              plan->used_dx,
              &plan->num_x)) {
        if (errorMessage) {
          *errorMessage =
              "coarsened Hough grid dimension is invalid";
        }
        return false;
      }

      // In case platform rounding still lands one bin above the cap, enlarge
      // dx incrementally until it is safe.
      int guard = 0;
      while (plan->num_x > maxNumX && guard < 32) {
        plan->used_dx *= 1.0001;
        if (!roundedNumX(
                plan->range_x,
                plan->used_dx,
                &plan->num_x)) {
          if (errorMessage) {
            *errorMessage =
                "could not compute a safe coarsened Hough grid";
          }
          return false;
        }
        ++guard;
      }

      if (plan->num_x > maxNumX) {
        if (errorMessage) {
          *errorMessage =
              "failed to coarsen Hough grid below memory cap";
        }
        return false;
      }
    }
  }

  std::size_t nxSquared = 0;
  if (!checkedMultiply(plan->num_x, plan->num_x, &nxSquared)) {
    if (errorMessage) {
      *errorMessage = "num_x * num_x overflows size_t";
    }
    return false;
  }

  if (!checkedMultiply(nxSquared, plan->num_b, &plan->num_cells)) {
    if (errorMessage) {
      *errorMessage = "Hough cell-count multiplication overflows size_t";
    }
    return false;
  }

  if (!checkedMultiply(
          plan->num_cells,
          sizeof(unsigned int),
          &plan->bytes)) {
    if (errorMessage) {
      *errorMessage = "Hough byte-count multiplication overflows size_t";
    }
    return false;
  }

  if (maxBytes > 0 && plan->bytes > maxBytes) {
    if (errorMessage) {
      *errorMessage =
          "planned Hough voting space still exceeds configured memory cap";
    }
    return false;
  }

  return true;
}

Hough::Hough(const Vector3d& minP, const Vector3d& maxP, double var_dx,
             unsigned int sphereGranularity)
    : sphere(NULL),
      num_b(0),
      dx(0.0),
      max_x(0.0),
      num_x(0) {

  sphere = new Sphere();
  sphere->fromIcosahedron(sphereGranularity);
  num_b = sphere->vertices.size();

  if (num_b == 0) {
    delete sphere;
    sphere = NULL;
    throw std::runtime_error(
        "Hough: direction sphere contains zero vertices");
  }

  max_x = std::max(maxP.norm(), minP.norm());
  const double range_x = 2.0 * max_x;

  dx = var_dx;
  if (!std::isfinite(dx) || dx <= 0.0) {
    dx = range_x / 64.0;
  }

  if (!roundedNumX(range_x, dx, &num_x)) {
    delete sphere;
    sphere = NULL;
    throw std::length_error(
        "Hough: invalid or overflowing Hough grid dimension");
  }

  std::size_t nxSquared = 0;
  std::size_t numCells = 0;

  if (!checkedMultiply(num_x, num_x, &nxSquared) ||
      !checkedMultiply(nxSquared, num_b, &numCells)) {
    delete sphere;
    sphere = NULL;
    throw std::length_error(
        "Hough: voting-space cell count overflows size_t");
  }

  if (numCells > VotingSpace.max_size()) {
    delete sphere;
    sphere = NULL;
    throw std::length_error(
        "Hough: voting-space cell count exceeds vector::max_size()");
  }

  try {
    VotingSpace.assign(numCells, 0u);
  } catch (...) {
    delete sphere;
    sphere = NULL;
    throw;
  }
}

Hough::~Hough() {
  delete sphere;
  sphere = NULL;
}

void Hough::add(const PointCloud &pc) {
  for (std::vector<PointCloud::Point>::const_iterator it = pc.points.begin();
       it != pc.points.end(); ++it) {
    pointVote(it->p, true);
  }
}

void Hough::subtract(const PointCloud &pc) {
  for (std::vector<PointCloud::Point>::const_iterator it = pc.points.begin();
       it != pc.points.end(); ++it) {
    pointVote(it->p, false);
  }
}

void Hough::pointVote(const Vector3d& point, bool add) {

  if (VotingSpace.empty() || num_x == 0 || num_b == 0 || dx <= 0.0) {
    return;
  }

  for (std::size_t j = 0; j < sphere->vertices.size(); ++j) {
    const Vector3d& b = sphere->vertices[j];

    const double denom = 1.0 + b.z;
    if (!std::isfinite(denom) || std::fabs(denom) < 1.0e-15) {
      continue;
    }

    const double beta = 1.0 / denom;

    const double x_new =
        ((1.0 - (beta * (b.x * b.x))) * point.x)
        - ((beta * (b.x * b.y)) * point.y)
        - (b.x * point.z);

    const double y_new =
        ((-beta * (b.x * b.y)) * point.x)
        + ((1.0 - (beta * (b.y * b.y))) * point.y)
        - (b.y * point.z);

    const double x_bin =
        roundToNearest((x_new + max_x) / dx);
    const double y_bin =
        roundToNearest((y_new + max_x) / dx);

    // IMPORTANT: test the signed floating-point result before converting to
    // size_t.  Converting a negative bin directly to size_t would wrap to a
    // very large positive integer.
    if (!std::isfinite(x_bin) || !std::isfinite(y_bin) ||
        x_bin < 0.0 || y_bin < 0.0 ||
        x_bin >= static_cast<double>(num_x) ||
        y_bin >= static_cast<double>(num_x)) {
      continue;
    }

    const std::size_t x_i =
        static_cast<std::size_t>(x_bin);
    const std::size_t y_i =
        static_cast<std::size_t>(y_bin);

    const std::size_t index =
        (x_i * num_x * num_b) +
        (y_i * num_b) +
        j;

    if (index >= VotingSpace.size()) {
      continue;
    }

    if (add) {
      ++VotingSpace[index];
    } else {
      // Do not allow an unsigned underflow to turn a zero-vote cell into
      // UINT_MAX, which would then falsely become the strongest Hough line.
      if (VotingSpace[index] > 0u) {
        --VotingSpace[index];
      }
    }
  }
}

unsigned int Hough::getLine(Vector3d* a, Vector3d* b) {

  if (!a || !b || VotingSpace.empty() ||
      num_x == 0 || num_b == 0) {
    return 0u;
  }

  unsigned int votes = 0u;
  std::size_t bestIndex = 0u;

  // size_t is required here.  The previous unsigned-int loop counter can wrap
  // when VotingSpace has more than UINT_MAX entries.
  for (std::size_t i = 0; i < VotingSpace.size(); ++i) {
    if (VotingSpace[i] > votes) {
      votes = VotingSpace[i];
      bestIndex = i;
    }
  }

  const std::size_t xyStride = num_x * num_b;

  const std::size_t xIndex = bestIndex / xyStride;
  std::size_t remainder = bestIndex % xyStride;

  const std::size_t yIndex = remainder / num_b;
  const std::size_t bIndex = remainder % num_b;

  const double x =
      static_cast<double>(xIndex) * dx - max_x;
  const double y =
      static_cast<double>(yIndex) * dx - max_x;

  *b = sphere->vertices[bIndex];

  const double denom = 1.0 + b->z;
  if (!std::isfinite(denom) || std::fabs(denom) < 1.0e-15) {
    a->x = a->y = a->z = 0.0;
    return 0u;
  }

  a->x =
      x * (1.0 - ((b->x * b->x) / denom))
      - y * ((b->x * b->y) / denom);

  a->y =
      x * (-(b->x * b->y) / denom)
      + y * (1.0 - ((b->y * b->y) / denom));

  a->z = -x * b->x - y * b->y;

  return votes;
}
