#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace saturn {
namespace vof {

// Two-dimensional PLIC volume-of-fluid prototype.
//
// Cell-centered volume fractions on a periodic Cartesian grid. Each cell is
// reconstructed as a half-plane
//
//   n . (x - center) <= alpha
//
// with a Youngs normal from central differences of c and alpha chosen so that
// the clipped area equals c. Advection is dimensionally split (Strang:
// x half-step, y full-step, x half-step) with geometric fluxes: the fluid area
// crossing a face is the intersection of the upwind cell's PLIC polygon with
// the swept strip.

struct Point {
  double x;
  double y;
};

inline double polygonArea(const std::vector<Point>& polygon) {
  double area = 0.0;
  const int n = static_cast<int>(polygon.size());
  for (int i = 0; i < n; ++i) {
    const Point& a = polygon[i];
    const Point& b = polygon[(i + 1) % n];
    area += a.x * b.y - b.x * a.y;
  }
  return 0.5 * std::abs(area);
}

// Clip a convex polygon by the half-plane sign*(p.x - c) <= 0 along one axis.
inline std::vector<Point> clipAxis(const std::vector<Point>& polygon, int axis,
                                   double value, double sign) {
  std::vector<Point> output;
  if (polygon.empty()) {
    return output;
  }
  const int n = static_cast<int>(polygon.size());
  for (int i = 0; i < n; ++i) {
    const Point& current = polygon[i];
    const Point& next = polygon[(i + 1) % n];
    auto coord = [axis](const Point& p) { return axis == 0 ? p.x : p.y; };
    const double dc = sign * (coord(current) - value);
    const double dn = sign * (coord(next) - value);
    const bool inside_current = dc <= 0.0;
    const bool inside_next = dn <= 0.0;
    if (inside_current) {
      output.push_back(current);
    }
    if (inside_current != inside_next) {
      const double t = dc / (dc - dn);
      output.push_back(Point{current.x + t * (next.x - current.x),
                             current.y + t * (next.y - current.y)});
    }
  }
  return output;
}

inline std::vector<Point> clipHalfPlane(const std::vector<Point>& polygon,
                                        double nx, double ny, double alpha) {
  std::vector<Point> output;
  if (polygon.empty()) {
    return output;
  }
  const int n = static_cast<int>(polygon.size());
  for (int i = 0; i < n; ++i) {
    const Point& current = polygon[i];
    const Point& next = polygon[(i + 1) % n];
    auto f = [&](const Point& p) {
      return nx * (p.x - 0.5) + ny * (p.y - 0.5) - alpha;
    };
    const double dc = f(current);
    const double dn = f(next);
    const bool inside_current = dc <= 0.0;
    const bool inside_next = dn <= 0.0;
    if (inside_current) {
      output.push_back(current);
    }
    if (inside_current != inside_next) {
      const double t = dc / (dc - dn);
      output.push_back(Point{current.x + t * (next.x - current.x),
                             current.y + t * (next.y - current.y)});
    }
  }
  return output;
}

inline std::vector<Point> unitSquare() {
  return {Point{0.0, 0.0}, Point{1.0, 0.0}, Point{1.0, 1.0}, Point{0.0, 1.0}};
}

struct Reconstruction {
  double nx = 1.0;
  double ny = 0.0;
  double alpha = 0.0;
};

inline Reconstruction reconstruct(double c, double nx, double ny) {
  Reconstruction result;
  const double norm = std::sqrt(nx * nx + ny * ny);
  if (norm < 1.0e-14 || c <= 0.0 || c >= 1.0) {
    result.nx = 1.0;
    result.ny = 0.0;
    result.alpha = c - 0.5;
    return result;
  }
  result.nx = nx / norm;
  result.ny = ny / norm;
  double lo = -0.5 * (std::abs(result.nx) + std::abs(result.ny));
  double hi = 0.5 * (std::abs(result.nx) + std::abs(result.ny));
  const auto square = unitSquare();
  for (int iter = 0; iter < 50; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double area =
        polygonArea(clipHalfPlane(square, result.nx, result.ny, mid));
    if (area < c) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  result.alpha = 0.5 * (lo + hi);
  return result;
}

inline std::vector<Point> fluidPolygon(const Reconstruction& r) {
  return clipHalfPlane(unitSquare(), r.nx, r.ny, r.alpha);
}

class Grid2D {
 public:
  Grid2D(int nx, int ny, double lx = 1.0, double ly = 1.0)
      : nx_(nx),
        ny_(ny),
        hx_(lx / nx),
        hy_(ly / ny),
        c_(static_cast<std::size_t>(nx) * ny, 0.0) {}

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  double hx() const { return hx_; }
  double hy() const { return hy_; }
  double& operator()(int i, int j) {
    return c_[static_cast<std::size_t>(wrap(i, nx_)) * ny_ + wrap(j, ny_)];
  }
  double operator()(int i, int j) const {
    return c_[static_cast<std::size_t>(wrap(i, nx_)) * ny_ + wrap(j, ny_)];
  }
  std::vector<double>& data() { return c_; }
  const std::vector<double>& data() const { return c_; }

  // Reconstruct all cells; normals are Youngs gradients.
  std::vector<Reconstruction> reconstructions() const {
    std::vector<Reconstruction> r(static_cast<std::size_t>(nx_) * ny_);
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        // The normal points from fluid to empty, i.e. opposite the gradient.
        const double nx = -0.5 * ((*this)(i + 1, j) - (*this)(i - 1, j));
        const double ny = -0.5 * ((*this)(i, j + 1) - (*this)(i, j - 1));
        r[static_cast<std::size_t>(i) * ny_ + j] =
            reconstruct((*this)(i, j), nx, ny);
      }
    }
    return r;
  }

  // One directional sweep. `velocity` is face-centered in the sweep direction
  // (nx*ny for x, nx*ny for y); periodic indexing.
  void sweepX(const std::vector<double>& u, double dt) {
    const auto rec = reconstructions();
    std::vector<double> flux(static_cast<std::size_t>(nx_) * ny_, 0.0);
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        const double vel = u[static_cast<std::size_t>(i) * ny_ + j];
        if (vel == 0.0) {
          continue;
        }
        const int up = vel > 0.0 ? i - 1 : i;
        const auto& r = rec[static_cast<std::size_t>(wrap(up, nx_)) * ny_ + j];
        const double width = std::min(std::abs(vel) * dt / hx_, 1.0);
        const double x0 = vel > 0.0 ? 1.0 - width : 0.0;
        const double x1 = vel > 0.0 ? 1.0 : width;
        std::vector<Point> piece = fluidPolygon(r);
        piece = clipAxis(piece, 0, x0, -1.0);
        piece = clipAxis(piece, 0, x1, 1.0);
        const double area = polygonArea(piece);
        flux[static_cast<std::size_t>(i) * ny_ + j] =
            (vel > 0.0 ? 1.0 : -1.0) * area * hx_ * hy_;
      }
    }
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        const double left = flux[static_cast<std::size_t>(i) * ny_ + j];
        const double right =
            flux[static_cast<std::size_t>(wrap(i + 1, nx_)) * ny_ + j];
        (*this)(i, j) -= (right - left) / (hx_ * hy_);
      }
    }
  }

  void sweepY(const std::vector<double>& v, double dt) {
    const auto rec = reconstructions();
    std::vector<double> flux(static_cast<std::size_t>(nx_) * ny_, 0.0);
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        const double vel = v[static_cast<std::size_t>(i) * ny_ + j];
        if (vel == 0.0) {
          continue;
        }
        const int up = vel > 0.0 ? j - 1 : j;
        const auto& r = rec[static_cast<std::size_t>(i) * ny_ + wrap(up, ny_)];
        const double width = std::min(std::abs(vel) * dt / hy_, 1.0);
        const double y0 = vel > 0.0 ? 1.0 - width : 0.0;
        const double y1 = vel > 0.0 ? 1.0 : width;
        std::vector<Point> piece = fluidPolygon(r);
        piece = clipAxis(piece, 1, y0, -1.0);
        piece = clipAxis(piece, 1, y1, 1.0);
        const double area = polygonArea(piece);
        flux[static_cast<std::size_t>(i) * ny_ + j] =
            (vel > 0.0 ? 1.0 : -1.0) * area * hx_ * hy_;
      }
    }
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        const double lower = flux[static_cast<std::size_t>(i) * ny_ + j];
        const double upper =
            flux[static_cast<std::size_t>(i) * ny_ + wrap(j + 1, ny_)];
        (*this)(i, j) -= (upper - lower) / (hx_ * hy_);
      }
    }
  }

  // Strang-split advection over dt.
  void advance(const std::vector<double>& u, const std::vector<double>& v,
               double dt) {
    sweepX(u, 0.5 * dt);
    sweepY(v, dt);
    sweepX(u, 0.5 * dt);
  }

  double mass() const {
    double sum = 0.0;
    for (double c : c_) {
      sum += c;
    }
    return sum * hx_ * hy_;
  }

 private:
  static int wrap(int i, int n) { return ((i % n) + n) % n; }

  int nx_;
  int ny_;
  double hx_;
  double hy_;
  std::vector<double> c_;
};

}  // namespace vof
}  // namespace saturn
