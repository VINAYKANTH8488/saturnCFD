#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace saturn {
namespace vof {

// Three-dimensional PLIC volume-of-fluid prototype on a periodic grid.
//
// Each cell is a unit cube reconstructed as a half-plane
//   n . (x - 0.5) <= alpha
// with a Youngs normal from the volume-fraction gradient and alpha chosen so
// that the clipped volume equals c. Advection is Strang-split in x, y, z with
// geometric fluxes: the fluid volume crossing a face is the volume of the
// upwind cell's PLIC polyhedron intersected with the swept slab.

struct Vec3 {
  double x;
  double y;
  double z;
};

using Face = std::vector<Vec3>;
using Polyhedron = std::vector<Face>;

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
  return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3 operator+(const Vec3& a, const Vec3& b) {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3 operator*(double s, const Vec3& a) {
  return Vec3{s * a.x, s * a.y, s * a.z};
}
inline double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x};
}

inline Polyhedron unitCube() {
  Polyhedron cube;
  cube.push_back({{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}});  // -x
  cube.push_back({{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}});  // +x
  cube.push_back({{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}});  // -y
  cube.push_back({{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}});  // +y
  cube.push_back({{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}});  // -z
  cube.push_back({{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}});  // +z
  return cube;
}

inline double polyhedronVolume(const Polyhedron& poly) {
  double volume = 0.0;
  for (const auto& face : poly) {
    if (face.size() < 3) {
      continue;
    }
    const Vec3& a = face[0];
    for (std::size_t i = 1; i + 1 < face.size(); ++i) {
      volume += dot(a, cross(face[i], face[i + 1])) / 6.0;
    }
  }
  return std::abs(volume);
}

// Clip a convex polyhedron by n.x <= alpha. Intersection points are collected
// into a new cap face, sorted around the plane normal.
inline Polyhedron clipHalfPlane(const Polyhedron& poly, const Vec3& n,
                                double alpha) {
  Polyhedron result;
  std::vector<Vec3> cap;
  for (const auto& face : poly) {
    if (face.size() < 3) {
      continue;
    }
    Face clipped;
    const int count = static_cast<int>(face.size());
    for (int i = 0; i < count; ++i) {
      const Vec3& current = face[i];
      const Vec3& next = face[(i + 1) % count];
      const double dc = dot(n, current) - alpha;
      const double dn = dot(n, next) - alpha;
      const bool inside_current = dc <= 0.0;
      const bool inside_next = dn <= 0.0;
      if (inside_current) {
        clipped.push_back(current);
      }
      if (inside_current != inside_next) {
        const double t = dc / (dc - dn);
        const Vec3 point = current + t * (next - current);
        clipped.push_back(point);
        cap.push_back(point);
      }
    }
    if (clipped.size() >= 3) {
      result.push_back(std::move(clipped));
    }
  }

  if (cap.size() >= 3) {
    Vec3 center{0, 0, 0};
    for (const auto& p : cap) {
      center = center + p;
    }
    center = (1.0 / static_cast<double>(cap.size())) * center;
    Vec3 axis = n;
    const double norm = std::sqrt(dot(axis, axis));
    if (norm > 0.0) {
      axis = (1.0 / norm) * axis;
    }
    Vec3 helper = std::abs(axis.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 e1 = cross(axis, helper);
    const double e1n = std::sqrt(dot(e1, e1));
    if (e1n > 1e-14) {
      e1 = (1.0 / e1n) * e1;
    }
    Vec3 e2 = cross(axis, e1);
    std::sort(cap.begin(), cap.end(), [&](const Vec3& a, const Vec3& b) {
      const double aa = std::atan2(dot(a - center, e2), dot(a - center, e1));
      const double bb = std::atan2(dot(b - center, e2), dot(b - center, e1));
      return aa < bb;
    });
    result.push_back(std::move(cap));
  }
  return result;
}

struct Reconstruction3D {
  double nx = 1.0;
  double ny = 0.0;
  double nz = 0.0;
  double alpha = 0.0;
};

inline Reconstruction3D reconstruct(double c, double nx, double ny, double nz) {
  Reconstruction3D result;
  const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
  if (norm < 1e-14 || c <= 0.0 || c >= 1.0) {
    result.nx = 1.0;
    result.ny = 0.0;
    result.nz = 0.0;
    result.alpha = c - 0.5;
    return result;
  }
  result.nx = nx / norm;
  result.ny = ny / norm;
  result.nz = nz / norm;
  const Vec3 normal{result.nx, result.ny, result.nz};
  const double center_offset = 0.5 * (result.nx + result.ny + result.nz);
  const Polyhedron cube = unitCube();
  double lo =
      -0.5 * (std::abs(result.nx) + std::abs(result.ny) + std::abs(result.nz));
  double hi = -lo;
  for (int iter = 0; iter < 42; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double volume =
        polyhedronVolume(clipHalfPlane(cube, normal, mid + center_offset));
    if (volume < c) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  result.alpha = 0.5 * (lo + hi);
  return result;
}

inline Polyhedron fluidPolyhedron(const Reconstruction3D& r) {
  const double plane_alpha = r.alpha + 0.5 * (r.nx + r.ny + r.nz);
  return clipHalfPlane(unitCube(), Vec3{r.nx, r.ny, r.nz}, plane_alpha);
}

class Grid3D {
 public:
  Grid3D(int nx, int ny, int nz, double lx = 1.0, double ly = 1.0,
         double lz = 1.0)
      : nx_(nx),
        ny_(ny),
        nz_(nz),
        hx_(lx / nx),
        hy_(ly / ny),
        hz_(lz / nz),
        c_(static_cast<std::size_t>(nx) * ny * nz, 0.0) {}

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }
  double hx() const { return hx_; }
  double hy() const { return hy_; }
  double hz() const { return hz_; }

  double& operator()(int i, int j, int k) { return c_[index(i, j, k)]; }
  double operator()(int i, int j, int k) const { return c_[index(i, j, k)]; }
  std::vector<double>& data() { return c_; }
  const std::vector<double>& data() const { return c_; }

  double cellVolume() const { return hx_ * hy_ * hz_; }

  std::vector<Reconstruction3D> reconstructions() const {
    std::vector<Reconstruction3D> r(c_.size());
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < nz_; ++k) {
          const double nx =
              -0.5 * ((*this)(i + 1, j, k) - (*this)(i - 1, j, k));
          const double ny =
              -0.5 * ((*this)(i, j + 1, k) - (*this)(i, j - 1, k));
          const double nz =
              -0.5 * ((*this)(i, j, k + 1) - (*this)(i, j, k - 1));
          r[index(i, j, k)] = reconstruct((*this)(i, j, k), nx, ny, nz);
        }
      }
    }
    return r;
  }

  void sweepX(const std::vector<double>& u, double dt) {
    const auto rec = reconstructions();
    std::vector<double> flux(c_.size(), 0.0);
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < nz_; ++k) {
          const double vel = u[index(i, j, k)];
          if (vel == 0.0) {
            continue;
          }
          const int up = vel > 0.0 ? i - 1 : i;
          const auto& r = rec[index(up, j, k)];
          const double width = std::min(std::abs(vel) * dt / hx_, 1.0);
          Polyhedron poly = fluidPolyhedron(r);
          if (vel > 0.0) {
            poly = clipHalfPlane(poly, Vec3{1, 0, 0}, 1.0);
            poly = clipHalfPlane(poly, Vec3{-1, 0, 0}, -(1.0 - width));
          } else {
            poly = clipHalfPlane(poly, Vec3{-1, 0, 0}, 0.0);
            poly = clipHalfPlane(poly, Vec3{1, 0, 0}, width);
          }
          const double fraction = polyhedronVolume(poly);
          flux[index(i, j, k)] =
              (vel > 0.0 ? 1.0 : -1.0) * fraction * cellVolume();
        }
      }
    }
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < nz_; ++k) {
          const double left = flux[index(i, j, k)];
          const double right = flux[index(i + 1, j, k)];
          (*this)(i, j, k) -= (right - left) / cellVolume();
        }
      }
    }
  }

  void sweepY(const std::vector<double>& v, double dt) {
    const auto rec = reconstructions();
    std::vector<double> flux(c_.size(), 0.0);
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < nz_; ++k) {
          const double vel = v[index(i, j, k)];
          if (vel == 0.0) {
            continue;
          }
          const int up = vel > 0.0 ? j - 1 : j;
          const auto& r = rec[index(i, up, k)];
          const double width = std::min(std::abs(vel) * dt / hy_, 1.0);
          Polyhedron poly = fluidPolyhedron(r);
          if (vel > 0.0) {
            poly = clipHalfPlane(poly, Vec3{0, 1, 0}, 1.0);
            poly = clipHalfPlane(poly, Vec3{0, -1, 0}, -(1.0 - width));
          } else {
            poly = clipHalfPlane(poly, Vec3{0, -1, 0}, 0.0);
            poly = clipHalfPlane(poly, Vec3{0, 1, 0}, width);
          }
          const double fraction = polyhedronVolume(poly);
          flux[index(i, j, k)] =
              (vel > 0.0 ? 1.0 : -1.0) * fraction * cellVolume();
        }
      }
    }
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < nz_; ++k) {
          const double lower = flux[index(i, j, k)];
          const double upper = flux[index(i, j + 1, k)];
          (*this)(i, j, k) -= (upper - lower) / cellVolume();
        }
      }
    }
  }

  void sweepZ(const std::vector<double>& w, double dt) {
    const auto rec = reconstructions();
    std::vector<double> flux(c_.size(), 0.0);
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < nz_; ++k) {
          const double vel = w[index(i, j, k)];
          if (vel == 0.0) {
            continue;
          }
          const int up = vel > 0.0 ? k - 1 : k;
          const auto& r = rec[index(i, j, up)];
          const double width = std::min(std::abs(vel) * dt / hz_, 1.0);
          Polyhedron poly = fluidPolyhedron(r);
          if (vel > 0.0) {
            poly = clipHalfPlane(poly, Vec3{0, 0, 1}, 1.0);
            poly = clipHalfPlane(poly, Vec3{0, 0, -1}, -(1.0 - width));
          } else {
            poly = clipHalfPlane(poly, Vec3{0, 0, -1}, 0.0);
            poly = clipHalfPlane(poly, Vec3{0, 0, 1}, width);
          }
          const double fraction = polyhedronVolume(poly);
          flux[index(i, j, k)] =
              (vel > 0.0 ? 1.0 : -1.0) * fraction * cellVolume();
        }
      }
    }
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < nz_; ++k) {
          const double lower = flux[index(i, j, k)];
          const double upper = flux[index(i, j, k + 1)];
          (*this)(i, j, k) -= (upper - lower) / cellVolume();
        }
      }
    }
  }

  void advance(const std::vector<double>& u, const std::vector<double>& v,
               const std::vector<double>& w, double dt) {
    sweepX(u, 0.5 * dt);
    sweepY(v, 0.5 * dt);
    sweepZ(w, dt);
    sweepY(v, 0.5 * dt);
    sweepX(u, 0.5 * dt);
  }

  double mass() const {
    double sum = 0.0;
    for (double c : c_) {
      sum += c;
    }
    return sum * cellVolume();
  }

 private:
  std::size_t index(int i, int j, int k) const {
    return (static_cast<std::size_t>(wrap(i, nx_)) * ny_ +
            static_cast<std::size_t>(wrap(j, ny_))) *
               nz_ +
           static_cast<std::size_t>(wrap(k, nz_));
  }
  static int wrap(int i, int n) { return ((i % n) + n) % n; }

  int nx_;
  int ny_;
  int nz_;
  double hx_;
  double hy_;
  double hz_;
  std::vector<double> c_;
};

}  // namespace vof
}  // namespace saturn
