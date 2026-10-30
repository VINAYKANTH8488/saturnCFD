// 1D ideal-gas Euler verification driver.
//
// The solver is deliberately small: finite-volume update on a uniform grid,
// dimension-by-dimension reconstruction, a Riemann solver, and SSP-RK3. Its
// purpose is to gate the reconstruction and flux modules against the exact Sod
// solution before they are wired into the 3D compressible solver.

#include "numerics/compressible/reconstruction.hpp"
#include "numerics/compressible/riemann.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using saturn::ConservativeState;
using saturn::Flux;
using saturn::PrimitiveState;
using saturn::Reconstruction;
using saturn::RiemannSolver;

// ---------------------------------------------------------------------------
// Exact Sod solution (Toro, Riemann Solvers and Numerical Methods for Fluid
// Dynamics, chapter 4).
// ---------------------------------------------------------------------------

struct ExactRiemann {
  double gamma = 1.4;
  double x0 = 0.5;
  PrimitiveState left{1.0, 0.0, 0.0, 0.0, 1.0};
  PrimitiveState right{0.125, 0.0, 0.0, 0.0, 0.1};

  double p_star = 0.0;
  double u_star = 0.0;
  double rho_star_left = 0.0;
  double rho_star_right = 0.0;
  double c_star_left = 0.0;
  double c_star_right = 0.0;

  void solve() {
    const double c_left = std::sqrt(gamma * left.p / left.rho);
    const double c_right = std::sqrt(gamma * right.p / right.rho);

    const auto f = [&](double p, const PrimitiveState& state, double c) {
      if (p > state.p) {
        const double a = 2.0 / ((gamma + 1.0) * state.rho);
        const double b = (gamma - 1.0) / (gamma + 1.0) * state.p;
        return (p - state.p) * std::sqrt(a / (p + b));
      }
      return 2.0 * c / (gamma - 1.0) *
             (std::pow(p / state.p, (gamma - 1.0) / (2.0 * gamma)) - 1.0);
    };

    const auto total = [&](double p) {
      return f(p, left, c_left) + f(p, right, c_right) + right.u - left.u;
    };

    double low = 1.0e-12;
    double high = 10.0 * std::max(left.p, right.p) + 1.0;
    while (total(low) * total(high) > 0.0) {
      high *= 2.0;
    }
    for (int iteration = 0; iteration < 200; ++iteration) {
      const double mid = 0.5 * (low + high);
      if (total(mid) > 0.0) {
        high = mid;
      } else {
        low = mid;
      }
    }
    p_star = 0.5 * (low + high);
    u_star = 0.5 * (left.u + right.u) +
             0.5 * (f(p_star, right, c_right) - f(p_star, left, c_left));

    const auto starDensity = [&](const PrimitiveState& state) {
      if (p_star > state.p) {
        return state.rho * (p_star / state.p + (gamma - 1.0) / (gamma + 1.0)) /
               ((gamma - 1.0) / (gamma + 1.0) * p_star / state.p + 1.0);
      }
      return state.rho * std::pow(p_star / state.p, 1.0 / gamma);
    };
    rho_star_left = starDensity(left);
    rho_star_right = starDensity(right);
    c_star_left = std::sqrt(gamma * p_star / rho_star_left);
    c_star_right = std::sqrt(gamma * p_star / rho_star_right);
  }

  PrimitiveState sample(double x, double time) const {
    if (time <= 0.0) {
      return x < x0 ? left : right;
    }
    const double xi = (x - x0) / time;
    const double c_left = std::sqrt(gamma * left.p / left.rho);
    const double c_right = std::sqrt(gamma * right.p / right.rho);

    if (xi <= u_star) {
      if (p_star > left.p) {
        const double shock_speed =
            left.u -
            c_left * std::sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / left.p +
                               (gamma - 1.0) / (2.0 * gamma));
        if (xi >= shock_speed) {
          return PrimitiveState{rho_star_left, u_star, 0.0, 0.0, p_star};
        }
        return left;
      }

      const double head = left.u - c_left;
      const double tail = u_star - c_star_left;
      if (xi < head) {
        return left;
      }
      if (xi > tail) {
        return PrimitiveState{rho_star_left, u_star, 0.0, 0.0, p_star};
      }
      const double u =
          2.0 / (gamma + 1.0) * (c_left + 0.5 * (gamma - 1.0) * left.u + xi);
      const double c =
          2.0 / (gamma + 1.0) * (c_left + 0.5 * (gamma - 1.0) * (left.u - xi));
      return PrimitiveState{
          left.rho * std::pow(c / c_left, 2.0 / (gamma - 1.0)), u, 0.0, 0.0,
          left.p * std::pow(c / c_left, 2.0 * gamma / (gamma - 1.0))};
    }

    if (p_star > right.p) {
      const double shock_speed =
          right.u +
          c_right * std::sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / right.p +
                              (gamma - 1.0) / (2.0 * gamma));
      if (xi <= shock_speed) {
        return PrimitiveState{rho_star_right, u_star, 0.0, 0.0, p_star};
      }
      return right;
    }

    const double head = right.u + c_right;
    const double tail = u_star + c_star_right;
    if (xi > head) {
      return right;
    }
    if (xi < tail) {
      return PrimitiveState{rho_star_right, u_star, 0.0, 0.0, p_star};
    }
    const double u =
        2.0 / (gamma + 1.0) * (-c_right + 0.5 * (gamma - 1.0) * right.u + xi);
    const double c =
        2.0 / (gamma + 1.0) * (c_right - 0.5 * (gamma - 1.0) * (right.u - xi));
    return PrimitiveState{
        right.rho * std::pow(c / c_right, 2.0 / (gamma - 1.0)), u, 0.0, 0.0,
        right.p * std::pow(c / c_right, 2.0 * gamma / (gamma - 1.0))};
  }
};

// ---------------------------------------------------------------------------
// Small conservative-state helpers.
// ---------------------------------------------------------------------------

ConservativeState addScaled(const ConservativeState& base,
                            const ConservativeState& increment, double scale) {
  return ConservativeState{base.rho + scale * increment.rho,
                           base.mom_n + scale * increment.mom_n,
                           base.mom_t1 + scale * increment.mom_t1,
                           base.mom_t2 + scale * increment.mom_t2,
                           base.energy + scale * increment.energy};
}

ConservativeState linearCombination(double a, const ConservativeState& x,
                                    double b, const ConservativeState& y) {
  return ConservativeState{a * x.rho + b * y.rho, a * x.mom_n + b * y.mom_n,
                           a * x.mom_t1 + b * y.mom_t1,
                           a * x.mom_t2 + b * y.mom_t2,
                           a * x.energy + b * y.energy};
}

// ---------------------------------------------------------------------------
// 1D solver.
// ---------------------------------------------------------------------------

class Euler1D {
 public:
  Euler1D(int n, double length, double gamma, Reconstruction scheme,
          RiemannSolver solver)
      : n_(n),
        dx_(length / static_cast<double>(n)),
        gamma_(gamma),
        scheme_(scheme),
        solver_(solver),
        cells_(n) {}

  void initialize(const ExactRiemann& reference) {
    for (int i = 0; i < n_; ++i) {
      const double x = (static_cast<double>(i) + 0.5) * dx_;
      const PrimitiveState state = reference.sample(x, 0.0);
      cells_[i] = saturn::riemann::toConservative(state, gamma_);
    }
  }

  // Returns false if a non-positive state is produced.
  bool advance(double t_end, double cfl) {
    double time = 0.0;
    double last_dt = 0.0;
    int step = 0;
    while (time < t_end - 1.0e-14) {
      double maxSpeed = 0.0;
      for (const auto& cell : cells_) {
        const PrimitiveState state = saturn::riemann::toPrimitive(cell, gamma_);
        if (!(state.rho > 0.0) || !(state.p > 0.0)) {
          std::fprintf(stderr, "non-positive state at step %d\n", step);
          return false;
        }
        const double c = std::sqrt(gamma_ * state.p / state.rho);
        maxSpeed = std::max(maxSpeed, std::fabs(state.u) + c);
      }
      if (!(maxSpeed > 0.0)) {
        break;
      }
      double dt = cfl * dx_ / maxSpeed;
      if (time + dt > t_end) {
        dt = t_end - time;
      }
      last_dt = dt;

      std::vector<ConservativeState> stage(n_);
      std::vector<ConservativeState> stageRhs(n_);
      std::vector<ConservativeState> next(n_);

      if (!computeRhs(cells_, stageRhs)) {
        return false;
      }
      for (int i = 0; i < n_; ++i) {
        stage[i] = addScaled(cells_[i], stageRhs[i], dt);
      }
      if (!computeRhs(stage, stageRhs)) {
        return false;
      }
      for (int i = 0; i < n_; ++i) {
        next[i] = linearCombination(0.75, cells_[i], 0.25,
                                    addScaled(stage[i], stageRhs[i], dt));
      }
      if (!computeRhs(next, stageRhs)) {
        return false;
      }
      for (int i = 0; i < n_; ++i) {
        cells_[i] = linearCombination(1.0 / 3.0, cells_[i], 2.0 / 3.0,
                                      addScaled(next[i], stageRhs[i], dt));
      }

      time += dt;
      ++step;
    }
    std::printf("steps: %d, final time: %.6f, dt(last): %.3e\n", step, time,
                last_dt);
    return true;
  }

  const std::vector<ConservativeState>& cells() const { return cells_; }
  double dx() const { return dx_; }

 private:
  bool computeRhs(const std::vector<ConservativeState>& state,
                  std::vector<ConservativeState>& rhs) {
    constexpr int kGhost = 3;
    const int extended_size = n_ + 2 * kGhost;
    std::vector<PrimitiveState> extended(extended_size);

    for (int i = 0; i < n_; ++i) {
      extended[kGhost + i] = saturn::riemann::toPrimitive(state[i], gamma_);
    }
    for (int g = 0; g < kGhost; ++g) {
      extended[kGhost - 1 - g] = extended[kGhost];
      extended[kGhost + n_ + g] = extended[kGhost + n_ - 1];
    }

    std::vector<Flux> faces(n_ + 1);
    for (int face = 0; face <= n_; ++face) {
      // Stencil offsets -2..3 relative to the cell left of the face.
      const int base = kGhost + face - 3;
      double rho[6];
      double u[6];
      double v[6];
      double w[6];
      double p[6];
      for (int j = 0; j < 6; ++j) {
        const PrimitiveState& value = extended[base + j];
        rho[j] = value.rho;
        u[j] = value.u;
        v[j] = value.v;
        w[j] = value.w;
        p[j] = value.p;
      }

      PrimitiveState left{
          saturn::recon::reconstructLeft(scheme_, rho[0], rho[1], rho[2],
                                         rho[3], rho[4], rho[5]),
          saturn::recon::reconstructLeft(scheme_, u[0], u[1], u[2], u[3], u[4],
                                         u[5]),
          saturn::recon::reconstructLeft(scheme_, v[0], v[1], v[2], v[3], v[4],
                                         v[5]),
          saturn::recon::reconstructLeft(scheme_, w[0], w[1], w[2], w[3], w[4],
                                         w[5]),
          saturn::recon::reconstructLeft(scheme_, p[0], p[1], p[2], p[3], p[4],
                                         p[5])};
      PrimitiveState right{
          saturn::recon::reconstructRight(scheme_, rho[0], rho[1], rho[2],
                                          rho[3], rho[4], rho[5]),
          saturn::recon::reconstructRight(scheme_, u[0], u[1], u[2], u[3], u[4],
                                          u[5]),
          saturn::recon::reconstructRight(scheme_, v[0], v[1], v[2], v[3], v[4],
                                          v[5]),
          saturn::recon::reconstructRight(scheme_, w[0], w[1], w[2], w[3], w[4],
                                          w[5]),
          saturn::recon::reconstructRight(scheme_, p[0], p[1], p[2], p[3], p[4],
                                          p[5])};

      // First-order fallback if a high-order reconstruction leaves the
      // physical state space.
      if (!saturn::riemann::valid(left) || !saturn::riemann::valid(right)) {
        left = extended[kGhost + face - 1];
        right = extended[kGhost + face];
      }
      faces[face] = saturn::riemann::flux(solver_, left, right, gamma_);
    }

    for (int i = 0; i < n_; ++i) {
      rhs[i] =
          ConservativeState{-(faces[i + 1].rho - faces[i].rho) / dx_,
                            -(faces[i + 1].mom_n - faces[i].mom_n) / dx_,
                            -(faces[i + 1].mom_t1 - faces[i].mom_t1) / dx_,
                            -(faces[i + 1].mom_t2 - faces[i].mom_t2) / dx_,
                            -(faces[i + 1].energy - faces[i].energy) / dx_};
    }
    return true;
  }

  int n_;
  double dx_;
  double gamma_;
  Reconstruction scheme_;
  RiemannSolver solver_;
  std::vector<ConservativeState> cells_;
};

Reconstruction parseScheme(const std::string& name) {
  if (name == "central4") {
    return Reconstruction::Central4;
  }
  if (name == "central6") {
    return Reconstruction::Central6;
  }
  if (name == "teno6") {
    return Reconstruction::Teno6;
  }
  return Reconstruction::Weno5Z;
}

const char* schemeName(Reconstruction scheme) {
  switch (scheme) {
    case Reconstruction::Central2:
      return "central2";
    case Reconstruction::Central4:
      return "central4";
    case Reconstruction::Central6:
      return "central6";
    case Reconstruction::Weno5Z:
      return "weno5z";
    case Reconstruction::Teno6:
      return "teno6";
  }
  return "weno5z";
}

RiemannSolver parseSolver(const std::string& name) {
  if (name == "llf") {
    return RiemannSolver::Llf;
  }
  if (name == "hllc_lm") {
    return RiemannSolver::HllcLowMach;
  }
  if (name == "ausm_up") {
    return RiemannSolver::AusmPlusUp;
  }
  return RiemannSolver::Hllc;
}

const char* solverName(RiemannSolver solver) {
  switch (solver) {
    case RiemannSolver::Llf:
      return "llf";
    case RiemannSolver::Hllc:
      return "hllc";
    case RiemannSolver::HllcLowMach:
      return "hllc_lm";
    case RiemannSolver::AusmPlusUp:
      return "ausm_up";
  }
  return "hllc";
}

}  // namespace

int main(int argc, char** argv) {
  int n = 400;
  double t_end = 0.2;
  double cfl = 0.4;
  bool check = false;
  std::string output;
  Reconstruction scheme = Reconstruction::Weno5Z;
  RiemannSolver solver = RiemannSolver::Hllc;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto next = [&]() -> const char* {
      return i + 1 < argc ? argv[++i] : nullptr;
    };
    if (argument == "--n") {
      const char* value = next();
      if (value) {
        n = std::atoi(value);
      }
    } else if (argument == "--tend") {
      const char* value = next();
      if (value) {
        t_end = std::atof(value);
      }
    } else if (argument == "--cfl") {
      const char* value = next();
      if (value) {
        cfl = std::atof(value);
      }
    } else if (argument == "--scheme") {
      const char* value = next();
      if (value) {
        scheme = parseScheme(value);
      }
    } else if (argument == "--solver") {
      const char* value = next();
      if (value) {
        solver = parseSolver(value);
      }
    } else if (argument == "--output") {
      const char* value = next();
      if (value) {
        output = value;
      }
    } else if (argument == "--check") {
      check = true;
    } else if (argument == "--help") {
      std::printf(
          "usage: saturn_euler1d [--n N] [--tend T] [--cfl C]\n"
          "                       [--scheme central4|central6|weno5z|teno6]\n"
          "                       [--solver llf|hllc|hllc_lm|ausm_up]\n"
          "                       [--output FILE] [--check]\n");
      return 0;
    }
  }

  if (n < 16) {
    std::fprintf(stderr, "n must be at least 16\n");
    return 2;
  }

  ExactRiemann exact;
  exact.gamma = 1.4;
  exact.solve();

  Euler1D simulation(n, 1.0, exact.gamma, scheme, solver);
  simulation.initialize(exact);
  if (!simulation.advance(t_end, cfl)) {
    return 1;
  }

  const auto& cells = simulation.cells();
  const double dx = simulation.dx();
  double l1_rho = 0.0;
  double l1_u = 0.0;
  double l1_p = 0.0;
  double linf_rho = 0.0;

  std::FILE* file = nullptr;
  if (!output.empty()) {
    file = std::fopen(output.c_str(), "w");
    if (!file) {
      std::fprintf(stderr, "cannot open %s\n", output.c_str());
      return 1;
    }
    std::fprintf(file, "x,rho,u,p,rho_exact,u_exact,p_exact\n");
  }

  for (int i = 0; i < n; ++i) {
    const double x = (static_cast<double>(i) + 0.5) * dx;
    const PrimitiveState numerical =
        saturn::riemann::toPrimitive(cells[i], exact.gamma);
    const PrimitiveState reference = exact.sample(x, t_end);
    const double drho = std::fabs(numerical.rho - reference.rho);
    const double du = std::fabs(numerical.u - reference.u);
    const double dp = std::fabs(numerical.p - reference.p);
    l1_rho += drho * dx;
    l1_u += du * dx;
    l1_p += dp * dx;
    linf_rho = std::max(linf_rho, drho);
    if (file) {
      std::fprintf(file, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n", x,
                   numerical.rho, numerical.u, numerical.p, reference.rho,
                   reference.u, reference.p);
    }
  }
  if (file) {
    std::fclose(file);
  }

  std::printf("scheme: %s, solver: %s, n: %d, t: %.3f\n", schemeName(scheme),
              solverName(solver), n, t_end);
  std::printf("exact: p*=%.8f u*=%.8f rho*L=%.8f rho*R=%.8f\n", exact.p_star,
              exact.u_star, exact.rho_star_left, exact.rho_star_right);
  std::printf("L1: rho %.6e, u %.6e, p %.6e; Linf rho %.6e\n", l1_rho, l1_u,
              l1_p, linf_rho);

  // Plateau checks, well inside the star regions and away from the waves.
  const auto sampleValue = [&](double x) {
    const int index = std::min(n - 1, std::max(0, static_cast<int>(x / dx)));
    return saturn::riemann::toPrimitive(cells[index], exact.gamma).rho;
  };
  const double left_plateau = sampleValue(0.60);
  const double right_plateau = sampleValue(0.75);
  std::printf("rho(0.60)=%.6f (exact %.6f), rho(0.75)=%.6f (exact %.6f)\n",
              left_plateau, exact.rho_star_left, right_plateau,
              exact.rho_star_right);

  bool passed = true;
  if (check) {
    if (!(std::fabs(left_plateau - exact.rho_star_left) < 0.03)) {
      std::fprintf(stderr, "FAIL: left star density plateau off by %.3e\n",
                   std::fabs(left_plateau - exact.rho_star_left));
      passed = false;
    }
    if (!(std::fabs(right_plateau - exact.rho_star_right) < 0.03)) {
      std::fprintf(stderr, "FAIL: right star density plateau off by %.3e\n",
                   std::fabs(right_plateau - exact.rho_star_right));
      passed = false;
    }
    if (!(l1_rho < 0.02)) {
      std::fprintf(stderr, "FAIL: L1 density error %.3e exceeds 0.02\n",
                   l1_rho);
      passed = false;
    }
  }

  if (!passed) {
    return 1;
  }
  std::printf("saturnCFD Sod verification passed\n");
  return 0;
}
