/**
 * @file compressible_ns.hpp
 * @brief Compressible Navier-Stokes fluxes and source terms.
 */
#pragma once

#include <Kokkos_Core.hpp>

#include "core/mpi.hpp"
#include "core/parallel.hpp"
#include "fields/patch_field.hpp"
#include "mesh/box.hpp"
#include "mesh/uniform_grid.hpp"
#include "numerics/compressible/reconstruction.hpp"
#include "numerics/compressible/riemann.hpp"

namespace saturn {
namespace compressible {

// Ideal-gas compressible Navier-Stokes in conservative variables:
//
//   U = [rho, rho u, rho v, rho w, E]
//   E = p/(gamma-1) + 0.5 rho (u^2 + v^2 + w^2)
//   p = rho R T
//
// Inviscid fluxes come from the reconstruction + Riemann modules. Viscous
// fluxes use fourth-order cell-centered gradients averaged to faces:
//
//   tau_ij = mu (du_i/dx_j + du_j/dx_i - 2/3 delta_ij div u)
//   q_j    = -k dT/dx_j,   k = mu cp / Pr,
//   Fv_x   = [0, tau_xx, tau_xy, tau_xz,
//             u tau_xx + v tau_xy + w tau_xz - q_x]
//
// with cp = gamma R/(gamma-1). Constant mu and Pr are assumed for now.
struct GasProperties {
  double gamma = 1.4;
  double gas_constant = 1.0;
  double mu = 0.0;
  double prandtl = 0.72;
};

KOKKOS_INLINE_FUNCTION
double soundSpeed(const PrimitiveState& state, const GasProperties& gas) {
  return Kokkos::sqrt(gas.gamma * state.p / state.rho);
}

KOKKOS_INLINE_FUNCTION
double temperature(const PrimitiveState& state, const GasProperties& gas) {
  return state.p / (state.rho * gas.gas_constant);
}

template <class PatchType>
KOKKOS_INLINE_FUNCTION PrimitiveState loadPrimitive(const PatchType& q, int i,
                                                    int j, int k,
                                                    const GasProperties& gas) {
  const ConservativeState state{q(0, i, j, k), q(1, i, j, k), q(2, i, j, k),
                                q(3, i, j, k), q(4, i, j, k)};
  return riemann::toPrimitive(state, gas.gamma);
}

// Rotate a state so that its first velocity is normal to the face.
KOKKOS_INLINE_FUNCTION
PrimitiveState toNormal(const PrimitiveState& state, int direction) {
  if (direction == 0) {
    return state;
  }
  if (direction == 1) {
    return PrimitiveState{state.rho, state.v, state.u, state.w, state.p};
  }
  return PrimitiveState{state.rho, state.w, state.u, state.v, state.p};
}

// Rotate a normal-coordinate flux back to Cartesian components.
KOKKOS_INLINE_FUNCTION
Flux fromNormal(const Flux& flux, int direction) {
  if (direction == 0) {
    return flux;
  }
  if (direction == 1) {
    return Flux{flux.rho, flux.mom_t1, flux.mom_n, flux.mom_t2, flux.energy};
  }
  return Flux{flux.rho, flux.mom_t1, flux.mom_t2, flux.mom_n, flux.energy};
}

// Reconstruct the left/right primitive states at the face between cell
// (i,j,k) and its neighbor in `direction`.
template <class PatchType>
KOKKOS_INLINE_FUNCTION void reconstructFace(const PatchType& q, int i, int j,
                                            int k, int direction,
                                            Reconstruction scheme,
                                            const GasProperties& gas,
                                            PrimitiveState& left,
                                            PrimitiveState& right) {
  const int di = direction == 0 ? 1 : 0;
  const int dj = direction == 1 ? 1 : 0;
  const int dk = direction == 2 ? 1 : 0;

  double rho[6];
  double u[6];
  double v[6];
  double w[6];
  double p[6];
  for (int offset = -2; offset <= 3; ++offset) {
    const PrimitiveState value = loadPrimitive(
        q, i + offset * di, j + offset * dj, k + offset * dk, gas);
    const int index = offset + 2;
    rho[index] = value.rho;
    u[index] = value.u;
    v[index] = value.v;
    w[index] = value.w;
    p[index] = value.p;
  }

  left = PrimitiveState{
      recon::reconstructLeft(scheme, rho[0], rho[1], rho[2], rho[3], rho[4],
                             rho[5]),
      recon::reconstructLeft(scheme, u[0], u[1], u[2], u[3], u[4], u[5]),
      recon::reconstructLeft(scheme, v[0], v[1], v[2], v[3], v[4], v[5]),
      recon::reconstructLeft(scheme, w[0], w[1], w[2], w[3], w[4], w[5]),
      recon::reconstructLeft(scheme, p[0], p[1], p[2], p[3], p[4], p[5])};
  right = PrimitiveState{
      recon::reconstructRight(scheme, rho[0], rho[1], rho[2], rho[3], rho[4],
                              rho[5]),
      recon::reconstructRight(scheme, u[0], u[1], u[2], u[3], u[4], u[5]),
      recon::reconstructRight(scheme, v[0], v[1], v[2], v[3], v[4], v[5]),
      recon::reconstructRight(scheme, w[0], w[1], w[2], w[3], w[4], w[5]),
      recon::reconstructRight(scheme, p[0], p[1], p[2], p[3], p[4], p[5])};

  if (!riemann::valid(left) || !riemann::valid(right)) {
    left = loadPrimitive(q, i, j, k, gas);
    right = loadPrimitive(q, i + di, j + dj, k + dk, gas);
  }
}

template <class PatchType>
KOKKOS_INLINE_FUNCTION Flux inviscidFaceFlux(const PatchType& q, int i, int j,
                                             int k, int direction,
                                             Reconstruction scheme,
                                             RiemannSolver solver,
                                             const GasProperties& gas) {
  PrimitiveState left;
  PrimitiveState right;
  reconstructFace(q, i, j, k, direction, scheme, gas, left, right);
  const Flux normal_flux = riemann::flux(solver, toNormal(left, direction),
                                         toNormal(right, direction), gas.gamma);
  return fromNormal(normal_flux, direction);
}

KOKKOS_INLINE_FUNCTION
double central4Derivative(double minus2, double minus1, double plus1,
                          double plus2, double h) {
  return (minus2 - 8.0 * minus1 + 8.0 * plus1 - plus2) / (12.0 * h);
}

// Cell-centered 4th-order gradients of (u,v,w,T) along one direction.
template <class PatchType>
KOKKOS_INLINE_FUNCTION void gradientLine(const PatchType& q, int i, int j,
                                         int k, int direction, double h,
                                         const GasProperties& gas,
                                         double derivative[4]) {
  const int di = direction == 0 ? 1 : 0;
  const int dj = direction == 1 ? 1 : 0;
  const int dk = direction == 2 ? 1 : 0;
  const PrimitiveState minus2 =
      loadPrimitive(q, i - 2 * di, j - 2 * dj, k - 2 * dk, gas);
  const PrimitiveState minus1 = loadPrimitive(q, i - di, j - dj, k - dk, gas);
  const PrimitiveState plus1 = loadPrimitive(q, i + di, j + dj, k + dk, gas);
  const PrimitiveState plus2 =
      loadPrimitive(q, i + 2 * di, j + 2 * dj, k + 2 * dk, gas);

  derivative[0] = central4Derivative(minus2.u, minus1.u, plus1.u, plus2.u, h);
  derivative[1] = central4Derivative(minus2.v, minus1.v, plus1.v, plus2.v, h);
  derivative[2] = central4Derivative(minus2.w, minus1.w, plus1.w, plus2.w, h);
  derivative[3] =
      central4Derivative(temperature(minus2, gas), temperature(minus1, gas),
                         temperature(plus1, gas), temperature(plus2, gas), h);
}

// Gradient layout:
//   0..2  du/dx, du/dy, du/dz
//   3..5  dv/dx, dv/dy, dv/dz
//   6..8  dw/dx, dw/dy, dw/dz
//   9..11 dT/dx, dT/dy, dT/dz
template <class ExecSpace>
void computeGradients(const PatchField<ExecSpace>& q,
                      PatchField<ExecSpace>& grad, const UniformGrid& grid,
                      const GasProperties& gas, const Box& region) {
  if (region.empty()) {
    return;
  }
  const auto policy = makeMdrange3<ExecSpace>(
      Int3{{region.lo[2], region.lo[1], region.lo[0]}},
      Int3{{region.hi[2] + 1, region.hi[1] + 1, region.hi[0] + 1}});
  const double hx = grid.h[0];
  const double hy = grid.h[1];
  const double hz = grid.h[2];

  Kokkos::parallel_for(
      "compressible_gradients", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        double derivative[4];
        gradientLine(q, i, j, k, 0, hx, gas, derivative);
        grad(0, i, j, k) = derivative[0];
        grad(3, i, j, k) = derivative[1];
        grad(6, i, j, k) = derivative[2];
        grad(9, i, j, k) = derivative[3];

        gradientLine(q, i, j, k, 1, hy, gas, derivative);
        grad(1, i, j, k) = derivative[0];
        grad(4, i, j, k) = derivative[1];
        grad(7, i, j, k) = derivative[2];
        grad(10, i, j, k) = derivative[3];

        gradientLine(q, i, j, k, 2, hz, gas, derivative);
        grad(2, i, j, k) = derivative[0];
        grad(5, i, j, k) = derivative[1];
        grad(8, i, j, k) = derivative[2];
        grad(11, i, j, k) = derivative[3];
      });
}

// Cell-centered viscous stress. Layout:
//   0 tau_xx, 1 tau_yy, 2 tau_zz, 3 tau_xy, 4 tau_xz, 5 tau_yz
template <class ExecSpace>
void computeStress(const PatchField<ExecSpace>& grad,
                   PatchField<ExecSpace>& stress, const GasProperties& gas,
                   const Box& region) {
  if (region.empty()) {
    return;
  }
  const auto policy = makeMdrange3<ExecSpace>(
      Int3{{region.lo[2], region.lo[1], region.lo[0]}},
      Int3{{region.hi[2] + 1, region.hi[1] + 1, region.hi[0] + 1}});
  const double mu = gas.mu;

  Kokkos::parallel_for(
      "compressible_stress", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const double du_dx = grad(0, i, j, k);
        const double du_dy = grad(1, i, j, k);
        const double du_dz = grad(2, i, j, k);
        const double dv_dx = grad(3, i, j, k);
        const double dv_dy = grad(4, i, j, k);
        const double dv_dz = grad(5, i, j, k);
        const double dw_dx = grad(6, i, j, k);
        const double dw_dy = grad(7, i, j, k);
        const double dw_dz = grad(8, i, j, k);
        const double div = du_dx + dv_dy + dw_dz;
        stress(0, i, j, k) = mu * (2.0 * du_dx - (2.0 / 3.0) * div);
        stress(1, i, j, k) = mu * (2.0 * dv_dy - (2.0 / 3.0) * div);
        stress(2, i, j, k) = mu * (2.0 * dw_dz - (2.0 / 3.0) * div);
        stress(3, i, j, k) = mu * (du_dy + dv_dx);
        stress(4, i, j, k) = mu * (du_dz + dw_dx);
        stress(5, i, j, k) = mu * (dv_dz + dw_dy);
      });
}

// Workspace holding gradients and one face-flux array per direction.
template <class ExecSpace>
class CompressibleWorkspace {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using face_view = Kokkos::View<double****, Kokkos::LayoutRight, memory_space>;

  explicit CompressibleWorkspace(const Box& box)
      : grad(box, 12, 3, "compressible_grad"),
        stress(box, 6, 2, "compressible_stress"),
        fx("compressible_fx", 5, box.length(2), box.length(1),
           box.length(0) + 1),
        fy("compressible_fy", 5, box.length(2), box.length(1) + 1,
           box.length(0)),
        fz("compressible_fz", 5, box.length(2) + 1, box.length(1),
           box.length(0)) {}

  PatchField<ExecSpace> grad;
  PatchField<ExecSpace> stress;
  face_view fx;
  face_view fy;
  face_view fz;
};

template <class ExecSpace>
void computeFaceFluxesX(const PatchField<ExecSpace>& q,
                        const CompressibleWorkspace<ExecSpace>& workspace,
                        const UniformGrid& grid, const GasProperties& gas,
                        Reconstruction scheme, RiemannSolver solver) {
  const Box box = grid.local;
  const int nx = box.length(0);
  const int ny = box.length(1);
  const int nz = box.length(2);
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{nz, ny, nx + 1}});
  auto fx = workspace.fx;

  Kokkos::parallel_for(
      "compressible_faces_x", policy, KOKKOS_LAMBDA(int k, int j, int face) {
        const int i_left = box.lo[0] + face - 1;
        const int j_global = box.lo[1] + j;
        const int k_global = box.lo[2] + k;

        const Flux inviscid = inviscidFaceFlux(q, i_left, j_global, k_global, 0,
                                               scheme, solver, gas);
        fx(0, k, j, face) = inviscid.rho;
        fx(1, k, j, face) = inviscid.mom_n;
        fx(2, k, j, face) = inviscid.mom_t1;
        fx(3, k, j, face) = inviscid.mom_t2;
        fx(4, k, j, face) = inviscid.energy;
      });
}

template <class ExecSpace>
void computeFaceFluxesY(const PatchField<ExecSpace>& q,
                        const CompressibleWorkspace<ExecSpace>& workspace,
                        const UniformGrid& grid, const GasProperties& gas,
                        Reconstruction scheme, RiemannSolver solver) {
  const Box box = grid.local;
  const int nx = box.length(0);
  const int ny = box.length(1);
  const int nz = box.length(2);
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{nz, ny + 1, nx}});
  auto fy = workspace.fy;

  Kokkos::parallel_for(
      "compressible_faces_y", policy, KOKKOS_LAMBDA(int k, int face, int i) {
        const int j_left = box.lo[1] + face - 1;
        const int i_global = box.lo[0] + i;
        const int k_global = box.lo[2] + k;

        const Flux inviscid = inviscidFaceFlux(q, i_global, j_left, k_global, 1,
                                               scheme, solver, gas);
        fy(0, k, face, i) = inviscid.rho;
        fy(1, k, face, i) = inviscid.mom_n;
        fy(2, k, face, i) = inviscid.mom_t1;
        fy(3, k, face, i) = inviscid.mom_t2;
        fy(4, k, face, i) = inviscid.energy;
      });
}

template <class ExecSpace>
void computeFaceFluxesZ(const PatchField<ExecSpace>& q,
                        const CompressibleWorkspace<ExecSpace>& workspace,
                        const UniformGrid& grid, const GasProperties& gas,
                        Reconstruction scheme, RiemannSolver solver) {
  const Box box = grid.local;
  const int nx = box.length(0);
  const int ny = box.length(1);
  const int nz = box.length(2);
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{nz + 1, ny, nx}});
  auto fz = workspace.fz;

  Kokkos::parallel_for(
      "compressible_faces_z", policy, KOKKOS_LAMBDA(int face, int j, int i) {
        const int k_left = box.lo[2] + face - 1;
        const int i_global = box.lo[0] + i;
        const int j_global = box.lo[1] + j;

        const Flux inviscid = inviscidFaceFlux(q, i_global, j_global, k_left, 2,
                                               scheme, solver, gas);
        fz(0, face, j, i) = inviscid.rho;
        fz(1, face, j, i) = inviscid.mom_n;
        fz(2, face, j, i) = inviscid.mom_t1;
        fz(3, face, j, i) = inviscid.mom_t2;
        fz(4, face, j, i) = inviscid.energy;
      });
}

// Fourth-order central derivative of one component of a cell-centered field.
template <class FieldType>
KOKKOS_INLINE_FUNCTION double fieldDerivative(const FieldType& field,
                                              int component, int i, int j,
                                              int k, int di, int dj, int dk,
                                              double h) {
  return central4Derivative(
      field(component, i - 2 * di, j - 2 * dj, k - 2 * dk),
      field(component, i - di, j - dj, k - dk),
      field(component, i + di, j + dj, k + dk),
      field(component, i + 2 * di, j + 2 * dj, k + 2 * dk), h);
}

// Assemble -div(F_inviscid) + div(tau) and the viscous energy terms. The
// viscous operator is cell-centered and fourth-order:
//
//   div(tau)_i = d tau_ij / dx_j
//   energy     = u . div(tau) + tau : grad(u) - div(q)
//
// If `dissipation_sink` is true, tau : grad(u) is removed from the energy
// equation, making the manufactured shear test isothermal.
template <class ExecSpace>
void assembleRhs(const PatchField<ExecSpace>& q,
                 const CompressibleWorkspace<ExecSpace>& workspace,
                 PatchField<ExecSpace>& rhs, const UniformGrid& grid,
                 const GasProperties& gas, bool dissipation_sink) {
  const Box box = grid.local;
  const double hx = grid.h[0];
  const double hy = grid.h[1];
  const double hz = grid.h[2];
  auto fx = workspace.fx;
  auto fy = workspace.fy;
  auto fz = workspace.fz;
  const auto grad_patch = workspace.grad;
  const auto stress = workspace.stress;
  const double conductivity =
      gas.mu * gas.gamma * gas.gas_constant / ((gas.gamma - 1.0) * gas.prandtl);

  const auto policy = makeMdrange3<ExecSpace>(
      Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});

  Kokkos::parallel_for(
      "compressible_assemble_rhs", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int il = i - box.lo[0];
        const int jl = j - box.lo[1];
        const int kl = k - box.lo[2];

        const double div_x = (fx(0, kl, jl, il + 1) - fx(0, kl, jl, il)) / hx;
        const double div_y = (fy(0, kl, jl + 1, il) - fy(0, kl, jl, il)) / hy;
        const double div_z = (fz(0, kl + 1, jl, il) - fz(0, kl, jl, il)) / hz;
        rhs(0, i, j, k) = -(div_x + div_y + div_z);

        const double mom_x = (fx(1, kl, jl, il + 1) - fx(1, kl, jl, il)) / hx +
                             (fy(1, kl, jl + 1, il) - fy(1, kl, jl, il)) / hy +
                             (fz(1, kl + 1, jl, il) - fz(1, kl, jl, il)) / hz;
        rhs(1, i, j, k) = -mom_x;

        const double mom_y = (fx(2, kl, jl, il + 1) - fx(2, kl, jl, il)) / hx +
                             (fy(2, kl, jl + 1, il) - fy(2, kl, jl, il)) / hy +
                             (fz(2, kl + 1, jl, il) - fz(2, kl, jl, il)) / hz;
        rhs(2, i, j, k) = -mom_y;

        const double mom_z = (fx(3, kl, jl, il + 1) - fx(3, kl, jl, il)) / hx +
                             (fy(3, kl, jl + 1, il) - fy(3, kl, jl, il)) / hy +
                             (fz(3, kl + 1, jl, il) - fz(3, kl, jl, il)) / hz;
        rhs(3, i, j, k) = -mom_z;

        const double energy = (fx(4, kl, jl, il + 1) - fx(4, kl, jl, il)) / hx +
                              (fy(4, kl, jl + 1, il) - fy(4, kl, jl, il)) / hy +
                              (fz(4, kl + 1, jl, il) - fz(4, kl, jl, il)) / hz;
        rhs(4, i, j, k) = -energy;

        const double div_tau_x =
            fieldDerivative(stress, 0, i, j, k, 1, 0, 0, hx) +
            fieldDerivative(stress, 3, i, j, k, 0, 1, 0, hy) +
            fieldDerivative(stress, 4, i, j, k, 0, 0, 1, hz);
        const double div_tau_y =
            fieldDerivative(stress, 3, i, j, k, 1, 0, 0, hx) +
            fieldDerivative(stress, 1, i, j, k, 0, 1, 0, hy) +
            fieldDerivative(stress, 5, i, j, k, 0, 0, 1, hz);
        const double div_tau_z =
            fieldDerivative(stress, 4, i, j, k, 1, 0, 0, hx) +
            fieldDerivative(stress, 5, i, j, k, 0, 1, 0, hy) +
            fieldDerivative(stress, 2, i, j, k, 0, 0, 1, hz);

        const double du_dx = grad_patch(0, i, j, k);
        const double du_dy = grad_patch(1, i, j, k);
        const double du_dz = grad_patch(2, i, j, k);
        const double dv_dx = grad_patch(3, i, j, k);
        const double dv_dy = grad_patch(4, i, j, k);
        const double dv_dz = grad_patch(5, i, j, k);
        const double dw_dx = grad_patch(6, i, j, k);
        const double dw_dy = grad_patch(7, i, j, k);
        const double dw_dz = grad_patch(8, i, j, k);
        const double dissipation =
            stress(0, i, j, k) * du_dx + stress(1, i, j, k) * dv_dy +
            stress(2, i, j, k) * dw_dz + stress(3, i, j, k) * (du_dy + dv_dx) +
            stress(4, i, j, k) * (du_dz + dw_dx) +
            stress(5, i, j, k) * (dv_dz + dw_dy);

        const double lap_temperature =
            fieldDerivative(grad_patch, 9, i, j, k, 1, 0, 0, hx) +
            fieldDerivative(grad_patch, 10, i, j, k, 0, 1, 0, hy) +
            fieldDerivative(grad_patch, 11, i, j, k, 0, 0, 1, hz);

        const PrimitiveState state = loadPrimitive(q, i, j, k, gas);
        rhs(1, i, j, k) += div_tau_x;
        rhs(2, i, j, k) += div_tau_y;
        rhs(3, i, j, k) += div_tau_z;
        rhs(4, i, j, k) += state.u * div_tau_x + state.v * div_tau_y +
                           state.w * div_tau_z + dissipation +
                           conductivity * lap_temperature;
        if (dissipation_sink) {
          rhs(4, i, j, k) -= dissipation;
        }
      });
}

// Full compressible RHS. The caller must have filled q's ghost cells.
template <class ExecSpace>
void compressibleRhs(const PatchField<ExecSpace>& q,
                     CompressibleWorkspace<ExecSpace>& workspace,
                     PatchField<ExecSpace>& rhs, const UniformGrid& grid,
                     const GasProperties& gas, Reconstruction scheme,
                     RiemannSolver solver, bool dissipation_sink = false) {
  const Box box = grid.local;
  computeGradients(q, workspace.grad, grid, gas, box.grow(2));
  computeStress(workspace.grad, workspace.stress, gas, box.grow(2));
  computeFaceFluxesX(q, workspace, grid, gas, scheme, solver);
  computeFaceFluxesY(q, workspace, grid, gas, scheme, solver);
  computeFaceFluxesZ(q, workspace, grid, gas, scheme, solver);
  assembleRhs(q, workspace, rhs, grid, gas, dissipation_sink);
}

template <class ExecSpace>
double maxWaveSpeed(const PatchField<ExecSpace>& q, const GasProperties& gas) {
  const Box box = q.box();
  const auto policy = makeMdrange3<ExecSpace>(
      Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  double local_max = 0.0;
  Kokkos::parallel_reduce(
      "compressible_max_speed", policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
        const PrimitiveState state = loadPrimitive(q, i, j, k, gas);
        double normal_speed = Kokkos::fabs(state.u);
        const double speed_v = Kokkos::fabs(state.v);
        const double speed_w = Kokkos::fabs(state.w);
        normal_speed = speed_v > normal_speed ? speed_v : normal_speed;
        normal_speed = speed_w > normal_speed ? speed_w : normal_speed;
        const double speed = normal_speed + soundSpeed(state, gas);
        acc = speed > acc ? speed : acc;
      },
      Kokkos::Max<double>{local_max});
  return mpiMax(local_max);
}

template <class ExecSpace>
double maxKinematicDiffusivity(const PatchField<ExecSpace>& q,
                               const GasProperties& gas) {
  if (gas.mu <= 0.0) {
    return 0.0;
  }
  const Box box = q.box();
  const auto policy = makeMdrange3<ExecSpace>(
      Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  double local_max = 0.0;
  Kokkos::parallel_reduce(
      "compressible_max_diffusivity", policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
        const PrimitiveState state = loadPrimitive(q, i, j, k, gas);
        const double nu_momentum = (4.0 / 3.0) * gas.mu / state.rho;
        const double nu_thermal = gas.mu / (state.rho * gas.prandtl);
        const double nu = nu_momentum > nu_thermal ? nu_momentum : nu_thermal;
        acc = nu > acc ? nu : acc;
      },
      Kokkos::Max<double>{local_max});
  return mpiMax(local_max);
}

}  // namespace compressible
}  // namespace saturn
