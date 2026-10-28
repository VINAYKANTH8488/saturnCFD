#pragma once

#include <Kokkos_Core.hpp>

#include "core/parallel.hpp"
#include "fields/patch_field.hpp"

namespace saturn {

template <class Field>
void field_combine(Field& destination, double a, const Field& x, double b,
                   const Field& y) {
  using ExecSpace = typename Field::execution_space;
  const Box box = destination.box();
  const auto policy = make_mdrange3<ExecSpace>(
      Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  Kokkos::parallel_for(
      "field_combine", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        for (int v = 0; v < destination.nvar(); ++v) {
          destination(v, i, j, k) = a * x(v, i, j, k) + b * y(v, i, j, k);
        }
      });
}

template <class Field>
void field_add_scaled(Field& destination, double scale, const Field& x) {
  using ExecSpace = typename Field::execution_space;
  const Box box = destination.box();
  const auto policy = make_mdrange3<ExecSpace>(
      Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  Kokkos::parallel_for(
      "field_add_scaled", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        for (int v = 0; v < destination.nvar(); ++v) {
          destination(v, i, j, k) += scale * x(v, i, j, k);
        }
      });
}

template <class Field>
void field_copy(const Field& source, Field& destination) {
  using ExecSpace = typename Field::execution_space;
  const Box box = source.box();
  const auto policy = make_mdrange3<ExecSpace>(
      Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  Kokkos::parallel_for(
      "field_copy", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        for (int v = 0; v < source.nvar(); ++v) {
          destination(v, i, j, k) = source(v, i, j, k);
        }
      });
}

// SSP-RK3. `compute_rhs(state, rhs, time)` must fill rhs with
// L(state, time). Boundary conditions that depend on time are evaluated at
// the stage time.
template <class Field, class ComputeRhs>
void advance_ssprk3(Field& q, Field& stage, Field& next, Field& rhs, double dt,
                    double time, ComputeRhs& compute_rhs) {
  compute_rhs(q, rhs, time);
  field_combine(stage, 1.0, q, dt, rhs);

  compute_rhs(stage, rhs, time + dt);
  field_combine(next, 0.75, q, 0.25, stage);
  field_add_scaled(next, 0.25 * dt, rhs);

  compute_rhs(next, rhs, time + 0.5 * dt);
  field_combine(q, 1.0 / 3.0, q, 2.0 / 3.0, next);
  field_add_scaled(q, (2.0 / 3.0) * dt, rhs);
}

// Classical RK4. `compute_rhs(state, rhs, time)` must fill rhs with
// L(state, time).
template <class Field, class ComputeRhs>
void advance_rk4(Field& q, Field& stage, Field& rhs, Field& k1, Field& k2,
                 Field& k3, double dt, double time, ComputeRhs& compute_rhs) {
  compute_rhs(q, rhs, time);
  field_copy(rhs, k1);
  field_combine(stage, 1.0, q, 0.5 * dt, k1);

  compute_rhs(stage, rhs, time + 0.5 * dt);
  field_copy(rhs, k2);
  field_combine(stage, 1.0, q, 0.5 * dt, k2);

  compute_rhs(stage, rhs, time + 0.5 * dt);
  field_copy(rhs, k3);
  field_combine(stage, 1.0, q, dt, k3);

  compute_rhs(stage, rhs, time + dt);
  field_add_scaled(q, dt / 6.0, k1);
  field_add_scaled(q, dt / 3.0, k2);
  field_add_scaled(q, dt / 3.0, k3);
  field_add_scaled(q, dt / 6.0, rhs);
}

}  // namespace saturn
