#pragma once

#include <Kokkos_Core.hpp>

#include "core/parallel.hpp"
#include "fields/patch_field.hpp"

namespace saturn {

template <class Field>
void fieldCombine(Field& destination, double a, const Field& x, double b,
                  const Field& y) {
  using ExecSpace = typename Field::execution_space;
  const Box box = destination.box();
  const auto policy = makeMdrange3<ExecSpace>(
      Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  auto dest = destination.data();
  auto xd = x.data();
  auto yd = y.data();
  const int nvar = destination.nvar();
  const int dng = destination.ng();
  const int xng = x.ng();
  const int yng = y.ng();
  const int lo0 = box.lo[0];
  const int lo1 = box.lo[1];
  const int lo2 = box.lo[2];
  Kokkos::parallel_for(
      "fieldCombine", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int ip = i - lo0 + dng;
        const int jp = j - lo1 + dng;
        const int kp = k - lo2 + dng;
        const int xip = i - lo0 + xng;
        const int xjp = j - lo1 + xng;
        const int xkp = k - lo2 + xng;
        const int yip = i - lo0 + yng;
        const int yjp = j - lo1 + yng;
        const int ykp = k - lo2 + yng;
        for (int v = 0; v < nvar; ++v) {
          dest(v, kp, jp, ip) =
              a * xd(v, xkp, xjp, xip) + b * yd(v, ykp, yjp, yip);
        }
      });
}

template <class Field>
void fieldAddScaled(Field& destination, double scale, const Field& x) {
  using ExecSpace = typename Field::execution_space;
  const Box box = destination.box();
  const auto policy = makeMdrange3<ExecSpace>(
      Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  auto dest = destination.data();
  auto xd = x.data();
  const int nvar = destination.nvar();
  const int dng = destination.ng();
  const int xng = x.ng();
  const int lo0 = box.lo[0];
  const int lo1 = box.lo[1];
  const int lo2 = box.lo[2];
  Kokkos::parallel_for(
      "fieldAddScaled", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int ip = i - lo0 + dng;
        const int jp = j - lo1 + dng;
        const int kp = k - lo2 + dng;
        const int xip = i - lo0 + xng;
        const int xjp = j - lo1 + xng;
        const int xkp = k - lo2 + xng;
        for (int v = 0; v < nvar; ++v) {
          dest(v, kp, jp, ip) += scale * xd(v, xkp, xjp, xip);
        }
      });
}

template <class Field>
void fieldCopy(const Field& source, Field& destination) {
  using ExecSpace = typename Field::execution_space;
  const Box box = source.box();
  const auto policy = makeMdrange3<ExecSpace>(
      Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  auto src = source.data();
  auto dest = destination.data();
  const int nvar = source.nvar();
  const int sng = source.ng();
  const int dng = destination.ng();
  const int lo0 = box.lo[0];
  const int lo1 = box.lo[1];
  const int lo2 = box.lo[2];
  Kokkos::parallel_for(
      "fieldCopy", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int sip = i - lo0 + sng;
        const int sjp = j - lo1 + sng;
        const int skp = k - lo2 + sng;
        const int dip = i - lo0 + dng;
        const int djp = j - lo1 + dng;
        const int dkp = k - lo2 + dng;
        for (int v = 0; v < nvar; ++v) {
          dest(v, dkp, djp, dip) = src(v, skp, sjp, sip);
        }
      });
}

// SSP-RK3. `computeRhs(state, rhs, time)` must fill rhs with
// L(state, time). Boundary conditions that depend on time are evaluated at
// the stage time.
template <class Field, class ComputeRhs>
void advanceSsprk3(Field& q, Field& stage, Field& next, Field& rhs, double dt,
                   double time, ComputeRhs& computeRhs) {
  computeRhs(q, rhs, time);
  fieldCombine(stage, 1.0, q, dt, rhs);

  computeRhs(stage, rhs, time + dt);
  fieldCombine(next, 0.75, q, 0.25, stage);
  fieldAddScaled(next, 0.25 * dt, rhs);

  computeRhs(next, rhs, time + 0.5 * dt);
  fieldCombine(q, 1.0 / 3.0, q, 2.0 / 3.0, next);
  fieldAddScaled(q, (2.0 / 3.0) * dt, rhs);
}

// Classical RK4. `computeRhs(state, rhs, time)` must fill rhs with
// L(state, time).
template <class Field, class ComputeRhs>
void advanceRk4(Field& q, Field& stage, Field& rhs, Field& k1, Field& k2,
                Field& k3, double dt, double time, ComputeRhs& computeRhs) {
  computeRhs(q, rhs, time);
  fieldCopy(rhs, k1);
  fieldCombine(stage, 1.0, q, 0.5 * dt, k1);

  computeRhs(stage, rhs, time + 0.5 * dt);
  fieldCopy(rhs, k2);
  fieldCombine(stage, 1.0, q, 0.5 * dt, k2);

  computeRhs(stage, rhs, time + 0.5 * dt);
  fieldCopy(rhs, k3);
  fieldCombine(stage, 1.0, q, dt, k3);

  computeRhs(stage, rhs, time + dt);
  fieldAddScaled(q, dt / 6.0, k1);
  fieldAddScaled(q, dt / 3.0, k2);
  fieldAddScaled(q, dt / 3.0, k3);
  fieldAddScaled(q, dt / 6.0, rhs);
}

}  // namespace saturn
