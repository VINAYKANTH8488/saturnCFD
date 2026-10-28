// Boundary-condition unit test: periodic wrap, transmissive clamp, and
// slip-wall normal-momentum reflection, including corners.

#include <Kokkos_Core.hpp>

#include "fields/boundary_conditions.hpp"
#include "fields/patch_field.hpp"

#include <cmath>
#include <cstdio>

namespace {

using saturn::BoundaryType;

void fill_interior(saturn::PatchField<Kokkos::DefaultExecutionSpace>& q) {
  const saturn::Box box = q.box();
  auto host = Kokkos::create_mirror_view(q.data());
  for (int k = box.lo[2]; k <= box.hi[2]; ++k) {
    for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        const int ip = i - box.lo[0] + q.ng();
        const int jp = j - box.lo[1] + q.ng();
        const int kp = k - box.lo[2] + q.ng();
        host(0, kp, jp, ip) = 2.0;
        host(1, kp, jp, ip) = 3.0;
        host(2, kp, jp, ip) = 4.0;
        host(3, kp, jp, ip) = 5.0;
        host(4, kp, jp, ip) = 6.0;
      }
    }
  }
  Kokkos::deep_copy(q.data(), host);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    using exec_space = Kokkos::DefaultExecutionSpace;
    const saturn::Box box = saturn::Box::from_extents(8, 8, 8);
    saturn::PatchField<exec_space> q(box, 5, 4, "q");
    fill_interior(q);

    BoundaryType bc[3][2];
    bc[0][0] = bc[0][1] = BoundaryType::Periodic;
    bc[1][0] = bc[1][1] = BoundaryType::Transmissive;
    bc[2][0] = bc[2][1] = BoundaryType::SlipWall;
    saturn::apply_boundary_conditions(q, bc);

    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    q.data());
    const int ng = q.ng();
    int failures = 0;
    const auto at = [&](int v, int i, int j, int k) {
      return host(v, k - box.lo[2] + ng, j - box.lo[1] + ng,
                  i - box.lo[0] + ng);
    };
    const auto expect = [&](const char* what, double actual, double expected) {
      if (std::fabs(actual - expected) > 1.0e-12) {
        std::printf("FAIL %s: actual %.6f expected %.6f\n", what, actual,
                    expected);
        ++failures;
      }
    };

    // Periodic x: ghost at lo-1 is the interior cell at hi.
    expect("periodic rho", at(0, box.lo[0] - 1, 3, 3), 2.0);
    expect("periodic momx", at(1, box.lo[0] - 1, 3, 3), 3.0);

    // Transmissive y: ghost at hi+1 is the interior cell at hi.
    expect("transmissive rho", at(0, 3, box.hi[1] + 1, 3), 2.0);
    expect("transmissive momy", at(2, 3, box.hi[1] + 1, 3), 4.0);

    // Slip wall z: normal momentum flips, other components copy.
    expect("slip rho", at(0, 3, 3, box.lo[2] - 1), 2.0);
    expect("slip momx", at(1, 3, 3, box.lo[2] - 1), 3.0);
    expect("slip momz", at(3, 3, 3, box.lo[2] - 1), -5.0);

    // Corner: y transmissive + z slip.
    expect("corner rho", at(0, 3, box.hi[1] + 1, box.lo[2] - 2), 2.0);
    expect("corner momz", at(3, 3, box.hi[1] + 1, box.lo[2] - 2), -5.0);

    if (failures == 0) {
      std::printf("boundary-condition test passed\n");
    } else {
      std::printf("boundary-condition test FAILED: %d\n", failures);
      status = 1;
    }
  }
  Kokkos::finalize();
  return status;
}
