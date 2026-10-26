#pragma once

#include <Kokkos_Core.hpp>

#include <string>

#include "mesh/box.hpp"

namespace saturn {

// Cell-centered field on one box, with ng ghost layers on every side.
//
// Layout: [variable][k][j][i], i fastest (LayoutRight), so the inner dimension
// is contiguous and vectorizes/coalesces.
template <class ExecSpace>
class PatchField {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using view_type =
      Kokkos::View<double****, Kokkos::LayoutRight, memory_space>;

  PatchField() = default;

  PatchField(const Box& box, int nvar, int ng,
             const std::string& label = "q")
      : box_(box),
        nvar_(nvar),
        ng_(ng),
        data_(label, nvar, box.length(2) + 2 * ng,
              box.length(1) + 2 * ng, box.length(0) + 2 * ng) {}

  KOKKOS_INLINE_FUNCTION
  double& operator()(int v, int i, int j, int k) const {
    return data_(v, k - box_.lo[2] + ng_, j - box_.lo[1] + ng_,
                 i - box_.lo[0] + ng_);
  }

  const Box& box() const { return box_; }
  int nvar() const { return nvar_; }
  int ng() const { return ng_; }
  view_type& data() { return data_; }
  const view_type& data() const { return data_; }

  // Periodic ghost fill for a patch that spans the full periodic domain.
  // Multi-rank exchanges will replace this with halo messages.
  void fill_periodic() {
    const int n0 = box_.length(0);
    const int n1 = box_.length(1);
    const int n2 = box_.length(2);
    const int np0 = n0 + 2 * ng_;
    const int np1 = n1 + 2 * ng_;
    const int np2 = n2 + 2 * ng_;

    using policy_type =
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;
    auto data = data_;
    const int ng = ng_;
    const int nvar = nvar_;

    Kokkos::parallel_for(
        "periodic_fill", policy_type({0, 0, 0}, {np2, np1, np0}),
        KOKKOS_LAMBDA(int kp, int jp, int ip) {
          const int ii = wrap(ip - ng, n0);
          const int jj = wrap(jp - ng, n1);
          const int kk = wrap(kp - ng, n2);
          for (int v = 0; v < nvar; ++v) {
            data(v, kp, jp, ip) = data(v, kk + ng, jj + ng, ii + ng);
          }
        });
  }

 private:
  KOKKOS_INLINE_FUNCTION
  static int wrap(int i, int n) { return ((i % n) + n) % n; }

  Box box_{};
  int nvar_ = 1;
  int ng_ = 0;
  view_type data_{};
};

}  // namespace saturn
