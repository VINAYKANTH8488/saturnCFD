#pragma once

#include <Kokkos_Core.hpp>

#include "fields/patch_field.hpp"
#include "io/vtk_writer.hpp"
#include "mesh/decomposition.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <type_traits>

namespace saturn {

struct CheckpointHeader {
  char magic[8] = {'S', 'A', 'T', 'U', 'R', 'N', 'C', 'K'};
  int version = 1;
  int rank = 0;
  int size = 1;
  int nvar = 1;
  int ng = 1;
  int local_lo[3] = {0, 0, 0};
  int local_hi[3] = {-1, -1, -1};
  int global_lo[3] = {0, 0, 0};
  int global_hi[3] = {-1, -1, -1};
  int step = 0;
  double time = 0.0;
};
static_assert(std::is_trivially_copyable_v<CheckpointHeader>);

template <class ExecSpace>
bool writeCheckpoint(const std::string& base, const PatchField<ExecSpace>& q,
                     int step, double time,
                     const Decomposition& decomposition) {
  const Box box = q.box();
  CheckpointHeader header;
  header.rank = decomposition.rank;
  header.size = decomposition.size;
  header.nvar = q.nvar();
  header.ng = q.ng();
  for (int d = 0; d < 3; ++d) {
    header.local_lo[d] = box.lo[d];
    header.local_hi[d] = box.hi[d];
    header.global_lo[d] = decomposition.global.lo[d];
    header.global_hi[d] = decomposition.global.hi[d];
  }
  header.step = step;
  header.time = time;

  const std::string path = base + rankSuffix(decomposition.rank) + ".bin";
  ensureParent(path);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "could not open %s for writing\n", path.c_str());
    return false;
  }
  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
  if (!out) {
    return false;
  }

  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q.data());
  for (int v = 0; v < q.nvar(); ++v) {
    for (int k = box.lo[2]; k <= box.hi[2]; ++k) {
      for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          const double value =
              host(v, k - box.lo[2] + q.ng(), j - box.lo[1] + q.ng(),
                   i - box.lo[0] + q.ng());
          out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
      }
    }
  }
  return static_cast<bool>(out);
}

template <class ExecSpace>
bool readCheckpoint(const std::string& base, PatchField<ExecSpace>& q,
                    int& step, double& time,
                    const Decomposition& decomposition) {
  const std::string path = base + rankSuffix(decomposition.rank) + ".bin";
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }

  CheckpointHeader header;
  in.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!in) {
    return false;
  }
  const bool magic_ok = header.magic[0] == 'S' && header.magic[1] == 'A' &&
                        header.magic[2] == 'T' && header.magic[3] == 'U' &&
                        header.magic[4] == 'R' && header.magic[5] == 'N' &&
                        header.magic[6] == 'C' && header.magic[7] == 'K';
  if (!magic_ok || header.version != 1 || header.rank != decomposition.rank ||
      header.size != decomposition.size || header.nvar != q.nvar() ||
      header.ng != q.ng()) {
    return false;
  }
  const Box box = q.box();
  for (int d = 0; d < 3; ++d) {
    if (header.local_lo[d] != box.lo[d] || header.local_hi[d] != box.hi[d] ||
        header.global_lo[d] != decomposition.global.lo[d] ||
        header.global_hi[d] != decomposition.global.hi[d]) {
      return false;
    }
  }

  auto host = Kokkos::create_mirror_view(q.data());
  for (int v = 0; v < q.nvar(); ++v) {
    for (int k = box.lo[2]; k <= box.hi[2]; ++k) {
      for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          double value = 0.0;
          in.read(reinterpret_cast<char*>(&value), sizeof(value));
          if (!in) {
            return false;
          }
          host(v, k - box.lo[2] + q.ng(), j - box.lo[1] + q.ng(),
               i - box.lo[0] + q.ng()) = value;
        }
      }
    }
  }
  Kokkos::deep_copy(q.data(), host);
  step = header.step;
  time = header.time;
  return true;
}

}  // namespace saturn
