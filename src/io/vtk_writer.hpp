#pragma once

#include <Kokkos_Core.hpp>

#include "fields/patch_field.hpp"
#include "mesh/box.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#ifdef SATURN_ENABLE_MPI
#include <mpi.h>
#endif

namespace saturn {

inline std::string rankSuffix(int rank) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), ".rank%04d", rank);
  return buffer;
}

inline void ensureParent(const std::string& path) {
  const std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
}

// Write a cell-centered multi-variable field as VTK ImageData.
//
// Single rank:   <base>.vti
// Multiple ranks: <base>.rankNNNN.vti + <base>.pvti (rank 0)
//
// `names` must contain at least q.nvar() entries.
template <class ExecSpace>
bool writeVtiFields(const std::string& base, const PatchField<ExecSpace>& q,
                    const std::vector<std::string>& names,
                    const std::array<double, 3>& origin,
                    const std::array<double, 3>& spacing, const Box& global,
                    double time) {
  (void)time;  // a time-series .pvd can be added later
  const Box local = q.box();
  const int ng = q.ng();
  const int nvar = q.nvar();
  if (nvar < 1) {
    return false;
  }

  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q.data());

  int rank = 0;
  int size = 1;
#ifdef SATURN_ENABLE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

  const std::string vti =
      (size > 1) ? (base + rankSuffix(rank) + ".vti") : (base + ".vti");
  ensureParent(vti);
  std::ofstream out(vti);
  if (!out) {
    std::fprintf(stderr, "could not open %s for writing\n", vti.c_str());
    return false;
  }

  const int whole[6] = {global.lo[0],     global.hi[0] + 1, global.lo[1],
                        global.hi[1] + 1, global.lo[2],     global.hi[2] + 1};
  const int piece[6] = {local.lo[0],     local.hi[0] + 1, local.lo[1],
                        local.hi[1] + 1, local.lo[2],     local.hi[2] + 1};

  auto nameOf = [&](int v) -> const std::string& {
    static const std::string fallback = "var";
    if (v < static_cast<int>(names.size())) {
      return names[static_cast<std::size_t>(v)];
    }
    return fallback;
  };

  out << std::setprecision(17);
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\" version=\"0.1\" "
         "byte_order=\"LittleEndian\">\n"
      << "  <ImageData WholeExtent=\"" << whole[0] << " " << whole[1] << " "
      << whole[2] << " " << whole[3] << " " << whole[4] << " " << whole[5]
      << "\" Origin=\"" << origin[0] << " " << origin[1] << " " << origin[2]
      << "\" Spacing=\"" << spacing[0] << " " << spacing[1] << " " << spacing[2]
      << "\">\n"
      << "    <Piece Extent=\"" << piece[0] << " " << piece[1] << " "
      << piece[2] << " " << piece[3] << " " << piece[4] << " " << piece[5]
      << "\">\n"
      << "      <CellData Scalars=\"" << nameOf(0) << "\">\n";

  for (int v = 0; v < nvar; ++v) {
    out << "        <DataArray type=\"Float64\" Name=\"" << nameOf(v)
        << "\" format=\"ascii\">\n         ";
    for (int k = local.lo[2]; k <= local.hi[2]; ++k) {
      for (int j = local.lo[1]; j <= local.hi[1]; ++j) {
        for (int i = local.lo[0]; i <= local.hi[0]; ++i) {
          out << " "
              << host(v, k - local.lo[2] + ng, j - local.lo[1] + ng,
                      i - local.lo[0] + ng);
        }
      }
    }
    out << "\n        </DataArray>\n";
  }
  out << "      </CellData>\n"
      << "    </Piece>\n"
      << "  </ImageData>\n"
      << "</VTKFile>\n";

  if (size == 1) {
    return true;
  }

  std::vector<int> extents;
#ifdef SATURN_ENABLE_MPI
  int local_extent[6] = {piece[0], piece[1], piece[2],
                         piece[3], piece[4], piece[5]};
  if (rank == 0) {
    extents.resize(static_cast<std::size_t>(6) * size);
  }
  MPI_Gather(local_extent, 6, MPI_INT, (rank == 0) ? extents.data() : nullptr,
             6, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  if (rank == 0) {
    const std::string pvti = base + ".pvti";
    std::ofstream collection(pvti);
    if (!collection) {
      std::fprintf(stderr, "could not open %s for writing\n", pvti.c_str());
      return false;
    }
    collection << std::setprecision(17);
    collection << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"PImageData\" version=\"0.1\" "
                  "byte_order=\"LittleEndian\">\n"
               << "  <PImageData WholeExtent=\"" << whole[0] << " " << whole[1]
               << " " << whole[2] << " " << whole[3] << " " << whole[4] << " "
               << whole[5] << "\" GhostLevel=\"0\" "
               << "Origin=\"" << origin[0] << " " << origin[1] << " "
               << origin[2] << "\" Spacing=\"" << spacing[0] << " "
               << spacing[1] << " " << spacing[2] << "\">\n"
               << "    <PCellData Scalars=\"" << nameOf(0) << "\">\n";
    for (int v = 0; v < nvar; ++v) {
      collection << "      <PDataArray type=\"Float64\" Name=\"" << nameOf(v)
                 << "\"/>\n";
    }
    collection << "    </PCellData>\n";
    for (int r = 0; r < size; ++r) {
      const int* e = &extents[static_cast<std::size_t>(r) * 6];
      collection << "    <Piece Extent=\"" << e[0] << " " << e[1] << " " << e[2]
                 << " " << e[3] << " " << e[4] << " " << e[5] << "\" Source=\""
                 << base << rankSuffix(r) << ".vti\"/>\n";
    }
    collection << "  </PImageData>\n"
               << "</VTKFile>\n";
  }

  return true;
}

// Scalar convenience wrapper.
template <class ExecSpace>
bool writeVti(const std::string& base, const PatchField<ExecSpace>& q,
              const std::array<double, 3>& origin,
              const std::array<double, 3>& spacing, const Box& global,
              double time) {
  return writeVtiFields(base, q, std::vector<std::string>{"c"}, origin, spacing,
                        global, time);
}

}  // namespace saturn
