#pragma once

#ifdef SATURN_ENABLE_MPI
#include <mpi.h>
#endif

namespace saturn {

inline int mpi_rank() {
#ifdef SATURN_ENABLE_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif
}

inline int mpi_size() {
#ifdef SATURN_ENABLE_MPI
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
#else
  return 1;
#endif
}

inline double mpi_sum(double value) {
#ifdef SATURN_ENABLE_MPI
  double result = 0.0;
  MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return result;
#else
  return value;
#endif
}

inline double mpi_max(double value) {
#ifdef SATURN_ENABLE_MPI
  double result = 0.0;
  MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return result;
#else
  return value;
#endif
}

inline bool mpi_all(bool value) {
#ifdef SATURN_ENABLE_MPI
  int local = value ? 1 : 0;
  int result = 0;
  MPI_Allreduce(&local, &result, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  return result != 0;
#else
  return value;
#endif
}

}  // namespace saturn
