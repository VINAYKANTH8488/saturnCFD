add_executable(saturn_test_smoke test_smoke.cpp)
target_link_libraries(saturn_test_smoke PRIVATE saturn_core)

add_executable(saturn_test_halo test_halo.cpp)
target_link_libraries(saturn_test_halo PRIVATE saturn_core)

add_executable(saturn_test_io test_io.cpp)
target_link_libraries(saturn_test_io PRIVATE saturn_core)

add_executable(saturn_test_reconstruction test_reconstruction.cpp)
target_link_libraries(saturn_test_reconstruction PRIVATE saturn_core)

add_executable(saturn_test_riemann test_riemann.cpp)
target_link_libraries(saturn_test_riemann PRIVATE saturn_core)

add_executable(saturn_test_box_exchange test_box_exchange.cpp)
target_link_libraries(saturn_test_box_exchange PRIVATE saturn_core)

add_executable(saturn_test_boundary_conditions test_boundary_conditions.cpp)
target_link_libraries(saturn_test_boundary_conditions PRIVATE saturn_core)

add_test(NAME saturn.smoke COMMAND saturn_test_smoke)
set_tests_properties(saturn.smoke PROPERTIES LABELS "unit")

add_test(NAME saturn.halo COMMAND saturn_test_halo)
set_tests_properties(saturn.halo PROPERTIES LABELS "unit;halo")

add_test(NAME saturn.io COMMAND saturn_test_io)
set_tests_properties(saturn.io PROPERTIES LABELS "unit;io")

add_test(NAME saturn.reconstruction COMMAND saturn_test_reconstruction)
set_tests_properties(saturn.reconstruction PROPERTIES LABELS "unit;numerics")

add_test(NAME saturn.riemann COMMAND saturn_test_riemann)
set_tests_properties(saturn.riemann PROPERTIES LABELS "unit;numerics")

add_test(NAME saturn.box_exchange COMMAND saturn_test_box_exchange)
set_tests_properties(saturn.box_exchange PROPERTIES LABELS "unit;halo")

add_test(NAME saturn.boundary_conditions COMMAND saturn_test_boundary_conditions)
set_tests_properties(saturn.boundary_conditions
  PROPERTIES LABELS "unit;numerics")

add_test(NAME saturn.scalar_mms
  COMMAND saturn_scalar_mms --n-list 16 32 --nu 0.01 --tend 0.2 --check)
set_tests_properties(saturn.scalar_mms PROPERTIES LABELS "verification")

add_test(NAME saturn.sod
  COMMAND saturn_euler1d --n 400 --check)
set_tests_properties(saturn.sod PROPERTIES LABELS "verification;sod")

add_test(NAME saturn.sod_teno6_lowmach
  COMMAND saturn_euler1d --n 400 --scheme teno6 --solver hllc_lm --check)
set_tests_properties(saturn.sod_teno6_lowmach
  PROPERTIES LABELS "verification;sod")

add_test(NAME saturn.compressible_entropy
  COMMAND saturn_compressible_mms --case entropy --n-list 8 16 32
          --scheme central4 --time rk4 --cfl 0.3 --check)
set_tests_properties(saturn.compressible_entropy
  PROPERTIES LABELS "verification;compressible")

add_test(NAME saturn.compressible_shear
  COMMAND saturn_compressible_mms --case shear --n-list 8 16 32
          --scheme central4 --time rk4 --cfl 0.3 --check)
set_tests_properties(saturn.compressible_shear
  PROPERTIES LABELS "verification;compressible")

add_test(NAME saturn.compressible_weno5
  COMMAND saturn_compressible_mms --case entropy --n-list 8 16
          --scheme weno5z --time rk4 --cfl 0.3 --min-order 4.5 --check)
set_tests_properties(saturn.compressible_weno5
  PROPERTIES LABELS "verification;compressible")

add_test(NAME saturn.compressible_energy
  COMMAND saturn_compressible_mms --case entropy --n 16 --tend 0.5
          --scheme weno5z --time rk4 --cfl 0.3 --energy-tol 1e-11 --check)
set_tests_properties(saturn.compressible_energy
  PROPERTIES LABELS "verification;compressible")

add_test(NAME saturn.compressible_tgv
  COMMAND saturn_compressible_sim --case tgv --n 16 --tend 0.05 --check)
set_tests_properties(saturn.compressible_tgv
  PROPERTIES LABELS "verification;compressible")

add_test(NAME saturn.compressible_tgv_lowmach
  COMMAND saturn_compressible_sim --case tgv --n 16 --tend 0.01
          --mach 0.1 --check)
set_tests_properties(saturn.compressible_tgv_lowmach
  PROPERTIES LABELS "verification;compressible")

add_test(NAME saturn.compressible_acoustic
  COMMAND saturn_compressible_sim --case acoustic --n 16 --tend 0.05 --check)
set_tests_properties(saturn.compressible_acoustic
  PROPERTIES LABELS "verification;compressible")

add_test(NAME saturn.compressible_vortex
  COMMAND saturn_compressible_sim --case vortex --nx 16 --ny 16 --nz 4
          --tend 0.05 --check)
set_tests_properties(saturn.compressible_vortex
  PROPERTIES LABELS "verification;compressible")

add_test(NAME saturn.compressible_double_mach
  COMMAND saturn_compressible_sim --case dm --nx 32 --ny 16 --nz 4
          --tend 0.005 --cfl 0.2 --check)
set_tests_properties(saturn.compressible_double_mach
  PROPERTIES LABELS "verification;compressible")

if(SATURN_HAVE_MPI)
  add_test(NAME saturn.halo_mpi
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_test_halo>
            --kokkos-num-threads=1)
  set_tests_properties(saturn.halo_mpi PROPERTIES
    LABELS "unit;halo;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.box_exchange_mpi2
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_test_box_exchange>
            --kokkos-num-threads=1)
  set_tests_properties(saturn.box_exchange_mpi2 PROPERTIES
    LABELS "unit;halo;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.box_exchange_mpi4
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_test_box_exchange>
            --kokkos-num-threads=1)
  set_tests_properties(saturn.box_exchange_mpi4 PROPERTIES
    LABELS "unit;halo;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.compressible_tgv_mpi2
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_compressible_sim>
            --case tgv --n 16 --tend 0.05 --check
            --kokkos-num-threads=1)
  set_tests_properties(saturn.compressible_tgv_mpi2 PROPERTIES
    LABELS "verification;compressible;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.compressible_tgv_mpi4
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_compressible_sim>
            --case tgv --n 16 --tend 0.05 --check
            --kokkos-num-threads=1)
  set_tests_properties(saturn.compressible_tgv_mpi4 PROPERTIES
    LABELS "verification;compressible;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.compressible_entropy_mpi2
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_compressible_mms>
            --case entropy --n-list 8 16 --scheme central4 --time rk4
            --cfl 0.3 --check --kokkos-num-threads=1)
  set_tests_properties(saturn.compressible_entropy_mpi2 PROPERTIES
    LABELS "verification;compressible;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.compressible_double_mach_mpi2
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_compressible_sim>
            --case dm --nx 32 --ny 16 --nz 4 --tend 0.005 --cfl 0.2
            --check --kokkos-num-threads=1)
  set_tests_properties(saturn.compressible_double_mach_mpi2 PROPERTIES
    LABELS "verification;compressible;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.compressible_double_mach_mpi4
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_compressible_sim>
            --case dm --nx 64 --ny 16 --nz 4 --tend 0.02 --cfl 0.2
            --check --kokkos-num-threads=1)
  set_tests_properties(saturn.compressible_double_mach_mpi4 PROPERTIES
    LABELS "verification;compressible;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.compressible_shear_mpi2
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_compressible_mms>
            --case shear --n-list 8 16 --scheme central4 --time rk4
            --cfl 0.3 --check --kokkos-num-threads=1)
  set_tests_properties(saturn.compressible_shear_mpi2 PROPERTIES
    LABELS "verification;compressible;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.scalar_mms_mpi
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_scalar_mms>
            --n-list 16 32 --nu 0.01 --tend 0.2 --check
            --kokkos-num-threads=1)
  set_tests_properties(saturn.scalar_mms_mpi PROPERTIES
    LABELS "verification;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.scalar_mms_overlap_mpi
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_scalar_mms>
            --n-list 16 32 --nu 0.01 --tend 0.2 --overlap --check
            --kokkos-num-threads=1)
  set_tests_properties(saturn.scalar_mms_overlap_mpi PROPERTIES
    LABELS "verification;mpi;overlap"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.io_mpi
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_test_io>
            --kokkos-num-threads=1)
  set_tests_properties(saturn.io_mpi PROPERTIES
    LABELS "unit;io;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
endif()
