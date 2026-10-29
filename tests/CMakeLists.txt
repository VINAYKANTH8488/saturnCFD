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

add_executable(saturn_test_poisson test_poisson.cpp)
target_link_libraries(saturn_test_poisson PRIVATE saturn_core)

add_executable(saturn_test_projection test_projection.cpp)
target_link_libraries(saturn_test_projection PRIVATE saturn_core)

add_executable(saturn_test_fft_poisson test_fft_poisson.cpp)
target_link_libraries(saturn_test_fft_poisson PRIVATE saturn_core)

add_executable(saturn_test_vof test_vof.cpp)
target_link_libraries(saturn_test_vof PRIVATE saturn_core)

add_executable(saturn_test_vof3d test_vof3d.cpp)
target_link_libraries(saturn_test_vof3d PRIVATE saturn_core)

add_executable(saturn_test_vof_curvature test_vof_curvature.cpp)
target_link_libraries(saturn_test_vof_curvature PRIVATE saturn_core)

add_executable(saturn_test_vof_curvature3d test_vof_curvature3d.cpp)
target_link_libraries(saturn_test_vof_curvature3d PRIVATE saturn_core)

add_executable(saturn_test_vof_redistribute test_vof_redistribute.cpp)
target_link_libraries(saturn_test_vof_redistribute PRIVATE saturn_core)

add_executable(saturn_test_twophase test_twophase.cpp)
target_link_libraries(saturn_test_twophase PRIVATE saturn_core)

add_executable(saturn_test_vof_curvature_levelset test_vof_curvature_levelset.cpp)
target_link_libraries(saturn_test_vof_curvature_levelset PRIVATE saturn_core)

add_executable(saturn_test_variable_poisson test_variable_poisson.cpp)
target_link_libraries(saturn_test_variable_poisson PRIVATE saturn_core)

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

add_test(NAME saturn.poisson COMMAND saturn_test_poisson)
set_tests_properties(saturn.poisson PROPERTIES LABELS "unit;numerics;poisson")

add_test(NAME saturn.projection COMMAND saturn_test_projection)
set_tests_properties(saturn.projection
  PROPERTIES LABELS "unit;numerics;poisson")

add_test(NAME saturn.fft_poisson COMMAND saturn_test_fft_poisson)
set_tests_properties(saturn.fft_poisson PROPERTIES LABELS "unit;numerics;poisson")

add_test(NAME saturn.vof COMMAND saturn_test_vof)
set_tests_properties(saturn.vof PROPERTIES LABELS "unit;numerics;vof")

add_test(NAME saturn.vof3d COMMAND saturn_test_vof3d)
set_tests_properties(saturn.vof3d PROPERTIES LABELS "unit;numerics;vof")

add_test(NAME saturn.vof_curvature COMMAND saturn_test_vof_curvature)
set_tests_properties(saturn.vof_curvature
  PROPERTIES LABELS "unit;numerics;vof")

add_test(NAME saturn.vof_curvature3d COMMAND saturn_test_vof_curvature3d)
set_tests_properties(saturn.vof_curvature3d
  PROPERTIES LABELS "unit;numerics;vof")

add_test(NAME saturn.vof_redistribute COMMAND saturn_test_vof_redistribute)
set_tests_properties(saturn.vof_redistribute
  PROPERTIES LABELS "unit;numerics;vof")

add_test(NAME saturn.twophase COMMAND saturn_test_twophase)
set_tests_properties(saturn.twophase
  PROPERTIES LABELS "unit;numerics;vof;twophase")

add_test(NAME saturn.vof_curvature_levelset
  COMMAND saturn_test_vof_curvature_levelset)
set_tests_properties(saturn.vof_curvature_levelset
  PROPERTIES LABELS "unit;numerics;vof")

add_test(NAME saturn.variable_poisson COMMAND saturn_test_variable_poisson)
set_tests_properties(saturn.variable_poisson
  PROPERTIES LABELS "unit;numerics;twophase")

add_test(NAME saturn.incompressible_tgv
  COMMAND saturn_incompressible_sim --n-list 16 32 64 --tend 0.5
          --nu 0.05 --cfl 0.3 --check)
set_tests_properties(saturn.incompressible_tgv
  PROPERTIES LABELS "verification;incompressible")

if(SATURN_HAVE_FFTW)
  add_test(NAME saturn.incompressible_tgv_fft
    COMMAND saturn_incompressible_sim --n-list 16 32 64 --tend 0.5
            --nu 0.05 --cfl 0.3 --poisson fft --check)
  set_tests_properties(saturn.incompressible_tgv_fft
    PROPERTIES LABELS "verification;incompressible;fft")
endif()

add_test(NAME saturn.incompressible_abc
  COMMAND saturn_incompressible_sim --case abc --n-list 16 32 --tend 0.5
          --nu 0.1 --cfl 0.3 --check)
set_tests_properties(saturn.incompressible_abc
  PROPERTIES LABELS "verification;incompressible")

add_test(NAME saturn.channel_couette
  COMMAND saturn_channel_sim --case couette --n-list 16 32 --tend 0.1
          --nu 0.1 --u-top 1 --check)
set_tests_properties(saturn.channel_couette
  PROPERTIES LABELS "verification;incompressible;channel")

add_test(NAME saturn.channel_poiseuille
  COMMAND saturn_channel_sim --case poiseuille --n-list 16 32 --tend 0.1
          --nu 0.1 --gx 1 --check)
set_tests_properties(saturn.channel_poiseuille
  PROPERTIES LABELS "verification;incompressible;channel")

add_test(NAME saturn.channel_stats
  COMMAND saturn_channel_sim --case poiseuille --n-list 16 --nx 8 --nz 4
          --tend 0.05 --nu 0.1 --gx 1
          --stats ${CMAKE_CURRENT_BINARY_DIR}/channel_stats.csv --check)
set_tests_properties(saturn.channel_stats
  PROPERTIES LABELS "verification;incompressible;channel")

if(SATURN_HAVE_FFTW)
  add_test(NAME saturn.channel_poiseuille_fft
    COMMAND saturn_channel_sim --case poiseuille --n-list 16 32 --tend 0.1
            --nu 0.1 --gx 1 --poisson fft --check)
  set_tests_properties(saturn.channel_poiseuille_fft
    PROPERTIES LABELS "verification;incompressible;channel;fft")
endif()

add_test(NAME saturn.incompressible_tgv3d_smoke
  COMMAND saturn_incompressible_sim --case tgv3d --n 16 --tend 1
          --nu 0.000625 --cfl 0.3 --check)
set_tests_properties(saturn.incompressible_tgv3d_smoke
  PROPERTIES LABELS "verification;incompressible")

add_test(NAME saturn.incompressible_tgv3d_filtered
  COMMAND saturn_incompressible_sim --case tgv3d --n 32 --tend 5
          --nu 0.000625 --cfl 0.3 --filter 0.1 --check)
set_tests_properties(saturn.incompressible_tgv3d_filtered
  PROPERTIES LABELS "verification;incompressible")

add_test(NAME saturn.cavity
  COMMAND saturn_cavity_sim --n 16 --re 100 --tend 5 --rms-tol 0.08 --check)
set_tests_properties(saturn.cavity
  PROPERTIES LABELS "verification;incompressible;cavity")

add_test(NAME saturn.edac_tgv
  COMMAND saturn_edac_sim --n 32 --mc 0.1 --tau 0.01 --tend 0.5 --check)
set_tests_properties(saturn.edac_tgv
  PROPERTIES LABELS "verification;incompressible;edac")

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
