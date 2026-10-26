add_executable(saturn_test_smoke test_smoke.cpp)
target_link_libraries(saturn_test_smoke PRIVATE saturn_core)

add_executable(saturn_test_halo test_halo.cpp)
target_link_libraries(saturn_test_halo PRIVATE saturn_core)

add_test(NAME saturn.smoke COMMAND saturn_test_smoke)
set_tests_properties(saturn.smoke PROPERTIES LABELS "unit")

add_test(NAME saturn.halo COMMAND saturn_test_halo)
set_tests_properties(saturn.halo PROPERTIES LABELS "unit;halo")

add_test(NAME saturn.scalar_mms
  COMMAND saturn_scalar_mms --n-list 16 32 --nu 0.01 --tend 0.2 --check)
set_tests_properties(saturn.scalar_mms PROPERTIES LABELS "verification")

if(SATURN_HAVE_MPI)
  add_test(NAME saturn.halo_mpi
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_test_halo>
            --kokkos-num-threads=1)
  set_tests_properties(saturn.halo_mpi PROPERTIES
    LABELS "unit;halo;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

  add_test(NAME saturn.scalar_mms_mpi
    COMMAND ${SATURN_MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:saturn_scalar_mms>
            --n-list 16 32 --nu 0.01 --tend 0.2 --check
            --kokkos-num-threads=1)
  set_tests_properties(saturn.scalar_mms_mpi PROPERTIES
    LABELS "verification;mpi"
    ENVIRONMENT "DISPLAY=;OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
endif()
