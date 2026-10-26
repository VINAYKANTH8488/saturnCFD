add_executable(saturn_test_smoke test_smoke.cpp)
target_link_libraries(saturn_test_smoke PRIVATE saturn_core)

add_test(NAME saturn.smoke COMMAND saturn_test_smoke)
set_tests_properties(saturn.smoke PROPERTIES LABELS "unit")

add_test(NAME saturn.scalar_mms
  COMMAND saturn_scalar_mms --n-list 16 32 --nu 0.01 --tend 0.2 --check)
set_tests_properties(saturn.scalar_mms PROPERTIES LABELS "verification")
