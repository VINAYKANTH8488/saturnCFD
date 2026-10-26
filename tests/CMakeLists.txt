add_executable(saturn_test_smoke test_smoke.cpp)
target_link_libraries(saturn_test_smoke PRIVATE saturn_core)

add_test(NAME saturn.smoke COMMAND saturn_test_smoke)
set_tests_properties(saturn.smoke PROPERTIES LABELS "unit")
