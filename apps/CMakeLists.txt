foreach(app stencil stream halo)
  add_executable(saturn_${app} ${app}_bench.cpp)
  target_link_libraries(saturn_${app} PRIVATE saturn_core)
  target_compile_definitions(saturn_${app} PRIVATE
    SATURN_DEFAULT_SPACE="${SATURN_DEFAULT_SPACE}")
endforeach()

add_executable(saturn_scalar_mms scalar_mms.cpp)
target_link_libraries(saturn_scalar_mms PRIVATE saturn_core)
target_compile_definitions(saturn_scalar_mms PRIVATE
  SATURN_DEFAULT_SPACE="${SATURN_DEFAULT_SPACE}")

add_executable(saturn_euler1d euler1d.cpp)
target_link_libraries(saturn_euler1d PRIVATE saturn_core)

add_executable(saturn_compressible_mms compressible_mms.cpp)
target_link_libraries(saturn_compressible_mms PRIVATE saturn_core)
target_compile_definitions(saturn_compressible_mms PRIVATE
  SATURN_DEFAULT_SPACE="${SATURN_DEFAULT_SPACE}")

add_executable(saturn_compressible_sim compressible_sim.cpp)
target_link_libraries(saturn_compressible_sim PRIVATE saturn_core)
target_compile_definitions(saturn_compressible_sim PRIVATE
  SATURN_DEFAULT_SPACE="${SATURN_DEFAULT_SPACE}")

add_executable(saturn_incompressible_sim incompressible_sim.cpp)
target_link_libraries(saturn_incompressible_sim PRIVATE saturn_core)
target_compile_definitions(saturn_incompressible_sim PRIVATE
  SATURN_DEFAULT_SPACE="${SATURN_DEFAULT_SPACE}")

add_executable(saturn_channel_sim channel_sim.cpp)
target_link_libraries(saturn_channel_sim PRIVATE saturn_core)
target_compile_definitions(saturn_channel_sim PRIVATE
  SATURN_DEFAULT_SPACE="${SATURN_DEFAULT_SPACE}")

add_executable(saturn_cavity_sim cavity_sim.cpp)
target_link_libraries(saturn_cavity_sim PRIVATE saturn_core)
target_compile_definitions(saturn_cavity_sim PRIVATE
  SATURN_DEFAULT_SPACE="${SATURN_DEFAULT_SPACE}")
