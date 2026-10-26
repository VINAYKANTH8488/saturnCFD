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
