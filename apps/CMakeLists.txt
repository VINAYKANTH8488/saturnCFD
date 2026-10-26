foreach(app stencil stream)
  add_executable(saturn_${app} ${app}_bench.cpp)
  target_link_libraries(saturn_${app} PRIVATE saturn_core)
  target_compile_definitions(saturn_${app} PRIVATE
    SATURN_DEFAULT_SPACE="${SATURN_DEFAULT_SPACE}")
endforeach()
