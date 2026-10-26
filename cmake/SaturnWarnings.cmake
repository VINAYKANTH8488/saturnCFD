function(saturn_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} INTERFACE /W4)
    if(SATURN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} INTERFACE /WX)
    endif()
  else()
    target_compile_options(${target} INTERFACE
      -Wall
      -Wextra
      -Wpedantic
      -Wshadow
      -Wnon-virtual-dtor
      -Wold-style-cast
      -Wcast-align
      -Wunused
      -Woverloaded-virtual)
    if(SATURN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} INTERFACE -Werror)
    endif()
  endif()
endfunction()
