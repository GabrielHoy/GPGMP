cmake_minimum_required(VERSION 3.18)

# -*-*-*-*-*-*-*- OPERATING SYSTEM SPECIFIC PATHS - OVERWRITE THESE WITH YOUR LOCATIONS OR YOU MAY BE ROYALLY FUCKED -*-*-*-*-*-*-*-

# Wish I knew why CMake can't find pkgconfig on its own, you can try removing this line but you may need to point it to your pkgconf executable as well - I imagine I'm doing something wrong but I don't know what.
set(PKG_CONFIG_EXECUTABLE "C:/Users/gbhy2/Desktop/repos/vcpkg/installed/x64-windows-static/tools/pkgconf/pkgconf.exe")

# set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};C:/dev/cpp")


# -*-*-*-*-*-*-*- END OF OPERATING SYSTEM SPECIFIC PATHS -*-*-*-*-*-*-*-

project(GPGMP VERSION 1.0 LANGUAGES C CXX CUDA)

#list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# A few quick build type checks
if(CMAKE_GENERATOR MATCHES "Visual Studio")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
else()
  message(STATUS "CMAKE GENERATOR DEFINITION: ${CMAKE_GENERATOR}")
  # For single-config generators like Ninja
  if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified, using Release - this can lead to confusing build results and issues! Please specify a build type.")
    set(CMAKE_BUILD_TYPE Release)
  endif()
endif()

# Enable the usage of CUDA C because CUDA is love CUDA is life
enable_language(CUDA)

# Verify CUDA is available (required)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "CUDA compiler not found. CUDA is required to build GPGMP.")
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard and architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA parallel compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -t0")  # Use all available threads for CUDA compilation

# Template depth limits for compilation? What are those? :)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ftemplate-depth=4096")

# Enable CUDA debugging for Debug configuration only
if(MSVC)
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0 -Xcompiler=/Zi")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O2 -g -Xcompiler=/Zi")
  set(CMAKE_CUDA_FLAGS_MINSIZEREL "${CMAKE_CUDA_FLAGS_MINSIZEREL} -O1")
endif()

# GMP tends to be required for a GMP based library...
find_package(PkgConfig REQUIRED)
pkg_check_modules(gmp REQUIRED IMPORTED_TARGET gmp)
pkg_check_modules(gmpxx REQUIRED IMPORTED_TARGET gmpxx)

# Include directories
include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)




# Collect C/++ source files
file(GLOB_RECURSE PROJECT_SOURCES
  "src/*.cpp"
  "src/*.c"
)

# Collect CUDA source files
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")




# Create executable
add_executable(GPGMP ${PROJECT_SOURCES} ${CUDA_SOURCES})

# Set CUDA properties
set_target_properties(GPGMP PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "60"
)

# Base compiler options that apply to all configurations - MSVC Specific
if(MSVC)
  target_compile_options(GPGMP PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /MP /wd4996>
  )
  target_compile_definitions(GPGMP PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:_CRT_SECURE_NO_WARNINGS>
  )
else()
  target_compile_options(GPGMP PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  )
endif()

# Per-configuration compiler/linker options - MSVC Specific
if(MSVC)
  target_compile_options(GPGMP PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /Ob2 /Oi /Ot /GL>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Zi /Ob0 /Od /RTC1>
  )

  target_link_options(GPGMP PRIVATE
    $<$<CONFIG:Release>:/LTCG>
  )
endif()

# Link libraries
target_link_libraries(GPGMP PRIVATE
  cudart
  PkgConfig::gmp
  PkgConfig::gmpxx
)

