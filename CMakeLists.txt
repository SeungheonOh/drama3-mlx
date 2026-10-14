cmake_minimum_required(VERSION 3.20)
project(drama3 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-function")

find_path(MLXC_INCLUDE_DIR mlx/c/mlx.h HINTS /opt/homebrew/include /usr/local/include)
find_library(MLXC_LIBRARY mlxc HINTS /opt/homebrew/lib /usr/local/lib)
if(NOT MLXC_INCLUDE_DIR OR NOT MLXC_LIBRARY)
  message(FATAL_ERROR "mlx-c not found (brew install mlx-c)")
endif()

set(CORE_SOURCES
  src/mlxu.c
  src/json.c
  src/unicode.c
  src/unicode_data.c
  src/tokenizer.c
  src/wav.c
  src/resample.c
  src/weights.c
  src/backbone.c
  src/attention.c
  src/tower.c
  src/sampling.c
  src/prompt.c
  src/chunking.c
  src/generate.c
  src/codec.c
  src/model.c
  src/npy.c
  src/reference.c
)

add_library(drama3_core STATIC ${CORE_SOURCES})
target_include_directories(drama3_core PUBLIC src ${MLXC_INCLUDE_DIR})
target_link_libraries(drama3_core PUBLIC ${MLXC_LIBRARY} m)

# Shared library with the same code, used by the parity tests through ctypes.
add_library(drama3 SHARED ${CORE_SOURCES} src/testapi.c)
target_include_directories(drama3 PRIVATE src ${MLXC_INCLUDE_DIR})
target_link_libraries(drama3 PRIVATE ${MLXC_LIBRARY} m)

add_executable(drama3-tts src/main.c)
target_link_libraries(drama3-tts PRIVATE drama3_core)

add_executable(drama3-bench tools/bench.c)
target_link_libraries(drama3-bench PRIVATE drama3_core)
