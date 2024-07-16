project(odas3)
cmake_minimum_required(VERSION 2.4.6)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

find_package(PkgConfig REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/include")

set(CMAKE_BUILD_TYPE "Release")

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")

#Add base directory for includes (global)
include_directories(include/dnn-c)

add_subdirectory(include)

set(SRC

    src/layers/gru.c
    src/layers/linear.c

    src/utils/functional.c
    src/utils/tensor.c

)

add_library(dnn-c SHARED
    ${SRC}
)

target_link_libraries(dnn-c
    m
)

add_executable(tests
    test/tests.c
    test/units/test_layers_linear.c
    test/units/test_utils_functional.c
)

target_link_libraries(tests dnn-c)

install(TARGETS dnn-c DESTINATION lib)