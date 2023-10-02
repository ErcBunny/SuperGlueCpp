cmake_minimum_required(VERSION 3.21)
project(supergluecpp)

# set vars
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/libtorch)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# find packs
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# set include dirs
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/libtorch/include
)

# superglue python wrapper ver shared lib
add_library(superglue_pywrap SHARED src/SuperGluePyWrap.cpp)
target_link_libraries(superglue_pywrap
    Python3::Python
    Python3::NumPy
    ${OpenCV_LIBS}
)

# superglue torch C++ API ver shared lib
add_library(superglue_libtorch SHARED src/SuperGlueLibtorch.cpp)
target_link_libraries(superglue_libtorch
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# demo applications
add_executable(demo_pywrap src/demoPyWrap.cpp)
target_link_libraries(demo_pywrap superglue_pywrap)

add_executable(demo_libtorch src/demoLibtorch.cpp)
target_link_libraries(demo_libtorch superglue_libtorch)
