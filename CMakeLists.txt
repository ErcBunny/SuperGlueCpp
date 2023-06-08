cmake_minimum_required(VERSION 3.21)
project(superglue)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(superglue SHARED src/SuperGlue.cpp)
target_link_libraries(superglue Python3::Python Python3::NumPy ${OpenCV_LIBS})

add_executable(demo src/demo.cpp)
target_link_libraries(demo superglue)
