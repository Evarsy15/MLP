# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name and C++ standard
project(MLP VERSION 1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRCS main.cpp mlp.cpp mlp_cpu.cpp mlp_cuda.cu)
set(HDRS mlp.h mlp_cpu.h mlp_cuda.h common.h)

find_package(OpenCV REQUIRED core imgcodecs)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRCS} ${HDRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} stdc++fs)
