# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name and C++ standard
project(MLPProgram VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify source files
set(SRCS main.cpp mlp.cpp)
set(HEADERS mlp.h common.h)

# Find OpenCV including imgcodecs for imread functionality
find_package(OpenCV REQUIRED core imgcodecs)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable target
add_executable(${PROJECT_NAME} ${SRCS} ${HEADERS})

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} stdc++fs)