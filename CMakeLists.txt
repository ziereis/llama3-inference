cmake_minimum_required(VERSION 3.12)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable targets
add_executable(quantize quantize.cpp)