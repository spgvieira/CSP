cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project (assignment1)

SET (CMAKE_CXX_FLAGS "-std=c++20")
add_executable(independent independent.cpp)
add_executable(concurrent concurrent.cpp)