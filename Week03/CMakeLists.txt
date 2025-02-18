cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project (affinity)

SET (CMAKE_CXX_FLAGS "-std=c++20")
add_executable(affinity affinity.cpp)
add_executable(affinityperformanceexample affinityperformanceexample.cpp)
add_executable(noaffinity no-affinity.cpp)