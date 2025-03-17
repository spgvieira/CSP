cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project (assignment1)

set (CMAKE_CXX_FLAGS "-std=c++20")
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(independent independent.cpp)
add_executable(concurrent concurrent.cpp)
add_executable(indep_core_aff_1 indep_core_aff_1.cpp)
add_executable(indep_core_aff_2 indep_core_aff_2.cpp)

# target_compile_options(independent PRIVATE -Wall) ## we don't want wall right?
target_link_libraries(indep_core_aff_1 PRIVATE pthread) # only actually relevant for core affinity
target_link_libraries(indep_core_aff_2 PRIVATE pthread) 