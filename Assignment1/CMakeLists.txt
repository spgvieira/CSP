cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project (assignment1)

#SET (CMAKE_CXX_FLAGS "-std=c++20")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(independent independent.cpp)
add_executable(concurrent concurrent.cpp)
add_executable(indep_core_aff_1 indep_core_aff_1.cpp)

target_compile_options(my_executable PRIVATE -Wall)
target_link_libraries(my_executable PRIVATE pthread)