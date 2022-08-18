cmake_minimum_required(VERSION 2.8)

project(water_pes Fortran
)

#set(CMAKE_Fortran_COMPILER "path_to_your_compiler/ifort")

add_library(h2opes
    h2opot.f90
)

target_compile_definitions(h2opes PUBLIC -O3 -mkl)

add_executable(pes.out main.f90)
target_link_libraries(pes.out h2opes)

