cmake_minimum_required(VERSION 3.10..4.1)
project(genivf)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(genivf SHARED genivf.cc)

add_executable(main main.cc)
target_link_libraries(main PRIVATE genivf)
