cmake_minimum_required(VERSION 3.14)
project(genivf)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library
add_library(genivf SHARED genivf.cpp)
target_include_directories(genivf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Demo executable
add_executable(main main.cc)
target_compile_options(main PRIVATE -Wall -Werror -Wextra)
target_link_libraries(main PRIVATE genivf)

