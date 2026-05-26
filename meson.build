project(
    'genivf',
    'cpp',
    version: '0.1.0',
    license: 'MIT',
    default_options: [
        'cpp_std=c++23',
        'warning_level=2', # -Wall -Wextra on GCC/Clang
        'buildtype=debug',
        'b_lto=true'
    ],
)

inc = include_directories('include')

lib_args = ['-Werror']

genivf_lib = static_library(
    'genivf',
    ['src/ivf.cpp', 'src/flat.cpp', 'src/seq.cpp', 'src/io.cpp', 'src/bsivf.cpp'],
    include_directories: inc,
    cpp_args: lib_args,
)

genivf_dep = declare_dependency(
    include_directories: inc,
    link_with: genivf_lib,
)

executable(
    'genivf',
    'main.cpp',
    dependencies: genivf_dep,
    cpp_args: lib_args,
)

subdir('tests')
subdir('examples')

