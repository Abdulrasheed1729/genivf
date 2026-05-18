doctest_dep = declare_dependency(
    include_directories: include_directories('.'),
)

foreach tc : [
    ['point_test', 'point.test.cpp'],
    ['genivf_test', 'genivf.test.cpp'],
    ['io_test', 'io.test.cpp'],
]
    exe = executable(
        tc[0],
        tc[1],
        dependencies: [genivf_dep, doctest_dep],
    )
    test(tc[0], exe)
endforeach
