bt = get_option('buildtype')
if bt == 'debug' or bt == 'debugoptimized'
    extra_args = ['-DDEBUG']
else
    extra_args = []
endif

foreach ex : [
    ['simple', 'simple.cpp'],
    ['build_ivf', 'build_ivf.cpp'],
    ['query_ivf', 'query_ivf.cpp'],
    ['build_flat_index', 'build_flat_index.cpp'],
    ['measure_accuracy', 'measure_accuracy.cpp'],
    ['measure_accuracy_bsivf', 'measure_accuracy_bsivf.cpp'],
]
    executable(
        ex[0],
        ex[1],
        dependencies: genivf_dep,
        cpp_args: lib_args + extra_args,
    )
endforeach
