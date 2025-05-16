openmp_dep = dependency('OpenMP', language: 'c', required: false)

cython_args = [
  '-X language_level=3', '-X boundscheck=False', '-X wraparound=False',
  '-X initializedcheck=False', '-X nonecheck=False', '-X cdivision=True',
  '-X profile=False',
]

py.extension_module(
    '_cancorr_fast',
    '_cancorr_fast.pyx',
    dependencies: [openmp_dep],
    subdir: 'fastcan',
    cython_args: cython_args,
    install: true
)

py.extension_module(
    '_narx_fast',
    '_narx_fast.pyx',
    subdir: 'fastcan',
    cython_args: cython_args,
    install: true
)