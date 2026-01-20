project(
  'fastcan',
  'c', 'cython',
  version: '0.5.0',
  license: 'MIT',
  meson_version: '>= 1.10.0',
  default_options: [
    'buildtype=release',
  ],
)

py = import('python').find_installation(pure: false)

install_subdir(
  'fastcan',
  install_dir: py.get_install_dir(),
  install_tag: 'python-runtime',
  exclude_files: ['_cancorr_fast.pyx', 'narx/_narx_fast.pyx'],
)

subdir('fastcan')
