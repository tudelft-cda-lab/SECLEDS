from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("eval_pairwise",       ["eval_pairwise.pyx"]),
    Extension("fast_assign",       ["fast_assign.pyx"]),
    Extension("fast_update",       ["fast_update.pyx"]),
]

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)

