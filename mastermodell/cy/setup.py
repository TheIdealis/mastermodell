from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("density", ["density.pyx", "nicerdicer.cpp"], 
               language='c++', extra_compile_args=["-std=c++11", "-Ofast"])]

setup(cmdclass = {'build_ext': build_ext}, ext_modules = ext_modules)
