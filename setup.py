import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension
from Cython.Distutils import build_ext


def readme():
    with open('README.rst') as f:
        return f.read()


PACKAGES = ['mastermodell', 'mastermodell/cy']

PACKAGE_DATA = {
    '.': ['README.md'],
    'mastermodell/cy': ['*.cpp', '*.h', '*.pyx']
}

cy_exts = ['density']

_compiler_flags = ["-std=c++11", "-Ofast"]

EXT_MODULES = []
for ext in cy_exts:
    _mod = Extension("mastermodell.cy." + ext,
                     sources=['mastermodell/cy/' + ext + '.pyx', 'mastermodell/cy/nicerdicer.cpp'],
                     include_dirs=[np.get_include()],
                     extra_compile_args=_compiler_flags,
                     extra_link_args=[],
                     language='c++')
    EXT_MODULES.append(_mod)

import distutils.sysconfig

cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

setup(name='mastermodell',
      version='0.1',
      description='Class which stores the data of a mastermodell',
      long_description=readme(),
      ext_modules=cythonize(EXT_MODULES),
      cmdclass={'build_ext': build_ext},
      classifiers=[
          'Development Status :: 1 - Beta',
          'License :: ',
          'Programming Language :: Python :: 3.6',
          'Topic :: mastermodall :: Sparse matrix integration',
      ],
      url='http://github.com/',
      author='Thomas Lettau',
      author_email='thomas_lettau@gmx.de',
      license='MIT',
      packages=PACKAGES,
      install_requires=[
          'qutip','numpy', 'scipy'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      package_data=PACKAGE_DATA,
      zip_safe=False)
