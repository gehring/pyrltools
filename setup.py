#from numpy.distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages
import numpy

setup(name = 'rltools',
      version = '0.1',
      include_dirs=[numpy.get_include(), '/usr/include'],
      packages = find_packages(),
      ext_modules= [Extension('rltools.ext_neuro',
                              ['c-src/neurosftd_libmod.c',
                               'c-src/neurosftd_lib.h'],
                              libraries = ['cblas'])])