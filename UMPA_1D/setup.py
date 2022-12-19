from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
import os

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

NAME = "UMPA_1D_PB"
VERSION = "0.1"
DESCR = "Unified Modulated Pattern Analysis"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Pierre Thibault"
EMAIL = "pierre.thibault@soton.ac.uk"

LICENSE = "Apache 2.0"

SRC_DIR = "UMPA_1D"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + ".model",
                  [SRC_DIR + "/model.pyx"],
                  language="c++",
                  #libraries=["m"],
                  #extra_compile_args=["-std=c++17", "-O3", "-ffast-math", "-march=native", "-fopenmp" ],
                  extra_compile_args=["/std:c++17", "/O2", "/fp:fast", "/favor:INTEL64", "/openmp" ],
                  #extra_link_args=['-fopenmp'],
                  include_dirs=[np.get_include()])
EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          #url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS,
          include_package_data=True,
          package_data={'': ['test/logo.npy']}
          )
