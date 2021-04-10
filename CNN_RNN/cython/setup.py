from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os

os.environ["C_INCLUDE_PATH"] = np.get_include()

setup(
    ext_modules = cythonize("cython_functions.pyx")
    )
