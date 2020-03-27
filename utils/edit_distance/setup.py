from distutils.core import setup
from Cython.Build import cythonize
from setuptools import Extension

app = Extension(
        name='edit_distance',
        language='c++',
        sources=['edit_distance_modified.pyx']
)

setup(
    ext_modules = cythonize(app),
)
