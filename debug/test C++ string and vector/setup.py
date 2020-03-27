from distutils.core import setup
from Cython.Build import cythonize
from setuptools import Extension

app = Extension(
        name='test',
        language='c++',
        sources=['edit_distance_modified.pyx']
)

setup(
    ext_modules = cythonize(app),
)
