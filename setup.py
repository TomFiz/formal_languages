from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension
extensions = [
    Extension(
        "clanguages",  # Output module name (import as clanguages)
        ["languages.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],  # Optimization flag
        language="c++"
    )
]

# Setup
setup(
    name="formal_languages",
    version="0.1",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        }
    )
)