
import os
from setuptools import setup, find_packages

# There is a two step process here. We need to run setup.py to know that we need to install pybind and pyarrow
# (setup_requires)
# So the first time setup.py is invoked, it goes through the exception, finds the setup_requires, installs pybind and
# pyarrow, then restarts and goes through the try statement successfully, and can build the cpp extensions properly.
# try:

try:
    import pyarrow
    from pybind11.setup_helpers import Pybind11Extension
    pyarrow_location = os.path.dirname(pyarrow.__file__)
    # For now, assume that we build against bundled pyarrow releases.
    pyarrow_include_dir = os.path.join(pyarrow_location, 'include')
    print(pyarrow_location)
    print(pyarrow_include_dir)
except ImportError as e:
    print(e)
    from setuptools import Extension as Pybind11Extension
    pyarrow_location = ""
    pyarrow_include_dir = ""

from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "wicker.l5ml_datastore.cpp_extensions",
        [
            "wicker/l5ml_datastore/cpp/arrow_util.cpp",
            "wicker/l5ml_datastore/cpp/cpp_extensions.cpp",
            "wicker/l5ml_datastore/cpp/sampling.cpp",
            "wicker/l5ml_datastore/cpp/temporal_windowing.cpp",
        ],
        cxx_std=17,
        include_dirs=[pyarrow_include_dir, pyarrow_location],
        extra_compile_args=[
            "-fvisibility=hidden",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-parameter",
            "-Wnon-virtual-dtor",
            "-I $pyarrow_include_dir/include" 
        ],
        library_dirs=[pyarrow_location, pyarrow_include_dir],
        libraries=[":libarrow.so.500", ":libarrow_python.so.500"],
        extra_link_args=["-Wl,-rpath,$ORIGIN", "-Wl,-rpath,$ORIGIN/pyarrow"],
    ),
]

# wicker's setup

setup(
    name="wicker",
    version="1.1.1",
    install_requires=[
        "boto3",
        "numpy",
        "pyarrow",
        "boto3",
        "tqdm"
    ],
    extras_require={
        "spark": ["pyspark"],
        "pyarrow": ["pyarrow==5.0.0"],
        "flyte": ["flytekit"],
        "dynamodb" : ["pynamodb"],
        "wandb" : ["wandb"]
    },
    dependency_links=[
        "https://artifactory.pdx.l5.woven-planet.tech/repository/pypi-internal/simple"
    ],
    python_requires='>=3.8',
    ext_modules=ext_modules,
    packages=find_packages(include=["*"]),
    package_data={
        '': [
            "*",
        ],
    },
    include_package_data=True,
)
