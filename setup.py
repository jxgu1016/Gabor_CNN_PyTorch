#!/usr/bin/env python
import os
from setuptools import setup, find_packages

setup(
    name="gcn",
    version="1.0",
    description="Gabor Convolutional Networks, WACV2017",
    url="https://github.com/jxgu1016/GCN_PyTorch/tree/para_product",
    author="Jiaxin Gu",
    author_email="jxgu1016@gmail.com",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(os.path.dirname(__file__), "build.py:ffi")
    ],
)