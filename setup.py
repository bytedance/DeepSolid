# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

REQUIRED_PACKAGES = (
    "absl-py",
    'attrs',
    "dataclasses",
    "networkx",
    "scipy==1.9.3",
    "numpy",
    "ordered-set",
    "typing",
    "chex==0.1.5",
    "jax==0.2.26",
    "jaxlib==0.1.75",
    "pandas",
    "ml_collections",
    "pyscf",
    "tables",
    "h5py==3.2.1",
    "optax==0.0.9",

)

setup(
    name="DeepSolid",
    version="1.0",
    description="A library combining solid quantum Monte Carlo and neural network.",
    author='ByteDance',
    author_email='lixiang.62770689@bytedance.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    scripts=['bin/deepsolid'],
    license='Apache 2.0',
)
