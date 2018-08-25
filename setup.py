from setuptools import setup, find_packages

import os

def load_requires_from_file(filepath):
    with open(filepath) as fp:
        return [pkg_name.strip() for pkg_name in fp.readlines()]

setup(
    name='torchcv',
    version='0.0.1',
    desctription='Collection of Deep Learning Computer Vision Algorithms implemented in PyTorch',
    url='https://github.com/doiken23/torchcv',
    install_requires=load_requires_from_file('requirements.txt'),
    licence='MIT',
    # exclude the test
    packages=find_packages(exclude=['tests']),
)
