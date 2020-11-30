#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='nnsaliency',
    version='0.1',
    description='',
    author='Katharina Anderer',
    author_email='k.anderer@t-online.de',
    packages=find_packages(exclude=[]),
    install_requires=[],
)
