# -*- coding: utf-8 -*-
# Copyright 2020 PyPAL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dispersant Screener"""
from __future__ import absolute_import

from setuptools import find_packages, setup

import versioneer

with open('requirements.txt', 'r') as fh:
    requirements = [line.strip() for line in fh]  # pylint:disable=invalid-name

setup(
    name='dispersant_screener',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    url='',
    license='Apache 2.0',
    python_requires='>3.6',
    install_requires=requirements,
    extras_require={
        'testing': ['pytest', 'pytest-cov<2.13'],
        'docs': ['guzzle_sphinx_theme'],
        'pre-commit': ['pre-commit', 'yapf', 'prospector', 'pylint', 'versioneer', 'isort'],
    },
    author='Kevin M. Jablonka, Brian Yoo, Berend Smit',
    author_email='kevin.jablonka@epfl.ch, brian.yoo@basf.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
