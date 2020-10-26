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
    license='MIT',
    python_requires='<3.8>=3.7',
    install_requires=requirements,
    extras_require={
        'testing': ['pytest', 'pytest-cov<2.11'],
        'docs': ['guzzle_sphinx_theme'],
        'pre-commit': ['pre-commit', 'yapf', 'prospector', 'pylint', 'versioneer', 'isort'],
    },
    author='Kevin M. Jablonka, Brian Yoo, Berend Smit',
    author_email='kevin.jablonka@epfl.ch, brian.yoo@basf.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
