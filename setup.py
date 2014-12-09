#!/usr/bin/env python

import platform
import os

from setuptools import setup,find_packages

requires = ['scikit-learn', 'scipy', 'numpy']

conffile_path = '/tmp'

setup_options = dict(
    name='biclustering',
    version='0.0.1',
    description='Biclustering experiments',
    author='Lucas F. Brunialti - 23/11/2014',
    author_email='lucas.brunialti@usp.br',
    url='<lucas.brunialti@usp.br>',
    #scripts=['bin/biclustering-experiments', 'bin/'],
    packages=find_packages('.', exclude=['tests*','*ipynb*', 'bin*']),
    package_dir={'biclustering-experiments': 'biclustering-experiments'},
    #data_files=[(conffile_path, ['conf/config.yaml'])],
    install_requires=requires
)

setup(**setup_options)

