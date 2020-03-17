#!/usr/bin/env python
from distutils.core import setup
from setuptools import setup, find_packages
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='sprcom',
      version='0.2',
      description='Bayesian spatial regression of communities',
      author='Christopher Krapu',
      author_email='ckrapu@gmail.com',
      packages=find_packages(),
      download_url='https://github.com/ckrapu/sprcom/archive/v0.1-beta.tar.gz',
      install_requires=['scipy','numpy','pymc3','theano','matplotlib','seaborn'],
      long_description=long_description,
      long_description_content_type='text/markdown')
