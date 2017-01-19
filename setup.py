#!/usr/bin/env python

from distutils.core import setup

setup(name='emfa',
      version='0.1',
      description='Factor Analysis via Expectation Maximization',
      author='Yannis Liapis',
      author_email='yliapis44@gmail.com',
      url='https://github.com/yliapis/emfa',
      packages=['emfa'],
      install_requires=['numpy', 'scipy', 'matplotlib', 'jupyter'],
     )
