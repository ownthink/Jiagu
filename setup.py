#!/usr/bin/env python
# -*- coding:utf-8 -*-
from setuptools import setup

setup(name='jiagu',
      version='0.2.3',
      description='Jiagu Natural Language Processing',
      author='Yener(Zheng Wenyu)',
      author_email='help@ownthink.com',
      url='https://github.com/ownthink/Jiagu',
      license='MIT',
      packages=['jiagu'],
      package_dir={'jiagu': 'jiagu'},
      package_data={'jiagu': ['*.*', 'cluster/*', 'data/*', 'model/*',
		'normal/*', 'segment/*', 'segment/dict/*','segment/model/*',
		'sentiment/*', 'sentiment/model/*', 'topic/*']}
      )
