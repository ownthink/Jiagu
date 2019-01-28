#!/usr/bin/env python
# -*- coding:utf-8 -*-
from setuptools import setup

setup(name='jiagu',
      version='0.1.1',
      description='Jiagu Natural Language Processing',
      author='Yener(Zheng Wenyu)',
      author_email='yener@ownthink.com',
      url='https://github.com/ownthink/Jiagu',
      license="MIT",
      install_requires=['tensorflow>=1.3.0', 'numpy>=1.12.1', 'networkx>=2.1'],
      packages=['jiagu'],
      package_dir={"jiagu": "jiagu"},
      package_data={'jiagu': ['*.*', 'model/*', 'data/*']}
      )
