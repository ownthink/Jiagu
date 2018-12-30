#!/usr/bin/env python
# -*- coding:utf-8 -*-
from distutils.core import setup

setup(name = 'pixiu',
    version = '0.1',
    description = 'Pixiu Natural Language Processing ',
    author = 'Yener(Zheng Wenyu)',
    author_email = 'yener@ownthink.com',
    url = 'https://github.com/ownthink/pixiu',
    install_requires=['tensorflow>=1.3.0', 'numpy>=1.12.1'],
    packages = ['pixiu'],
    package_dir = {"pixiu": "pixiu"},
	package_data={'pixiu':['*.*','data/*']}
    )
