#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import re

version = ''
with open('idtrackerai_gui/__init__.py', 'r') as fd: 
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE).group(1)

setup(
    name='idtrackerai-gui',
    version=version,
    description="""""",
    author=['Ricardo Ribeiro'],
    author_email='info@idtracker.ai, ricardojvr@gmail.com',
    url='https://http://idtrackerai-gui.readthedocs.org',
    packages=find_packages(),
    install_requires=[
        'coloredlogs',
    ]   
)
