#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import re

version = ""
with open("idtrackerai_app/__init__.py", "r") as fd:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE
    ).group(1)

setup(
    name="idtrackerai-app",
    version=version,
    description="""""",
    author=["Ricardo Ribeiro", "Francisco Romero Ferrero"],
    author_email="idtrackerai@gmail.com, ricardojvr@gmail.com, paco.romero.ferrero@gmail.com",
    url="https://idtrackerai-app.readthedocs.org",
    packages=find_packages(),
    install_requires=["coloredlogs", "idtrackerai", "pyforms-terminal"],
    extras_require={
        "gui": [
            "pyforms-gui",
            "python-video-annotator",
            "python-video-annotator-module-idtrackerai==0.6.67",
        ],
        "only-gui": ["pyforms-gui"],
    },
    entry_points={
        "console_scripts": [
            "idtrackerai=idtrackerai_app.__main__:start",
        ],
    },
)
