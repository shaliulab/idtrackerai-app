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
    name="idtrackerai-app-shaliulab",
    version=version,
    description="""""",
    author=["Ricardo Ribeiro", "Francisco Romero Ferrero"],
    author_email="idtrackerai@gmail.com, ricardojvr@gmail.com, paco.romero.ferrero@gmail.com",
    url="https://idtrackerai-app.readthedocs.org",
    packages=find_packages(),
    install_requires=[
        "rich",
        "pyforms-terminal==4.*",
        "imgstore-shaliulab>=0.4.9",
    ],
    extras_require={
        "gui": [
            "pyforms-gui==4.904.152",
            "python-video-annotator==3.306",
            "python-video-annotator-module-idtrackerai == 1.0.0a0",
        ],
        "only-gui": ["pyforms-gui==4.904.152"],
    },
    entry_points={
        "console_scripts": [
            "idtrackerai=idtrackerai_app.__main__:start",
            "idtrackerai_cli=idtrackerai_app.cli.__main__:start",
        ],
    },
)
