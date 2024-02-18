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
        #"rich",
        #"pyforms-terminal==4.*",
        #"imgstore-shaliulab>=0.4.9",
    ],
    extras_require={
        "gui": [
            "pyforms-gui==4.904.152",
            "python-video-annotator==3.306",
            "python-video-annotator-module-idtrackerai == 1.0.0a0",
        ],
        "only-gui": ["pyforms-gui==4.904.152"],
    },
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.json'],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "idtrackerai=idtrackerai_app.__main__:start",
            "idtrackerai_cli=idtrackerai_app.cli.__main__:start_ts",
            "idtrackerai_nextflow=idtrackerai_app.cli.__main__:start_nextflow",
            "integrate-yolov7=idtrackerai_app.cli.yolov7.__main__:main",
            "detect-incomplete-frames=idtrackerai_app.cli.yolov7.detect_incomplete_frames:main",
            "save-frames=idtrackerai_app.cli.yolov7.save_frames:main",
            "show-fragment-structure=idtrackerai_app.cli.utils.fragmentation:main",
            "load-blobs=idtrackerai_app.cli.utils.fragmentation:load_list_of_blobs",
            "load-video=idtrackerai_app.cli.loaders:main",
            "ls-accuracy=idtrackerai_app.cli.metrics.accuracy:main",
            "annotate-scenes=idtrackerai_app.cli.annotation:main",
            "load2fiftyone=idtrackerai_app.cli.yolov7.load2fiftyone:main",
            "concatenate-chunks=idtrackerai_app.cli.utils.overlap:main",
            "correct-scenes=idtrackerai_app.cli.annotation:correct_scenes_main",
            "view-corrections=idtrackerai_app.cli.annotation:view_corrections_main",
            "init-idtrackerai=idtrackerai_app.cli.init:init_idtrackerai",
            "upload-blobs-to-vsc=idtrackerai_app.cli.transfer:upload_blob_collections",
            "download-blobs-from-vsc=idtrackerai_app.cli.transfer:download_blob_collections",
            "qc-identity-zero=idtrackerai_app.cli.qc:qc_identity_zero",
            "preprocess-sample=idtrackerai_app.cli.qc.preprocess_sample:main"
        ],
    },
)
