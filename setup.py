import os

from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup


base_path = os.path.dirname(__file__) + "/.."
nlp_metrics_path = base_path + "/NLP_Metrics/"

setup(
    name="nlp_datasets",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/nlp_datasets/*.py")],
    install_requires=[
        "nlp_metrics @ file://localhost/%s" % nlp_metrics_path,
        ],
)