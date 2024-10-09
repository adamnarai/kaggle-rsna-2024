import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


requirements = [
    # use environment.yml
]


setup(
    name="rsna2024",
    version="0.0.1",
    url="https://github.com/adamnarai/kaggle-rsna-2024",
    author="Adam Narai",
    author_email="narai.adam@gmail.com",
    description="https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests",)),
    entry_points={"console_scripts": ["rsna2024=rsna2024.cli:cli"]},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
)
