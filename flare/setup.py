from setuptools import setup, find_packages

import flare

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="flare-torch",
    version=flare.__version__,
    license=flare.__license__,
    author=flare.__author__,
    description=flare.__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="sungjin.code@gmail.com",
    packages=find_packages(include=["flare", "flare.*"]),
    url="https://github.com/denev6/deep-learning-codes/tree/main/flare",
    install_requires=[
        "torch",
        "tqdm",
        "numpy",
        "scikit-learn",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
