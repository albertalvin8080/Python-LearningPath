from setuptools import setup, find_packages

# pip install -e .
# The -e flag installs the package in "editable" mode, allowing you to modify the source code without reinstalling it.
setup(
    name="albert_utils",
    version="0.1",
    description="Utility functions for OpenCV",
    author="Albert Alvin",
    packages=find_packages(),
)
