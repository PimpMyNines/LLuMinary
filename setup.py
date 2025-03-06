# This file is maintained for backward compatibility.
# For modern Python packaging, all configuration is defined in pyproject.toml.

from setuptools import setup

# Allow package to be built with pip
# All configuration is handled in pyproject.toml
setup(
    # Minimal requirement for setuptools to recognize this as a package
    name="lluminary",
    # All other metadata is in pyproject.toml
)
