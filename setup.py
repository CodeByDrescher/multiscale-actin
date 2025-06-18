import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", 'r') as readme:
    long_description = readme.read()

setup(
    name='multiscale_actin',
    version='0.0.1',
    packages=[
        'multiscale_actin',
        'multiscale_actin.processes',
        'multiscale_actin.composites',
        'multiscale_actin.experiments',
    ],
    author='',  # TODO: Put your name here.
    author_email='',  # TODO: Put your email here.
    url='',  # TODO: Put your project URL here.
    license='',  # TODO: Choose a license.
    entry_points={
        'console_scripts': []},
    short_description='',  # TODO: Describe your project briefely.
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={},
    include_package_data=True,
    install_requires=[
        'vivarium-core',
        'process-bigraph',
        'simularium_readdy_models @ git+https://github.com/blairlyons/readdy-models.git',
    ],
)
