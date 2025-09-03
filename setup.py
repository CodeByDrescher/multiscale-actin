import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", 'r') as readme:
    long_description = readme.read()

setup(
    name='multiscale_actin',
    version='1.0.0',
    packages=[
        'multiscale_actin',
        'multiscale_actin.processes',
        'multiscale_actin.composites',
        'multiscale_actin.experiments',
    ],
    author='Blair Lyons',
    author_email='blair208@gmail.com',
    url='',
    license='Apache Software License 2.0',
    entry_points={
        'console_scripts': []},
    short_description='Simulate actin and membranes multiscale with Vivarium as connector',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={},
    include_package_data=True,
    install_requires=[
        'vivarium-core',
        'process-bigraph',
        'simularium-readdy-models',
    ],
)
