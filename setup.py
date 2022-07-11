#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
import os

package_dirs = {
    'models': os.path.join('src', 'models'),
    'ocr_engine': os.path.join('src', 'ocr_engine'),
    'tools': os.path.join('src', 'tools'),
}

d = generate_distutils_setup(
    packages=package_dirs.keys(),
    package_dir=package_dirs,
)
setup(**d)
