"""
Transform the Cython files to their C-counterpart by running this setup-file from root.

This file must be called from build.py, or called via the following command:
python3 <folder_from_root>setup.py build_ext --inplace
"""

from distutils.core import setup

from Cython.Build import cythonize

folder = 'environment/cy_entities/'
setup(ext_modules=cythonize('{}god_class_cy.pyx'.format(folder)))
