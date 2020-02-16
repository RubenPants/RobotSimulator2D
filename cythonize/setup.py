"""
Transform the Cython files to their C-counterpart by running this setup-file from root.

Note: This file must be called from build.py, or called via the following command:
python3 <folder_from_root>setup.py build_ext --inplace
"""

from distutils.core import setup

from Cython.Build import cythonize

setup(ext_modules=cythonize('utils/cy/vec2d_cy.pyx'))
setup(ext_modules=cythonize('utils/cy/line2d_cy.pyx'))
setup(ext_modules=cythonize('utils/cy/intersection_cy.pyx'))
setup(ext_modules=cythonize('environment/entities/cy/sensors_cy.pyx'))
setup(ext_modules=cythonize('environment/entities/cy/robots_cy.pyx'))
setup(ext_modules=cythonize('environment/entities/cy/game_cy.pyx'))
