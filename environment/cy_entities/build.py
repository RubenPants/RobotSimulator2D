"""
Call this file (in Linux environment) to build all the Cython files via setup.py

Note: Cython must installed beforehand (i.e. pip3 install cython)
"""
import os

os.system('python3 setup.py build_ext --inplace')
os.system('mv environment/cy_entities/* .')
os.system('rm -r environment/')
