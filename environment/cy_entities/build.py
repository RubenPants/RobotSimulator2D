"""
Call this file (in Linux environment) to build all the Cython files via setup.py

Note: Cython must installed beforehand (i.e. pip3 install cython)
"""
import argparse
import os

if __name__ == '__main__':
    # Parse arguments from call
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--folder', type=str, default='environment/cy_entities/')
    args = parser.parse_args()
    folder = args.folder
    
    # Build the setup first
    os.system('python3 {}setup.py build_ext --inplace'.format(folder))
