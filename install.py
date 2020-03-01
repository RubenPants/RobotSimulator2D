"""
install.py

Install all needed dependencies.
"""
import os
import sys

if __name__ == '__main__':
    os.system("pip3 install cython")
    os.system("pip3 install graphviz")
    os.system("pip3 install matplotlib")
    os.system("pip3 install neat")
    os.system("pip3 install neat-python")
    os.system("pip3 install pyglet")
    os.system("pip3 install pymunk")
    os.system("pip3 install pytest")
    os.system("pip3 install scipy")
    os.system("pip3 install tqdm")
    os.system("pip3 install pylint")
    if sys.platform == 'linux' or sys.platform == 'darwin':  # Linux or Mac
        os.system("pip install torch torchvision")
    elif sys.platform == 'win32':  # Windows
        os.system("pip3 install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html")
    else:
        raise Exception("Platform not supported, please download PyTorch manually on: https://pytorch.org/")
