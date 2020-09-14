import setuptools
from codecs import open
from os import path

__version__ = '0.0.1'

setuptools.setup(
    name="Neural-Machine-Translation-Scratch", 
    version= __version__,
    author="Jesse Khaira",
    author_email="jesse.khaira15@gmail.com",
    license = 'MIT',
    description="Python implementations using only NumPy of a seq2seq Recurrent Neural Network architecture",
    url="https://github.com/13jk59/Neural-Machine-Translation-Scratch",
    packages=setuptools.find_packages(),
)
