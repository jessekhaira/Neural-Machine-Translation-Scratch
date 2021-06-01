import setuptools
from os import path
from pip._internal.req import parse_requirements

__version__ = "0.0.1"

here = path.abspath(path.dirname(__file__))

production_requirements = parse_requirements("requirements.txt", session="hack")
production_install = [str(ir.req) for ir in production_requirements]

development_requirements = parse_requirements("requirements/dev.txt",
                                              session="hack")
development_install = [str(ir.req) for ir in development_requirements]

setuptools.setup(
    name="Sequence to Sequence Model",
    version=__version__,
    author="Jesse Khaira",
    author_email="jesse.khaira10@gmail.com",
    license="MIT",
    description=
    "Python implementations using only NumPy of a sequence to sequence architecture",
    url="https://github.com/13jk59/MachineLearning_Scratch.git",
    packages=setuptools.find_packages(),
    install_requires=production_install,
    extras_require={"dev": development_install})
