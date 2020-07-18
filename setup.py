from setuptools import setup
from setuptools import find_packages
import os


def find_requirements(setup_file):
    requirement_file = os.path.dirname(
        os.path.realpath(setup_file)) + '/requirements.txt'
    install_requires = []

    if os.path.isfile(requirement_file):
        with open(requirement_file) as f:
            install_requires = f.read().splitlines()
    return install_requires


setup(
    name='face-finding-dqn',
    version='0.1.0',
    author='Cleison C. Amorim',
    author_email='cca5@cin.ufpe.br',
    license='GPL-3.0',
    install_requires=find_requirements(__file__),
    extras_require={'tests': ['pytest', 'requests', 'markdown']},
    packages=find_packages())
