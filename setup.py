from setuptools import setup, find_packages

setup(name='QDNN',
      version='0.1',
      author='Danny Rosen',
      install_requires=['numpy',
                        'matplotlib',
                        'qiskit'],
      packages= find_packages())