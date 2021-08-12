from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='calo-cluster',
      version='0.0.2',
      author='Alex Schuy',
      author_email='alexjschuy@gmail.com',
      description='implementation of spvcnn for physics clustering datasets.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages())
