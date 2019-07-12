from setuptools import setup, find_packages
import os

# from causallib import __version__ as cl_version

GIT_URL = "https://github.com/IBM/causallib"

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='causallib',
      version="0.5.0-beta",
      # version=cl_version
      # packages=find_packages(exclude=['scripts', 'data', 'tests']),
      packages=find_packages(),
      description='A library of causal inference tools by IBM Haifa Research Labs',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url=GIT_URL,
      author='IBM Research Haifa Labs - Machine Learning for Healthcare and Life Sciences',
      # author_email=None,
      license="Apache License 2.0",
      keywords="causal inference effect estimation causality",
      install_requires=open("requirements.txt", "r").read().splitlines(),
      extra_requires={'docs': open(os.path.join("docs", "requirements.txt"), "r").read().splitlines()},
      project_urls={
          'Documentation': 'https://causallib.readthedocs.io/en/latest/',
          'Source Code': GIT_URL,
          'Bug Tracker': GIT_URL + '/issues',
      },
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "License :: OSI Approved :: Apache Software License",
          "Development Status :: 4 - Beta",
          "Topic :: Scientific/Engineering"
      ]
      )
