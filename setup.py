from setuptools import setup, find_packages
import os


GIT_URL = "https://github.com/BiomedSciAI/causallib"

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def get_lines(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


def get_valid_packages_from_requirement_file(file_path):
    lines = get_lines(file_path)
    # Filter out non-package lines that are legal for `pip install -r` but fail for setuptools' `require`:
    pkg_list = [p for p in lines if p.lstrip()[0].isalnum()]
    return pkg_list


def get_version(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                quotes_type = '"' if '"' in line else "'"
                version = line.split(quotes_type)[1]
                return version
    raise RuntimeError("Unable to find version string.")


setup(name='causallib',
      version=get_version(os.path.join('causallib', '__init__.py')),
      # packages=find_packages(exclude=['scripts', 'data', 'tests']),
      packages=find_packages(),
      description='A Python package for flexible and modular causal inference modeling',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url=GIT_URL,
      author='Causal Machine Learning for Healthcare and Life Sciences, IBM Research Israel',
      # author_email=None,
      license="Apache License 2.0",
      keywords="causal inference effect estimation causality",
      install_requires=get_valid_packages_from_requirement_file("requirements.txt"),
      extras_require={
          'contrib': get_valid_packages_from_requirement_file(os.path.join("causallib", "contrib", "requirements.txt")),
          'docs': get_valid_packages_from_requirement_file(os.path.join("docs", "requirements.txt"))
      },
      # include_package_data=True,
      package_data={
          'causallib': [os.path.join('datasets', 'data', '*/*.csv')]
      },
      project_urls={
          'Documentation': 'https://causallib.readthedocs.io/en/latest/',
          'Source Code': GIT_URL,
          'Bug Tracker': GIT_URL + '/issues',
      },
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "License :: OSI Approved :: Apache Software License",
          "Development Status :: 4 - Beta",
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research"
      ]
      )
