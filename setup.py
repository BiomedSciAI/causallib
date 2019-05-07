from setuptools import setup, find_packages

# from causallib import __version__ as cl_version

GIT_URL = "https://github.com/IBM/causallib"

setup(name='causallib',
      version="1.0.0",
      # version=cl_version
      # packages=find_packages(exclude=['scripts', 'data', 'tests']),
      packages=find_packages(),
      description='A library of causal inference tools by IBM Haifa Research Labs',
      # long_description=None,
      url=GIT_URL,
      author='IBM Research Haifa Labs - Machine Learning for Healthcare and Life Sciences',
      # author_email=None,
      license="Apache License 2.0",
      keywords="causal inference effect estimation causality",
      install_requires=open("requirements.txt", "r").read().splitlines(),
      project_urls={'Bug Reports': GIT_URL + '/issues',
                    'Source Code': GIT_URL},
      classifiers=[
              "Programming Language :: Python :: 3.6",
              "License :: OSI Approved :: Apache Software License",
          ]
      )
