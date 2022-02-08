To generate the `source/causallib.*` files run:
```bash
sphinx-apidoc -o source ../causallib --separate --force
```
(from within this directory)

To generate html build run:
```bash
make html
```

#### requirements
* [sphinx v2.1.0](http://www.sphinx-doc.org/en/master/): to generate documentation
* [m2r v0.2.1](https://github.com/miyakogi/m2r): to support inline inclusion of the modules' README markdown files
* [nbsphinx v0.4.2](https://nbsphinx.readthedocs.io): to support inclusion of Jupyter Notebooks inside the html 
  documentation

`requirement.txt` is a requirement file necessary for [readthedocs.org](readthedocs.org) build.
Pointed by `../.readthedocs.yml` configuration file.

The `source/conf.py` file also includes some arbitrary code for the automatic 
inclusion of README files within the documentation.
