=========================================
Scikit-learn wrapper for Python CRF-Suite 
=========================================

General Information
-------------------

ace-sklearn-crfsuite is a fork of `sklearn-crfsuite <https://github.com/TeamHG-Memex/sklearn-crfsuite>`_, adapted to fit modern versions of scikit-learn.

This version went for the strict minimum, and only support pyhton 3.10.

License is MIT.

How to install
--------------

The package is available as a *pip* package.


    pip install ace-sklearn-srfsuite


How to use
----------

We provide a `tutorial <https://github.com/ace-design/ace-sklearn-crfsuite/blob/master/tutorial.md>`_ to demonstrate how the package supports the integration of sklearn recent improvements to the existing code base. 

This tutorial is heavily inspired by the original one available in the `sklearn-crfsuite` documentation.


How to contribute
-----------------

The project use *pipenv* to allow easy external contributions. Using the `--dev` option install the development tools (tests, build and deploy)

    pipenv install --dev 


One can start an environment will all dependencies satisfied using the following command:

    pipenv shell

Inside this environment, to run the tests:

    python -m pytest tests/test_*.py

To check code coverage, one need to fiurst run the tests with coveragem, and then ask for a report.

    coverage run -m pytest tests/test_*.py

    coverage report 


To build the code as a deployable package:

    python -m build
    
To upload the freshly built packages to PyPi:

    twine upload -r testpypi dist/*

Remove `-r testpypi` if the deployment went well, to publish to the real PyPi repository

