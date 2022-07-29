# Sklearn CRF tutorial

## Installation

First install the dependencies (including Jupyter):

```
pipenv install
```

Then start Jupyter inside the virtual environment:

```
pipenv run python -m jupyter notebook tutorial.ipynb   
```

And open the `tutorial.ipynb` file.

Remark: we need to use "python -m" to be sure we are using our _pipenv-ed_ version of Jupyter and not the one on th ehost machine. 

## Pre-trained models

I you want to skip the training steps (can take a while, as in... hours), you might want to move the `.pkl` files located in the `pre_trained` directory to the current one. The notebook will use these files instead of training its own models.

## Example of execution

The `output` directory contains an export of the notebook, so that you don't even have to run it yourself!
