# AutoML techniques in Federated Learning

This repository is used to evaluate how federated learning can be improved using AutoML techniques.

> :warning: Please use **Python 3.8** to run this project, PyTorch pip packaging is currently not working with Python 3.9 (https://github.com/pytorch/pytorch/issues/47116).

> :warning: Due to a Windows bug do not upgrade the current numpy and stay with version 1.19.3 (https://github.com/numpy/numpy/wiki/FMod-Bug-on-Windows)

## Installation

Create a new virtual environment and install the requirements in `requirements.txt`

## Running experiments

Add the repository root to your python path e.g. on windows `set PYTHONPATH=%cd%`

Run the sacred experiments in ``mlmi/experiments``

e.g.

run fedavg with the MNIST dataset
``python mlmi/experiments/fedavg.py with mnist``

run fedavg with the FEMNIST dataset
``python mlmi/experiments/fedavg.py``
