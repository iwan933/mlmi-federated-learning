# AutoML techniques in Federated Learning

This repository is used to evaluate how federated learning can be improved using AutoML techniques.

> :warning: Please use **Python 3.8** to run this project, PyTorch pip packaging is currently not working with Python 3.9 (https://github.com/pytorch/pytorch/issues/47116).

> :warning: Due to a Windows bug do not upgrade the current numpy and stay with version 1.19.3 (https://github.com/numpy/numpy/wiki/FMod-Bug-on-Windows)

## Installation on windows

The given implementation is based on https://github.com/FedML-AI/FedML thus to execute the code, make the repository 
available in your python path.

For windows with virtualenv & virtualenvwrapper (https://timmyreilly.azurewebsites.net/python-pip-virtualenv-installation-on-windows/) use the following commands to get started:

1. Clone the **forked repository** ``git clone https://github.com/iwan933/FedML``
    > the forked repository fixes some pathing issues that need to be addressed to use the data loaders. 

2. Create a new virtual python environment ``mkvirtualenv myproject-env``

3. Activate the virtualenv ``workon myproject-env``

4. Add the FedML repository as a library to your python path virtualenvs ``add2virtualenv /path/to/FedML``

5. Install the packages ``pip install -r requirements.txt``

## Download the testdata

FedML provides shell scripts to download the data, which do not work in windows.
To test your set up download the files manually and locate them in your repository.

### Unix

1. Run the download ``./download_datasets.sh``

### Windows

1. The only required package for minimal setup is fed_cifar100. 
Manually download the files from the googledrive, links can be found 
here ``FedML/data/fed_cifar100/download_fedcifar100.sh``, then copy these
to ``FedML/data/fed_cifar100``.

## Test the setup

To test the setup run the FedAvg algorithm from the FedML package by using the ``run`` script
provided in this repository.

1. Set create a login at wandb.com and copy your api key

2. Set the environment variable Windows: ``set WANDB_API_KEY=[my-api-key]`` Linux: ``export WANDB_API_KEY=[my-api-key]``
    > you might want to persist the environment variable for future runs
3. Adapt the FedML repository path ``--data_dir [my-path]`` in Windows: ``run.cmd`` Linux: ``run.sh``
