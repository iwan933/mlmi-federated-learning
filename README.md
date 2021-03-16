## Load the data
Create a directory ``data`` in ``/path/to/project`` and download the ham10k dataset from kaggle: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

Extract the data set such that the following structure is present:

````
|- data
|----/ ham10k
|-------- /HAM10000_images_part_1
|-------- /HAM10000_images_part_2
|-------- HAM10000_metadata.csv
|-------- hmnist_8_8_L.csv
|-------- hmnist_8_8_RGB.csv
|-------- hmnist_28_28_L.csv
|-------- hmnist_28_28_RGB.csv 
|- mlmi
...
````

## Execute the experiments
Experiments are provided with sacred and can be directly called with the papers parameter from commandline.

1. install python requirements ``pip install -r requirements.txt`` (for pytorch & tensorflow installations refer to the respective project)
2. add the project to python path ``cd /path/to/project/ && set pythonpath=%cd%``
3. run the experiments:

* the centralized standard setting ``python mlmi\experiments\ham10k_full_dataset.py``
* reptile (default) ``python mlmi\experiments\federated.py``
* fedavg ``python mlmi\experiments\federated.py with fedavg``
* reptile + clustering (default) ``python mlmi\experiments\federated_clustering.py``
* fedavg + clustering ``python mlmi\experiments\federated_clustering.py with fedavg``
