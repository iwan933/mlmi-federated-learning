from mlmi.structs import ClusterArgs, ModelArgs, OptimizerArgs, TrainArgs


class FedAvgExperimentContext(object):
    """
    Structure to hold experiment context information
    """

    def __init__(self, name: str, client_fraction: float, local_epochs: int, lr: float,
                 batch_size: int, optimizer_args: 'OptimizerArgs', train_args: 'TrainArgs', model_args: 'ModelArgs',
                 dataset_name: str):
        self.name = name
        self.client_fraction = client_fraction
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer_args = optimizer_args
        self.train_args = train_args
        self.model_args = model_args
        self.dataset_name = dataset_name
        self._cluster_args = None
        self._experiment_logger = None

    @property
    def cluster_args(self) -> 'ClusterArgs':
        return self._cluster_args

    @cluster_args.setter
    def cluster_args(self, value: 'ClusterArgs'):
        self._cluster_args = value

    @property
    def experiment_logger(self):
        return self._experiment_logger

    @experiment_logger.setter
    def experiment_logger(self, value):
        self._experiment_logger = value

    def __str__(self):
        """
        String identifying experiment. Used for model loading and saving.
        :return:
        """
        id = f'{self.dataset_name}_bs{self.batch_size}lr{self.lr:.2E}cf{self.client_fraction:.2f}{self.train_args}' \
             f'_{self.optimizer_args}'
        return id
