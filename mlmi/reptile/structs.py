from mlmi.structs import ModelArgs, OptimizerArgs, TrainArgs, ClusterArgs


class ReptileExperimentContext(object):
    """
    Structure to hold experiment context information
    """

    def __init__(self, name: str, inner_training_steps: int, inner_learning_rate: float,
                 inner_batch_size: int, inner_optimizer_args: OptimizerArgs, inner_train_args: TrainArgs,
                 inner_model_args: ModelArgs, meta_model_args: ModelArgs, meta_batch_size: int,
                 dataset_name: str, meta_optimizer_args: OptimizerArgs, num_clients_train: int, num_clients_test: int,
                 num_classes_per_client: int, num_shots_per_class: int, eval_iters: int,
                 meta_learning_rate_initial: int, meta_learning_rate_final: int, meta_num_steps: int,
                 meta_training_args: TrainArgs):
        self.name = name

        # trainer arguments
        self.inner_training_steps = inner_training_steps
        self.inner_learning_rate = inner_learning_rate
        self.inner_batch_size = inner_batch_size
        self.inner_optimizer_args = inner_optimizer_args
        self.inner_train_args = inner_train_args
        self.inner_model_args = inner_model_args

        self.dataset_name = dataset_name

        # aggregator arguments
        self.meta_batch_size = meta_batch_size  # number of clients per round
        self.meta_model_args = meta_model_args
        self.meta_optimizer_args = meta_optimizer_args
        self.meta_learning_rate_initial = meta_learning_rate_initial
        self.meta_learning_rate_final = meta_learning_rate_final
        self.meta_num_steps = meta_num_steps
        self.meta_training_args = meta_training_args

        self.num_clients_train = num_clients_train
        self.num_clients_test = num_clients_test
        self.num_classes_per_client = num_classes_per_client
        self.num_shots_per_class = num_shots_per_class

        self.eval_iters = eval_iters

        self.weighted_aggregation: bool = True
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
        w = '1' if self.weighted_aggregation else '0'
        id = (
            f'{self.dataset_name}_ibs{self.inner_batch_size}ilr{self.inner_learning_rate:.2E}'
            f'is{self.inner_training_steps}'
            f'mbs{self.meta_batch_size}mlr{self.meta_learning_rate_initial}w{w}'
        )
        return id
