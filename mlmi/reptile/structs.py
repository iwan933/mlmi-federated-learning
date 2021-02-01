import torch

from mlmi.structs import ModelArgs, OptimizerArgs, TrainArgs, ClusterArgs


class ReptileTrainingArgs:
    """
    Container for meta-learning parameters
    :param model: Class of base model
    :param inner_optimizer: Optimizer on task level
    :param inner_learning_rate: Learning rate for task level optimizer
    :param num_inner_steps: Number of training steps on task level
    :param num_inner_steps_eval: Number of training steps for training on test
        set
    :param log_every_n_steps:
    :param inner_batch_size: Batch size for training on task level. A value of -1
        means batch size is equal to local training set size (full batch
        training)
    :param meta_batch_size: Batch size of tasks for single meta-training step.
        A value of -1 means meta batch size is equal to total number of training
        tasks (full batch meta training)
    :param meta_learning_rate_initial: Learning rate for meta training (initial
        value). Learning rate decreases linearly with training progress to reach
        meta_learning_rate_final at end of training.
    :param meta_learning_rate_final: Final value for learning rate for meta
        training. If None, this will be equal to meta_learning_rate_initial and
        learning rate will remain constant over training.
    :param num_meta_steps: Number of total meta training steps
    :return:
    """
    def __init__(self,
                 model_class,
                 inner_optimizer,
                 inner_learning_rate=0.03,
                 num_inner_steps=1,
                 num_inner_steps_eval=50,
                 log_every_n_steps=3,
                 meta_learning_rate_initial=0.03,
                 meta_learning_rate_final=None,
                 num_classes_per_client=5):
        self.model_class = model_class
        self.inner_optimizer = inner_optimizer
        self.inner_learning_rate = inner_learning_rate
        self.num_inner_steps = num_inner_steps
        self.num_inner_steps_eval = num_inner_steps_eval
        self.log_every_n_steps = log_every_n_steps
        self.meta_learning_rate_initial = meta_learning_rate_initial
        self.meta_learning_rate_final = meta_learning_rate_final
        if self.meta_learning_rate_final is None:
            self.meta_learning_rate_final = self.meta_learning_rate_initial
        self.num_classes_per_client = num_classes_per_client

    def get_inner_model_args(self):
        inner_optimizer_args = OptimizerArgs(
            optimizer_class=self.inner_optimizer,
            lr=self.inner_learning_rate,
            betas=(0, 0.999)
        )
        return ModelArgs(
            model_class=self.model_class,
            optimizer_args=inner_optimizer_args,
            num_classes=self.num_classes_per_client
        )

    def get_meta_model_args(self):
        dummy_optimizer_args = OptimizerArgs(
            optimizer_class=torch.optim.SGD
        )
        return ModelArgs(
            model_class=self.model_class,
            optimizer_args=dummy_optimizer_args,
            num_classes=self.num_classes_per_client
        )

    def get_inner_training_args(self, eval=False):
        """
        Return TrainArgs for inner training (training on task level)
        """
        inner_training_args = TrainArgs(
            min_steps=self.num_inner_steps if not eval else self.num_inner_steps_eval,
            max_steps=self.num_inner_steps if not eval else self.num_inner_steps_eval,
            log_every_n_steps=self.log_every_n_steps,
            weights_summary=None,  # Do not show model summary
            progress_bar_refresh_rate=0  # Do not show training progress bar
        )
        if torch.cuda.is_available():
            inner_training_args.kwargs['gpus'] = 1
        return inner_training_args

    def get_meta_training_args(self, frac_done: float):
        """
        Return TrainArgs for meta training
        :param frac_done: Fraction of meta training steps already done
        """
        return TrainArgs(
            meta_learning_rate=frac_done * self.meta_learning_rate_final + \
                                 (1 - frac_done) * self.meta_learning_rate_initial
        )


class ReptileExperimentContext(object):
    """
    Structure to hold experiment context information
    """

    def __init__(self,
                 name: str, inner_training_steps: int, inner_learning_rate: float,
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
