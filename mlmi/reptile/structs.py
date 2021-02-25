import torch

from mlmi.structs import ModelArgs, OptimizerArgs, TrainArgs, ClusterArgs


class ReptileExperimentContext(object):
    """
    Structure to hold experiment context information
    """

    def __init__(self,
                 name: str,
                 dataset_name: str,
                 swap_labels: bool,
                 num_classes_per_client: int,
                 num_shots_per_class: int,
                 seed: int,
                 model_class,
                 sgd: bool,
                 adam_betas: tuple,
                 num_clients_train: int,
                 num_clients_test: int,
                 meta_batch_size: int,
                 num_meta_steps: int,
                 meta_learning_rate_initial: float,
                 meta_learning_rate_final: float,
                 eval_interval: int,
                 num_eval_clients_training: int,
                 do_final_evaluation: bool,
                 num_eval_clients_final: int,
                 inner_batch_size: int,
                 inner_learning_rate: float,
                 num_inner_steps: int,
                 num_inner_steps_eval: int):
        self.name = name
        self.seed = seed

        # Arguments pertaining to data set
        self.dataset_name = dataset_name
        self.swap_labels = swap_labels
        self.num_classes_per_client = num_classes_per_client
        self.num_shots_per_class = num_shots_per_class
        self.num_clients_train = num_clients_train
        self.num_clients_test = num_clients_test

        # Model arguments
        self.model_class = model_class
        self.sgd = sgd
        self.adam_betas = adam_betas

        # Arguments for inner training
        self.num_inner_steps = num_inner_steps
        self.inner_learning_rate = inner_learning_rate
        self.inner_batch_size = inner_batch_size
        if self.sgd:
            inner_optimizer_args = OptimizerArgs(
                optimizer_class=torch.optim.SGD,
                lr=self.inner_learning_rate
            )
        else:
            inner_optimizer_args = OptimizerArgs(
                optimizer_class=torch.optim.Adam,
                lr=self.inner_learning_rate,
                betas=self.adam_betas
            )
        if self.dataset_name == 'omniglot':
            self.inner_model_args = ModelArgs(
                model_class=self.model_class,
                optimizer_args=inner_optimizer_args,
                num_classes=self.num_classes_per_client
            )
        elif self.dataset_name == 'ham10k':
            self.inner_model_args = ModelArgs(
                model_class=self.model_class,
                optimizer_args=inner_optimizer_args,
                num_classes=7
            )
        else:
            self.inner_model_args = ModelArgs(
                model_class=self.model_class,
                optimizer_args=inner_optimizer_args,
            )

        # Arguments for meta training
        self.num_meta_steps = num_meta_steps
        self.meta_learning_rate_initial = meta_learning_rate_initial
        self.meta_learning_rate_final = meta_learning_rate_final
        self.meta_batch_size = meta_batch_size  # number of clients per round
        if self.dataset_name == 'omniglot':
            self.meta_model_args = ModelArgs(
                model_class=self.model_class,
                optimizer_args=OptimizerArgs(  # Dummy optimizer args
                    optimizer_class=torch.optim.SGD
                ),
                num_classes=self.num_classes_per_client
            )
        elif self.dataset_name == 'ham10k':
            self.meta_model_args = ModelArgs(
                model_class=self.model_class,
                optimizer_args=OptimizerArgs(  # Dummy optimizer args
                    optimizer_class=torch.optim.SGD
                ),
                num_classes=7
            )
        else:
            self.meta_model_args = ModelArgs(
                model_class=self.model_class,
                optimizer_args=OptimizerArgs(  # Dummy optimizer args
                    optimizer_class=torch.optim.SGD
                )
            )

        # Arguments for evaluation
        self.num_inner_steps_eval = num_inner_steps_eval
        self.eval_interval = eval_interval
        self.num_eval_clients_training = num_eval_clients_training
        if num_eval_clients_training > num_clients_train or \
                (num_clients_test and num_eval_clients_training > num_clients_test):
            raise ValueError(
                "num_eval_clients_training must be lower or equal to "
                "num_clients_train and num_clients_test"
            )
        self.do_final_evaluation = do_final_evaluation
        if do_final_evaluation and (
                num_eval_clients_final > num_clients_train or
                (num_clients_test and num_eval_clients_final > num_clients_test)
        ):
            raise ValueError(
                "num_eval_clients_final must be lower or equal to "
                "num_clients_train and num_clients_test"
            )
        self.num_eval_clients_final = num_eval_clients_final

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

    def get_inner_training_args(self, eval=False):
        """
        Return TrainArgs for inner training (training on task level)
        """
        inner_training_args = TrainArgs(
            min_steps=self.num_inner_steps if not eval else self.num_inner_steps_eval,
            max_steps=self.num_inner_steps if not eval else self.num_inner_steps_eval,
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

    def __str__(self):
        """
        String identifying experiment. Used for model loading and saving.
        :return:
        """
        dataset_string = ""
        if self.dataset_name == 'omniglot':
            dataset_string = (
                f"{self.num_classes_per_client}-way{self.num_shots_per_class}-shot"
            )
        elif self.dataset_name == 'femnist':
            dataset_string = (
                f"{'swap' if self.swap_labels else ''}"
            )
        data_size_string = (
            f"trainc{self.num_clients_train}"
            f"testc{self.num_clients_test}"
        )
        model_string = f"{'sgd' if self.sgd else 'adam'}"
        inner_learning_string = (
            f"is{self.num_inner_steps}"
            f"ilr{str(self.inner_learning_rate).replace('.', '')}"
            f"ib{self.inner_batch_size}"
        )
        meta_learning_string = (
            f"ms{self.num_meta_steps}"
            f"mlri{str(self.meta_learning_rate_initial).replace('.', '')}"
            f"mlrf{str(self.meta_learning_rate_final).replace('.', '')}"
            f"mb{self.meta_batch_size}"
        )
        evaluation_string = (
            f"isev{self.num_inner_steps_eval}"
            f"ect{self.num_eval_clients_training}"
            f"ecf{self.num_eval_clients_final if self.do_final_evaluation else 'N'}"
        )
        experiment_string = (
            f"{self.dataset_name}_seed{self.seed}_"
            f"{dataset_string}{data_size_string}_"
            f"{model_string}_{inner_learning_string}_"
            f"{meta_learning_string}_{evaluation_string}"
        )
        return experiment_string
