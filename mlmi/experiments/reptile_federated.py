import sys
sys.path.append('C:/Users/Richard/Desktop/Informatik/Semester_5/MLMI/git/mlmi-federated-learning')

from typing import Callable, Dict, List, Optional

from sacred import Experiment
from functools import partial

from torch import Tensor, optim

from mlmi.experiments.log import log_goal_test_acc, log_loss_and_acc
from mlmi.reptile.omniglot import load_omniglot_datasets
from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import CNNLightning
from mlmi.reptile.model import OmniglotLightning
from mlmi.reptile.structs import ReptileExperimentContext
from mlmi.reptile.run_reptile_experiment import run_reptile

from mlmi.plot import generate_data_label_heatmap
from mlmi.settings import REPO_ROOT
from mlmi.structs import FederatedDatasetData
from mlmi.utils import create_tensorboard_logger, fix_random_seeds

ex = Experiment('reptile')


@ex.config
def femnist():
    name = 'reptile'
    dataset = 'femnist'  # Options: 'omniglot', 'femnist'
    classes = 0  # Only used with dataset='omniglot'
    shots = 0  # Only used with dataset='omniglot'
    seed = 123123123

    model_class = CNNLightning
    sgd = True  # True -> Use SGD as inner optimizer; False -> Use Adam
    adam_betas = None  # Used only if sgd = False

    num_clients_train = 367
    num_clients_test = 0  # Used only with dataset='omniglot'
    meta_batch_size = 30
    num_meta_steps = 10000
    meta_learning_rate_initial = 1
    meta_learning_rate_final = 0

    eval_interval = 100
    num_eval_clients_training = 100
    do_final_evaluation = True
    num_eval_clients_final = -1

    inner_batch_size = 10
    inner_learning_rate = [0.005]
    num_inner_steps = 5
    num_inner_steps_eval = 50

@ex.named_config
def omniglot():
    name = 'reptile'
    dataset = 'omniglot'  # Options: 'omniglot', 'femnist'
    classes = 5  # Only used with dataset='omniglot'
    shots = 5  # Only used with dataset='omniglot'
    seed = 0

    model_class = OmniglotLightning
    sgd = True  # True -> Use SGD as inner optimizer; False -> Use Adam
    adam_betas = None  # Used only if sgd = False

    num_clients_train = 10000
    num_clients_test = 1000  # Used only with dataset='omniglot'
    meta_batch_size = 5
    num_meta_steps = 100000
    meta_learning_rate_initial = 1
    meta_learning_rate_final = 0

    eval_interval = 10
    num_eval_clients_training = 1
    do_final_evaluation = True
    num_eval_clients_final = 1000  # Applies only when do_final_evaluation=True

    inner_batch_size = 10
    inner_learning_rate = [0.001]
    num_inner_steps = 5
    num_inner_steps_eval = 50

def log_after_round_evaluation(
        experiment_logger,
        tag: str,
        loss_train_test: Tensor,
        acc_train_test: Tensor,
        loss_test_test: Tensor,
        acc_test_test: Tensor,
        step: int
    ):
    log_loss_and_acc(
        f'{tag}train-test',
        loss_train_test,
        acc_train_test,
        experiment_logger,
        step
    )
    if loss_test_test is not None and acc_test_test is not None:
        log_loss_and_acc(
            f'{tag}test-test',
            loss_test_test,
            acc_test_test,
            experiment_logger,
            step
        )


def log_dataset_distribution(experiment_logger, tag: str, dataset: FederatedDatasetData):
    dataloaders = list(dataset.train_data_local_dict.values())
    image = generate_data_label_heatmap(tag, dataloaders, dataset.class_num)
    experiment_logger.experiment.add_image('label distribution', image.numpy())


@ex.automain
def run_reptile_experiment(
    name,
    dataset,
    classes,
    shots,
    seed,
    model_class,
    sgd,
    adam_betas,
    num_clients_train,
    num_clients_test,
    meta_batch_size,
    num_meta_steps,
    meta_learning_rate_initial,
    meta_learning_rate_final,
    eval_interval,
    num_eval_clients_training,
    do_final_evaluation,
    num_eval_clients_final,
    inner_batch_size,
    inner_learning_rate,
    num_inner_steps,
    num_inner_steps_eval
):
    fix_random_seeds(seed)

    if dataset == 'femnist':
        fed_dataset_train = load_femnist_dataset(
            data_dir=str((REPO_ROOT / 'data').absolute()),
            num_clients=num_clients_train,
            batch_size=inner_batch_size,
            random_seed=seed
        )
        fed_dataset_test = None
    elif dataset == 'omniglot':
        fed_dataset_train, fed_dataset_test = load_omniglot_datasets(
            data_dir=str((REPO_ROOT / 'data' / 'omniglot').absolute()),
            num_clients_train=num_clients_train,
            num_clients_test=num_clients_test,
            num_classes_per_client=classes,
            num_shots_per_class=shots,
            inner_batch_size=inner_batch_size,
            random_seed=seed
        )
    else:
        raise ValueError(f'dataset "{dataset}" unknown')

    #data_distribution_logged = False
    for lr in inner_learning_rate:
        reptile_context = ReptileExperimentContext(
            name=name,
            dataset_name=dataset,
            num_classes_per_client=classes,
            num_shots_per_class=shots,
            seed=seed,
            model_class=model_class,
            sgd=sgd,
            adam_betas=adam_betas,
            num_clients_train=num_clients_train,
            num_clients_test=num_clients_test,
            meta_batch_size=meta_batch_size,
            num_meta_steps=num_meta_steps,
            meta_learning_rate_initial=meta_learning_rate_initial,
            meta_learning_rate_final=meta_learning_rate_final,
            eval_interval=eval_interval,
            num_eval_clients_training=num_eval_clients_training,
            do_final_evaluation=do_final_evaluation,
            num_eval_clients_final=num_eval_clients_final,
            inner_batch_size=inner_batch_size,
            inner_learning_rate=lr,
            num_inner_steps=num_inner_steps,
            num_inner_steps_eval=num_inner_steps_eval
        )

        experiment_specification = f'{reptile_context}'
        experiment_logger = create_tensorboard_logger(
            reptile_context.name, experiment_specification
        )
        reptile_context.experiment_logger = experiment_logger

        #if not data_distribution_logged:
        #    log_dataset_distribution(
        #        experiment_logger, 'full dataset', fed_dataset_train
        #    )
        #    data_distribution_logged = True

        log_after_round_evaluation_fns = [
            partial(log_after_round_evaluation, experiment_logger)
        ]
        run_reptile(
            context=reptile_context,
            dataset_train=fed_dataset_train,
            dataset_test=fed_dataset_test,
            initial_model_state=None,
            after_round_evaluation=log_after_round_evaluation_fns
        )
