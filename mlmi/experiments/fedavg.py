from typing import Callable, Dict, List, Optional

from sacred import Experiment
from functools import partial

from torch import Tensor, optim

from mlmi.clustering import ModelFlattenWeightsPartitioner
from mlmi.experiments.log import log_goal_test_acc, log_loss_and_acc
from mlmi.fedavg.data import scratch_labels
from mlmi.fedavg.femnist import load_femnist_dataset, load_mnist_dataset
from mlmi.fedavg.model import CNNLightning, CNNMnistLightning, FedAvgServer
from mlmi.fedavg.run import run_fedavg
from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.fedavg.util import load_fedavg_state, run_fedavg_round, run_fedavg_train_round
from mlmi.hierarchical.run import run_fedavg_hierarchical
from mlmi.participant import BaseTrainingParticipant
from mlmi.plot import generate_client_label_heatmap, generate_data_label_heatmap
from mlmi.settings import REPO_ROOT
from mlmi.structs import ClusterArgs, FederatedDatasetData, ModelArgs, OptimizerArgs, TrainArgs
from mlmi.utils import create_tensorboard_logger, fix_random_seeds, overwrite_participants_models

ex = Experiment('fedavg')


@ex.config
def femnist():
    seed = 123123123
    lr = 0.1
    name = 'hptest_lr'
    total_fedavg_rounds = 20
    client_fraction = [0.1]
    local_epochs = 3
    batch_size = 10
    num_clients = 367
    sample_threshold = 250
    num_label_limit = 15
    num_classes = 62
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    train_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs, progress_bar_refresh_rate=0)
    model_args = ModelArgs(CNNLightning, optimizer_args=optimizer_args, only_digits=False)
    dataset = 'femnist'


@ex.named_config
def mnist():
    seed = 123123123
    lr = 0.1
    name = 'mnist'
    total_fedavg_rounds = 50
    client_fraction = [0.1]
    local_epochs = 3
    batch_size = 10
    num_clients = 100
    num_classes = 10
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    train_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs, progress_bar_refresh_rate=0)
    model_args = ModelArgs(CNNMnistLightning, optimizer_args=optimizer_args, num_classes=num_classes)
    dataset = 'mnist'


def log_after_round_evaluation(
        experiment_logger,
        tag: str,
        loss: Tensor,
        acc: Tensor,
        step: int
):
    log_loss_and_acc(tag, loss, acc, experiment_logger, step)
    log_goal_test_acc(tag, acc, experiment_logger, step)


def log_dataset_distribution(experiment_logger, tag: str, dataset: FederatedDatasetData):
    dataloaders = list(dataset.train_data_local_dict.values())
    image = generate_data_label_heatmap(tag, dataloaders, dataset.class_num)
    experiment_logger.experiment.add_image('label distribution', image.numpy())


@ex.automain
def run_fedavg_experiment(
        seed,
        lr,
        name,
        total_fedavg_rounds,
        client_fraction,
        local_epochs,
        batch_size,
        num_clients,
        sample_threshold,
        num_label_limit,
        optimizer_args,
        train_args,
        model_args,
        dataset,
):
    fix_random_seeds(seed)

    if dataset == 'femnist':
        fed_dataset = load_femnist_dataset(str((REPO_ROOT / 'data').absolute()),
                                           num_clients=num_clients, batch_size=batch_size,
                                           sample_threshold=sample_threshold)

        if num_label_limit != -1:
            fed_dataset = scratch_labels(fed_dataset, num_label_limit)
    elif dataset == 'mnist':
        fed_dataset = load_mnist_dataset(str((REPO_ROOT / 'data').absolute()),
                                         num_clients=num_clients, batch_size=batch_size)
    else:
        raise ValueError(f'dataset "{dataset}" unknown')


    data_distribution_logged = False
    for cf in client_fraction:
        fedavg_context = FedAvgExperimentContext(name=name, client_fraction=cf, local_epochs=local_epochs,
                                                 lr=lr, batch_size=batch_size, optimizer_args=optimizer_args,
                                                 model_args=model_args, train_args=train_args,
                                                 dataset_name=dataset)
        experiment_specification = f'{fedavg_context}'
        experiment_logger = create_tensorboard_logger(fedavg_context.name, experiment_specification)

        if not data_distribution_logged:
            log_dataset_distribution(experiment_logger, 'full dataset', fed_dataset)
            data_distribution_logged = True

        log_after_round_evaluation_fns = [
            partial(log_after_round_evaluation, experiment_logger, 'fedavg')
        ]
        run_fedavg(context=fedavg_context, num_rounds=total_fedavg_rounds, dataset=fed_dataset,
                   save_states=True, restore_state=False, after_round_evaluation=log_after_round_evaluation_fns)
