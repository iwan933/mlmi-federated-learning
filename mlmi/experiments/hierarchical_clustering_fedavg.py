from typing import Callable, Dict, List, Optional

from sacred import Experiment
from functools import partial

from torch import Tensor, optim

from mlmi.clustering import ModelFlattenWeightsPartitioner
from mlmi.experiments.log import log_goal_test_acc, log_loss_and_acc
from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import CNNLightning, CNNMnistLightning, FedAvgServer
from mlmi.fedavg.run import run_fedavg
from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.fedavg.util import load_fedavg_state, run_fedavg_round, run_fedavg_train_round
from mlmi.hierarchical.run import run_fedavg_hierarchical
from mlmi.participant import BaseTrainingParticipant
from mlmi.settings import REPO_ROOT
from mlmi.structs import ClusterArgs, ModelArgs, OptimizerArgs, TrainArgs
from mlmi.utils import create_tensorboard_logger, fix_random_seeds, overwrite_participants_models

ex = Experiment('hierachical_clustering')


@ex.named_config
def briggs():
    seed = 123123123
    lr = 0.1
    name = 'briggs'
    total_fedavg_rounds = 50
    cluster_initialization_rounds = [1, 3, 5, 10]
    client_fraction = [0.1]
    local_epochs = 3
    batch_size = 10
    num_clients = 367
    num_classes = 62
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    train_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs, progress_bar_refresh_rate=0)
    model_args = ModelArgs(CNNLightning, optimizer_args=optimizer_args, only_digits=False)
    dataset = 'femnist'
    partitioner_class = ModelFlattenWeightsPartitioner
    linkage_mech = 'ward'
    criterion = 'distance'
    dis_metric = 'euclidean'
    max_value_criterion = 10.0


def log_after_round_evaluation(
        experiment_logger,
        tag: str,
        loss: Tensor,
        acc: Tensor,
        step: int
):
    log_loss_and_acc(tag, loss, acc, experiment_logger, step)
    log_goal_test_acc(tag, acc, experiment_logger, step)


def log_cluster_distribution(
        cluster_clients_dic: Dict[str, List['BaseTrainingParticipant']]
):
    # TODO: log the heatmap
    return


@ex.automain
def run_hierarchical_clustering(
        seed,
        lr,
        name,
        total_fedavg_rounds,
        cluster_initialization_rounds,
        client_fraction,
        local_epochs,
        batch_size,
        num_clients,
        optimizer_args,
        train_args,
        model_args,
        dataset,
        partitioner_class,
        linkage_mech,
        criterion,
        dis_metric,
        max_value_criterion
):
    fix_random_seeds(seed)
    global_tag = 'global_performance'

    if dataset == 'femnist':
        fed_dataset = load_femnist_dataset(str((REPO_ROOT / 'data').absolute()),
                                           num_clients=num_clients, batch_size=batch_size)
    else:
        raise ValueError(f'dataset "{dataset}" unknown')

    # TODO: log the heatmap
    # log_data_distribution_by_dataset('fedavg', dataset, context.experiment_logger)

    for cf in client_fraction:
        fedavg_context = FedAvgExperimentContext(name=name, client_fraction=cf, local_epochs=local_epochs,
                                                 lr=lr, batch_size=batch_size, optimizer_args=optimizer_args,
                                                 model_args=model_args, train_args=train_args,
                                                 dataset_name=dataset)
        experiment_specification = f'{fedavg_context}'
        experiment_logger = create_tensorboard_logger(fedavg_context.name, experiment_specification)

        log_after_round_evaluation_fns = [
            partial(log_after_round_evaluation, experiment_logger, 'fedavg'),
            partial(log_after_round_evaluation, experiment_logger, global_tag)
        ]
        server, clients = run_fedavg(context=fedavg_context, num_rounds=total_fedavg_rounds, dataset=fed_dataset,
                                     save_states=True, restore_state=True,
                                     after_round_evaluation=log_after_round_evaluation_fns)

        for init_rounds in cluster_initialization_rounds:
            # load the model state
            round_model_state = load_fedavg_state(fedavg_context, init_rounds)
            overwrite_participants_models(round_model_state, clients)
            # initialize the cluster configuration
            round_configuration = {
                'num_rounds_init': init_rounds,
                'num_rounds_cluster': total_fedavg_rounds - init_rounds
            }
            cluster_args = ClusterArgs(partitioner_class, linkage_mech=linkage_mech,
                                       criterion=criterion, dis_metric=dis_metric,
                                       max_value_criterion=max_value_criterion,
                                       plot_dendrogram=False, **round_configuration)
            # create new logger for cluster experiment
            experiment_specification = f'{fedavg_context}_{cluster_args}'
            experiment_logger = create_tensorboard_logger(fedavg_context.name, experiment_specification)
            fedavg_context.experiment_logger = experiment_logger

            initial_train_fn = partial(run_fedavg_train_round, round_model_state, training_args=train_args)
            create_aggregator_fn = partial(FedAvgServer, model_args=model_args, context=fedavg_context)
            federated_round_fn = partial(run_fedavg_round, training_args=train_args, client_fraction=cf)








            after_post_clustering_evaluation = [
                partial(log_after_round_evaluation, experiment_logger, 'post_clustering')
            ]
            after_clustering_round_evaluation = [
                partial(log_after_round_evaluation, experiment_logger)
            ]
            after_federated_round_evaluation = [
                partial(log_after_round_evaluation, experiment_logger, 'final hierarchical'),
                partial(log_after_round_evaluation, experiment_logger, global_tag)
            ]
            after_clustering_fn = [
                partial(log_cluster_distribution)
            ]
            run_fedavg_hierarchical(server, clients, cluster_args,
                                    initial_train_fn,
                                    federated_round_fn,
                                    create_aggregator_fn,
                                    after_post_clustering_evaluation,
                                    after_clustering_round_evaluation,
                                    after_federated_round_evaluation,
                                    after_clustering_fn)

