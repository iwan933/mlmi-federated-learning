from typing import Callable, Dict, List, Optional

from sacred import Experiment
from functools import partial

import torch
from torch import Tensor, optim

from mlmi.clustering import DatadependentPartitioner, FixedAlternativePartitioner, ModelFlattenWeightsPartitioner, \
    AlternativePartitioner, \
    RandomClusterPartitioner
from mlmi.datasets.ham10k import load_ham10k_federated
from mlmi.log import getLogger
from mlmi.experiments.log import log_goal_test_acc, log_loss_and_acc
from mlmi.fedavg.data import load_n_of_each_class, scratch_labels
from mlmi.fedavg.femnist import load_femnist_colored_dataset, load_femnist_dataset
from mlmi.fedavg.ham10k import initialize_ham10k_clients
from mlmi.fedavg.model import CNNLightning, CNNMnistLightning, FedAvgServer
from mlmi.fedavg.run import DEFAULT_CLIENT_INIT_FN, run_fedavg
from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.fedavg.util import evaluate_cluster_models, load_fedavg_state, run_fedavg_round, run_fedavg_train_round
from mlmi.hierarchical.run import run_fedavg_hierarchical
from mlmi.models.ham10k import GlobalConfusionMatrix, MobileNetV2Lightning
from mlmi.participant import BaseParticipant, BaseTrainingParticipant
from mlmi.plot import generate_client_label_heatmap, generate_confusion_matrix_heatmap, generate_data_label_heatmap
from mlmi.settings import REPO_ROOT
from mlmi.structs import ClusterArgs, FederatedDatasetData, ModelArgs, OptimizerArgs, TrainArgs
from mlmi.utils import create_tensorboard_logger, evaluate_local_models, fix_random_seeds, overwrite_participants_models


logger = getLogger(__name__)
ex = Experiment('hierachical_clustering')


@ex.named_config
def ham10k():
    local_evaluation_steps = 7
    seed = 123123123
    lr = [0.01]
    name = 'ham10k'
    total_fedavg_rounds = 150
    cluster_initialization_rounds = [20]
    client_fraction = [0.3]
    local_epochs = 1
    batch_size = 16
    num_clients = 27
    sample_threshold = -1  # we need clients with at least 250 samples to make sure all labels are present
    num_label_limit = -1
    num_classes = 7
    train_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs, progress_bar_refresh_rate=0)
    train_cluster_args = TrainArgs(max_epochs=3, min_epochs=3, progress_bar_refresh_rate=0)
    dataset = 'ham10k'
    partitioner_class = FixedAlternativePartitioner
    linkage_mech = 'ward'
    criterion = 'distance'
    dis_metric = 'euclidean'
    max_value_criterion = [300.00]
    reallocate_clients = False
    threshold_min_client_cluster = -1
    use_colored_images = False
    use_pattern = False
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def log_after_round_evaluation(
        experiment_logger,
        tags,
        loss: Tensor,
        acc: Tensor,
        step: int
):
    if type(tags) is not list:
        tags = [tags]
    try:
        global_confusion_matrix = GlobalConfusionMatrix()
        if global_confusion_matrix.has_data:
            matrix = global_confusion_matrix.compute()
            for tag in tags:
                image = generate_confusion_matrix_heatmap(matrix, title=tag)
                experiment_logger.experiment.add_image(tag, image.numpy(), step)
    except Exception as e:
        logger.error('failed to log confusion matrix', e)

    for tag in tags:
        log_loss_and_acc(tag, loss, acc, experiment_logger, step)
        log_goal_test_acc(tag, acc, experiment_logger, step)


def log_personalized_performance(
        experiment_logger,
        tags,
        max_train_steps: int,
        server: BaseParticipant,
        clients,
        step: int
):
    if type(tags) is not list:
        tags = [tags]
    train_args = TrainArgs(max_steps=max_train_steps, progress_bar_refresh_rate=0)
    run_fedavg_train_round(server.model.state_dict(), clients, train_args)
    result = evaluate_local_models(participants=clients)
    loss, acc = result.get('test/loss'), result.get('test/acc')
    for tag in tags:
        log_after_round_evaluation(experiment_logger, tag, loss, acc, step)


def log_personalized_global_cluster_performance(
        experiment_logger,
        tags,
        max_train_steps: int,
        cluster_server_dic,
        cluster_clients_dic,
        step: int
):
    if type(tags) is not list:
        tags = [tags]
    train_args = TrainArgs(max_steps=max_train_steps, progress_bar_refresh_rate=0)
    for cluster_id in cluster_clients_dic.keys():
        server = cluster_server_dic[cluster_id]
        clients = cluster_clients_dic[cluster_id]
        run_fedavg_train_round(server.model.state_dict(), clients, train_args)
    loss, acc = evaluate_cluster_models(cluster_server_dic, cluster_clients_dic, evaluate_local=True)
    for tag in tags:
        log_after_round_evaluation(experiment_logger, tag, loss, acc, step)


def log_cluster_distribution(
        experiment_logger,
        cluster_clients_dic: Dict[str, List['BaseTrainingParticipant']],
        num_classes
):
    for cluster_id, clients in cluster_clients_dic.items():
        image = generate_client_label_heatmap(f'cluster {cluster_id}', clients, num_classes)
        experiment_logger.experiment.add_image(f'label distribution/cluster_{cluster_id}', image.numpy())


def log_sample_images_from_each_client(
        experiment_logger,
        cluster_clients_dic: Dict[str, List['BaseTrainingParticipant']]
):
    import numpy as np
    for cluster_id, clients in cluster_clients_dic.items():
        images = []
        for c in clients:
            x, y = next(c.train_data_loader.__iter__())
            images.append(x[0].numpy())
        images_array = np.stack(images, axis=0)
        experiment_logger.experiment.add_image(f'color distribution/cluster_{cluster_id}',
                                               images_array,
                                               dataformats='NCHW')


def log_dataset_distribution(experiment_logger, tag: str, dataset: FederatedDatasetData):
    dataloaders = list(dataset.train_data_local_dict.values())
    image = generate_data_label_heatmap(tag, dataloaders, dataset.class_num)
    experiment_logger.experiment.add_image('label distribution', image.numpy())


def generate_configuration(init_rounds_list, max_value_criterion_list):
    for ri in init_rounds_list:
        for mv in max_value_criterion_list:
            yield ri, mv


@ex.automain
def run_hierarchical_clustering(
        local_evaluation_steps,
        seed,
        lr,
        name,
        total_fedavg_rounds,
        cluster_initialization_rounds,
        client_fraction,
        local_epochs,
        batch_size,
        num_clients,
        sample_threshold,
        num_label_limit,
        train_args,
        dataset,
        partitioner_class,
        linkage_mech,
        criterion,
        dis_metric,
        max_value_criterion,
        reallocate_clients,
        threshold_min_client_cluster,
        use_colored_images,
        use_pattern,
        train_cluster_args=None,
        mean=None,
        std=None
):
    fix_random_seeds(seed)
    global_tag = 'global_performance'
    global_tag_local = 'global_performance_personalized'
    initialize_clients_fn = DEFAULT_CLIENT_INIT_FN
    if dataset == 'ham10k':
        fed_dataset = load_ham10k_federated(partitions=num_clients, batch_size=batch_size, mean=mean, std=std)
        initialize_clients_fn = initialize_ham10k_clients
    else:
        raise ValueError(f'dataset "{dataset}" unknown')

    if not hasattr(max_value_criterion, '__iter__'):
        max_value_criterion = [max_value_criterion]
    if not hasattr(lr, '__iter__'):
        lr = [lr]

    for cf in client_fraction:
        for lr_i in lr:
            optimizer_args = OptimizerArgs(optim.SGD, lr=lr_i)
            model_args = ModelArgs(MobileNetV2Lightning, optimizer_args=optimizer_args, num_classes=7)
            fedavg_context = FedAvgExperimentContext(name=name, client_fraction=cf, local_epochs=local_epochs,
                                                     lr=lr_i, batch_size=batch_size, optimizer_args=optimizer_args,
                                                     model_args=model_args, train_args=train_args,
                                                     dataset_name=dataset)
            experiment_specification = f'{fedavg_context}'
            experiment_logger = create_tensorboard_logger(fedavg_context.name, experiment_specification)
            fedavg_context.experiment_logger = experiment_logger
            for init_rounds, max_value in generate_configuration(cluster_initialization_rounds, max_value_criterion):
                # load the model state
                round_model_state = load_fedavg_state(fedavg_context, init_rounds)

                server = FedAvgServer('initial_server', fedavg_context.model_args, fedavg_context)
                server.overwrite_model_state(round_model_state)
                logger.info('initializing clients ...')
                clients = initialize_clients_fn(fedavg_context, fed_dataset, server.model.state_dict())

                overwrite_participants_models(round_model_state, clients)
                # initialize the cluster configuration
                round_configuration = {
                    'num_rounds_init': init_rounds,
                    'num_rounds_cluster': total_fedavg_rounds - init_rounds
                }
                if partitioner_class == DatadependentPartitioner:
                    clustering_dataset = load_femnist_colored_dataset(str((REPO_ROOT / 'data').absolute()),
                                                              num_clients=num_clients, batch_size=batch_size,
                                                              sample_threshold=sample_threshold)
                    dataloader = load_n_of_each_class(clustering_dataset, n=5,
                                                      tabu=list(fed_dataset.train_data_local_dict.keys()))
                    cluster_args = ClusterArgs(partitioner_class, linkage_mech=linkage_mech,
                                               criterion=criterion, dis_metric=dis_metric,
                                               max_value_criterion=max_value,
                                               plot_dendrogram=False, reallocate_clients=reallocate_clients,
                                               threshold_min_client_cluster=threshold_min_client_cluster,
                                               dataloader=dataloader,
                                               **round_configuration)
                else:
                    cluster_args = ClusterArgs(partitioner_class, linkage_mech=linkage_mech,
                                               criterion=criterion, dis_metric=dis_metric,
                                               max_value_criterion=max_value,
                                               plot_dendrogram=False, reallocate_clients=reallocate_clients,
                                               threshold_min_client_cluster=threshold_min_client_cluster,
                                               **round_configuration)
                # create new logger for cluster experiment
                experiment_specification = f'{fedavg_context}_{cluster_args}'
                experiment_logger = create_tensorboard_logger(fedavg_context.name, experiment_specification)
                fedavg_context.experiment_logger = experiment_logger

                initial_train_fn = partial(run_fedavg_train_round, round_model_state, training_args=train_cluster_args)
                create_aggregator_fn = partial(FedAvgServer, model_args=model_args, context=fedavg_context)
                federated_round_fn = partial(run_fedavg_round, training_args=train_args, client_fraction=cf)

                after_post_clustering_evaluation = [
                    partial(log_after_round_evaluation, experiment_logger, 'post_clustering')
                ]
                after_clustering_round_evaluation = [
                    partial(log_after_round_evaluation, experiment_logger)
                ]
                after_federated_round_evaluation = [
                    partial(log_after_round_evaluation, experiment_logger, ['final hierarchical', global_tag])
                ]
                after_clustering_fn = [
                    partial(log_cluster_distribution, experiment_logger, num_classes=fed_dataset.class_num),
                    partial(log_sample_images_from_each_client, experiment_logger)
                ]
                after_federated_round_fn = [
                    partial(log_personalized_global_cluster_performance, experiment_logger,
                            ['final hierarchical personalized', global_tag_local], local_evaluation_steps)
                ]
                run_fedavg_hierarchical(server, clients, cluster_args,
                                        initial_train_fn,
                                        federated_round_fn,
                                        create_aggregator_fn,
                                        after_post_clustering_evaluation,
                                        after_clustering_round_evaluation,
                                        after_federated_round_evaluation,
                                        after_clustering_fn,
                                        after_federated_round=after_federated_round_fn
                                        )
