from sacred import Experiment
from functools import partial
from typing import Dict, List
from torch import optim

from mlmi.clustering import AlternativePartitioner
from mlmi.datasets.ham10k import load_ham10k_federated
from mlmi.experiments.fedavg import log_dataset_distribution
from mlmi.fedavg.ham10k import initialize_ham10k_clients
from mlmi.fedavg.run import run_fedavg
from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.fedavg.util import load_fedavg_state, run_fedavg_train_round
from mlmi.models.ham10k import MobileNetV2Lightning
from mlmi.plot import generate_client_label_heatmap
from mlmi.structs import TrainArgs, ModelArgs, OptimizerArgs, ClusterArgs
from mlmi.utils import fix_random_seeds, create_tensorboard_logger, overwrite_participants_models

ex = Experiment('clustering test')


@ex.config
def DefaultConfig():
    seed = 123123123
    lr = 0.01
    name = 'clustering_test'
    total_fedavg_rounds = 50
    client_fraction = 0.1
    local_epochs = 1
    batch_size = 16
    num_clients = 27
    num_classes = 7
    train_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs, progress_bar_refresh_rate=0)
    train_cluster_args = TrainArgs(max_epochs=3, min_epochs=3, progress_bar_refresh_rate=0)
    dataset = 'ham10k'
    partitioner_class = AlternativePartitioner
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    model_args = ModelArgs(MobileNetV2Lightning, optimizer_args=optimizer_args, num_classes=num_classes)
    initialization_rounds = [25, 50]
    linkage_mech = 'ward'
    criterion = 'distance'
    dis_metric = 'euclidean'
    max_value_criterion = [100, 200, 300]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def generate_configuration(init_rounds_list, max_value_criterion_list):
    for ri in init_rounds_list:
        for mv in max_value_criterion_list:
            yield ri, mv


def log_cluster_distribution(
        experiment_logger,
        cluster_clients_dic: Dict[str, List['BaseTrainingParticipant']],
        num_classes
):
    for cluster_id, clients in cluster_clients_dic.items():
        image = generate_client_label_heatmap(f'cluster {cluster_id}', clients, num_classes)
        experiment_logger.experiment.add_image(f'label distribution/cluster_{cluster_id}', image.numpy())


@ex.automain
def clustering_test(
        mean,
        std,
        seed,
        lr,
        local_epochs,
        client_fraction,
        optimizer_args,
        total_fedavg_rounds,
        batch_size,
        num_clients,
        model_args,
        train_args,
        train_cluster_args,
        initialization_rounds,
        partitioner_class,
        linkage_mech,
        criterion,
        dis_metric,
        max_value_criterion
):
    fix_random_seeds(seed)

    fed_dataset = load_ham10k_federated(partitions=num_clients, batch_size=batch_size, mean=mean, std=std)
    initialize_clients_fn = initialize_ham10k_clients

    fedavg_context = FedAvgExperimentContext(name='ham10k_clustering', client_fraction=client_fraction,
                                             local_epochs=local_epochs,
                                             lr=lr, batch_size=batch_size, optimizer_args=optimizer_args,
                                             model_args=model_args, train_args=train_args,
                                             dataset_name='ham10k')
    experiment_specification = f'{fedavg_context}'
    experiment_logger = create_tensorboard_logger(fedavg_context.name, experiment_specification)

    log_dataset_distribution(experiment_logger, 'full dataset', fed_dataset)

    server, clients = run_fedavg(context=fedavg_context, num_rounds=total_fedavg_rounds, dataset=fed_dataset,
               save_states=True, restore_state=True, evaluate_rounds=False, initialize_clients_fn=initialize_clients_fn)

    for init_rounds in initialization_rounds:
        # load the model state
        round_model_state = load_fedavg_state(fedavg_context, init_rounds)
        overwrite_participants_models(round_model_state, clients)
        run_fedavg_train_round(round_model_state, training_args=train_cluster_args, participants=clients)
        for max_value in max_value_criterion:
            # initialize the cluster configuration
            round_configuration = {
                'num_rounds_init': init_rounds,
                'num_rounds_cluster': total_fedavg_rounds - init_rounds
            }
            cluster_args = ClusterArgs(partitioner_class, linkage_mech=linkage_mech,
                                       criterion=criterion, dis_metric=dis_metric,
                                       max_value_criterion=max_value,
                                       plot_dendrogram=False, reallocate_clients=False,
                                       threshold_min_client_cluster=-1,
                                       **round_configuration)
            experiment_logger = create_tensorboard_logger(fedavg_context.name, f'{experiment_specification}{cluster_args}')
            partitioner = cluster_args()
            cluster_clients_dic = partitioner.cluster(clients, server)
            log_cluster_distribution(experiment_logger, cluster_clients_dic, 7)
