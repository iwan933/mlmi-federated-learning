import sys
sys.path.append('C:/Users/Richard/Desktop/Informatik/Semester_5/MLMI/git/mlmi-federated-learning')

from typing import Callable, Dict, List, Optional

from sacred import Experiment
from functools import partial

from torch import Tensor, optim

from mlmi.clustering import DatadependentPartitioner, ModelFlattenWeightsPartitioner, AlternativePartitioner, \
    RandomClusterPartitioner
from mlmi.experiments.log import log_goal_test_acc, log_loss_and_acc
from mlmi.fedavg.data import load_n_of_each_class, scratch_labels
from mlmi.fedavg.femnist import load_femnist_colored_dataset, load_femnist_dataset
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

from mlmi.reptile.structs import ReptileExperimentContext

ex = Experiment('hierachical_clustering_reptile')


@ex.config
def default_configuration():
    seed = 123123123
    name = 'hierarchical_reptile'
    dataset = 'femnist'
    num_clients = 367
    batch_size = 10
    num_label_limit = -1
    use_colored_images = False
    sample_threshold = -1

    hc_lr = 0.068
    hc_total_fedavg_rounds = 20
    hc_cluster_initialization_rounds = [3, 5, 10]
    hc_client_fraction = [0.1]
    hc_local_epochs = 3
    hc_train_args = TrainArgs(max_epochs=hc_local_epochs, min_epochs=hc_local_epochs, progress_bar_refresh_rate=0)
    hc_train_cluster_args = TrainArgs(max_epochs=hc_local_epochs, min_epochs=hc_local_epochs, progress_bar_refresh_rate=0)
    hc_partitioner_class = DatadependentPartitioner
    hc_linkage_mech = 'ward'
    hc_criterion = 'distance'
    hc_dis_metric = 'euclidean'
    hc_max_value_criterion = 15.0
    hc_reallocate_clients = False
    hc_threshold_min_client_cluster = 80

    rp_sgd = True  # True -> Use SGD as inner optimizer; False -> Use Adam
    rp_adam_betas = (0.9, 0.999)  # Used only if sgd = False
    rp_meta_batch_size = 5
    rp_num_meta_steps = 20000
    rp_meta_learning_rate_initial = 1
    rp_meta_learning_rate_final = 0
    rp_eval_interval = 250
    rp_inner_learning_rate = 0.1
    rp_num_inner_steps = 5
    rp_num_inner_steps_eval = 50


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
def run_hierarchical_clustering_reptile(
        seed,
        name,
        dataset,
        num_clients,
        batch_size,
        num_label_limit,
        use_colored_images,
        sample_threshold,
        hc_lr,
        hc_total_fedavg_rounds,
        hc_cluster_initialization_rounds,
        hc_client_fraction,
        hc_local_epochs,
        hc_train_args,
        hc_partitioner_class,
        hc_linkage_mech,
        hc_criterion,
        hc_dis_metric,
        hc_max_value_criterion,
        hc_reallocate_clients,
        hc_threshold_min_client_cluster,
        hc_train_cluster_args,
        rp_sgd,  # True -> Use SGD as inner optimizer; False -> Use Adam
        rp_adam_betas,  # Used only if sgd = False
        rp_meta_batch_size,
        rp_num_meta_steps,
        rp_meta_learning_rate_initial,
        rp_meta_learning_rate_final,
        rp_eval_interval,
        rp_inner_learning_rate,
        rp_num_inner_steps,
        rp_num_inner_steps_eval
):
    fix_random_seeds(seed)
    global_tag = 'global_performance'

    if dataset == 'femnist':
        if use_colored_images:
            fed_dataset = load_femnist_colored_dataset(
                data_dir=str((REPO_ROOT / 'data').absolute()),
                num_clients=num_clients,
                batch_size=batch_size,
                sample_threshold=sample_threshold
            )
        else:
            fed_dataset = load_femnist_dataset(
                data_dir=str((REPO_ROOT / 'data').absolute()),
                num_clients=num_clients,
                batch_size=batch_size,
                sample_threshold=sample_threshold
            )
        if num_label_limit != -1:
            fed_dataset = scratch_labels(fed_dataset, num_label_limit)
    else:
        raise ValueError(f'dataset "{dataset}" unknown')

    if not hasattr(hc_max_value_criterion, '__iter__'):
        hc_max_value_criterion = [hc_max_value_criterion]
    if not hasattr(hc_lr, '__iter__'):
        hc_lr = [hc_lr]
    input_channels = 3 if use_colored_images else 1
    data_distribution_logged = False
    for cf in hc_client_fraction:
        for lr_i in hc_lr:
            # Initialize experiment context parameters
            fedavg_optimizer_args = OptimizerArgs(optim.SGD, lr=lr_i)
            fedavg_model_args = ModelArgs(
                CNNLightning,
                optimizer_args=fedavg_optimizer_args,
                input_channels=input_channels,
                only_digits=False
            )
            fedavg_context = FedAvgExperimentContext(
                name=name,
                client_fraction=cf,
                local_epochs=hc_local_epochs,
                lr=lr_i,
                batch_size=batch_size,
                optimizer_args=fedavg_optimizer_args,
                model_args=fedavg_model_args,
                train_args=hc_train_args,
                dataset_name=dataset
            )
            reptile_context = ReptileExperimentContext(
                name=name,
                dataset_name=dataset,
                swap_labels=False,
                num_classes_per_client=0,
                num_shots_per_class=0,
                seed=seed,
                model_class=CNNLightning,
                sgd=rp_sgd,
                adam_betas=rp_adam_betas,
                num_clients_train=num_clients,
                num_clients_test=0,
                meta_batch_size=rp_meta_batch_size,
                num_meta_steps=rp_num_meta_steps,
                meta_learning_rate_initial=rp_meta_learning_rate_initial,
                meta_learning_rate_final=rp_meta_learning_rate_final,
                eval_interval=rp_eval_interval,
                num_eval_clients_training=-1,
                do_final_evaluation=True,
                num_eval_clients_final=-1,
                inner_batch_size=batch_size,
                inner_learning_rate=rp_inner_learning_rate,
                num_inner_steps=rp_num_inner_steps,
                num_inner_steps_eval=rp_num_inner_steps_eval
            )
            experiment_specification = f'{fedavg_context}_{reptile_context}'
            experiment_logger = create_tensorboard_logger(name, experiment_specification)
            if not data_distribution_logged:
                log_dataset_distribution(experiment_logger, 'full dataset', fed_dataset)
                data_distribution_logged = True

            log_after_round_evaluation_fns = [
                partial(log_after_round_evaluation, experiment_logger, 'fedavg'),
                partial(log_after_round_evaluation, experiment_logger, global_tag)
            ]
            server, clients = run_fedavg(
                context=fedavg_context,
                num_rounds=hc_total_fedavg_rounds,
                dataset=fed_dataset,
                save_states=True,
                restore_state=True,
                after_round_evaluation=log_after_round_evaluation_fns
            )

            for init_rounds, max_value in generate_configuration(hc_cluster_initialization_rounds, hc_max_value_criterion):
                # load the model state
                round_model_state = load_fedavg_state(fedavg_context, init_rounds)
                overwrite_participants_models(round_model_state, clients)
                # initialize the cluster configuration
                round_configuration = {
                    'num_rounds_init': init_rounds,
                    'num_rounds_cluster': hc_total_fedavg_rounds - init_rounds
                }
                if partitioner_class == DatadependentPartitioner:
                    clustering_dataset = load_femnist_colored_dataset(
                        data_dir=str((REPO_ROOT / 'data').absolute()),
                        num_clients=num_clients,
                        batch_size=batch_size,
                        sample_threshold=sample_threshold
                    )
                    dataloader = load_n_of_each_class(clustering_dataset, n=5,
                                                      tabu=list(fed_dataset.train_data_local_dict.keys()))
                    cluster_args = ClusterArgs(partitioner_class, linkage_mech=hc_linkage_mech,
                                               criterion=hc_criterion, dis_metric=hc_dis_metric,
                                               max_value_criterion=max_value,
                                               plot_dendrogram=False, reallocate_clients=hc_reallocate_clients,
                                               threshold_min_client_cluster=hc_threshold_min_client_cluster,
                                               dataloader=dataloader,
                                               **round_configuration)
                else:
                    cluster_args = ClusterArgs(partitioner_class, linkage_mech=hc_linkage_mech,
                                               criterion=hc_criterion, dis_metric=hc_dis_metric,
                                               max_value_criterion=max_value,
                                               plot_dendrogram=False, reallocate_clients=hc_reallocate_clients,
                                               threshold_min_client_cluster=hc_threshold_min_client_cluster,
                                               **round_configuration)
                # create new logger for cluster experiment
                experiment_specification = f'{fedavg_context}_{cluster_args}'
                experiment_logger = create_tensorboard_logger(name, experiment_specification)
                fedavg_context.experiment_logger = experiment_logger

                initial_train_fn = partial(run_fedavg_train_round, round_model_state, training_args=train_cluster_args)
                create_aggregator_fn = partial(FedAvgServer, model_args=model_args, context=fedavg_context)

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
                    partial(log_cluster_distribution, experiment_logger, num_classes=fed_dataset.class_num),
                    partial(log_sample_images_from_each_client, experiment_logger)
                ]

                # HIERARCHICAL CLUSTERING
                num_rounds_init = cluster_args.num_rounds_init
                num_rounds_cluster = cluster_args.num_rounds_cluster
                logger.debug('starting local training before clustering.')
                trained_participants = initial_train_fn(clients)
                if len(trained_participants) != len(clients):
                    raise ValueError(
                        'not all clients successfully participated in the clustering round')

                # Clustering of participants by model updates
                partitioner = cluster_args()
                cluster_clients_dic = partitioner.cluster(clients, server)
                _cluster_clients_dic = dict()
                for cluster_id, participants in cluster_clients_dic.items():
                    _cluster_clients_dic[cluster_id] = [c._name for c in participants]

                # Initialize cluster models
                cluster_server_dic = {}
                for cluster_id, participants in cluster_clients_dic.items():
                    intermediate_cluster_server = create_aggregator_fn('cluster_server' + cluster_id)
                    intermediate_cluster_server.aggregate(participants)
                    cluster_server = ReptileServer(
                        participant_name='cluster_server' + cluster_id,
                        model_args=model_args,
                        context=reptile_context,
                        initial_model_state=intermediate_cluster_server.model.state_dict()
                    )
                    #create_aggregator_fn('cluster_server' + cluster_id)
                    #cluster_server.aggregate(participants)
                    cluster_server_dic[cluster_id] = cluster_server

                # REPTILE TRAINING INSIDE CLUSTERS
                RANDOM = random.Random(context.seed)

                # Perform training
                for i in range(reptile_context.num_meta_steps):
                    for cluster_id, participants in cluster_clients_dic.items():

                        if reptile_context.meta_batch_size == -1:
                            meta_batch = participants
                        else:
                            meta_batch = [
                                participants[k] for k in cyclerange(
                                    start=i * reptile_context.meta_batch_size % len(participants),
                                    stop=(i + 1) * reptile_context.meta_batch_size % len(participants),
                                    len=len(participants)
                                )
                            ]
                        # Meta training step
                        reptile_train_step(
                            aggregator=cluster_server_dic[cluster_id],
                            participants=meta_batch,
                            inner_training_args=reptile_context.get_inner_training_args(),
                            meta_training_args=reptile_context.get_meta_training_args(
                                frac_done=i / reptile_context.num_meta_steps
                            )
                        )

                        # Evaluation on train and test clients
                        if i % reptile_context.eval_interval == 0:
                            # Test on all clients inside clusters
                            reptile_train_step(
                                aggregator=cluster_server_dic[cluster_id],
                                participants=participants,
                                inner_training_args=reptile_context.get_inner_training_args(eval=True),
                                evaluation_mode=True
                            )
                            result = evaluate_local_models(participants=participants)
                            loss = result.get('test/loss')
                            accs = result.get('test/acc')

                            # Log
                            if after_round_evaluation is not None:
                                for c in after_round_evaluation:
                                    c(f'cluster_{cluster_id}', loss, acc, i)

                    logger.info('finished training round')

                # Final evaluation at end of training
                if reptile_context.do_final_evaluation:
                    global_loss, global_acc = [], []

                    for cluster_id, participants in cluster_clients_dic.items():
                        # Final evaluation on train and test clients
                        # Test on all clients inside clusters
                        reptile_train_step(
                            aggregator=cluster_server_dic[cluster_id],
                            participants=participants,
                            inner_training_args=reptile_context.get_inner_training_args(
                                eval=True),
                            evaluation_mode=True
                        )
                        result = evaluate_local_models(participants=participants)
                        loss = result.get('test/loss')
                        accs = result.get('test/acc')
                        global_loss.extend(loss.tolist())
                        global_acc.extend(acc.tolist())

                        # Log
                        if after_round_evaluation is not None:
                            for c in after_round_evaluation:
                                c(f'cluster_{cluster_id}', loss, acc, reptile_context.num_meta_steps)

                    experiment_logger.add_scalar('Final_loss', mean(global_loss), global_step=0)
                    experiment_logger.add_scalar('Final_acc', mean(global_acc), global_step=0)
