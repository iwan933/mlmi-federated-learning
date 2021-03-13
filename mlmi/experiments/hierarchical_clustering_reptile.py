from mlmi.datasets.ham10k import load_ham10k_federated
from mlmi.fedavg.ham10k import initialize_ham10k_clients

from typing import Callable, Dict, List, Optional
import random
import copy

from sacred import Experiment
from functools import partial

import numpy as np
import torch
from torch import Tensor, optim

from mlmi.datasets.ham10k import load_ham10k_federated, load_ham10k_few_big_many_small_federated, \
    load_ham10k_partition_by_two_labels_federated
from mlmi.models.ham10k import GlobalConfusionMatrix, GlobalTestTestConfusionMatrix, GlobalTrainTestConfusionMatrix, \
    MobileNetV2Lightning
from mlmi.plot import generate_confusion_matrix_heatmap, generate_data_label_heatmap

from mlmi.clustering import DatadependentPartitioner, FixedAlternativePartitioner, ModelFlattenWeightsPartitioner, \
    AlternativePartitioner, \
    RandomClusterPartitioner
from mlmi.experiments.log import log_goal_test_acc, log_loss_and_acc
from mlmi.fedavg.data import load_n_of_each_class, scratch_labels
from mlmi.fedavg.femnist import load_femnist_colored_dataset, load_femnist_dataset
from mlmi.fedavg.model import CNNLightning, CNNMnistLightning, FedAvgServer
from mlmi.fedavg.run import DEFAULT_CLIENT_INIT_FN, run_fedavg
from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.fedavg.util import load_fedavg_state, run_fedavg_round, run_fedavg_train_round
from mlmi.hierarchical.run import run_fedavg_hierarchical
from mlmi.participant import BaseTrainingParticipant
from mlmi.plot import generate_client_label_heatmap, generate_data_label_heatmap
from mlmi.settings import REPO_ROOT
from mlmi.structs import ClusterArgs, FederatedDatasetData, ModelArgs, OptimizerArgs, TrainArgs
from mlmi.utils import create_tensorboard_logger, fix_random_seeds, overwrite_participants_models, \
    evaluate_local_models

from mlmi.reptile.structs import ReptileExperimentContext
from mlmi.reptile.model import ReptileServer
from mlmi.reptile.run_reptile_experiment import run_reptile, cyclerange
from mlmi.reptile.util import reptile_train_step, run_train_round
from mlmi.log import getLogger

ex = Experiment('hierachical_clustering_reptile')
logger = getLogger(__name__)


@ex.config
def default_configuration():
    seed = 123123123
    name = 'ham10khcreptile'
    dataset = 'ham10k2label'
    num_clients = 0 # Not used here
    batch_size = 8
    model_class = MobileNetV2Lightning
    do_balancing = False

    hc_lr = 0.0008
    hc_cluster_initialization_rounds = [2]
    hc_meta_batch_size = 5
    hc_local_epochs = 1
    hc_train_cluster_args = TrainArgs(max_epochs=3, min_epochs=3, progress_bar_refresh_rate=0)
    hc_partitioner_class = FixedAlternativePartitioner
    hc_linkage_mech = 'ward'
    hc_criterion = 'distance'
    hc_dis_metric = 'euclidean'
    hc_max_value_criterion = 300.00
    hc_reallocate_clients = False
    hc_threshold_min_client_cluster = 1

    rp_sgd = True  # True -> Use SGD as inner optimizer; False -> Use Adam
    rp_adam_betas = None  # Used only if sgd = False
    rp_meta_batch_size = 5
    rp_num_meta_steps = 2401
    rp_meta_learning_rate_initial = 2
    rp_meta_learning_rate_final = 0
    rp_eval_interval = 120
    rp_inner_learning_rate = 0.0008
    rp_num_inner_epochs = 1
    rp_num_inner_epochs_eval = 3
    rp_personalize_before_eval = True

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


@ex.named_config
def ham10k():
    seed = 123123123
    name = 'ham10k'
    dataset = 'ham10k'
    num_clients = 27
    batch_size = 16
    num_label_limit = -1
    use_colored_images = False
    sample_threshold = -1
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    hc_lr = 0.007
    hc_cluster_initialization_rounds = [20]
    hc_client_fraction = [0.3]
    hc_local_epochs = 1
    hc_train_args = TrainArgs(max_epochs=hc_local_epochs, min_epochs=hc_local_epochs, progress_bar_refresh_rate=0)
    hc_train_cluster_args = TrainArgs(max_epochs=3, min_epochs=3, progress_bar_refresh_rate=0)
    hc_partitioner_class = FixedAlternativePartitioner
    hc_linkage_mech = 'ward'
    hc_criterion = 'distance'
    hc_dis_metric = 'euclidean'
    hc_max_value_criterion = [300.00]
    hc_reallocate_clients = False
    hc_threshold_min_client_cluster = 1

    rp_sgd = True  # True -> Use SGD as inner optimizer; False -> Use Adam
    rp_adam_betas = (0.9, 0.999)  # Used only if sgd = False
    rp_meta_batch_size = 5
    rp_num_meta_steps = 2000
    rp_meta_learning_rate_initial = 1
    rp_meta_learning_rate_final = 0
    rp_eval_interval = 20
    rp_inner_learning_rate = 0.05
    rp_num_inner_steps = 70
    rp_num_inner_steps_eval = 70


def log_after_round_evaluation(
        experiment_logger,
        tag: str,
        loss_train_test: Tensor,
        acc_train_test: Tensor,
        balanced_acc_train_test: Tensor,
        loss_test_test: Tensor,
        acc_test_test: Tensor,
        balanced_acc_test_test: Tensor,
        step: int
):
    try:
        global_confusion_matrices = [(GlobalConfusionMatrix(), 'global'),
                                     (GlobalTrainTestConfusionMatrix(), 'train-test'),
                                     (GlobalTestTestConfusionMatrix(), 'test-test')]
        for global_confusion_matrix, matrix_type in global_confusion_matrices:
            if global_confusion_matrix.has_data:
                matrix = global_confusion_matrix.compute()
                image = generate_confusion_matrix_heatmap(matrix, title=tag)
                experiment_logger.experiment.add_image(f'{tag}-{matrix_type}', image.numpy(), step)
    except Exception as e:
        print('failed to log confusion matrix (global)', e)

    log_loss_and_acc(
        f'{tag}train-test',
        loss_train_test,
        acc_train_test,
        experiment_logger,
        step
    )
    log_loss_and_acc(
        f'{tag}balanced-train-test',
        loss_train_test,
        balanced_acc_train_test,
        experiment_logger,
        step
    )
    log_goal_test_acc(f'{tag}train-test', acc_train_test, experiment_logger, step)
    log_goal_test_acc(f'{tag}balanced-train-test', balanced_acc_train_test, experiment_logger, step)
    if loss_test_test is not None and acc_test_test is not None:
        log_loss_and_acc(
            f'{tag}test-test',
            loss_test_test,
            acc_test_test,
            experiment_logger,
            step
        )
        log_loss_and_acc(
            f'{tag}balanced-test-test',
            loss_test_test,
            balanced_acc_test_test,
            experiment_logger,
            step
        )
        log_goal_test_acc(f'{tag}test-test', acc_test_test, experiment_logger, step)
        log_goal_test_acc(f'{tag}balanced-test-test', balanced_acc_test_test, experiment_logger, step)


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
        model_class,
        do_balancing,

        hc_lr,
        hc_cluster_initialization_rounds,
        hc_meta_batch_size,
        hc_local_epochs,
        hc_partitioner_class,
        hc_linkage_mech,
        hc_criterion,
        hc_dis_metric,
        hc_max_value_criterion,  # distance threshold
        hc_reallocate_clients,  #
        hc_threshold_min_client_cluster,  # only with hc_reallocate_clients = True,
                                          # results in clusters having at least this number of clients
        hc_train_cluster_args,

        rp_sgd,  # True -> Use SGD as inner optimizer; False -> Use Adam
        rp_adam_betas,  # Used only if sgd = False
        rp_meta_batch_size,
        rp_num_meta_steps,
        rp_meta_learning_rate_initial,
        rp_meta_learning_rate_final,
        rp_eval_interval,
        rp_inner_learning_rate,
        rp_num_inner_epochs,
        rp_num_inner_epochs_eval,
        rp_personalize_before_eval,
        mean=None,
        std=None
):
    fix_random_seeds(seed)
    global_tag = 'global_performance'
    initialize_clients_fn = DEFAULT_CLIENT_INIT_FN

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
    elif dataset == 'ham10k':
        fed_dataset = load_ham10k_federated(partitions=num_clients, batch_size=batch_size, mean=mean, std=std)
        initialize_clients_fn = initialize_ham10k_clients
    elif dataset == 'ham10k2label':
        fed_dataset = load_ham10k_partition_by_two_labels_federated(
            batch_size=batch_size,  mean=mean, std=std
        )
    else:
        raise ValueError(f'dataset "{dataset}" unknown')

    if not hasattr(hc_max_value_criterion, '__iter__'):
        hc_max_value_criterion = [hc_max_value_criterion]
    if not hasattr(hc_lr, '__iter__'):
        hc_lr = [hc_lr]
    data_distribution_logged = False

    for lr_i in hc_lr:
        # Initialize experiment context parameters
        fedavg_context = ReptileExperimentContext(
            name=name,
            dataset_name=dataset,
            swap_labels=False,
            num_classes_per_client=0,
            num_shots_per_class=0,
            seed=seed,
            model_class=model_class,
            sgd=True,
            adam_betas=None,
            num_clients_train=num_clients,
            num_clients_test=0,
            meta_batch_size=hc_meta_batch_size,
            num_meta_steps=hc_cluster_initialization_rounds[0],
            meta_learning_rate_initial=1.0,
            meta_learning_rate_final=1.0,
            eval_interval=1,
            num_eval_clients_training=-1,
            do_final_evaluation=False,
            num_eval_clients_final=-1,
            inner_batch_size=batch_size,
            inner_learning_rate=lr_i,
            num_inner_epochs=hc_local_epochs,
            num_inner_epochs_eval=0,
            do_balancing=do_balancing
        )
        reptile_context = ReptileExperimentContext(
            name=name,
            dataset_name=dataset,
            swap_labels=False,
            num_classes_per_client=0,
            num_shots_per_class=0,
            seed=seed,
            model_class=model_class,
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
            num_inner_epochs=rp_num_inner_epochs,
            num_inner_epochs_eval=rp_num_inner_epochs_eval,
            do_balancing=do_balancing
        )
        experiment_specification = f'{fedavg_context}'
        experiment_logger = create_tensorboard_logger(name, experiment_specification)
        # TODO: Fix this
        #if not data_distribution_logged:
        #    log_dataset_distribution(experiment_logger, 'full dataset', fed_dataset)
        #    data_distribution_logged = True

        log_after_round_evaluation_fns = [
            partial(log_after_round_evaluation, experiment_logger)
        ]
        server, clients, _ = run_reptile(
            context=fedavg_context,
            dataset_train=fed_dataset,
            dataset_test=None,
            initial_model_state=None,
            after_round_evaluation=log_after_round_evaluation_fns,
            start_round=0,
            personalize_before_eval=False
        )

        for init_rounds, max_value in generate_configuration(hc_cluster_initialization_rounds, hc_max_value_criterion):
            # load the model state
            #round_model_state = load_fedavg_state(fedavg_context, init_rounds)
            #overwrite_participants_models(round_model_state, clients)
            # initialize the cluster configuration
            round_configuration = {
                'num_rounds_init': init_rounds,
                'num_rounds_cluster': 0
            }
            cluster_args = ClusterArgs(hc_partitioner_class, linkage_mech=hc_linkage_mech,
                                       criterion=hc_criterion, dis_metric=hc_dis_metric,
                                       max_value_criterion=max_value,
                                       plot_dendrogram=False, reallocate_clients=hc_reallocate_clients,
                                       threshold_min_client_cluster=hc_threshold_min_client_cluster,
                                       **round_configuration)
            # create new logger for cluster experiment
            experiment_specification = f'{fedavg_context}_{cluster_args}_{reptile_context}'
            experiment_logger = create_tensorboard_logger(name, experiment_specification)
            fedavg_context.experiment_logger = experiment_logger

            # HIERARCHICAL CLUSTERING
            logger.debug('starting local training before clustering.')

            fedavg_server_state = copy.deepcopy(server.model.state_dict())
            overwrite_participants_models(
                model_state=fedavg_server_state,
                participants=clients
            )
            trained_participants = run_train_round(
                participants=clients,
                training_args=hc_train_cluster_args
            )
            if trained_participants != len(clients):
                raise ValueError(
                    'not all clients successfully participated in the clustering round')

            # Clustering of participants by model updates
            partitioner = cluster_args()
            cluster_clients_dic = partitioner.cluster(clients, server)
            _cluster_clients_dic = dict()
            for cluster_id, participants in cluster_clients_dic.items():
                _cluster_clients_dic[cluster_id] = [c._name for c in participants]
            log_cluster_distribution(experiment_logger, cluster_clients_dic, 62)

            # Initialize cluster models
            cluster_server_dic = {}
            for cluster_id, participants in cluster_clients_dic.items():
                intermediate_cluster_server = ReptileServer(
                    participant_name=f'cluster_server{cluster_id}',
                    model_args=fedavg_context.meta_model_args,
                    context=fedavg_context,
                    initial_model_state=fedavg_server_state
                )
                intermediate_cluster_server.aggregate(
                    participants=participants, meta_learning_rate=1
                )
                cluster_server = ReptileServer(
                    participant_name=f'cluster_server{cluster_id}',
                    model_args=reptile_context.meta_model_args,
                    context=reptile_context,
                    initial_model_state=intermediate_cluster_server.model.state_dict()
                )
                cluster_server_dic[cluster_id] = cluster_server

            # REPTILE TRAINING INSIDE CLUSTERS
            after_round_evaluation = [log_after_round_evaluation]

            def _evaluate(global_step: int):
                """Evaluate on all clients in each cluster"""
                global_loss, global_acc, global_balanced_acc = [], [], []

                for cluster_id, participants in cluster_clients_dic.items():
                    # Test on all clients inside clusters
                    if rp_personalize_before_eval:
                        reptile_train_step(
                            aggregator=cluster_server_dic[cluster_id],
                            participants=participants,
                            inner_training_args=reptile_context.get_inner_training_args(eval=True),
                            evaluation_mode=True
                        )
                    if rp_personalize_before_eval:
                        result = evaluate_local_models(participants=participants)
                    else:
                        result = evaluate_global_model(
                            global_model_participant=cluster_server_dic[cluster_id],
                            participants=participants
                        )
                    loss = result.get('test/loss')
                    acc = result.get('test/acc')
                    balanced_acc = result.get('test/balanced_acc')

                    # Log
                    if after_round_evaluation is not None:
                        for c in after_round_evaluation:
                            c(experiment_logger, f'cluster_{cluster_id}',
                              loss, acc, balanced_acc,
                              None, None, None,
                              global_step)
                    loss_list = loss.tolist()
                    acc_list = acc.tolist()
                    balanced_acc_list = balanced_acc.tolist()
                    global_loss.extend(loss_list if isinstance(loss_list, list) else [loss_list])
                    global_acc.extend(acc_list if isinstance(acc_list, list) else [acc_list])
                    global_balanced_acc.extend(
                        balanced_acc_list if isinstance(balanced_acc_list, list) else [
                            balanced_acc_list])

                return global_loss, global_acc, global_balanced_acc

            # Perform training
            for i in range(reptile_context.num_meta_steps):
                for cluster_id, participants in cluster_clients_dic.items():

                    if reptile_context.meta_batch_size == -1:
                        meta_batch = participants
                    else:
                        meta_batch = np.random.choice(participants, reptile_context.meta_batch_size, False)
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
                    global_step = init_rounds + i
                    global_loss, global_acc, global_balanced_acc = _evaluate(global_step=global_step)

                    if after_round_evaluation is not None:
                        for c in after_round_evaluation:
                            c(experiment_logger, 'mean_over_all_clients',
                              Tensor(global_loss), Tensor(global_acc),
                              Tensor(global_balanced_acc), None, None, None,
                              global_step)

                logger.info(f'Finished Reptile training round {i}')

            # Final evaluation at end of training
            if reptile_context.do_final_evaluation:
                global_loss, global_acc, global_balanced_acc = _evaluate(
                    global_step=reptile_context.num_meta_steps
                )
                if after_round_evaluation is not None:
                    for c in after_round_evaluation:
                        c(experiment_logger, 'overall_mean',
                          Tensor(global_loss), Tensor(global_acc),
                          Tensor(global_balanced_acc), None, None, None, 0)
