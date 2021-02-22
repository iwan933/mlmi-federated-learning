from sacred import Experiment

from mlmi.clustering import AlternativePartitioner
from mlmi.structs import TrainArgs

ex = Experiment('clustering test')


@ex.config
class DefaultConfig:
    seed = 123123123
    lr = 0.01
    name = 'clustering_test'
    total_fedavg_rounds = 13
    client_fraction = 0.1
    local_epochs = 1
    batch_size = 16
    num_clients = 27
    num_classes = 7
    train_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs, progress_bar_refresh_rate=0)
    train_cluster_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs, progress_bar_refresh_rate=0)
    dataset = 'ham10k'
    partitioner_class = AlternativePartitioner
    model_args = None
    initialization_rounds = [1, 3, 5, 8, 10, 13]
    linkage_mech = 'ward'
    criterion = 'distance'
    dis_metric = 'euclidean'
    max_value_criterion = 6.5


@ex.automain
def clustering_test(
        client_fraction,
        optimizer_args,
        total_fedavg_rounds,
        batch_size,
        num_clients,
        num_classes,
        model_args,
        train_args,
        intialization_rounds,
        partitioner_class,
        metric,
        linkage_mech,
        criterion,
        dis_metric,
        max_value_criterion
):
    fix_random_seeds(seed)
    global_tag = 'global_performance'
    global_tag_local = 'global_performance_personalized'

    if dataset == 'femnist':
        if use_colored_images:
            fed_dataset = load_femnist_colored_dataset(str((REPO_ROOT / 'data').absolute()),
                                                       num_clients=num_clients, batch_size=batch_size,
                                                       sample_threshold=sample_threshold,
                                                       add_pattern=use_pattern)
        else:
            fed_dataset = load_femnist_dataset(str((REPO_ROOT / 'data').absolute()),
                                               num_clients=num_clients, batch_size=batch_size,
                                               sample_threshold=sample_threshold)
        if num_label_limit != -1:
            fed_dataset = scratch_labels(fed_dataset, num_label_limit)
    else:
        raise ValueError(f'dataset "{dataset}" unknown')

    if not hasattr(max_value_criterion, '__iter__'):
        max_value_criterion = [max_value_criterion]
    if not hasattr(lr, '__iter__'):
        lr = [lr]
    input_channels = 3 if use_colored_images else 1
    data_distribution_logged = False
    for cf in client_fraction:
        for lr_i in lr:
            optimizer_args = OptimizerArgs(optim.SGD, lr=lr_i)
            model_args = ModelArgs(CNNLightning, optimizer_args=optimizer_args,
                                   input_channels=input_channels, only_digits=False)
            fedavg_context = FedAvgExperimentContext(name=name, client_fraction=cf, local_epochs=local_epochs,
                                                     lr=lr_i, batch_size=batch_size, optimizer_args=optimizer_args,
                                                     model_args=model_args, train_args=train_args,
                                                     dataset_name=dataset)
            experiment_specification = f'{fedavg_context}'
            experiment_logger = create_tensorboard_logger(fedavg_context.name, experiment_specification)
            if not data_distribution_logged:
                log_dataset_distribution(experiment_logger, 'full dataset', fed_dataset)
                data_distribution_logged = True

            log_after_round_evaluation_fns = [
                partial(log_after_round_evaluation, experiment_logger, 'fedavg'),
                partial(log_after_round_evaluation, experiment_logger, global_tag)
            ]
            log_after_aggregation = [
                partial(log_personalized_performance, experiment_logger, 'fedavg_personalized', local_evaluation_steps),
                partial(log_personalized_performance, experiment_logger, global_tag_local, local_evaluation_steps)
            ]
            server, clients = run_fedavg(context=fedavg_context, num_rounds=total_fedavg_rounds, dataset=fed_dataset,
                                         save_states=True, restore_state=True,
                                         after_round_evaluation=log_after_round_evaluation_fns,
                                         after_aggregation=log_after_aggregation)

            for init_rounds, max_value in generate_configuration(cluster_initialization_rounds, max_value_criterion):
