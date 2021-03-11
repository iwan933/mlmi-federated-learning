from typing import Callable, Dict, List, Optional

from sacred import Experiment
from functools import partial

from torch import Tensor, optim

from mlmi.clustering import ModelFlattenWeightsPartitioner
from mlmi.datasets.ham10k import load_ham10k_federated, load_ham10k_few_big_many_small_federated
from mlmi.experiments.log import log_goal_test_acc, log_loss_and_acc
from mlmi.fedavg.femnist import load_femnist_dataset, load_mnist_dataset
from mlmi.fedavg.ham10k import initialize_ham10k_clients
from mlmi.fedavg.model import CNNLightning, CNNMnistLightning, FedAvgServer
from mlmi.fedavg.run import DEFAULT_CLIENT_INIT_FN, run_fedavg
from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.fedavg.util import load_fedavg_state, run_fedavg_round, run_fedavg_train_round
from mlmi.hierarchical.run import run_fedavg_hierarchical
from mlmi.log import getLogger
from mlmi.models.ham10k import Densenet121Lightning, GlobalConfusionMatrix, MobileNetV2Lightning, ResNet18Lightning
from mlmi.participant import BaseParticipant, BaseTrainingParticipant
from mlmi.plot import generate_client_label_heatmap, generate_confusion_matrix_heatmap, generate_data_label_heatmap
from mlmi.settings import REPO_ROOT
from mlmi.structs import ClusterArgs, FederatedDatasetData, ModelArgs, OptimizerArgs, TrainArgs
from mlmi.utils import create_tensorboard_logger, evaluate_local_models, fix_random_seeds, overwrite_participants_models

ex = Experiment('fedavg')
logger = getLogger(__name__)


@ex.config
def femnist():
    mean = None
    std = None
    seed = 123123123
    lr = 0.1
    name = 'femnist'
    total_fedavg_rounds = 50
    eval_interval = 10
    client_fraction = [0.1]
    local_epochs = 3
    batch_size = 10
    num_clients = 367
    num_classes = 62
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    train_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs, progress_bar_refresh_rate=0)
    model_args = ModelArgs(CNNLightning, optimizer_args=optimizer_args, only_digits=False, input_channels=1)
    dataset = 'femnist'


@ex.named_config
def mnist():
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


@ex.named_config
def ham10k_IDA_configuration():
    lr = 0.016
    name = 'ham10k_IDA'
    total_fedavg_rounds = 10
    client_fraction = [0.3]
    local_epochs = 1
    batch_size = 16  # original: 32
    num_clients = 11
    num_classes = 7
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    train_args = TrainArgs(max_epochs=local_epochs,
                           min_epochs=local_epochs,
                           progress_bar_refresh_rate=5)
    model_args = ModelArgs(Densenet121Lightning, optimizer_args=optimizer_args, num_classes=num_classes)
    dataset = 'ham10k'


@ex.named_config
def ham10k_MobileNetV2():
    lr = 0.001
    name = 'ham10k_mobilenet'
    total_fedavg_rounds = 2400
    client_fraction = [0.075]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    local_epochs = 1
    eval_interval = 10
    batch_size = 8  # original: 32 but requires 8gb gpu
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    train_args = TrainArgs(max_epochs=local_epochs,
                           min_epochs=local_epochs,
                           progress_bar_refresh_rate=0)
    eval_interval = 10
    model_args = ModelArgs(MobileNetV2Lightning, optimizer_args=optimizer_args, num_classes=7)
    dataset = 'ham10k'


def log_after_round_evaluation(
        experiment_logger,
        tags,
        loss: Tensor,
        acc: Tensor,
        train_loss: Tensor,
        train_acc: Tensor,
        test_loss: Tensor,
        test_acc: Tensor,
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
        log_loss_and_acc(f'{tag}-all', loss, acc, experiment_logger, step)
        log_loss_and_acc(f'{tag}-train', train_loss, train_acc, experiment_logger, step)
        log_loss_and_acc(f'{tag}-test', test_loss, test_acc, experiment_logger, step)
        log_goal_test_acc(f'{tag}-all', acc, experiment_logger, step)
        log_goal_test_acc(f'{tag}-train', train_acc, experiment_logger, step)
        log_goal_test_acc(f'{tag}-test', test_acc, experiment_logger, step)


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
        optimizer_args,
        train_args,
        model_args,
        eval_interval,
        dataset,
        mean,
        std
):
    fix_random_seeds(seed)
    initialize_clients_fn = DEFAULT_CLIENT_INIT_FN

    fed_dataset_test = None
    if dataset == 'femnist':
        fed_dataset = load_femnist_dataset(str((REPO_ROOT / 'data').absolute()),
                                           num_clients=num_clients, batch_size=batch_size)
    elif dataset == 'mnist':
        fed_dataset = load_mnist_dataset(str((REPO_ROOT / 'data').absolute()),
                                         num_clients=num_clients, batch_size=batch_size)
    elif dataset == 'ham10k':
        fed_dataset, fed_dataset_test = load_ham10k_few_big_many_small_federated(
            batch_size=8, mean=mean, std=std
        )
        initialize_clients_fn = initialize_ham10k_clients
    else:
        raise ValueError(f'dataset "{dataset}" unknown')

    data_distribution_logged = False
    for cf in client_fraction:
        fedavg_context = FedAvgExperimentContext(name=name, client_fraction=cf, local_epochs=local_epochs,
                                                 lr=lr, batch_size=batch_size, optimizer_args=optimizer_args,
                                                 model_args=model_args, train_args=train_args, train_args_eval=None,
                                                 dataset_name=dataset, eval_interval=eval_interval)
        experiment_specification = f'{fedavg_context}'
        experiment_logger = create_tensorboard_logger(fedavg_context.name, experiment_specification)

        if not data_distribution_logged:
            log_dataset_distribution(experiment_logger, 'full dataset', fed_dataset)
            data_distribution_logged = True

        log_after_round_evaluation_fns = [
            partial(log_after_round_evaluation, experiment_logger, 'fedavg')
        ]
        run_fedavg(context=fedavg_context, num_rounds=total_fedavg_rounds, dataset_train=fed_dataset,
                   dataset_test=fed_dataset_test, save_states=True, restore_state=True, eval_interval=eval_interval,
                   after_round_evaluation=log_after_round_evaluation_fns, initialize_clients_fn=initialize_clients_fn)
