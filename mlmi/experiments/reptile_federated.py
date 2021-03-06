
from mlmi.datasets.ham10k import load_ham10k_federated, load_ham10k_few_big_many_small_federated
from mlmi.models.ham10k import GlobalConfusionMatrix, GlobalTestTestConfusionMatrix, GlobalTrainTestConfusionMatrix, \
    MobileNetV2Lightning

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

from mlmi.plot import generate_confusion_matrix_heatmap, generate_data_label_heatmap
from mlmi.settings import REPO_ROOT
from mlmi.structs import FederatedDatasetData
from mlmi.utils import create_tensorboard_logger, fix_random_seeds

ex = Experiment('reptile')


@ex.config
def ham10k():
    name = 'ham10kreptile'
    dataset = 'ham10k'  # Options: 'omniglot', 'femnist'
    swap_labels = False  # Only used with dataset='femnist'
    classes = 0  # Only used with dataset='omniglot'
    shots = 0  # Only used with dataset='omniglot'
    seed = 123123123

    model_class = MobileNetV2Lightning
    sgd = True  # True -> Use SGD as inner optimizer; False -> Use Adam
    adam_betas = (0.9, 0.999)  # Used only if sgd = False

    num_clients_train = 0 # Not used here
    num_clients_test = 0  # Not used here
    meta_batch_size = 5
    num_meta_steps = 1000
    meta_learning_rate_initial = 1
    meta_learning_rate_final = 0.8

    eval_interval = 50
    num_eval_clients_training = -1
    do_final_evaluation = True
    num_eval_clients_final = -1

    inner_batch_size = 8
    inner_learning_rate = [0.001, 0.002, 0.004]
    num_inner_epochs = [1]
    num_inner_epochs_eval = [5]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


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
        loss_test_test,
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


def log_dataset_distribution(experiment_logger, tag: str, dataset: FederatedDatasetData):
    dataloaders = list(dataset.train_data_local_dict.values())
    image = generate_data_label_heatmap(tag, dataloaders, dataset.class_num)
    experiment_logger.experiment.add_image('label distribution', image.numpy())


@ex.automain
def run_reptile_experiment(
    name,
    dataset,
    swap_labels,
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
    num_inner_epochs,
    num_inner_epochs_eval,
    mean=None,
    std=None
):
    fix_random_seeds(seed)
    fed_dataset_test = None
    if dataset == 'femnist':
        fed_dataset_train = load_femnist_dataset(
            data_dir=str((REPO_ROOT / 'data').absolute()),
            num_clients=num_clients_train,
            batch_size=inner_batch_size,
            random_seed=seed
        )
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
    elif dataset == 'ham10k':
        fed_dataset_train, fed_dataset_test = load_ham10k_few_big_many_small_federated(batch_size=inner_batch_size, mean=mean, std=std)
    else:
        raise ValueError(f'dataset "{dataset}" unknown')

    if not hasattr(inner_learning_rate, '__iter__'):
        inner_learning_rate = [inner_learning_rate]
    if not hasattr(num_inner_epochs, '__iter__'):
        num_inner_epochs = [num_inner_epochs]
    if not hasattr(num_inner_epochs_eval, '__iter__'):
        num_inner_epochs = [num_inner_epochs_eval]
    #data_distribution_logged = False
    for lr in inner_learning_rate:
        for _is in num_inner_epochs:
            for _ieev in num_inner_epochs_eval:
                reptile_context = ReptileExperimentContext(
                    name=name,
                    dataset_name=dataset,
                    swap_labels=swap_labels,
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
                    num_inner_epochs=_is,
                    num_inner_epochs_eval=_ieev
                )

                experiment_specification = f'{reptile_context}'
                experiment_logger = create_tensorboard_logger(
                    reptile_context.name, experiment_specification
                )
                reptile_context.experiment_logger = experiment_logger

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
