import argparse
import sys
from typing import List

from mlmi.reptile.structs import ReptileExperimentContext
from mlmi.structs import FederatedDatasetData
from itertools import cycle

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import optim

from mlmi.log import getLogger
from mlmi.reptile.omniglot import load_omniglot_datasets
from mlmi.reptile.model import ReptileClient, ReptileServer, OmniglotLightning
from mlmi.reptile.util import reptile_train_step
from mlmi.structs import ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger, evaluate_local_models

logger = getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_const',
                        const=True, default=False)


def log_loss_and_acc(model_name: str, loss: torch.Tensor, acc: torch.Tensor, experiment_logger: LightningLoggerBase,
                     global_step: int):
    """
    Logs the loss and accuracy in an histogram as well as scalar
    :param model_name: name for logging
    :param loss: loss tensor
    :param acc: acc tensor
    :param experiment_logger: lightning logger
    :param global_step: global step
    :return:
    """
    experiment_logger.experiment.add_histogram('test-test/loss/{}'.format(model_name), loss, global_step=global_step)
    experiment_logger.experiment.add_scalar('test-test/loss/{}/mean'.format(model_name), torch.mean(loss),
                                            global_step=global_step)
    experiment_logger.experiment.add_histogram('test-test/acc/{}'.format(model_name), acc, global_step=global_step)
    experiment_logger.experiment.add_scalar('test-test/acc/{}/mean'.format(model_name), torch.mean(acc),
                                            global_step=global_step)


def initialize_reptile_clients(context: ReptileExperimentContext, fed_dataset: FederatedDatasetData):
    clients = []
    for c in fed_dataset.train_data_local_dict.keys():
        client = ReptileClient(
            client_id=str(c),
            model_args=context.inner_model_args,
            context=context,
            train_dataloader=fed_dataset.train_data_local_dict[c],
            num_train_samples=fed_dataset.data_local_train_num_dict[c],
            test_dataloader=fed_dataset.test_data_local_dict[c],
            num_test_samples=fed_dataset.data_local_test_num_dict[c],
            lightning_logger=context.experiment_logger
        )
        clients.append(client)
    return clients


def create_client_batches(clients: List[ReptileClient], batch_size: int) -> List[List[ReptileClient]]:
    if batch_size == -1:
        client_batches = [clients]
    else:
        client_batches = [
            clients[i:i + batch_size] for i in range(
                0, len(clients), batch_size
            )
        ]
    return client_batches


def get_meta_train_args_for_step(
        context: ReptileExperimentContext,
        steps: int
):
    fraction = steps / context.meta_num_steps
    context.meta_training_args.kwargs['meta_learning_rate'] = fraction * context.meta_learning_rate_final + \
                                                              (1 - fraction) * context.meta_learning_rate_initial
    return context.meta_training_args


def run_reptile(context: ReptileExperimentContext,
                train_datasets: FederatedDatasetData,
                test_datasets: FederatedDatasetData,
                initial_model_state=None
                ):
    # Set up clients
    # Since we are doing meta-learning, we need separate sets of training and
    # test clients
    train_clients = initialize_reptile_clients(context, train_datasets)
    test_clients = initialize_reptile_clients(context, test_datasets)

    # Set up server
    server = ReptileServer(
        participant_name='initial_server',
        model_args=context.meta_model_args,
        context=context,
        initial_model_state=initial_model_state
    )
    # Perform training
    client_batches = create_client_batches(train_clients, context.meta_batch_size)
    for i, client_batch in \
            zip(range(context.meta_num_steps), cycle(client_batches)):
        logger.info(f'starting meta training round {i + 1}')
        # train
        reptile_train_step(
            aggregator=server,
            participants=client_batch,
            inner_training_args=context.inner_train_args,
            meta_training_args=get_meta_train_args_for_step(context, i)
        )

        if i % context.eval_iters == context.eval_iters - 1:
            # test
            # Do one training step on test-train set
            reptile_train_step(
                aggregator=server,
                participants=test_clients,
                inner_training_args=context.inner_train_args,
                evaluation_mode=True
            )
            # Evaluate on test-test set
            result = evaluate_local_models(participants=test_clients)
            log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'),
                             context.experiment_logger, global_step=i + 1)

        logger.info('finished training round')


def create_reptile_omniglot_experiment_context() -> ReptileExperimentContext:
    num_client_classes = 5
    num_clients_train = 5000
    num_clients_test = 50
    num_classes_per_client = 5
    num_shots_per_class = 5
    eval_iters = 100
    inner_learning_rate = 0.001
    inner_training_steps = 5
    inner_batch_size = 10

    meta_batch_size = 5
    meta_learning_rate_initial = 1
    meta_learning_rate_final = 0
    meta_num_steps = 100000

    inner_training_args = TrainArgs(min_steps=inner_training_steps, max_steps=inner_training_steps)
    meta_training_args = TrainArgs(meta_learning_rate=meta_learning_rate_initial)
    inner_optimizer_args = OptimizerArgs(optimizer_class=optim.Adam, lr=inner_learning_rate)
    meta_optimizer_args = OptimizerArgs(optimizer_class=optim.SGD, lr=meta_learning_rate_initial)
    inner_model_args = ModelArgs(model_class=OmniglotLightning, num_classes=num_client_classes,
                                 optimizer_args=inner_optimizer_args)
    meta_model_args = ModelArgs(model_class=OmniglotLightning, num_classes=num_client_classes,
                                optimizer_args=meta_optimizer_args)
    context = ReptileExperimentContext(name='reptile',
                                       dataset_name='omniglot',
                                       eval_iters=eval_iters,
                                       inner_training_steps=inner_training_steps,
                                       inner_batch_size=inner_batch_size,
                                       inner_optimizer_args=inner_optimizer_args,
                                       inner_learning_rate=inner_learning_rate,
                                       inner_model_args=inner_model_args,
                                       inner_train_args=inner_training_args,
                                       num_clients_train=num_clients_train,
                                       num_clients_test=num_clients_test,
                                       num_classes_per_client=num_classes_per_client,
                                       num_shots_per_class=num_shots_per_class,
                                       meta_model_args=meta_model_args,
                                       meta_batch_size=meta_batch_size,
                                       meta_learning_rate_initial=meta_learning_rate_initial,
                                       meta_learning_rate_final=meta_learning_rate_final,
                                       meta_num_steps=meta_num_steps,
                                       meta_optimizer_args=meta_optimizer_args,
                                       meta_training_args=meta_training_args)
    experiment_logger = create_tensorboard_logger(
        context.name,
        str(context)
    )
    context.experiment_logger = experiment_logger
    return context


def load_omniglot_experiment_datasets(context: ReptileExperimentContext):
    # Load and prepare Omniglot data
    data_dir = REPO_ROOT / 'data' / 'omniglot'
    omniglot_train_datasets, omniglot_test_datasets = load_omniglot_datasets(
        str(data_dir.absolute()),
        num_clients_train=context.num_clients_train,
        num_clients_test=context.num_clients_test,
        num_classes_per_client=context.num_classes_per_client,
        num_shots_per_class=context.num_shots_per_class,
        inner_batch_size=context.inner_batch_size
    )
    return omniglot_train_datasets, omniglot_test_datasets


if __name__ == '__main__':
    def run():
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args()

        context = create_reptile_omniglot_experiment_context()
        train_datasets, test_datasets = load_omniglot_experiment_datasets(context)
        run_reptile(context, train_datasets, test_datasets)

    run()
