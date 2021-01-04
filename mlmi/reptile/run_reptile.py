import argparse

from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import optim

from mlmi.log import getLogger
from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.reptile.model import ReptileClient, ReptileServer
from mlmi.fedavg.model import CNNLightning
from mlmi.reptile.util import run_reptile_round
from mlmi.struct import ExperimentContext, ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger, evaluate_global_model


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
    experiment_logger.experiment.add_histogram('test/loss/{}'.format(model_name), loss, global_step=global_step)
    experiment_logger.experiment.add_scalar('test/loss/{}/mean'.format(model_name), torch.mean(loss), global_step=global_step)
    experiment_logger.experiment.add_histogram('test/acc/{}'.format(model_name), acc, global_step=global_step)
    experiment_logger.experiment.add_scalar('test/acc/{}/mean'.format(model_name), torch.mean(acc), global_step=global_step)


def run_reptile(context: ExperimentContext,
                initial_model_params):
    model = CNNLightning
    num_clients = 20
    meta_batch_size = -1
    meta_learning_rate = 0.03
    num_meta_steps = 1000
    inner_batch_size = -1
    inner_learning_rate = 0.03
    num_inner_steps = 3
    #experiment_logger = create_tensorboard_logger(context.name,
    #                                              'c{}s{}bs{}lr{}'.format(num_clients, steps, batch_size,
    #                                                                      str(learning_rate).replace('.', '')))

    inner_optimizer_args = OptimizerArgs(optim.Adam, lr=inner_learning_rate)
    inner_model_args = ModelArgs(model, inner_optimizer_args, only_digits=False)
    if torch.cuda.is_available():
        training_args = TrainArgs(
            min_steps=num_inner_steps, max_steps=num_inner_steps, log_every_n_steps=log_every_n_steps, gpus=1
        )
    else:
        training_args = TrainArgs(
            min_steps=num_inner_steps, max_steps=num_inner_steps, log_every_n_steps=log_every_n_steps
        )
    data_dir = REPO_ROOT / 'data' / 'reptile'
    fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=num_clients, batch_size=inner_batch_size)

    # Set up clients with non-i.i.d. data
    clients = []
    for c, dataset in fed_dataset.train_data_local_dict.items():
        client = ReptileClient(str(c), inner_model_args, context, fed_dataset.train_data_local_dict[c],
                              fed_dataset.data_local_train_num_dict[c], fed_dataset.test_data_local_dict[c],
                              fed_dataset.data_local_test_num_dict[c], experiment_logger)
        checkpoint_callback = ModelCheckpoint(filepath=str(client.get_checkpoint_path(suffix='cb').absolute()))
        client.set_trainer_callbacks([checkpoint_callback])
        clients.append(client)

    # Set up server
    meta_optimizer_args = OptimizerArgs(optim.Adam, lr=meta_learning_rate)
    meta_model_args = ModelArgs(model, meta_optimizer_args, only_digits=False)
    server = ReptileServer('initial_server', meta_model_args, context, initial_model_params)
    num_train_samples = [client.num_train_samples for client in clients]

    # Perform training
    for i in range(num_meta_steps):
        logger.info(f'starting meta training round {i+1}')
        # train
        run_reptile_round(server, clients, training_args, num_train_samples=num_train_samples)
        # test
        result = evaluate_global_model(global_model_participant=server, participants=clients)
        log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'), experiment_logger, i)
        logger.info('finished training round')


if __name__ == '__main__':
    def run():
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args()

        context = ExperimentContext(name='reptile')
        run_reptile(context)

    run()
