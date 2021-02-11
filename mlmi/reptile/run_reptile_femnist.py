import argparse
import sys
sys.path.append('C:/Users/Richard/Desktop/Informatik/Semester_5/MLMI/git/mlmi-federated-learning')

from mlmi.structs import FederatedDatasetData, ModelArgs

import random

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import optim

from mlmi.reptile.args import argument_parser
from mlmi.log import getLogger
from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import CNNLightning
from mlmi.reptile.model import ReptileClient, ReptileServer
from mlmi.reptile.util import reptile_train_step
from mlmi.reptile.structs import ReptileTrainingArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger, evaluate_local_models


logger = getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_const',
                        const=True, default=False)

def cyclerange(start, stop, len):
    assert start < len and stop < len, "Error: start and stop must be < len"
    if start > stop:
        return list(range(start, len)) + list(range(0, stop))
    return list(range(start, stop))


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
    experiment_logger.experiment.add_scalar('test-test/loss/{}/mean'.format(model_name), torch.mean(loss), global_step=global_step)
    experiment_logger.experiment.add_histogram('test-test/acc/{}'.format(model_name), acc, global_step=global_step)
    experiment_logger.experiment.add_scalar('test-test/acc/{}/mean'.format(model_name), torch.mean(acc), global_step=global_step)


def initialize_clients(dataset: FederatedDatasetData, model_args: ModelArgs, context, experiment_logger):
    clients = []
    for c in dataset.train_data_local_dict.keys():
        client = ReptileClient(
            client_id=str(c),
            model_args=model_args,
            context=context,
            train_dataloader=dataset.train_data_local_dict[c],
            num_train_samples=dataset.data_local_train_num_dict[c],
            test_dataloader=dataset.test_data_local_dict[c],
            num_test_samples=dataset.data_local_test_num_dict[c],
            lightning_logger=experiment_logger
        )
        clients.append(client)
    return clients


def run_reptile(context: str, initial_model_state=None):

    args = argument_parser().parse_args()
    RANDOM = random.Random(args.seed)

    # TODO: Possibly implement logic using ReptileExperimentContext
    reptile_args = ReptileTrainingArgs(
        model_class=CNNLightning,
        sgd=args.sgd,
        inner_learning_rate=args.learning_rate,
        num_inner_steps=args.inner_iters,
        num_inner_steps_eval=args.eval_iters,
        log_every_n_steps=3,
        meta_learning_rate_initial=args.meta_step,
        meta_learning_rate_final=args.meta_step_final,
        num_classes_per_client=62
    )
    experiment_logger = create_tensorboard_logger(
        'reptile',
        (
            f"{context};seed{args.seed};"
            f"train-clients{args.train_clients};"
            f"{args.classes}-way{args.shots}-shot;"
            f"ib{args.inner_batch}ii{args.inner_iters}"
            f"ilr{str(args.learning_rate).replace('.', '')}"
            f"ms{str(args.meta_step).replace('.', '')}"
            f"mb{args.meta_batch}ei{args.eval_iters}"
            f"{'sgd' if args.sgd else 'adam'}"
        )
    )

    # Load and prepare Omniglot data
    data_dir = REPO_ROOT / 'data' / 'omniglot'
    femnist_clients = load_femnist_dataset(
        data_dir=str(data_dir.absolute()),
        batch_size=args.inner_batch,
        random_seed=args.seed
    )

    # Set up clients
    clients = initialize_clients(femnist_clients, reptile_args.get_inner_model_args(), context,
                                 experiment_logger)

    # Set up server
    server = ReptileServer(
        participant_name='initial_server',
        model_args=reptile_args.get_meta_model_args(),
        context=context,  # TODO: Change to ReptileExperimentContext
        initial_model_state=initial_model_state
    )

    # Perform training
    for i in range(args.meta_iters):
        if args.meta_batch == -1:
            meta_batch = clients
        else:
            meta_batch = [
                clients[k] for k in cyclerange(
                    i*args.meta_batch % len(clients),
                    (i+1)*args.meta_batch % len(clients),
                    len(clients)
                )
            ]
        # Meta training step
        reptile_train_step(
            aggregator=server,
            participants=meta_batch,
            inner_training_args=reptile_args.get_inner_training_args(),
            meta_training_args=reptile_args.get_meta_training_args(
                frac_done=i / args.meta_iters
            )
        )

        # Evaluation on subsample of clients
        eval_clients = RANDOM.sample(population=clients, k=5)
        if i % args.eval_interval == 0:
            reptile_train_step(
                aggregator=server,
                participants=eval_clients,
                inner_training_args=reptile_args.get_inner_training_args(eval=True),
                evaluation_mode=True
            )
            result = evaluate_local_models(participants=eval_clients)
            log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'),
                             experiment_logger, global_step=i)
        logger.info('finished training round')

    # Final evaluation on all clients
    reptile_train_step(
        aggregator=server,
        participants=clients,
        inner_training_args=reptile_args.get_inner_training_args(eval=True),
        evaluation_mode=True
    )
    result = evaluate_local_models(participants=clients)
    log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'),
                     experiment_logger, global_step=args.meta_iters)


if __name__ == '__main__':
    def run():
        run_reptile(context='reptile_femnist')
    run()
