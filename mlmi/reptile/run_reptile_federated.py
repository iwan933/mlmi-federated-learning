import argparse
import sys
sys.path.insert(0, 'C:\\Users\\Richard\\Desktop\\Informatik\\Semester_5\\MLMI\\git\\mlmi-federated-learning')
import random

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import optim

from mlmi.reptile.args import argument_parser
from mlmi.log import getLogger
from mlmi.reptile.omniglot import load_omniglot_datasets
from mlmi.reptile.model import ReptileClient, ReptileServer, OmniglotLightning
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


def run_reptile(context: str, initial_model_state=None):

    args = argument_parser().parse_args()
    RANDOM = random.Random(args.seed)

    # TODO: Possibly implement logic using ReptileExperimentContext
    reptile_args = ReptileTrainingArgs(
        model_class=OmniglotLightning,
        inner_optimizer=optim.SGD,
        inner_learning_rate=args.learning_rate,
        num_inner_steps=args.inner_iters,
        num_inner_steps_eval=args.eval_iters,
        log_every_n_steps=3,
        meta_learning_rate_initial=args.meta_step,
        meta_learning_rate_final=args.meta_step_final,
        num_classes_per_client=args.classes
    )
    experiment_logger = create_tensorboard_logger(
        'reptile',
        (
            f"{context};seed{args.seed};"
            f"train-clients{args.train_clients};"
            f"{args.classes}-way{args.shots}-shot;"
            f"mlr{str(args.meta_step).replace('.', '')}"
            f"ilr{str(args.learning_rate).replace('.', '')}"
            f"is{args.inner_iters}"
        )
    )

    # Load and prepare Omniglot data
    data_dir = REPO_ROOT / 'data' / 'omniglot'
    omniglot_train_clients, omniglot_test_clients = load_omniglot_datasets(
        str(data_dir.absolute()),
        num_clients_train=args.train_clients,
        num_clients_test=args.test_clients,
        num_classes_per_client=args.classes,
        num_shots_per_class=args.shots,
        inner_batch_size=args.inner_batch,
        random_seed=args.seed
    )

    # Set up clients
    # Since we are doing meta-learning, we need separate sets of training and
    # test clients
    train_clients = []
    for c in omniglot_train_clients.train_data_local_dict.keys():
        client = ReptileClient(
            client_id=str(c),
            model_args=reptile_args.get_inner_model_args(),
            context=context,
            train_dataloader=omniglot_train_clients.train_data_local_dict[c],
            num_train_samples=omniglot_train_clients.data_local_train_num_dict[c],
            test_dataloader=omniglot_train_clients.test_data_local_dict[c],
            num_test_samples=omniglot_train_clients.data_local_test_num_dict[c],
            lightning_logger=experiment_logger
        )
        #checkpoint_callback = ModelCheckpoint(
        #    filepath=str(client.get_checkpoint_path(suffix='cb').absolute()))
        #client.set_trainer_callbacks([checkpoint_callback])
        train_clients.append(client)
    test_clients = []
    for c in omniglot_test_clients.train_data_local_dict.keys():
        client = ReptileClient(
            client_id=str(c),
            model_args=reptile_args.get_inner_model_args(),
            context=context,
            train_dataloader=omniglot_test_clients.train_data_local_dict[c],
            num_train_samples=omniglot_test_clients.data_local_train_num_dict[c],
            test_dataloader=omniglot_test_clients.test_data_local_dict[c],
            num_test_samples=omniglot_test_clients.data_local_test_num_dict[c],
            lightning_logger=experiment_logger
        )
        test_clients.append(client)

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
            meta_batch = train_clients
        else:
            meta_batch = [
                train_clients[k] for k in cyclerange(
                    i*args.meta_batch % len(train_clients),
                    (i+1)*args.meta_batch % len(train_clients),
                    len(train_clients)
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

        # Evaluation on train and test clients
        if i % args.eval_interval == 0:
            # train-test set
            # Pick one train client at random and test on it
            k = RANDOM.randrange(len(train_clients))
            client = [train_clients[k]]
            reptile_train_step(
                aggregator=server,
                participants=client,
                inner_training_args=reptile_args.get_inner_training_args(eval=True),
                evaluation_mode=True
            )
            result = evaluate_local_models(participants=client)
            experiment_logger.experiment.add_scalar(
                'train-test/acc/{}/mean'.format('global_model'),
                torch.mean(result.get('test/acc')),
                global_step=i + 1
            )
            # test-test set
            # Pick one test client at random and test on it
            k = RANDOM.randrange(len(test_clients))
            client = [test_clients[k]]
            reptile_train_step(
                aggregator=server,
                participants=client,
                inner_training_args=reptile_args.get_inner_training_args(eval=True),
                evaluation_mode=True
            )
            result = evaluate_local_models(participants=client)
            experiment_logger.experiment.add_scalar(
                'test-test/acc/{}/mean'.format('global_model'),
                torch.mean(result.get('test/acc')),
                global_step=i + 1
            )
        logger.info('finished training round')

    # Final evaluation on a sample of train/test clients
    for label, client_set in zip(['Train', 'Test'], [train_clients, test_clients]):
        eval_sample = RANDOM.sample(client_set, args.eval_samples)
        reptile_train_step(
            aggregator=server,
            participants=eval_sample,
            inner_training_args=reptile_args.get_inner_training_args(eval=True),
            evaluation_mode=True
        )
        result = evaluate_local_models(participants=eval_sample)
        log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'),
                         experiment_logger, global_step=args.meta_iters)
        experiment_logger.experiment.add_scalar(
            f'final_{label}_acc',
            torch.mean(result.get('test/acc')),
            global_step=0
        )
        print(f"{label} accuracy: {torch.mean(result.get('test/acc'))}")


if __name__ == '__main__':
    def run():
        run_reptile(context='reptile_federated')

    run()
