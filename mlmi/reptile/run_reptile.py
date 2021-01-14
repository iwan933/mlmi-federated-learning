import argparse
import sys
sys.path.insert(0, 'C:\\Users\\Richard\\Desktop\\Informatik\\Semester_5\\MLMI\\git\\mlmi-federated-learning')
from itertools import cycle

from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import optim

from mlmi.log import getLogger
from mlmi.reptile.omniglot import load_omniglot_datasets
from mlmi.reptile.model import ReptileClient, ReptileServer, OmniglotLightning
from mlmi.reptile.util import reptile_train_step
from mlmi.struct import ExperimentContext, ModelArgs, TrainArgs, OptimizerArgs
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
    experiment_logger.experiment.add_scalar('test-test/loss/{}/mean'.format(model_name), torch.mean(loss), global_step=global_step)
    experiment_logger.experiment.add_histogram('test-test/acc/{}'.format(model_name), acc, global_step=global_step)
    experiment_logger.experiment.add_scalar('test-test/acc/{}/mean'.format(model_name), torch.mean(acc), global_step=global_step)

class ReptileTrainingArgs:
    """
    Container for meta-learning parameters
    :param model: Base model
    :param inner_optimizer: Optimizer on task level
    :param inner_learning_rate: Learning rate for task level optimizer
    :param num_inner_steps: Number of training steps on task level
    :param log_every_n_steps:
    :param inner_batch_size: Batch size for training on task level. A value of -1
        means batch size is equal to local training set size (full batch
        training)
    :param meta_batch_size: Batch size of tasks for single meta-training step.
        A value of -1 means meta batch size is equal to total number of training
        tasks (full batch meta training)
    :param meta_learning_rate_initial: Learning rate for meta training (initial
        value). Learning rate decreases linearly with training progress to reach
        meta_learning_rate_final at end of training.
    :param meta_learning_rate_final: Final value for learning rate for meta
        training. If None, this will be equal to meta_learning_rate_initial and
        learning rate will remain constant over training.
    :param num_meta_steps: Number of total meta training steps
    :return:
    """
    def __init__(self,
                 model,
                 inner_optimizer,
                 inner_learning_rate=0.03,
                 num_inner_steps=1,
                 log_every_n_steps=3,
                 inner_batch_size=5,
                 meta_batch_size=-1,
                 meta_learning_rate_initial=0.03,
                 meta_learning_rate_final=None,
                 num_meta_steps=1000):
        self.model = model
        self.inner_optimizer = inner_optimizer
        self.inner_learning_rate = inner_learning_rate
        self.num_inner_steps = num_inner_steps
        self.log_every_n_steps = log_every_n_steps
        self.inner_batch_size = inner_batch_size
        self.meta_batch_size = meta_batch_size
        self.meta_learning_rate_initial = meta_learning_rate_initial
        self.meta_learning_rate_final = meta_learning_rate_final
        if self.meta_learning_rate_final is None:
            self.meta_learning_rate_final = self.meta_learning_rate_initial
        self.num_meta_steps = num_meta_steps

    def get_inner_training_args(self):
        """
        Return TrainArgs for inner training (training on task level)
        """
        inner_training_args = TrainArgs(
            min_steps=self.num_inner_steps,
            max_steps=self.num_inner_steps,
            log_every_n_steps=self.log_every_n_steps,
            #weights_summary=None,  # Do not show model summary
            #progress_bar_refresh_rate=0  # Do not show training progress bar
        )
        if torch.cuda.is_available():
            inner_training_args.kwargs['gpus'] = 1
        return inner_training_args

    def get_meta_training_args(self, frac_done: float):
        """
        Return TrainArgs for meta training
        :param frac_done: Fraction of meta training steps already done
        """
        return TrainArgs(
            meta_learning_rate=frac_done * self.meta_learning_rate_final + \
                                 (1 - frac_done) * self.meta_learning_rate_initial
        )


def run_reptile(context: ExperimentContext, initial_model_state=None):

    num_clients_train = 10000
    num_clients_test = 100
    num_classes_per_client = 5
    num_shots_per_class = 5

    eval_iters = 1

    reptile_args = ReptileTrainingArgs(
        model=OmniglotLightning,
        inner_optimizer=optim.Adam,
        inner_learning_rate=0.001,
        num_inner_steps=5,
        log_every_n_steps=3,
        inner_batch_size=10,
        meta_batch_size=5,
        meta_learning_rate_initial=1,
        meta_learning_rate_final=0,
        num_meta_steps=100000
    )
    experiment_logger = create_tensorboard_logger(
        context.name,
        (f"c{num_clients_train};{num_classes_per_client}-way{num_shots_per_class}-shot;"
         f"mlr{str(reptile_args.meta_learning_rate_initial).replace('.', '')}"
         f"ilr{str(reptile_args.inner_learning_rate).replace('.', '')}"
         f"is{reptile_args.num_inner_steps}")
    )

    # Load and prepare Omniglot data
    data_dir = REPO_ROOT / 'data' / 'omniglot'
    omniglot_train_clients, omniglot_test_clients = load_omniglot_datasets(
        str(data_dir.absolute()),
        num_clients_train=num_clients_train,
        num_clients_test=num_clients_test,
        num_classes_per_client=num_classes_per_client,
        num_shots_per_class=num_shots_per_class,
        inner_batch_size=reptile_args.inner_batch_size
    )
    # Prepare ModelArgs for task training
    inner_optimizer_args = OptimizerArgs(
        optimizer_class=reptile_args.inner_optimizer,
        lr=reptile_args.inner_learning_rate
    )
    inner_model_args = ModelArgs(
        reptile_args.model,
        inner_optimizer_args,
        num_classes=num_classes_per_client
    )
    dummy_optimizer_args = OptimizerArgs(
        optimizer_class=optim.SGD
    )
    meta_model_args = ModelArgs(
        reptile_args.model,
        dummy_optimizer_args,
        num_classes=num_classes_per_client
    )

    # Set up clients
    # Since we are doing meta-learning, we need separate sets of training and
    # test clients
    train_clients = []
    for c in omniglot_train_clients.train_data_local_dict.keys():
        client = ReptileClient(
            client_id=str(c),
            model_args=inner_model_args,
            context=context,
            train_dataloader=omniglot_train_clients.train_data_local_dict[c],
            num_train_samples=omniglot_train_clients.data_local_train_num_dict[c],
            test_dataloader=omniglot_train_clients.test_data_local_dict[c],
            num_test_samples=omniglot_train_clients.data_local_test_num_dict[c],
            lightning_logger=experiment_logger
        )
        checkpoint_callback = ModelCheckpoint(
            filepath=str(client.get_checkpoint_path(suffix='cb').absolute()))
        client.set_trainer_callbacks([checkpoint_callback])
        train_clients.append(client)
    test_clients = []
    for c in omniglot_test_clients.train_data_local_dict.keys():
        client = ReptileClient(
            client_id=str(c),
            model_args=inner_model_args,
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
        model_args=meta_model_args,
        context=context,
        initial_model_state=initial_model_state
    )

    # Perform training
    if reptile_args.meta_batch_size == -1:
        client_batches = [train_clients]
    else:
        client_batches = [
            train_clients[i:i + reptile_args.meta_batch_size] for i in range(
                0, len(train_clients), reptile_args.meta_batch_size
            )
        ]
    for i, client_batch in \
            zip(range(reptile_args.num_meta_steps), cycle(client_batches)):
        logger.info(f'starting meta training round {i + 1}')
        # train
        reptile_train_step(
            aggregator=server,
            participants=client_batch,
            inner_training_args=reptile_args.get_inner_training_args(),
            meta_training_args=reptile_args.get_meta_training_args(
                frac_done=i / reptile_args.num_meta_steps
            )
        )

        if i % eval_iters == eval_iters - 1:
            # test
            # Do one training step on test-train set
            reptile_train_step(
                aggregator=server,
                participants=test_clients,
                inner_training_args=reptile_args.get_inner_training_args(),
                evaluation_mode=True
            )
            # Evaluate on test-test set
            result = evaluate_local_models(participants=test_clients)
            log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'),
                             experiment_logger, global_step=i+1)

        logger.info('finished training round')


if __name__ == '__main__':
    def run():
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args()

        context = ExperimentContext(name='reptile')
        run_reptile(context)

    run()
