import argparse
import sys

sys.path.insert(0, '/')
import random

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import optim

from mlmi.log import getLogger
from mlmi.reptile.omniglot import load_omniglot_datasets

from mlmi.reptile.model import OmniglotModel
from mlmi.struct import ExperimentContext, ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger

from mlmi.reptile.debugging.dataloading_ours_model_ours.supervised_reptile_3 import Reptile
import tensorflow.compat.v1 as tf

RANDOM_SEED = 77
RANDOM = random.Random(RANDOM_SEED)

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
    experiment_logger.experiment.add_scalar('test-test/loss/{}/mean'.format(model_name), torch.mean(loss),
                                            global_step=global_step)
    experiment_logger.experiment.add_histogram('test-test/acc/{}'.format(model_name), acc, global_step=global_step)
    experiment_logger.experiment.add_scalar('test-test/acc/{}/mean'.format(model_name), torch.mean(acc),
                                            global_step=global_step)

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
                 num_inner_steps_eval=50,
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
        self.num_inner_steps_eval = num_inner_steps_eval
        self.log_every_n_steps = log_every_n_steps
        self.inner_batch_size = inner_batch_size
        self.meta_batch_size = meta_batch_size
        self.meta_learning_rate_initial = meta_learning_rate_initial
        self.meta_learning_rate_final = meta_learning_rate_final
        if self.meta_learning_rate_final is None:
            self.meta_learning_rate_final = self.meta_learning_rate_initial
        self.num_meta_steps = num_meta_steps

    def get_inner_training_args(self, eval=False):
        """
        Return TrainArgs for inner training (training on task level)
        """
        inner_training_args = TrainArgs(
            min_steps=self.num_inner_steps if not eval else self.num_inner_steps_eval,
            max_steps=self.num_inner_steps if not eval else self.num_inner_steps_eval,
            log_every_n_steps=self.log_every_n_steps,
            weights_summary=None,  # Do not show model summary
            progress_bar_refresh_rate=0  # Do not show training progress bar
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

    num_clients_train = 10000
    num_clients_test = 1000
    num_classes_per_client = 5
    num_shots_per_class = 5

    eval_iters = 10

    reptile_args = ReptileTrainingArgs(
        model=OmniglotModel,
        inner_optimizer=optim.Adam,
        inner_learning_rate=0.001,
        num_inner_steps=5,
        num_inner_steps_eval=50,
        log_every_n_steps=3,
        inner_batch_size=10,
        meta_batch_size=5,
        meta_learning_rate_initial=1,
        meta_learning_rate_final=0,
        num_meta_steps=3000
    )
    experiment_logger = create_tensorboard_logger(
        context.name,
        "dataloading_ours;models_ours"
    )

    # Load and prepare Omniglot data
    data_dir = REPO_ROOT / 'data' / 'omniglot'

    #######
    tf.disable_eager_execution()
    #######

    omniglot_train_clients, omniglot_test_clients = load_omniglot_datasets(
        str(data_dir.absolute()),
        num_clients_train=num_clients_train,
        num_clients_test=num_clients_test,
        num_classes_per_client=num_classes_per_client,
        num_shots_per_class=num_shots_per_class,
        inner_batch_size=reptile_args.inner_batch_size,
        random_seed=RANDOM_SEED
    )

    # Prepare ModelArgs for task training
    inner_optimizer_args = OptimizerArgs(
        optimizer_class=reptile_args.inner_optimizer,
        lr=reptile_args.inner_learning_rate,
        betas=(0, 0.999)
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
    """
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
    )"""

    torch_model = OmniglotModel(num_classes=num_classes_per_client)
    torch_optimizer = inner_optimizer_args.optimizer_class(
        torch_model.parameters(),
        **inner_optimizer_args.optimizer_kwargs
    )

    reptile = Reptile(
        model=torch_model,
        optimizer=torch_optimizer,
        inner_iterations=reptile_args.num_inner_steps,
        inner_iterations_eval=reptile_args.num_inner_steps_eval
    )

    for i in range(reptile_args.num_meta_steps):
        frac_done = i / reptile_args.num_meta_steps
        cur_meta_step_size = frac_done * reptile_args.meta_learning_rate_final + (1 - frac_done) * reptile_args.meta_learning_rate_initial

        meta_batch = {k:omniglot_train_clients.train_data_local_dict[k] for k in cyclerange(
                i*reptile_args.meta_batch_size % len(omniglot_train_clients.train_data_local_dict),
                (i+1)*reptile_args.meta_batch_size % len(omniglot_train_clients.train_data_local_dict),
                len(omniglot_train_clients.train_data_local_dict)
            )}

        reptile.train_step(
            meta_batch=meta_batch,
            meta_step_size=cur_meta_step_size
        )

        if i % eval_iters == 0:
            accuracies = []
            k = RANDOM.randrange(len(omniglot_train_clients.train_data_local_dict))
            train_train = omniglot_train_clients.train_data_local_dict[k]
            train_test = omniglot_train_clients.test_data_local_dict[k]
            k = RANDOM.randrange(len(omniglot_test_clients.train_data_local_dict))
            test_train = omniglot_test_clients.train_data_local_dict[k]
            test_test = omniglot_test_clients.test_data_local_dict[k]

            for train_dl, test_dl in [(train_train, train_test), (test_train, test_test)]:
                correct = reptile.evaluate(
                    train_dl,
                    test_dl
                )
                accuracies.append(correct / num_classes_per_client)
            print('batch %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))

            # Write to TensorBoard
            experiment_logger.experiment.add_scalar('train-test/acc/{}/mean'.format('global_model'), accuracies[0], global_step=i)
            experiment_logger.experiment.add_scalar('test-test/acc/{}/mean'.format('global_model'), accuracies[1], global_step=i)

    """
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

    # Final evaluation on all test clients
    reptile_train_step(
        aggregator=server,
        participants=test_clients,
        inner_training_args=reptile_args.get_inner_training_args(eval=True),
        evaluation_mode=True
    )
    result = evaluate_local_models(participants=test_clients)
    log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'),
                     experiment_logger, global_step=reptile_args.num_meta_steps)
    print(f"Final test accuracy: {torch.mean(result.get('test/acc'))}")
    """

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
