import argparse
import sys
sys.path.insert(0, '/')
import random

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import optim

from mlmi.log import getLogger
from mlmi.reptile.omniglot import load_omniglot_datasets
from mlmi.reptile.model import OmniglotLightning
from mlmi.struct import ExperimentContext, TrainArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger

from mlmi.reptile.debugging.nichol_reptile_our_dataloading.supervised_reptile import OmniglotModel, Reptile
from mlmi.reptile.debugging.nichol_reptile_our_dataloading import weight_decay
import tensorflow.compat.v1 as tf

logger = getLogger(__name__)

RANDOM_SEED = 77
RANDOM = random.Random(RANDOM_SEED)

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
    num_clients_test = 1000
    num_classes_per_client = 5
    num_shots_per_class = 5

    # Every *eval_iters* meta steps, evaluation is performed on one random
    # client in the training and test set, respectively
    eval_iters = 10

    reptile_args = ReptileTrainingArgs(
        model=OmniglotLightning,
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
        (f"nichol_reptile_our_dataloading;{num_clients_train}clients;{num_classes_per_client}"
         f"-way{num_shots_per_class}-shot;"
         f"mlr{str(reptile_args.meta_learning_rate_initial).replace('.', '')}"
         f"ilr{str(reptile_args.inner_learning_rate).replace('.', '')}"
         f"is{reptile_args.num_inner_steps}")
    )

    # Load and prepare Omniglot data
    data_dir = REPO_ROOT / 'data' / 'omniglot'
    tf.disable_eager_execution()
    omniglot_train_clients, omniglot_test_clients = load_omniglot_datasets(
        str(data_dir.absolute()),
        num_clients_train=num_clients_train,
        num_clients_test=num_clients_test,
        num_classes_per_client=num_classes_per_client,
        num_shots_per_class=num_shots_per_class,
        inner_batch_size=reptile_args.inner_batch_size,
        tensorflow=True,
        random_seed=RANDOM_SEED
    )

    with tf.Session() as sess:
        model = OmniglotModel(num_classes_per_client,
                              learning_rate=reptile_args.inner_learning_rate)
        reptile = Reptile(sess,
                          transductive=True,
                          pre_step_op=weight_decay(1))
        accuracy_ph = tf.placeholder(tf.float32, shape=())
        tf.summary.scalar('accuracy', accuracy_ph)
        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())
        for i in range(reptile_args.num_meta_steps):
            frac_done = i / reptile_args.num_meta_steps
            cur_meta_step_size = frac_done * reptile_args.meta_learning_rate_final + (1 - frac_done) * reptile_args.meta_learning_rate_initial

            meta_batch = {
                k: omniglot_train_clients.train_data_local_dict[k] for k in cyclerange(
                    i*reptile_args.meta_batch_size % len(omniglot_train_clients.train_data_local_dict),
                    (i+1)*reptile_args.meta_batch_size % len(omniglot_train_clients.train_data_local_dict),
                    len(omniglot_train_clients.train_data_local_dict)
                )
            }

            reptile.train_step(
                meta_batch=meta_batch,
                input_ph=model.input_ph,
                label_ph=model.label_ph,
                minimize_op=model.minimize_op,
                inner_iters=reptile_args.num_inner_steps,
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
                        train_data_loader=train_dl,
                        test_data_loader=test_dl,
                        input_ph=model.input_ph,
                        label_ph=model.label_ph,
                        minimize_op=model.minimize_op,
                        predictions=model.predictions,
                        inner_iters=reptile_args.num_inner_steps_eval
                    )
                    #summary = sess.run(merged, feed_dict={accuracy_ph: correct/num_classes_per_client})
                    accuracies.append(correct / num_classes_per_client)
                print('batch %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))

                # Write to TensorBoard
                experiment_logger.experiment.add_scalar('train-test/acc/{}/mean'.format('global_model'), accuracies[0], global_step=i)
                experiment_logger.experiment.add_scalar('test-test/acc/{}/mean'.format('global_model'), accuracies[1], global_step=i)


if __name__ == '__main__':
    def run():
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args()

        context = ExperimentContext(name='reptile')
        run_reptile(context)

    run()
