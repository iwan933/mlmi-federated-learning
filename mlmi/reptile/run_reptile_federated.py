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
from mlmi.struct import ExperimentContext, ModelArgs, TrainArgs, OptimizerArgs
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

class ReptileTrainingArgs:
    """
    Container for meta-learning parameters
    :param model: Class of base model
    :param inner_optimizer: Optimizer on task level
    :param inner_learning_rate: Learning rate for task level optimizer
    :param num_inner_steps: Number of training steps on task level
    :param num_inner_steps_eval: Number of training steps for training on test
        set
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
                 model_class,
                 inner_optimizer,
                 inner_learning_rate=0.03,
                 num_inner_steps=1,
                 num_inner_steps_eval=50,
                 log_every_n_steps=3,
                 meta_learning_rate_initial=0.03,
                 meta_learning_rate_final=None,
                 num_classes_per_client=5):
        self.model_class = model_class
        self.inner_optimizer = inner_optimizer
        self.inner_learning_rate = inner_learning_rate
        self.num_inner_steps = num_inner_steps
        self.num_inner_steps_eval = num_inner_steps_eval
        self.log_every_n_steps = log_every_n_steps
        self.meta_learning_rate_initial = meta_learning_rate_initial
        self.meta_learning_rate_final = meta_learning_rate_final
        if self.meta_learning_rate_final is None:
            self.meta_learning_rate_final = self.meta_learning_rate_initial
        self.num_classes_per_client = num_classes_per_client

    def get_inner_model_args(self):
        inner_optimizer_args = OptimizerArgs(
            optimizer_class=self.inner_optimizer,
            lr=self.inner_learning_rate,
            betas=(0, 0.999)
        )
        return ModelArgs(
            self.model_class,
            inner_optimizer_args,
            num_classes=self.num_classes_per_client
        )

    def get_meta_model_args(self):
        dummy_optimizer_args = OptimizerArgs(
            optimizer_class=optim.SGD
        )
        return ModelArgs(
            self.model_class,
            dummy_optimizer_args,
            num_classes=self.num_classes_per_client
        )

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

    args = argument_parser().parse_args()
    RANDOM = random.Random(args.seed)

    reptile_args = ReptileTrainingArgs(
        model_class=OmniglotLightning,
        inner_optimizer=optim.Adam,
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
            f"{context.name};seed{args.seed};"
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
        context=context,
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
            # log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'),
            #                 experiment_logger, global_step=i+1)
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
            # log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'),
            #                 experiment_logger, global_step=i+1)
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
        print(f"{label} accuracy: {torch.mean(result.get('test/acc'))}")


if __name__ == '__main__':
    def run():
        context = ExperimentContext(name='reptile_federated')
        run_reptile(context=context)

    run()
