import argparse
import sys

from mlmi.clustering import flatten_model_parameter
from mlmi.reptile.model import OmniglotLightning

sys.path.insert(0, 'C:\\Users\\Richard\\Desktop\\Informatik\\Semester_5\\MLMI\\git\\mlmi-federated-learning')
import random

import torch
from pytorch_lightning.loggers import LightningLoggerBase

from mlmi.reptile.args import argument_parser
from mlmi.log import getLogger
from mlmi.reptile.omniglot import load_omniglot_datasets
from mlmi.structs import ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger, evaluate_local_models, fix_random_seeds

from mlmi.reptile.reptile_original.models import OmniglotModel
from mlmi.reptile.reptile_original.variables import weight_decay
from mlmi.reptile.reptile_original_adapted.reptile import ReptileForFederatedData
from mlmi.reptile.reptile_original_adapted.eval import evaluate
import tensorflow.compat.v1 as tf
from mlmi.reptile.omniglot import read_dataset, split_dataset, augment_dataset


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


def convert_to_tensorflow_omniglot_model_state(model_dict):
    flat_model = flatten_model_parameter(model_dict, list(model_dict.keys())).numpy()
    pos = 0
    output_list = []
    channels = (1, 64, 64, 64)
    for c, _ in zip(channels, range(4)):
        next_pos = pos + 3*3*64*c
        conv_w = flat_model[pos:next_pos].reshape((3, 3, c, 64))
        output_list.append(conv_w)
        pos = next_pos
        next_pos = pos + 64
        conv_b = flat_model[pos:next_pos].reshape((64,))
        output_list.append(conv_b)
        pos = next_pos
        next_pos = pos + 64
        bn_w = flat_model[pos:next_pos].reshape((64,))
        output_list.append(bn_w)
        pos = next_pos
        next_pos = pos + 64
        bn_b = flat_model[pos:next_pos].reshape((64,))
        output_list.append(bn_b)
        pos = next_pos
    next_pos = pos + 256*5
    dense = flat_model[pos:next_pos].reshape((256, 5))
    output_list.append(dense)
    next_pos = pos + 5
    dense_out = flat_model[pos:next_pos].reshape((5,))
    output_list.append(dense_out)
    return output_list

def run_reptile(context: str, initial_model_state=None):
    fix_random_seeds(123123)
    args = argument_parser().parse_args()
    RANDOM = random.Random(args.seed)

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
    tf.disable_eager_execution()
    omniglot_train_clients, omniglot_test_clients = load_omniglot_datasets(
        str(data_dir.absolute()),
        num_clients_train=args.train_clients,
        num_clients_test=args.test_clients,
        num_classes_per_client=args.classes,
        num_shots_per_class=args.shots,
        inner_batch_size=args.inner_batch,
        tensorflow=True,
        random_seed=args.seed
    )

    dummy_optimizer_args = OptimizerArgs(
        optimizer_class=torch.optim.SGD
    )
    pytorch_model = OmniglotLightning(
        participant_name='variable_generator',
        optimizer_args=dummy_optimizer_args,
        num_classes=args.classes
    )
    pytorch_model_state = pytorch_model.state_dict()
    pytorch_as_tf_state = convert_to_tensorflow_omniglot_model_state(pytorch_model_state)

    model_kwargs = {
        'learning_rate': args.learning_rate
    }
    if args.sgd:
        model_kwargs['optimizer'] = tf.train.GradientDescentOptimizer
    model = OmniglotModel(
        num_classes=args.classes,
        **model_kwargs
    )

    with tf.Session() as sess:
        reptile = ReptileForFederatedData(
            session=sess,
            transductive=True,
            pre_step_op=weight_decay(1)
        )
        accuracy_ph = tf.placeholder(tf.float32, shape=())
        tf.summary.scalar('accuracy', accuracy_ph)
        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())

        tf.print(model)

        reptile.load_external_variable_state(pytorch_as_tf_state)

        for i in range(args.meta_iters):
            frac_done = i / args.meta_iters
            cur_meta_step_size = frac_done * args.meta_step_final + (1 - frac_done) * args.meta_step

            crange = cyclerange(
                start=i*args.meta_batch % len(omniglot_train_clients.train_data_local_dict),
                stop=(i+1)*args.meta_batch % len(omniglot_train_clients.train_data_local_dict),
                len=len(omniglot_train_clients.train_data_local_dict)
            )
            #print(f"Meta-step {i}: train clients {crange[0]}-{crange[-1]}")
            meta_batch = {
                k: omniglot_train_clients.train_data_local_dict[k] for k in crange
            }
            reptile.train_step(
                meta_batch=meta_batch,
                input_ph=model.input_ph,
                label_ph=model.label_ph,
                minimize_op=model.minimize_op,
                inner_iters=args.inner_iters,
                meta_step_size=cur_meta_step_size
            )
            if i % args.eval_interval == 0:
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
                        inner_iters=args.eval_iters
                    )
                    #summary = sess.run(merged, feed_dict={accuracy_ph: correct/num_classes_per_client})
                    accuracies.append(correct / args.classes)
                print('batch %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))

                # Write to TensorBoard
                experiment_logger.experiment.add_scalar('train-test/acc/{}/mean'.format('global_model'), accuracies[0], global_step=i)
                experiment_logger.experiment.add_scalar('test-test/acc/{}/mean'.format('global_model'), accuracies[1], global_step=i)

        # Final evaluation on a sample of training/test clients
        for label, dataset in zip(
                ['Train', 'Test'],
                [omniglot_train_clients, omniglot_test_clients]
        ):
            keys = RANDOM.sample(
                dataset.train_data_local_dict.keys(),
                args.eval_samples
            )
            train_eval_sample = {
                k: dataset.train_data_local_dict[k] for k in keys
            }
            test_eval_sample = {
                k: dataset.test_data_local_dict[k] for k in keys
            }
            accuracy = evaluate(
                sess=sess,
                model=model,
                train_dataloaders=train_eval_sample,
                test_dataloaders=test_eval_sample,
                num_classes=args.classes,
                eval_inner_iters=args.eval_iters,
                transductive=True
            )
            experiment_logger.experiment.add_scalar(
                f'final_{label}_acc',
                accuracy,
                global_step=0
            )
            print(f"{label} accuracy: {accuracy}")

if __name__ == '__main__':
    def run():
        run_reptile(context='reptile_orig_dl_fed')

    run()
