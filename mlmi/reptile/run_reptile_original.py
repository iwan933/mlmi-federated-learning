"""
Train a model on Omniglot.
"""

"""
Runs:
  version_1: python -u run_omniglot.py --train-shots 5 --inner-batch 10
             --inner-iters 5 --learning-rate 0.001 --meta-step 1
             --meta-step-final 0 --meta-batch 5 --meta-iters 3000
             --eval-batch 10 --eval-iters 50 --checkpoint ckpt_o55
             --transductive
  version_2: python -u run_omniglot.py --train-shots 5 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 3000 --eval-batch 10 --eval-iters 50 --checkpoint ckpt_o55 --transductive --eval-samples 1000
             Train accuracy: 0.9598
             Test accuracy: 0.9552

"""

import sys
sys.path.insert(0, 'C:\\Users\\Richard\\Desktop\\Informatik\\Semester_5\\MLMI\\git\\mlmi-federated-learning')
import random

import tensorflow.compat.v1 as tf
# The usage of tensorflow in this code needs eager execution disabled
tf.disable_eager_execution()

from pytorch_lightning.loggers import TensorBoardLogger

from mlmi.settings import REPO_ROOT
from mlmi.settings import RUN_DIR
from mlmi.struct import ExperimentContext
from mlmi.reptile.reptile_original.args import (
    argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
)
from mlmi.reptile.reptile_original.eval import evaluate
from mlmi.reptile.reptile_original.models import OmniglotModel
from mlmi.reptile.reptile_original.omniglot import (
    read_dataset, split_dataset, augment_dataset
)
from mlmi.reptile.reptile_original.train import train

DATA_DIR = REPO_ROOT / 'data' / 'omniglot'

def main(context: ExperimentContext):
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    experiment_path = RUN_DIR / 'reptile' / (
        f"{context.name};{args.classes}-way{args.shots}-shot;"
        f"mlr{str(args.meta_step).replace('.', '')}"
        f"ilr{str(args.learning_rate).replace('.', '')}"
        f"is{args.inner_iters}"
    )
    experiment_logger = TensorBoardLogger(experiment_path.absolute())

    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    model = OmniglotModel(args.classes, **model_kwargs(args))

    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, experiment_logger, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        for label, dataset in zip(['Train', 'Test'], [train_set, test_set]):
            accuracy = evaluate(sess, model, dataset, **eval_kwargs)
            experiment_logger.experiment.add_scalar(
                f'final_{label}_acc',
                accuracy,
                global_step=0
            )
            print(f'{label} accuracy: {accuracy}')

if __name__ == '__main__':

    context = ExperimentContext(name='reptile_original')
    main(context=context)
