"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import copy
import random
import numpy as np
import tensorflow.compat.v1 as tf
import torch
from itertools import cycle
import pytorch_lightning as pl

from .variables_3 import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                          VariableState)

from functools import partial

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, model, inner_train_args, eval_inner_train_args, variables=None, pre_step_op=None):
        self.model = model
        self.trainer = pl.Trainer(checkpoint_callback=False, limit_val_batches=0.0, **inner_train_args.kwargs)
        self.eval_trainer = pl.Trainer(checkpoint_callback=False, limit_val_batches=0.0, **eval_inner_train_args.kwargs)
        #self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        #self._full_state = VariableState(self.session,
        #                                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        #self._transductive = transductive
        #self._pre_step_op = pre_step_op

    # pylint: disable=R0913,R0914
    def train_step(self,
                   meta_batch,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        #print('saving old vars')
        old_vars = copy.deepcopy(self.model.state_dict())
        #print(f"Weight = {old_vars['model.conv2d_1.weight'][0, 0, 0, 0]}")
        new_vars = []
        #print('starting training')
        for key, data_loader in meta_batch.items():
            #print(f"Training on client {key}")
            self.trainer.fit(model=self.model, train_dataloader=data_loader)
            #for i, batch in zip(range(inner_iters), cycle(data_loader)):
            #    inputs = []
            #    for t in batch[0]:
            #        inputs.append(t.numpy())
            #    inputs = tuple(inputs)
            #    labels = tuple(batch[1].numpy())
            #    self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            #new_vars.append(self._model_state.export_variables())
            new_vars.append(copy.deepcopy(self.model.state_dict()))
            #print(f"  Weight client {key} = {self.model.state_dict()['model.conv2d_1.weight'][0, 0, 0, 0]}")
            #self._model_state.import_variables(old_vars)
            self.model.load_state_dict(old_vars)
            #print(f"  After update = {self.model.state_dict()['model.conv2d_1.weight'][0, 0, 0, 0]}")
        #print('Updating global model state')
        new_vars = average_vars(new_vars)

        #self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))
        self.model.load_state_dict(interpolate_vars(old_vars, new_vars, meta_step_size))
        #print(f"Weight after step = {self.model.state_dict()['model.conv2d_1.weight'][0, 0, 0, 0]}")

    def evaluate(self,
                 train_data_loader,
                 test_data_loader,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """

        #old_vars = self._full_state.export_variables()
        old_vars = copy.deepcopy(self.model.state_dict())

        #for i, batch in zip(range(inner_iters), cycle(train_data_loader)):
        #    inputs = []
        #    for t in batch[0]:
        #        inputs.append(t.numpy())
        #    inputs = tuple(inputs)
        #    labels = tuple(batch[1].numpy())
        #    if self._pre_step_op:
        #        self.session.run(self._pre_step_op)
        #    self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        self.eval_trainer.fit(model=self.model, train_dataloader=train_data_loader)

        #_test_set = list(test_data_loader)[0]
        #inputs = []
        #for t in _test_set[0]:
        #    inputs.append(t.numpy())
        #labels = list(_test_set[1].numpy())
        #test_set = list(zip(inputs, labels))

        inputs, labels = list(test_data_loader)[0]

        #test_preds = self._test_predictions(test_set, input_ph, predictions)
        test_preds = self.model(inputs).argmax(dim=1)
        #num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        num_correct = int(sum([pred == label for pred, label in zip(test_preds, labels)]))
        #self._full_state.import_variables(old_vars)
        self.model.load_state_dict(old_vars)
        return num_correct

    def _test_predictions(self, test_set, input_ph, predictions):
        inputs, _ = zip(*test_set)
        return self.session.run(predictions, feed_dict={input_ph: inputs})


def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)

def _mini_batches(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set
