"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import numpy as np
import tensorflow.compat.v1 as tf
from itertools import cycle

from mlmi.reptile.dataloading_ours_model_nichol.variables_2 import (interpolate_vars, average_vars,
                                                                    VariableState)

from mlmi.struct import OptimizerArgs

from functools import partial


# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, participant_name, optimizer_args: OptimizerArgs):
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
        self.minimize_op = optimizer_args.optimizer_class(**optimizer_args.optimizer_kwargs).minimize(self.loss)

        self.participant_name = participant_name


class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, session, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

    # pylint: disable=R0913,R0914
    def train_step(self,
                   meta_batch,
                   input_ph,
                   label_ph,
                   minimize_op,
                   inner_iters,
                   meta_step_size):
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
        old_vars = self._model_state.export_variables()
        new_vars = []
        #print('starting training')
        for key, data_loader in meta_batch.items():
            #print(f"Training on client {key}")
            for i, batch in zip(range(inner_iters), cycle(data_loader)):
                inputs = []
                for t in batch[0]:
                    inputs.append(t.numpy())
                inputs = tuple(inputs)
                labels = tuple(batch[1].numpy())
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        new_vars = average_vars(new_vars)

        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))


    def evaluate(self,
                 train_data_loader,
                 test_data_loader,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 inner_iters):
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

        old_vars = self._full_state.export_variables()

        for i, batch in zip(range(inner_iters), cycle(train_data_loader)):
            inputs = []
            for t in batch[0]:
                inputs.append(t.numpy())
            inputs = tuple(inputs)
            labels = tuple(batch[1].numpy())
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})

        _test_set = list(test_data_loader)[0]
        inputs = []
        for t in _test_set[0]:
            inputs.append(t.numpy())
        labels = list(_test_set[1].numpy())
        test_set = list(zip(inputs, labels))

        test_preds = self._test_predictions(test_set, input_ph, predictions)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        self._full_state.import_variables(old_vars)
        return num_correct

    def _test_predictions(self, test_set, input_ph, predictions):
        inputs, _ = zip(*test_set)
        return self.session.run(predictions, feed_dict={input_ph: inputs})
